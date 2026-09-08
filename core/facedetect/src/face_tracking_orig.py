#!/usr/bin/env python3
"""HTPal 六轴机械臂低速人脸视觉跟踪。

启动后机械臂先到 ``core/detect_test_pos.md`` 定义的 HOME 位，然后开启
RealSense RGB-D 和 SCRFD 检测。按 ``c`` 切换跟踪，按 ``q`` 或 Ctrl+C 退出并回零。

跟踪状态机参考 VisionGrab，但实时跟踪采用速度级 QP；HOME、退出动作和目标几何关系按 HTPal 定义：
屏幕中心是 link6 原点，屏幕中心与人脸的目标法向距离为 0.5 m，
屏幕姿态保持启动时姿态。默认补偿上沿相机的切向安装偏移，优先让
人脸靠近相机光轴；此时不再要求屏幕中心与人脸严格共线。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pyrealsense2 as rs
from insightface.model_zoo import get_model
from scipy.spatial.transform import Rotation

from tracking_qp import build_least_squares_qp, solve_bounded_qp


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MODEL = (
    REPO_ROOT
    / "core"
    / "facedetect"
    / "model"
    / "models"
    / "buffalo_sc"
    / "det_500m.onnx"
)
DEFAULT_POSITION_FILE = REPO_ROOT / "core" / "detect_test_pos.md"
DEFAULT_CONFIG_FILE = REPO_ROOT / "panthera_python" / "robot_param" / "Follower_tracking.yaml"
DEFAULT_CALIBRATION_FILE = REPO_ROOT / "panthera_python" / "config" / "hand_eye_calibration.json"
JOINT_COUNT = 6
DEFAULT_WINDOW = "HTPal face tracking"
CAMERA_TRANSLATION_LINK6 = np.array([0.0, 0.0, 0.16], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    """Parse runtime and conservative motion settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-motion", action="store_true", help="允许发送跟踪电机命令")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="SCRFD ONNX 权重路径")
    parser.add_argument("--position-file", type=Path, default=DEFAULT_POSITION_FILE, help="HOME 六关节位置文件")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE, help="Panthera 配置文件")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_FILE, help="手眼标定 JSON")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--det-thresh", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--screen-distance", type=float, default=0.5, help="屏幕中心到人脸的目标法向距离（米，非欧氏距离）")
    parser.add_argument(
        "--camera-centering-gain",
        type=float,
        default=1.0,
        help="补偿相机相对屏幕中心的平面内偏移（0=不补偿，1=完全补偿）",
    )
    parser.add_argument("--confirm-frames", type=int, default=3, help="首次检测/远距离重捕获所需连续帧数")
    parser.add_argument("--max-target-jump", type=float, default=0.25, help="相邻有效目标允许的最大三维跳变（米）")
    parser.add_argument("--target-timeout", type=float, default=0.5, help="目标数据有效时间（秒）")
    parser.add_argument("--lost-timeout", type=float, default=2.0, help="进入返回 HOME 的连续丢失时间（秒）")
    parser.add_argument("--control-period", type=float, default=0.01, help="控制周期（秒）")
    parser.add_argument(
        "--cartesian-step", type=float, default=0.002,
        help="tracking 每周期允许推进的笛卡尔步长（米，默认 0.002）",
    )
    parser.add_argument("--max-joint-speed", type=float, default=0.15, help="跟踪最大关节速度（rad/s）")
    parser.add_argument("--max-joint-accel", type=float, default=0.30, help="跟踪最大关节加速度（rad/s²）")
    parser.add_argument("--qp-position-gain", type=float, default=4.5, help="QP 末端位置反馈增益")
    parser.add_argument("--qp-rotation-gain", type=float, default=4.0, help="QP 末端姿态反馈增益")
    parser.add_argument("--max-cartesian-speed", type=float, default=0.15, help="QP 末端线速度上限（m/s）")
    parser.add_argument("--max-angular-speed", type=float, default=0.50, help="QP 末端角速度上限（rad/s）")
    parser.add_argument("--qp-home-gain", type=float, default=0.8, help="QP HOME 构型偏好增益")
    parser.add_argument("--qp-home-speed", type=float, default=0.05, help="回 HOME 偏好速度上限（rad/s）")
    parser.add_argument("--j1-home-deviation", type=float, default=0.75, help="J1 相对 HOME 的最大偏离范围（rad）")
    parser.add_argument("--j1-horizontal-threshold", type=float, default=0.01, help="触发 J1 左右优先的屏幕水平速度阈值（m/s）")
    parser.add_argument("--qp-slowdown-distance", type=float, default=0.15, help="关节限位线性减速距离（rad）")
    parser.add_argument("--tracking-rebase-threshold", type=float, default=0.08, help="实际关节与命令偏差保护阈值（rad）")
    parser.add_argument(
        "--joint-limit-margin",
        type=float,
        default=0.1,
        help="跟踪时在每个关节限位前预留的安全余量（rad）",
    )
    parser.add_argument(
        "--limit-release-hysteresis",
        type=float,
        default=0.03,
        help="法向 LIMIT_HOLD 解除所需的反向位移滞回（米）",
    )
    parser.add_argument(
        "--ik-eps",
        type=float,
        default=0.03,
        help="固定 link6 姿态 IK 的可接受综合误差（默认：0.03，约厘米级）",
    )
    parser.add_argument(
        "--cartesian-ik-eps",
        type=float,
        default=0.0005,
        help="笛卡尔小步 IK 的严格收敛误差（米，默认 0.0005）",
    )
    parser.add_argument("--ik-deadzone", type=float, default=0.01, help="重新求 IK 的目标位移死区（米）")
    parser.add_argument("--depth-radius", type=int, default=3, help="深度中值采样半径（像素）")
    parser.add_argument("--min-depth", type=float, default=0.08, help="有效深度下限（米）")
    parser.add_argument("--max-depth", type=float, default=2.0, help="有效深度上限（米）")
    parser.add_argument("--home-velocity", type=float, default=0.20, help="启动/丢失返回 HOME 的速度（rad/s）")
    parser.add_argument("--exit-velocity", type=float, default=0.20, help="退出回零速度（rad/s）")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate files, dimensions, and safety limits."""
    for path, label in (
        (args.model, "SCRFD 模型"),
        (args.position_file, "HOME 位置文件"),
        (args.config, "机械臂配置"),
        (args.calibration, "手眼标定文件"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}不存在：{path}")
    if min(args.width, args.height, args.fps, args.det_size) <= 0:
        raise ValueError("图像尺寸、帧率和 det-size 必须大于 0")
    if not 0.0 <= args.det_thresh <= 1.0:
        raise ValueError("--det-thresh 必须位于 [0, 1] 范围内")
    if not 0.0 <= args.camera_centering_gain <= 1.0:
        raise ValueError("--camera-centering-gain 必须位于 [0, 1] 范围内")
    if args.warmup < 0 or args.depth_radius < 0:
        raise ValueError("--warmup 和 --depth-radius 不能为负数")
    if args.confirm_frames <= 0:
        raise ValueError("--confirm-frames 必须大于 0")
    if min(
        args.screen_distance,
        args.target_timeout,
        args.lost_timeout,
        args.control_period,
        args.max_joint_speed,
        args.max_joint_accel,
        args.qp_position_gain,
        args.qp_rotation_gain,
        args.max_cartesian_speed,
        args.max_angular_speed,
        args.qp_home_gain,
        args.qp_home_speed,
        args.j1_home_deviation,
        args.j1_horizontal_threshold,
        args.qp_slowdown_distance,
        args.tracking_rebase_threshold,
        args.limit_release_hysteresis,
        args.ik_eps,
        args.cartesian_ik_eps,
        args.ik_deadzone,
        args.min_depth,
        args.max_depth,
        args.home_velocity,
        args.exit_velocity,
        args.max_target_jump,
        args.cartesian_step,
    ) <= 0.0:
        raise ValueError("距离、时间和速度/加速度参数必须大于 0")
    if args.joint_limit_margin < 0.0:
        raise ValueError("--joint-limit-margin 不能为负数")
    if args.min_depth >= args.max_depth:
        raise ValueError("--min-depth 必须小于 --max-depth")
    if args.cartesian_ik_eps > args.ik_eps:
        raise ValueError("--cartesian-ik-eps 不应大于 --ik-eps")


def load_home_position(path: Path) -> np.ndarray:
    """Read six joint positions from the existing detection test position file."""
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from hold_detect_test_pos import load_target_position

    return np.asarray(load_target_position(path.resolve()), dtype=np.float64)


def load_calibration_rotation(path: Path) -> np.ndarray:
    """Load only the VisionGrab rotation; translation is overridden by HTPal geometry."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload.get("T_tcp_camera"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"T_tcp_camera 不是有效的 4x4 矩阵：{path}")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError(f"手眼标定旋转矩阵不是正交矩阵：{path}")
    return rotation


def load_detector(model_path: Path, det_size: int, det_thresh: float):
    """Load SCRFD, preferring CUDA and falling back to CPU."""
    available = ort.get_available_providers()
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]
    if not providers:
        raise RuntimeError(f"ONNX Runtime 没有可用的 CUDA/CPU provider：{available}")
    detector = get_model(str(model_path), providers=providers)
    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    detector.prepare(ctx_id=ctx_id, input_size=(det_size, det_size), det_thresh=det_thresh)
    selected = "CUDAExecutionProvider" if ctx_id == 0 else "CPUExecutionProvider"
    print(f"SCRFD provider：{selected}；模型：{model_path}")
    return detector


def report_joint_safety_margin(robot, home_position: np.ndarray, margin: float) -> None:
    """Print raw/safe limits and the HOME clearance to the safe limits."""
    limits = getattr(robot, "joint_limits", None)
    if limits is None:
        return
    lower = np.asarray(limits["lower"], dtype=np.float64)
    upper = np.asarray(limits["upper"], dtype=np.float64)
    safe_lower = lower + margin
    safe_upper = upper - margin
    print(f"关节安全余量：{margin:.3f} rad")
    for index, (q, lo, hi, safe_lo, safe_hi) in enumerate(
        zip(home_position, lower, upper, safe_lower, safe_upper), start=1
    ):
        clearance = min(float(q - safe_lo), float(safe_hi - q))
        print(
            f"  J{index}: HOME={q:+.3f} | 原始=[{lo:+.3f}, {hi:+.3f}] "
            f"安全=[{safe_lo:+.3f}, {safe_hi:+.3f}] | "
            f"距安全边界={clearance:+.3f} rad"
        )


def sample_depth_m(depth_frame, center_x: float, center_y: float, radius: int, min_depth: float, max_depth: float):
    """Return a valid median depth around a facial landmark center."""
    width, height = depth_frame.get_width(), depth_frame.get_height()
    x0 = max(0, int(round(center_x)) - radius)
    x1 = min(width - 1, int(round(center_x)) + radius)
    y0 = max(0, int(round(center_y)) - radius)
    y1 = min(height - 1, int(round(center_y)) + radius)
    values = []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            distance = float(depth_frame.get_distance(x, y))
            if math.isfinite(distance) and min_depth <= distance <= max_depth:
                values.append(distance)
    return float(np.median(values)) if values else None


def deproject_pixel(center_x: float, center_y: float, depth_m: float, intrinsics) -> np.ndarray:
    """Deproject an aligned color pixel into the RealSense color-camera frame."""
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(center_x), float(center_y)], float(depth_m))
    return np.asarray(point, dtype=np.float64)


def detect_faces(detector, color: np.ndarray, depth_frame, intrinsics, args: argparse.Namespace):
    """Detect faces and attach a depth-backed camera point to each detection."""
    bboxes, keypoints = detector.detect(color)
    detections = []
    if bboxes is None:
        return detections
    image_height, image_width = color.shape[:2]
    for index, bbox in enumerate(bboxes):
        x1, y1, x2, y2, score = [float(value) for value in bbox[:5]]
        x1 = max(0.0, min(image_width - 1.0, x1))
        x2 = max(0.0, min(image_width - 1.0, x2))
        y1 = max(0.0, min(image_height - 1.0, y1))
        y2 = max(0.0, min(image_height - 1.0, y2))
        if keypoints is not None and index < len(keypoints):
            points = np.asarray(keypoints[index], dtype=np.float64).reshape(-1, 2)
            center_x, center_y = np.mean(points, axis=0)
        else:
            center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            points = None
        depth_m = sample_depth_m(
            depth_frame, center_x, center_y, args.depth_radius, args.min_depth, args.max_depth
        )
        point_camera = (
            deproject_pixel(center_x, center_y, depth_m, intrinsics)
            if depth_m is not None else None
        )
        detections.append({
            "bbox": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            "center": (float(center_x), float(center_y)),
            "score": score,
            "keypoints": points,
            "depth_m": depth_m,
            "point_camera": point_camera,
        })
    return detections


def select_nearest_target(detections):
    """Select the nearest valid face, matching VisionGrab's single-target policy."""
    valid = [
        item for item in detections
        if item["point_camera"] is not None and item["point_camera"][2] > 0.0
    ]
    return min(valid, key=lambda item: item["point_camera"][2], default=None)


def draw_detections(image: np.ndarray, detections, selected, status: str, fps: float) -> None:
    """Draw all detections and current state without changing control state."""
    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        is_selected = item is selected
        color = (0, 255, 255) if is_selected else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cx, cy = item["center"]
        cv2.circle(image, (int(round(cx)), int(round(cy))), 4, (0, 0, 255), -1, cv2.LINE_AA)
        if item["keypoints"] is not None:
            for px, py in item["keypoints"]:
                cv2.circle(image, (int(round(px)), int(round(py))), 3, (0, 128, 255), -1, cv2.LINE_AA)
        label = f"face {item['score']:.2f}"
        if item["depth_m"] is not None:
            label += f" {item['depth_m']:.3f}m"
        cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    cv2.putText(image, f"{status} | FPS {fps:.1f} | C: toggle | Q: quit", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)


class TargetGate:
    """Reject isolated detections and require consistency when a target jumps."""

    def __init__(self, confirm_frames: int, max_jump: float):
        self.confirm_frames = int(confirm_frames)
        self.max_jump = float(max_jump)
        self.accepted = None
        self.pending = None
        self.pending_count = 0

    def update(self, point_camera: np.ndarray | None):
        """Return a target only after first/reacquired observations are consistent."""
        if point_camera is None:
            self.pending = None
            self.pending_count = 0
            return None

        point = np.asarray(point_camera, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            self.pending = None
            self.pending_count = 0
            return None

        if self.accepted is not None and np.linalg.norm(point - self.accepted) <= self.max_jump:
            self.accepted = point.copy()
            self.pending = None
            self.pending_count = 0
            return self.accepted.copy()

        if self.pending is None or np.linalg.norm(point - self.pending) > self.max_jump:
            self.pending = point.copy()
            self.pending_count = 1
        else:
            self.pending = point.copy()
            self.pending_count += 1

        if self.pending_count >= self.confirm_frames:
            self.accepted = self.pending.copy()
            self.pending = None
            self.pending_count = 0
            return self.accepted.copy()
        return None


def step_joint_profile(command_joint, command_velocity, target_joint, dt, max_speed, max_accel):
    """VisionGrab-style velocity/acceleration-limited joint profile step."""
    command_joint = np.asarray(command_joint, dtype=np.float64)
    command_velocity = np.asarray(command_velocity, dtype=np.float64)
    target_joint = np.asarray(target_joint, dtype=np.float64)
    dt = float(np.clip(dt, 1e-4, 0.05))
    error = target_joint - command_joint
    braking_speed = np.sqrt(2.0 * max_accel * np.abs(error))
    desired_velocity = np.sign(error) * np.minimum(max_speed, braking_speed)
    velocity_step = np.clip(desired_velocity - command_velocity, -max_accel * dt, max_accel * dt)
    next_velocity = command_velocity + velocity_step
    next_joint = command_joint + next_velocity * dt
    # 离散积分可能在目标附近越过关节目标，直接钳制避免把 1e-3 rad
    # 级别的数值 overshoot 发送给 SDK 后触发原始限位拒绝。
    crossed = (
        (target_joint - command_joint) * (target_joint - next_joint) <= 0.0
    )
    next_joint = np.where(crossed, target_joint, next_joint)
    next_velocity = np.where(crossed, 0.0, next_velocity)
    return next_joint, next_velocity


def make_transforms(fk: dict, camera_rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build T_base_link6 and actual T_link6_camera from current FK."""
    t_base_link6 = np.asarray(fk["transform"], dtype=np.float64).copy()
    t_link6_camera = np.eye(4, dtype=np.float64)
    t_link6_camera[:3, :3] = camera_rotation
    t_link6_camera[:3, 3] = CAMERA_TRANSLATION_LINK6
    return t_base_link6, t_base_link6 @ t_link6_camera


def screen_normal_in_base(screen_rotation: np.ndarray, camera_rotation: np.ndarray) -> np.ndarray:
    """Return the common camera-optical-axis/screen-normal direction in base coordinates."""
    normal = np.asarray(screen_rotation, dtype=np.float64) @ np.asarray(
        camera_rotation, dtype=np.float64
    )[:, 2]
    norm = np.linalg.norm(normal)
    if not np.isfinite(norm) or norm <= 1e-9:
        raise ValueError("相机光轴方向无效，无法计算屏幕法向")
    return normal / norm


def face_to_screen_target(
    point_camera: np.ndarray,
    fk: dict,
    camera_rotation: np.ndarray,
    screen_distance: float,
    screen_rotation: np.ndarray,
    camera_centering_gain: float = 1.0,
) -> np.ndarray:
    """计算保留法向距离、补偿上沿相机切向偏移的屏幕目标位置。

    gain=0 时屏幕中心正对人脸；gain=1 时目标姿态下人脸位于相机光轴。
    此处只保证目标几何关系，不保证 IK 可达、运动途中的视野或人体间距。
    """
    _, t_base_camera = make_transforms(fk, camera_rotation)
    p_face_h = t_base_camera @ np.append(np.asarray(point_camera, dtype=np.float64), 1.0)
    # 相机光轴与屏幕法向同向，但相机光学中心与屏幕中心有独立的位置偏移。
    # camera_rotation[:, 2] 是相机 +Z 在 link6 中的方向，screen_rotation 将其变换到 base。
    screen_normal_base = screen_normal_in_base(screen_rotation, camera_rotation)
    camera_axis_link6 = screen_normal_in_base(np.eye(3), camera_rotation)
    camera_offset_tangent_link6 = CAMERA_TRANSLATION_LINK6 - camera_axis_link6 * float(
        np.dot(CAMERA_TRANSLATION_LINK6, camera_axis_link6)
    )
    camera_offset_tangent_base = (
        np.asarray(screen_rotation, dtype=np.float64) @ camera_offset_tangent_link6
    )
    return (
        p_face_h[:3]
        - screen_normal_base * screen_distance
        - float(camera_centering_gain) * camera_offset_tangent_base
    )


class TrackingController:
    """State machine and low-speed command producer shared with the vision loop."""

    def __init__(
        self,
        robot,
        home_position: np.ndarray,
        camera_rotation: np.ndarray,
        screen_rotation: np.ndarray,
        args: argparse.Namespace,
    ):
        self.robot = robot
        self.home_position = home_position
        self.camera_rotation = camera_rotation
        self.screen_rotation = np.asarray(screen_rotation, dtype=np.float64).copy()
        self.args = args
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.tracking_enabled = False
        self.latest_point_camera = None
        self.latest_target_time = None
        self.status = "HOLD"
        self.command_joint = np.asarray(robot.get_current_pos(), dtype=np.float64)
        self.command_velocity = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.cartesian_command_ready = False
        self.cartesian_command_velocity = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.desired_joint = self.command_joint.copy()
        self.last_ik_target = None
        self.smoothed_target = None
        self.lost_since = None
        self.normal_saturation = None
        self.normal_saturation_direction = 0.0
        self.limit_hold_reported = False
        self.limit_hold_reason = None
        self.last_ik_failure = None
        self.last_ik_report_time = 0.0
        self.thread = threading.Thread(target=self._loop, name="htpal-tracking-control", daemon=True)
        configured_limits = np.asarray(getattr(robot, "velocity_limits", [1.0] * JOINT_COUNT), dtype=np.float64)
        self.max_speed = np.minimum(configured_limits, args.max_joint_speed)
        self.max_accel = np.full(JOINT_COUNT, args.max_joint_accel, dtype=np.float64)
        limits = getattr(robot, "joint_limits", None)
        if limits is None:
            raise RuntimeError("跟踪限位保护需要 Panthera 配置中的 robot.joint_limits")
        self.safe_lower = np.asarray(limits["lower"], dtype=np.float64) + args.joint_limit_margin
        self.safe_upper = np.asarray(limits["upper"], dtype=np.float64) - args.joint_limit_margin
        if args.joint_limit_margin == 0.0:
            print("警告：已关闭额外关节安全余量，仅保留 SDK 原始关节限位。", file=sys.stderr)
        if self.safe_lower.shape != (JOINT_COUNT,) or self.safe_upper.shape != (JOINT_COUNT,):
            raise ValueError("robot.joint_limits 必须包含六个关节的上下限")
        if np.any(self.safe_lower >= self.safe_upper):
            raise ValueError("joint-limit-margin 过大，导致没有可用的安全关节范围")

    def set_tracking_enabled(self, enabled: bool) -> None:
        with self.lock:
            self.tracking_enabled = bool(enabled)
            self.last_ik_target = None
            self.cartesian_command_ready = False
            self.cartesian_command_velocity.fill(0.0)
            self.smoothed_target = None
            self.lost_since = None
            self.normal_saturation = None
            self.normal_saturation_direction = 0.0
            self.limit_hold_reported = False
            self.limit_hold_reason = None
            if not enabled:
                self.desired_joint = self.command_joint.copy()
                self.status = "HOLD"

    def update_target(self, point_camera: np.ndarray | None, timestamp: float | None) -> None:
        with self.lock:
            if point_camera is not None and timestamp is not None:
                self.latest_point_camera = np.asarray(point_camera, dtype=np.float64).copy()
                self.latest_target_time = float(timestamp)

    def snapshot(self) -> tuple[str, bool, float | None]:
        with self.lock:
            age = None if self.latest_target_time is None else time.monotonic() - self.latest_target_time
            return self.status, self.tracking_enabled, age

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _loop(self) -> None:
        previous_time = time.monotonic()
        while not self.stop_event.is_set():
            cycle_start = time.perf_counter()
            now = time.monotonic()
            dt = now - previous_time
            previous_time = now
            with self.lock:
                enabled = self.tracking_enabled
                point_camera = None if self.latest_point_camera is None else self.latest_point_camera.copy()
                target_time = self.latest_target_time
                status = self.status
                command_joint = self.command_joint.copy()
                command_velocity = self.command_velocity.copy()
            fresh = point_camera is not None and target_time is not None and now - target_time <= self.args.target_timeout

            try:
                if not enabled:
                    with self.lock:
                        self.status = "HOLD"
                        self.desired_joint = self.command_joint.copy()
                        self.cartesian_command_ready = False
                        self.cartesian_command_velocity.fill(0.0)
                        self.last_ik_target = None
                        self.smoothed_target = None
                        self.lost_since = None
                elif fresh:
                    self._handle_fresh_target(
                        point_camera, status, command_joint, command_velocity, dt
                    )
                else:
                    self._handle_lost_target(now)

                with self.lock:
                    if self.status == "RETURNING" and fresh:
                        self.status = "TRACKING"
                    if self.cartesian_command_ready and fresh and self.status in (
                        "TRACKING", "TRACKING_LIMITED", "QP_FAIL", "REBASE"
                    ):
                        self.command_joint = self.desired_joint.copy()
                        self.command_velocity = self.cartesian_command_velocity.copy()
                        self.cartesian_command_ready = False
                    else:
                        self.command_joint, self.command_velocity = step_joint_profile(
                            self.command_joint,
                            self.command_velocity,
                            self.desired_joint,
                            dt,
                            self.max_speed,
                            self.max_accel,
                        )
                    if self.status == "RETURNING":
                        at_home = (
                            np.max(np.abs(self.command_joint - self.home_position)) <= 0.01
                            and np.max(np.abs(self.command_velocity)) <= 0.02
                        )
                        if at_home:
                            self.command_joint = self.home_position.copy()
                            self.command_velocity.fill(0.0)
                            self.desired_joint = self.home_position.copy()
                            self.status = "HOME"
                            self.lost_since = None
                    command = self.command_joint.copy()
                    velocity = self.command_velocity.copy()
                    status = self.status
                if self.args.enable_motion:
                    if (
                        command.shape != (JOINT_COUNT,)
                        or velocity.shape != (JOINT_COUNT,)
                        or not np.all(np.isfinite(command))
                        or not np.all(np.isfinite(velocity))
                    ):
                        raise RuntimeError("QP 输出包含非有限关节命令")
                    command = np.clip(command, self.safe_lower, self.safe_upper)
                    velocity = np.clip(velocity, -self.max_speed, self.max_speed)
                    accepted = self.robot.Joint_Pos_Vel(
                        command.tolist(), velocity.tolist(), self.robot.max_torque.tolist(), iswait=False
                    )
                    if not accepted:
                        raise RuntimeError("跟踪关节命令被 SDK 拒绝")
            except Exception as exc:
                with self.lock:
                    self.status = "ERROR"
                    self.desired_joint = self.command_joint.copy()
                    self.command_velocity.fill(0.0)
                print(f"\n跟踪控制异常：{exc}", file=sys.stderr)
                self.stop_event.set()

            remaining = self.args.control_period - (time.perf_counter() - cycle_start)
            if remaining > 0:
                time.sleep(remaining)

    def _handle_fresh_target(
        self,
        point_camera: np.ndarray,
        previous_status: str,
        command_joint: np.ndarray,
        command_velocity: np.ndarray,
        dt: float,
    ) -> None:
        actual_q = np.asarray(self.robot.get_current_pos(), dtype=np.float64)
        if actual_q.shape != (JOINT_COUNT,) or not np.all(np.isfinite(actual_q)):
            raise RuntimeError("无法读取当前六关节反馈")
        if np.max(np.abs(actual_q - command_joint)) > self.args.tracking_rebase_threshold:
            rebased = np.clip(actual_q, self.safe_lower, self.safe_upper)
            with self.lock:
                self.command_joint = rebased.copy()
                self.desired_joint = rebased.copy()
                self.command_velocity = np.zeros(JOINT_COUNT, dtype=np.float64)
                self.cartesian_command_velocity = np.zeros(JOINT_COUNT, dtype=np.float64)
                self.cartesian_command_ready = True
                self.status = "REBASE"
            if time.monotonic() - self.last_ik_report_time >= 1.0:
                print("实际关节与软件命令偏差过大，已安全重新同步。", file=sys.stderr)
                self.last_ik_report_time = time.monotonic()
            return
        actual_fk = self.robot.forward_kinematics(actual_q)
        command_fk = self.robot.forward_kinematics(command_joint)
        if actual_fk is None or command_fk is None:
            raise RuntimeError("无法计算当前 link6 正运动学")
        raw_target = face_to_screen_target(
            point_camera,
            actual_fk,
            self.camera_rotation,
            self.args.screen_distance,
            self.screen_rotation,
            self.args.camera_centering_gain,
        )
        with self.lock:
            if previous_status == "RETURNING":
                self.last_ik_target = None
                self.smoothed_target = None
                self.normal_saturation = None
                self.normal_saturation_direction = 0.0
                self.limit_hold_reported = False
                self.limit_hold_reason = None
            if self.smoothed_target is None:
                self.smoothed_target = raw_target.copy()
            else:
                alpha = 0.12
                self.smoothed_target = alpha * raw_target + (1.0 - alpha) * self.smoothed_target

            target_position = self.smoothed_target.copy()
            desired_twist = self._desired_twist(command_fk, target_position)
            qdot, solver_status, _, active = self._solve_velocity_qp(
                command_joint, command_velocity, command_fk, desired_twist, dt
            )
            if solver_status == "SOLVED":
                qdot = np.asarray(qdot, dtype=np.float64)
                q_next = command_joint + qdot * float(np.clip(dt, 1e-4, 0.05))
                q_next = np.clip(q_next, self.safe_lower, self.safe_upper)
                self.desired_joint = q_next
                self.cartesian_command_velocity = qdot
                self.cartesian_command_ready = True
                self.status = "TRACKING_LIMITED" if np.any(active != 0) else "TRACKING"
            else:
                self.desired_joint = command_joint.copy()
                self.cartesian_command_velocity = np.zeros(JOINT_COUNT, dtype=np.float64)
                self.cartesian_command_ready = True
                self.status = "QP_FAIL"
                if time.monotonic() - self.last_ik_report_time >= 1.0:
                    print(f"速度级 QP 求解失败（{solver_status}），已减速保持当前位置。", file=sys.stderr)
                    self.last_ik_report_time = time.monotonic()
            self.lost_since = None

    def _desired_twist(self, current_fk: dict, target_position: np.ndarray) -> np.ndarray:
        current_position = np.asarray(current_fk["position"], dtype=np.float64)
        current_rotation = np.asarray(current_fk["rotation"], dtype=np.float64)
        linear = self.args.qp_position_gain * (np.asarray(target_position) - current_position)
        linear_norm = np.linalg.norm(linear)
        if linear_norm > self.args.max_cartesian_speed:
            linear *= self.args.max_cartesian_speed / linear_norm
        rotation_error = Rotation.from_matrix(self.screen_rotation @ current_rotation.T).as_rotvec()
        angular = self.args.qp_rotation_gain * rotation_error
        angular_norm = np.linalg.norm(angular)
        if angular_norm > self.args.max_angular_speed:
            angular *= self.args.max_angular_speed / angular_norm
        return np.concatenate((linear, angular))

    def _solve_velocity_qp(self, q_command, qdot_previous, current_fk, desired_twist, dt):
        jacobian = np.asarray(self.robot.get_jacobian(q_command), dtype=np.float64)
        if jacobian.shape != (6, JOINT_COUNT) or not np.all(np.isfinite(jacobian)):
            return np.zeros(JOINT_COUNT), "NUMERICAL_FAILURE", 0, np.zeros(JOINT_COUNT, dtype=int)
        linear = np.asarray(desired_twist[:3], dtype=np.float64)
        # 只有屏幕水平轴的速度才触发 J1 优先。此前使用全部切向速度，
        # 会把上下运动也误判成左右运动，导致 J1 持续单向偏转。
        horizontal_axis = np.asarray(self.screen_rotation[:, 0], dtype=np.float64)
        horizontal_speed = abs(float(np.dot(horizontal_axis, linear)))
        # HOME 是偏好构型，不是把每个关节强行拉回 HOME。J1 在水平
        # 跟踪中应当主动偏离 HOME 来完成转向，因此它的 HOME 拉力只
        # 在非水平运动时保留；J2~J6 继续保持 HOME 风格。
        velocity_weights = np.array([0.35, 1.0, 1.0, 4.0, 4.0, 4.0], dtype=np.float64)
        home_weights = np.array([0.0, 1.0, 1.0, 4.0, 4.0, 4.0], dtype=np.float64)
        accel_weights = np.array([0.4, 0.6, 0.6, 1.5, 1.5, 1.5], dtype=np.float64)
        if horizontal_speed > self.args.j1_horizontal_threshold:
            velocity_weights[0] = 0.12
            home_weights[0] = 0.0
        else:
            # 上下/前后运动时 J1 只做少量必要补偿，并缓慢回到 HOME。
            velocity_weights[0] = 2.0
            home_weights[0] = 2.5
        qdot_home = np.clip(
            self.args.qp_home_gain * (self.home_position - q_command),
            -self.args.qp_home_speed,
            self.args.qp_home_speed,
        )
        task_scale = np.sqrt(np.array([100.0] * 3 + [40.0] * 3, dtype=np.float64))
        rows = [np.diag(task_scale) @ jacobian, np.diag(np.sqrt(accel_weights)), np.diag(np.sqrt(velocity_weights)), np.diag(np.sqrt(home_weights))]
        targets = [task_scale * desired_twist, np.sqrt(accel_weights) * qdot_previous, np.zeros(JOINT_COUNT), np.sqrt(home_weights) * qdot_home]
        hessian, gradient = build_least_squares_qp(rows, targets)
        dt = float(np.clip(dt, 1e-4, 0.05))
        lower = np.maximum(-self.max_speed, qdot_previous - self.max_accel * dt)
        upper = np.minimum(self.max_speed, qdot_previous + self.max_accel * dt)
        position_lower = (self.safe_lower - q_command) / dt
        position_upper = (self.safe_upper - q_command) / dt
        lower = np.maximum(lower, position_lower)
        upper = np.minimum(upper, position_upper)
        slowdown = self.args.qp_slowdown_distance
        for index in range(JOINT_COUNT):
            upper[index] = min(upper[index], self.max_speed[index] * np.clip((self.safe_upper[index] - q_command[index]) / slowdown, 0.0, 1.0))
            lower[index] = max(lower[index], -self.max_speed[index] * np.clip((q_command[index] - self.safe_lower[index]) / slowdown, 0.0, 1.0))
        # J1 只允许在 HOME 附近的有限范围内工作。到达范围边缘时只冻结
        # 继续远离 HOME 的方向，反向回 HOME 仍然允许。
        j1_home = float(self.home_position[0])
        j1_delta = float(q_command[0] - j1_home)
        if j1_delta >= self.args.j1_home_deviation:
            upper[0] = min(upper[0], 0.0)
        elif j1_delta <= -self.args.j1_home_deviation:
            lower[0] = max(lower[0], 0.0)
        if np.any(lower > upper + 1e-9):
            # If the previous command velocity points outward at a boundary,
            # the acceleration box can conflict with the position box.  Safety
            # takes precedence: relax only that conflicting acceleration side
            # so that qdot=0 remains a valid braking command.
            lower = np.minimum(lower, 0.0)
            upper = np.maximum(upper, 0.0)
        qdot, status, iterations, active = solve_bounded_qp(hessian, gradient, lower, upper)
        position_active = np.zeros(JOINT_COUNT, dtype=int)
        position_active[(active == -1) & (qdot <= position_lower + 1e-7)] = -1
        position_active[(active == 1) & (qdot >= position_upper - 1e-7)] = 1
        return qdot, status, iterations, position_active

    def _report_limit_hold(
        self,
        joint_position: np.ndarray,
        requested_target: np.ndarray,
        requested_normal: float,
        limited_normal: float,
        failure: dict,
    ) -> None:
        """Print one compact diagnostic record for each normal-saturation episode."""
        if self.limit_hold_reported:
            return
        limits = getattr(self.robot, "joint_limits", None)
        if limits is None:
            return
        lower = np.asarray(limits["lower"], dtype=np.float64)
        upper = np.asarray(limits["upper"], dtype=np.float64)
        safe_clearance = np.minimum(
            np.asarray(joint_position, dtype=np.float64) - self.safe_lower,
            self.safe_upper - np.asarray(joint_position, dtype=np.float64),
        )
        raw_clearance = np.minimum(
            np.asarray(joint_position, dtype=np.float64) - lower,
            upper - np.asarray(joint_position, dtype=np.float64),
        )
        limiting = [
            index + 1
            for index, value in enumerate(safe_clearance)
            if value <= float(np.min(safe_clearance)) + 1e-6
        ]
        print(
            f"LIMIT_HOLD_{failure['kind']} 诊断："
            f"拒绝原因={failure['detail']}，"
            f"原请求越安全余量关节={failure['joints']}，"
            f"投影候选最近边界关节={limiting}（不等同于失败原因），"
            f"候选安全边界余量={float(np.min(safe_clearance)):+.3f} rad，"
            f"候选原始边界余量={float(np.min(raw_clearance)):+.3f} rad，"
            f"法向目标={requested_normal:+.3f}->{limited_normal:+.3f} m，"
            f"目标={np.round(requested_target, 3).tolist()}，"
            f"IK关节={np.round(joint_position, 3).tolist()}"
        )
        self.limit_hold_reported = True

    def _safe_inverse_kinematics(
        self, target_position: np.ndarray, init_q: np.ndarray, eps: float | None = None
    ):
        """Run fixed-link6-pose IK with an optional per-call tolerance."""
        ik_eps = self.args.ik_eps if eps is None else float(eps)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            joint_position = self.robot.inverse_kinematics(
                np.asarray(target_position, dtype=np.float64).tolist(),
                self.screen_rotation,
                np.asarray(init_q, dtype=np.float64),
                max_iter=1000,
                eps=ik_eps,
                damping=1e-2,
                adaptive_damping=True,
                multi_init=False,
            )
        if joint_position is None:
            self.last_ik_failure = {
                "kind": "IK_PROJECTION", "joints": [],
                "detail": "SDK未返回解；不能据此断言真实不可达或哪个关节越限",
            }
            return None
        joint_position = np.asarray(joint_position, dtype=np.float64)
        if joint_position.shape != (JOINT_COUNT,) or not np.all(np.isfinite(joint_position)):
            self.last_ik_failure = {
                "kind": "IK_PROJECTION", "joints": [], "detail": "SDK返回的关节解无效",
            }
            return None
        outside = (joint_position < self.safe_lower) | (joint_position > self.safe_upper)
        if np.any(outside):
            self.last_ik_failure = {
                "kind": "JOINT_MARGIN", "joints": (np.flatnonzero(outside) + 1).tolist(),
                "detail": "SDK已返回解，但解超出配置的关节安全余量范围",
            }
            return None
        self.last_ik_failure = None
        return joint_position

    def _search_reachable_target(
        self,
        target_position: np.ndarray,
        current_position: np.ndarray,
        normal: np.ndarray,
        init_q: np.ndarray,
    ):
        """Back off only along screen normal, preserving the requested lateral position."""
        desired_normal = float(np.dot(target_position, normal))
        current_normal = float(np.dot(current_position, normal))
        direction = float(np.sign(desired_normal - current_normal))
        if direction == 0.0:
            return None
        tangent = target_position - normal * desired_normal
        anchor = tangent + normal * current_normal
        anchor_q = self._safe_inverse_kinematics(anchor, init_q)
        if anchor_q is None:
            # 若横向目标本身不可达，沿当前末端到目标的整条线寻找最近可达点。
            low, high = 0.0, 1.0
            low_q = self._safe_inverse_kinematics(current_position, init_q)
            if low_q is None:
                return None
            best_position, best_q = current_position.copy(), low_q
            for _ in range(8):
                fraction = 0.5 * (low + high)
                candidate = current_position + fraction * (target_position - current_position)
                candidate_q = self._safe_inverse_kinematics(candidate, init_q)
                if candidate_q is None:
                    high = fraction
                else:
                    low = fraction
                    best_position, best_q = candidate, candidate_q
            return best_position, best_q

        low, high = current_normal, desired_normal
        if low > high:
            low, high = high, low
        best_position, best_q = anchor, anchor_q
        for _ in range(10):
            mid = 0.5 * (low + high)
            candidate = tangent + normal * mid
            candidate_q = self._safe_inverse_kinematics(candidate, init_q)
            if candidate_q is None:
                high = mid if direction > 0.0 else high
                low = low if direction > 0.0 else mid
            else:
                best_position, best_q = candidate, candidate_q
                low = mid if direction > 0.0 else low
                high = high if direction > 0.0 else mid
        return best_position, best_q

    def _handle_lost_target(self, now: float) -> None:
        with self.lock:
            if self.status == "HOME":
                self.desired_joint = self.home_position.copy()
                return
            if self.lost_since is None:
                self.lost_since = now
            self.desired_joint = self.command_joint.copy()
            if now - self.lost_since >= self.args.lost_timeout:
                self.status = "RETURNING"
                self.desired_joint = self.home_position.copy()
                self.last_ik_target = None
                self.smoothed_target = None
                self.normal_saturation = None
                self.normal_saturation_direction = 0.0
                self.limit_hold_reported = False
                self.limit_hold_reason = None
            else:
                self.status = "LOST"


def initialize_realsense(args: argparse.Namespace):
    """Start aligned RGB-D streams and return the color intrinsics."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = color_profile.get_intrinsics()
    align = rs.align(rs.stream.color)
    for _ in range(args.warmup):
        pipeline.wait_for_frames()
    print(f"RealSense RGB-D：{args.width}x{args.height}@{args.fps}，深度对齐到彩色")
    return pipeline, align, intrinsics


def return_to_zero(robot, velocity: float) -> None:
    """Exit action: return to true six-joint zero, then stop the motors."""
    zero = [0.0] * JOINT_COUNT
    print("退出：返回零位 [0, 0, 0, 0, 0, 0]...")
    try:
        reached = robot.Joint_Pos_Vel(
            zero,
            [velocity] * JOINT_COUNT,
            robot.max_torque.tolist(),
            iswait=True,
            tolerance=0.01,
            timeout=30.0,
        )
        if not reached:
            print("回零超时，发送停止命令。", file=sys.stderr)
    finally:
        robot.set_stop()


def main() -> int:
    args = parse_args()
    validate_args(args)
    home_position = load_home_position(args.position_file)
    camera_rotation = load_calibration_rotation(args.calibration)
    sdk_scripts = str((REPO_ROOT / "panthera_python" / "scripts").resolve())
    if sdk_scripts not in sys.path:
        sys.path.insert(0, sdk_scripts)
    from Panthera_lib import Panthera

    robot = Panthera(str(args.config.resolve()))
    if robot.motor_count != JOINT_COUNT or getattr(robot, "gripper_enabled", True):
        raise RuntimeError("跟踪脚本要求无夹爪六轴配置")
    pipeline = None
    try:
        print(f"HOME 位置：{home_position.tolist()}")
        report_joint_safety_margin(robot, home_position, args.joint_limit_margin)
        print("机械臂先移动到 HOME...")
        if not robot.Joint_Pos_Vel(
            home_position.tolist(), [args.home_velocity] * JOINT_COUNT,
            robot.max_torque.tolist(), iswait=True, tolerance=0.01, timeout=30.0
        ):
            raise RuntimeError("机械臂未能到达 detect_test_pos.md 的 HOME 位")

        detector = load_detector(args.model.resolve(), args.det_size, args.det_thresh)
        pipeline, align, intrinsics = initialize_realsense(args)
        fk_home = robot.forward_kinematics(home_position)
        if fk_home is None:
            raise RuntimeError("无法计算 HOME 位的 link6 正运动学")
        screen_rotation = np.asarray(fk_home["rotation"], dtype=np.float64)
    except KeyboardInterrupt:
        if pipeline is not None:
            pipeline.stop()
        return_to_zero(robot, args.exit_velocity)
        raise

    controller = TrackingController(
        robot, home_position, camera_rotation, screen_rotation, args
    )
    target_gate = TargetGate(args.confirm_frames, args.max_target_jump)
    controller.start()
    if args.enable_motion:
        print("已启用真实跟踪电机命令；按 c 开始/停止跟踪。")
    else:
        print("当前为观察模式；按 c 只模拟状态。需要真实运动请加 --enable-motion。")

    previous_time = time.perf_counter()
    fps = 0.0
    last_toggle_time = 0.0
    try:
        cv2.namedWindow(DEFAULT_WINDOW, cv2.WINDOW_NORMAL)
        while True:
            frames = align.process(pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            color = np.asanyarray(color_frame.get_data()).copy()
            detections = detect_faces(detector, color, depth_frame, intrinsics, args)
            selected = select_nearest_target(detections)
            accepted_point = target_gate.update(
                None if selected is None else selected["point_camera"]
            )
            controller.update_target(
                accepted_point,
                None if accepted_point is None else time.monotonic(),
            )
            status, enabled, _ = controller.snapshot()
            if status == "ERROR":
                raise RuntimeError("跟踪控制线程已停止，请检查终端中的控制异常")
            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps
            display_status = status if enabled else "HOLD"
            draw_detections(color, detections, selected, display_status, fps)
            cv2.imshow(DEFAULT_WINDOW, color)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("c"), ord("C")):
                now_monotonic = time.monotonic()
                if now_monotonic - last_toggle_time >= 0.30:
                    controller.set_tracking_enabled(not enabled)
                    print(f"跟踪开关：{'开启' if not enabled else '关闭'}")
                    last_toggle_time = now_monotonic
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        controller.stop()
        cv2.destroyAllWindows()
        pipeline.stop()
        return_to_zero(robot, args.exit_velocity)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C。", file=sys.stderr)
        raise SystemExit(130)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
