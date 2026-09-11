#!/usr/bin/env python3
"""HTPal 六轴机械臂低速人脸视觉跟踪。

启动后机械臂先到 ``core/detect_test_pos.md`` 定义的 HOME 位，然后开启
RealSense RGB-D 和 SCRFD 检测。按 ``c`` 切换跟踪，按 ``q`` 或 Ctrl+C 退出并回零。

跟踪状态机参考 VisionGrab，但实时跟踪采用速度级 QP；HOME、退出动作和目标几何关系按 HTPal 定义：
屏幕中心是 link6 原点，屏幕中心与人脸的目标法向距离为 0.5 m，
动态试验入口只跟踪人脸，不加载手势识别；水平偏航连续跟随，半躺时切换固定俯视档。
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--recline-pitch-angle", type=float, default=20.0, help="半躺档固定向下俯视角（度）")
    parser.add_argument("--recline-enter-pitch", type=float, default=15.0, help="进入半躺档所需的相对脸部俯仰（度）")
    parser.add_argument("--recline-height-drop", type=float, default=0.08, help="辅助确认半躺的双眼下降量（米）")
    parser.add_argument("--recline-exit-height", type=float, default=0.04, help="退出半躺档的双眼下降量滞回阈值（米）")
    parser.add_argument("--recline-enter-hold", type=float, default=0.5, help="进入半躺档的条件保持时间（秒）")
    parser.add_argument("--recline-exit-hold", type=float, default=0.7, help="退出半躺档的条件保持时间（秒）")
    parser.add_argument("--recline-baseline-frames", type=int, default=20, help="正常坐姿基准采样帧数")
    parser.add_argument("--recline-pose-alpha", type=float, default=0.12, help="脸部俯仰低通滤波系数")
    parser.add_argument("--recline-transition-time", type=float, default=1.5, help="正常与半躺模式的名义过渡时间（秒）")
    parser.add_argument("--recline-transition-timeout", type=float, default=4.0, help="模式过渡等待机械臂追赶的最长时间（秒）")
    parser.add_argument("--transition-joint-error", type=float, default=0.03, help="开始放慢模式过渡的软件/实际关节误差（rad）")
    parser.add_argument("--transition-governor-tau", type=float, default=0.15, help="自适应进度速度的平滑时间常数（秒）")
    parser.add_argument("--hold-horizontal-enter", type=float, default=0.04, help="离开 HOLD 的水平误差（米）")
    parser.add_argument("--hold-horizontal-exit", type=float, default=0.02, help="进入 HOLD 的水平误差（米）")
    parser.add_argument("--hold-vertical-enter", type=float, default=0.05, help="离开 HOLD 的垂直误差（米）")
    parser.add_argument("--hold-vertical-exit", type=float, default=0.025, help="进入 HOLD 的垂直误差（米）")
    parser.add_argument("--hold-distance-enter", type=float, default=0.05, help="离开 HOLD 的距离误差（米）")
    parser.add_argument("--hold-distance-exit", type=float, default=0.025, help="进入 HOLD 的距离误差（米）")
    parser.add_argument("--hold-yaw-enter", type=float, default=4.0, help="离开 HOLD 的水平偏航误差（度）")
    parser.add_argument("--hold-yaw-exit", type=float, default=2.0, help="进入 HOLD 的水平偏航误差（度）")
    parser.add_argument("--hold-blend-time", type=float, default=0.20, help="FOLLOW/HOLD 控制增益过渡时间（秒）")
    parser.add_argument(
        "--recline-pitch-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="半躺方向的脸部俯仰符号；现场方向相反时设为 -1",
    )
    parser.add_argument("--pose-max-reprojection-error", type=float, default=8.0, help="五点 PnP 最大重投影均方根误差（像素）")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--screen-distance", type=float, default=0.5, help="屏幕中心到人脸的目标法向距离（米，非欧氏距离）")
    parser.add_argument(
        "--screen-center-offset",
        type=float,
        default=-0.05,
        help="计算用屏幕中心沿屏幕局部垂直轴的偏移（米，可为负数）",
    )
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
    parser.add_argument("--max-joint-speed", type=float, default=0.22, help="跟踪最大关节速度（rad/s）")
    parser.add_argument("--max-joint-accel", type=float, default=0.50, help="跟踪最大关节加速度（rad/s²）")
    parser.add_argument("--qp-position-gain", type=float, default=4.5, help="QP 末端位置反馈增益")
    parser.add_argument("--qp-rotation-gain", type=float, default=4.0, help="QP 末端姿态反馈增益")
    parser.add_argument("--qp-jerk-weight", type=float, default=1.0, help="QP 关节速度变化连续性权重")
    parser.add_argument("--max-cartesian-speed", type=float, default=0.20, help="QP 末端线速度上限（m/s）")
    parser.add_argument("--max-angular-speed", type=float, default=0.65, help="QP 末端角速度上限（rad/s）")
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
    if not np.isfinite(args.screen_center_offset):
        raise ValueError("--screen-center-offset 必须是有限数值")
    if args.recline_baseline_frames <= 0:
        raise ValueError("--recline-baseline-frames 必须大于 0")
    if not 0.0 < args.recline_pose_alpha <= 1.0:
        raise ValueError("--recline-pose-alpha 必须位于 (0, 1]")
    if args.recline_pitch_angle > 45.0:
        raise ValueError("--recline-pitch-angle 首版限制为不超过 45 度")
    if args.recline_enter_pitch > 60.0:
        raise ValueError("半躺进入脸部俯仰阈值不能超过 60 度")
    if args.recline_exit_height >= args.recline_height_drop:
        raise ValueError("--recline-exit-height 必须小于 --recline-height-drop 以形成滞回")
    if args.recline_transition_timeout <= args.recline_transition_time:
        raise ValueError("--recline-transition-timeout 必须大于名义过渡时间")
    for axis in ("horizontal", "vertical", "distance", "yaw"):
        enter = getattr(args, f"hold_{axis}_enter")
        exit_value = getattr(args, f"hold_{axis}_exit")
        if exit_value >= enter:
            raise ValueError(f"--hold-{axis}-exit 必须小于对应 enter 阈值")
    if min(
        args.screen_distance,
        args.target_timeout,
        args.lost_timeout,
        args.control_period,
        args.max_joint_speed,
        args.max_joint_accel,
        args.qp_position_gain,
        args.qp_rotation_gain,
        args.qp_jerk_weight,
        args.max_cartesian_speed,
        args.max_angular_speed,
        args.qp_home_gain,
        args.qp_home_speed,
        args.j1_home_deviation,
        args.j1_horizontal_threshold,
        args.qp_slowdown_distance,
        args.tracking_rebase_threshold,
        args.min_depth,
        args.max_depth,
        args.home_velocity,
        args.exit_velocity,
        args.max_target_jump,
        args.recline_pitch_angle,
        args.recline_enter_pitch,
        args.recline_height_drop,
        args.recline_exit_height,
        args.recline_enter_hold,
        args.recline_exit_hold,
        args.recline_transition_time,
        args.recline_transition_timeout,
        args.transition_joint_error,
        args.transition_governor_tau,
        args.hold_horizontal_enter,
        args.hold_horizontal_exit,
        args.hold_vertical_enter,
        args.hold_vertical_exit,
        args.hold_distance_enter,
        args.hold_distance_exit,
        args.hold_yaw_enter,
        args.hold_yaw_exit,
        args.hold_blend_time,
        args.pose_max_reprojection_error,
    ) <= 0.0:
        raise ValueError("距离、时间和速度/加速度参数必须大于 0")
    if args.joint_limit_margin < 0.0:
        raise ValueError("--joint-limit-margin 不能为负数")
    if args.min_depth >= args.max_depth:
        raise ValueError("--min-depth 必须小于 --max-depth")
def load_home_position(path: Path) -> np.ndarray:
    """直接解析 HOME markdown，不依赖任何项目内辅助脚本。"""
    values = []
    for line in path.resolve().read_text(encoding="utf-8").splitlines():
        if "位置=" not in line or "rad" not in line:
            continue
        try:
            value_text = line.split("位置=", 1)[1].split("rad", 1)[0].strip()
            values.append(float(value_text))
        except (IndexError, ValueError) as exc:
            raise ValueError(f"无法解析 HOME 位置行：{line!r}") from exc
    if len(values) != JOINT_COUNT:
        raise ValueError(
            f"HOME 文件应包含 {JOINT_COUNT} 个关节位置，实际得到 {len(values)} 个：{path}"
        )
    return np.asarray(values, dtype=np.float64)


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


class SynchronizedJointTrajectory:
    """所有关节共享一个七次平滑进度的点到点轨迹。"""

    # s(u)=35u^4-84u^5+70u^6-20u^7 的归一化峰值。
    PEAK_NORMALIZED_SPEED = 2.1875
    PEAK_NORMALIZED_ACCEL = 7.5131884044

    def __init__(
        self,
        start_joint: np.ndarray,
        target_joint: np.ndarray,
        max_speed: np.ndarray,
        max_accel: np.ndarray,
        minimum_duration: float,
    ):
        self.start_joint = np.asarray(start_joint, dtype=np.float64).reshape(-1).copy()
        self.target_joint = np.asarray(target_joint, dtype=np.float64).reshape(-1).copy()
        speed = np.asarray(max_speed, dtype=np.float64).reshape(-1)
        accel = np.asarray(max_accel, dtype=np.float64).reshape(-1)
        if not (
            self.start_joint.shape == self.target_joint.shape == speed.shape == accel.shape
            and np.all(np.isfinite(self.start_joint))
            and np.all(np.isfinite(self.target_joint))
            and np.all(np.isfinite(speed))
            and np.all(np.isfinite(accel))
            and np.all(speed > 0.0)
            and np.all(accel > 0.0)
        ):
            raise ValueError("同步关节轨迹输入无效")
        self.delta = self.target_joint - self.start_joint
        abs_delta = np.abs(self.delta)
        speed_duration = float(np.max(
            self.PEAK_NORMALIZED_SPEED * abs_delta / speed
        ))
        accel_duration = float(np.max(np.sqrt(
            self.PEAK_NORMALIZED_ACCEL * abs_delta / accel
        )))
        self.duration = max(float(minimum_duration), speed_duration, accel_duration)

    def sample(self, elapsed: float) -> tuple[np.ndarray, np.ndarray, bool]:
        """返回当前关节位置、速度及轨迹是否完成。"""
        if elapsed <= 0.0:
            return self.start_joint.copy(), np.zeros_like(self.start_joint), False
        if elapsed >= self.duration:
            return self.target_joint.copy(), np.zeros_like(self.target_joint), True
        u = float(elapsed / self.duration)
        u2 = u * u
        u3 = u2 * u
        u4 = u3 * u
        u5 = u4 * u
        u6 = u5 * u
        u7 = u6 * u
        progress = 35.0 * u4 - 84.0 * u5 + 70.0 * u6 - 20.0 * u7
        progress_rate = (
            140.0 * u3 - 420.0 * u4 + 420.0 * u5 - 140.0 * u6
        ) / self.duration
        position = self.start_joint + progress * self.delta
        velocity = progress_rate * self.delta
        return position, velocity, False


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
        self.qdot_previous_previous = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.cartesian_command_ready = False
        self.cartesian_command_velocity = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.desired_joint = self.command_joint.copy()
        self.smoothed_target = None
        self.lost_since = None
        self.return_trajectory = None
        self.return_elapsed = 0.0
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
            self.cartesian_command_ready = False
            self.cartesian_command_velocity.fill(0.0)
            self.qdot_previous_previous.fill(0.0)
            self.smoothed_target = None
            self.lost_since = None
            self.return_trajectory = None
            self.return_elapsed = 0.0
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
                        self.qdot_previous_previous.fill(0.0)
                        self.smoothed_target = None
                        self.lost_since = None
                elif fresh:
                    if status == "RETURNING":
                        with self.lock:
                            self.return_trajectory = None
                            self.return_elapsed = 0.0
                    self._handle_fresh_target(
                        point_camera, status, command_joint, command_velocity, dt
                    )
                else:
                    self._handle_lost_target(now)

                with self.lock:
                    if self.status == "RETURNING" and fresh:
                        self.status = "TRACKING"
                    qp_command_applied = False
                    if self.cartesian_command_ready and fresh and self.status in (
                        "TRACKING", "TRACKING_LIMITED", "QP_FAIL", "REBASE"
                    ):
                        self.command_joint = self.desired_joint.copy()
                        self.command_velocity = self.cartesian_command_velocity.copy()
                        self.qdot_previous_previous = command_velocity.copy()
                        self.cartesian_command_ready = False
                        qp_command_applied = True
                    if self.status == "RETURNING":
                        if self.return_trajectory is None:
                            raise RuntimeError("RETURNING 状态缺少同步关节轨迹")
                        self.return_elapsed += float(np.clip(dt, 1e-4, 0.05))
                        (
                            self.command_joint,
                            self.command_velocity,
                            return_complete,
                        ) = self.return_trajectory.sample(self.return_elapsed)
                        self.desired_joint = self.home_position.copy()
                        if return_complete:
                            self.command_joint = self.home_position.copy()
                            self.command_velocity.fill(0.0)
                            self.qdot_previous_previous.fill(0.0)
                            self.desired_joint = self.home_position.copy()
                            self.return_trajectory = None
                            self.return_elapsed = 0.0
                            self.status = "HOME"
                            self.lost_since = None
                    elif not qp_command_applied:
                        self.command_joint = self.desired_joint.copy()
                        self.command_velocity.fill(0.0)
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
                self.qdot_previous_previous.fill(0.0)
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
        raw_target = self._face_target(
            point_camera,
            actual_fk,
        )
        with self.lock:
            if previous_status == "RETURNING":
                self.smoothed_target = None
            self.smoothed_target = self._smooth_face_target(raw_target)

            target_position = self.smoothed_target.copy()
            desired_twist = self._desired_twist(command_fk, target_position)
            qdot, solver_status, _, active = self._solve_velocity_qp(
                command_joint, command_velocity, self.qdot_previous_previous,
                command_fk, desired_twist, dt
            )
            if solver_status == "SOLVED":
                qdot = np.asarray(qdot, dtype=np.float64)
                q_next = command_joint + qdot * float(np.clip(dt, 1e-4, 0.05))
                q_next = np.clip(q_next, self.safe_lower, self.safe_upper)
                self.desired_joint = q_next
                self.cartesian_command_velocity = qdot
                self.qdot_previous_previous = command_velocity.copy()
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

    def _face_target(self, point_camera: np.ndarray, fk: dict) -> np.ndarray:
        """根据当前控制器模式计算屏幕中心目标。"""
        return face_to_screen_target(
            point_camera,
            fk,
            self.camera_rotation,
            self.args.screen_distance,
            self.screen_rotation,
            self.args.camera_centering_gain,
        )

    def _smooth_face_target(self, raw_target: np.ndarray) -> np.ndarray:
        if self.smoothed_target is None:
            return np.asarray(raw_target, dtype=np.float64).copy()
        alpha = 0.12
        return (
            alpha * np.asarray(raw_target, dtype=np.float64)
            + (1.0 - alpha) * self.smoothed_target
        )

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

    def _solve_velocity_qp(
        self, q_command, qdot_previous, qdot_previous_previous,
        current_fk, desired_twist, dt,
    ):
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

    def _handle_lost_target(self, now: float) -> None:
        with self.lock:
            if self.status == "HOME":
                self.desired_joint = self.home_position.copy()
                return
            if self.lost_since is None:
                self.lost_since = now
            self.desired_joint = self.command_joint.copy()
            if now - self.lost_since < self.args.lost_timeout:
                self.status = "LOST"
                return
            if self.status == "RETURNING":
                return
            self.return_trajectory = SynchronizedJointTrajectory(
                self.command_joint,
                self.home_position,
                self.max_speed,
                self.max_accel,
                minimum_duration=max(0.25, 2.0 * self.args.control_period),
            )
            self.return_elapsed = 0.0
            self.status = "RETURNING"
            self.desired_joint = self.home_position.copy()
            self.command_velocity.fill(0.0)
            self.cartesian_command_ready = False
            self.cartesian_command_velocity.fill(0.0)
            self.qdot_previous_previous.fill(0.0)
            self.smoothed_target = None
            print(
                f"目标持续丢失：同步返回 HOME，"
                f"预计 {self.return_trajectory.duration:.2f} s。"
            )


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


def return_to_zero(
    robot,
    velocity: float,
    acceleration: float,
    control_period: float,
) -> None:
    """以同步七次插值返回真实六关节零位，然后停止电机。"""
    zero = np.zeros(JOINT_COUNT, dtype=np.float64)
    current = np.asarray(robot.get_current_pos(), dtype=np.float64)
    if current.shape != (JOINT_COUNT,) or not np.all(np.isfinite(current)):
        robot.set_stop()
        raise RuntimeError("退出时无法读取有效的六关节位置")
    configured_speed = np.asarray(
        getattr(robot, "velocity_limits", [velocity] * JOINT_COUNT),
        dtype=np.float64,
    )
    configured_accel = np.asarray(
        getattr(robot, "acceleration_limits", [acceleration] * JOINT_COUNT),
        dtype=np.float64,
    )
    max_speed = np.minimum(configured_speed, float(velocity))
    max_accel = np.minimum(configured_accel, float(acceleration))
    trajectory = SynchronizedJointTrajectory(
        current,
        zero,
        max_speed,
        max_accel,
        minimum_duration=max(0.25, 2.0 * float(control_period)),
    )
    print(
        "退出：同步返回零位 [0, 0, 0, 0, 0, 0]，"
        f"预计 {trajectory.duration:.2f} s..."
    )
    try:
        started = time.perf_counter()
        next_tick = started
        while True:
            elapsed = time.perf_counter() - started
            position, joint_velocity, complete = trajectory.sample(elapsed)
            accepted = robot.Joint_Pos_Vel(
                position.tolist(),
                joint_velocity.tolist(),
                robot.max_torque.tolist(),
                iswait=False,
            )
            if not accepted:
                raise RuntimeError("同步回零命令被 SDK 拒绝")
            if complete:
                break
            next_tick += float(control_period)
            remaining = next_tick - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)

        settle_deadline = time.monotonic() + 2.0
        reached = False
        while time.monotonic() < settle_deadline:
            actual = np.asarray(robot.get_current_pos(), dtype=np.float64)
            if (
                actual.shape == (JOINT_COUNT,)
                and np.all(np.isfinite(actual))
                and np.max(np.abs(actual - zero)) <= 0.01
            ):
                reached = True
                break
            time.sleep(float(control_period))
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
        return_to_zero(
            robot,
            args.exit_velocity,
            args.max_joint_accel,
            args.control_period,
        )
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
            observation_time = time.monotonic() if accepted_point is not None else None
            controller.update_target(
                accepted_point,
                observation_time,
            )
            controller.update_face_observation(
                None if selected is None or accepted_point is None else selected.get("face_normal_camera"),
                observation_time,
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
            recline = controller.recline_snapshot()
            raw_pitch = recline["raw_pitch"]
            pose_error = None if selected is None else selected.get("pose_reprojection_error")
            raw_text = "--" if raw_pitch is None else f"{raw_pitch:+.1f}"
            filtered_text = "--" if recline["pitch"] is None else f"{recline['pitch']:+.1f}"
            error_text = "--" if pose_error is None else f"{pose_error:.1f}px"
            if recline["calibrated"]:
                detail = (
                    f"dP={recline['pitch_delta']:+.1f}deg "
                    f"dZ={recline['height_drop']:+.2f}m "
                    f"dD={recline['depth_shift']:+.2f}m "
                    f"gate={100.0 * recline['progress']:.0f}%"
                )
            else:
                detail = f"calibrating {100.0 * recline['progress']:.0f}%"
            cv2.putText(
                color,
                f"mode={recline['mode']} world={raw_text}/{filtered_text}deg err={error_text}",
                (16, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                color,
                detail,
                (16, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 165, 255) if recline["progress"] > 0.0 else (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                color,
                f"blend={100.0 * recline['blend']:.0f}% "
                f"transition-speed={100.0 * recline['governor']:.0f}%"
                f"{' TIMEOUT' if recline['timed_out'] else ''}",
                (16, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 165, 255) if recline["active"] else (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                color,
                f"tracking={recline['follow_state']} "
                f"gain={100.0 * recline['follow_gain']:.0f}%",
                (16, 136),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
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
        return_to_zero(
            robot,
            args.exit_velocity,
            args.max_joint_accel,
            args.control_period,
        )
    return 0


ORIGINAL_DETECT_FACES = detect_faces

# 动态姿态滤波，避免人脸深度噪声直接变成屏幕角度抖动。
POSE_FOLLOW_ALPHA = 0.08
FACE_POSITION_ALPHA = 0.12
# 只跟随水平偏航；俯仰和滚转保留 HOME，避免深度噪声导致大幅抬头/低头。
MAX_YAW_OFFSET = math.radians(45.0)
PITCH_TASK_WEIGHT = 20.0
YAW_TASK_WEIGHT = 50.0
ROLL_TASK_WEIGHT = 50.0

# SCRFD 五点顺序：左眼、右眼、鼻尖、左嘴角、右嘴角。数值只定义一个
# 通用脸部形状；solvePnP 在这里仅取朝向，不使用平移或人脸绝对尺度。
FACE_MODEL_5_POINTS = np.array(
    [
        [-30.0, 35.0, -30.0],
        [30.0, 35.0, -30.0],
        [0.0, 0.0, 0.0],
        [-25.0, -30.0, -25.0],
        [25.0, -30.0, -25.0],
    ],
    dtype=np.float64,
)


def estimate_face_normal_5point(points: np.ndarray, intrinsics, max_error: float):
    """用 SCRFD 五点粗估脸部朝向。

    返回重投影误差和相机坐标系脸部法向。控制器会把法向转到基坐标系后
    再做档位判定，避免屏幕俯仰改变测量基准。
    """
    image_points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if image_points.shape != (5, 2) or not np.all(np.isfinite(image_points)):
        return None, None
    camera_matrix = np.array(
        [
            [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
            [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(getattr(intrinsics, "coeffs", [0.0] * 5), dtype=np.float64)
    if distortion.size < 4 or not np.all(np.isfinite(distortion)):
        distortion = np.zeros(5, dtype=np.float64)
    try:
        solved, rotation_vector, translation = cv2.solvePnP(
            FACE_MODEL_5_POINTS,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not solved or float(np.asarray(translation).reshape(3)[2]) <= 0.0:
            return None, None
        solved, rotation_vector, translation = cv2.solvePnP(
            FACE_MODEL_5_POINTS,
            image_points,
            camera_matrix,
            distortion,
            rotation_vector,
            translation,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not solved:
            return None, None
        projected, _ = cv2.projectPoints(
            FACE_MODEL_5_POINTS,
            rotation_vector,
            translation,
            camera_matrix,
            distortion,
        )
        residual = projected.reshape(-1, 2) - image_points
        reprojection_error = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        if not np.isfinite(reprojection_error) or reprojection_error > float(max_error):
            return reprojection_error, None
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        face_normal = rotation_matrix @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # 通用模型的 +Z 指向脸外；它应朝向相机。PnP 偶尔给出翻转解时统一方向。
        if face_normal[2] > 0.0:
            face_normal = -face_normal
        return reprojection_error, face_normal
    except cv2.error:
        return None, None


def smoothstep5(value: float) -> float:
    """端点速度和加速度均为零的五次过渡曲线。"""
    value = float(np.clip(value, 0.0, 1.0))
    return value ** 3 * (10.0 + value * (-15.0 + 6.0 * value))


class AdaptiveModeTransition:
    """用单一进度同步位置与姿态，并在末端误差增大时整体放慢。"""

    def __init__(self, args):
        self.args = args
        self.reset()

    def reset(self, blend: float = 0.0) -> None:
        self.blend = float(np.clip(blend, 0.0, 1.0))
        self.start_blend = self.blend
        self.target_blend = self.blend
        self.phase = 1.0
        self.governor = 1.0
        self.elapsed = 0.0
        self.timed_out = False

    @property
    def active(self) -> bool:
        return self.phase < 1.0 and not self.timed_out

    def set_target(self, target: float) -> None:
        target = float(np.clip(target, 0.0, 1.0))
        if abs(target - self.target_blend) <= 1e-9:
            return
        self.start_blend = self.blend
        self.target_blend = target
        self.phase = 0.0
        self.elapsed = 0.0
        self.timed_out = False

    def update(self, dt: float, joint_execution_error: float) -> float:
        if self.timed_out:
            return self.blend
        if self.phase >= 1.0:
            self.blend = self.target_blend
            self.governor = 1.0
            return self.blend
        error_ratio = float(joint_execution_error) / float(
            self.args.transition_joint_error
        )
        if error_ratio <= 1.0:
            desired_governor = 1.0
        elif error_ratio >= 2.0:
            desired_governor = 0.0
        else:
            desired_governor = 1.0 - smoothstep5(error_ratio - 1.0)
        dt = float(np.clip(dt, 1e-4, 0.05))
        self.elapsed += dt
        if self.elapsed >= float(self.args.recline_transition_timeout):
            self.timed_out = True
            self.governor = 0.0
            return self.blend
        alpha = 1.0 - math.exp(-dt / float(self.args.transition_governor_tau))
        self.governor += alpha * (desired_governor - self.governor)
        distance = max(abs(self.target_blend - self.start_blend), 1e-6)
        duration = float(self.args.recline_transition_time) * distance
        self.phase = min(1.0, self.phase + self.governor * dt / duration)
        self.blend = self.start_blend + (
            self.target_blend - self.start_blend
        ) * smoothstep5(self.phase)
        if self.phase >= 1.0:
            self.blend = self.target_blend
            self.governor = 1.0
        return self.blend

    def snapshot(self) -> dict:
        return {
            "blend": self.blend,
            "target": self.target_blend,
            "phase": self.phase,
            "governor": self.governor,
            "active": self.active,
            "timed_out": self.timed_out,
        }


class TaskSpaceHold:
    """对位置三轴和水平偏航共用一个带滞回的 FOLLOW/HOLD 状态。"""

    FOLLOW = "FOLLOW"
    HOLD = "HOLD"

    def __init__(self, args):
        self.args = args
        self.reset()

    def reset(self) -> None:
        self.state = self.FOLLOW
        self.gain = 1.0

    def update(
        self,
        dt: float,
        horizontal_error: float,
        vertical_error: float,
        distance_error: float,
        yaw_error: float,
        transition_active: bool,
    ) -> float:
        enter_limits = np.array(
            [
                self.args.hold_horizontal_enter,
                self.args.hold_vertical_enter,
                self.args.hold_distance_enter,
                math.radians(self.args.hold_yaw_enter),
            ],
            dtype=np.float64,
        )
        exit_limits = np.array(
            [
                self.args.hold_horizontal_exit,
                self.args.hold_vertical_exit,
                self.args.hold_distance_exit,
                math.radians(self.args.hold_yaw_exit),
            ],
            dtype=np.float64,
        )
        errors = np.abs(np.array(
            [horizontal_error, vertical_error, distance_error, yaw_error],
            dtype=np.float64,
        ))
        if transition_active:
            self.state = self.FOLLOW
        elif self.state == self.HOLD:
            if np.any(errors >= enter_limits):
                self.state = self.FOLLOW
        elif np.all(errors <= exit_limits):
            self.state = self.HOLD

        target_gain = 1.0 if self.state == self.FOLLOW else 0.0
        dt = float(np.clip(dt, 1e-4, 0.05))
        alpha = 1.0 - math.exp(-dt / float(self.args.hold_blend_time))
        self.gain += alpha * (target_gain - self.gain)
        if self.state == self.HOLD and self.gain < 1e-3:
            self.gain = 0.0
        elif self.state == self.FOLLOW and self.gain > 1.0 - 1e-3:
            self.gain = 1.0
        return self.gain

    def snapshot(self) -> dict:
        return {"follow_state": self.state, "follow_gain": self.gain}


class ReclineModeTracker:
    """以脸部俯仰为主、基坐标三维位移为辅的两态锁存器。"""

    NORMAL = "NORMAL"
    RECLINED = "RECLINED"

    def __init__(self, args, forward_axis_base: np.ndarray):
        self.args = args
        axis = np.asarray(forward_axis_base, dtype=np.float64).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if not np.isfinite(axis_norm) or axis_norm <= 1e-9:
            raise ValueError("半躺检测的前向基准轴无效")
        self.forward_axis_base = axis / axis_norm
        self.reset()

    def reset(self) -> None:
        self.mode = self.NORMAL
        self.raw_pitch = None
        self.filtered_pitch = None
        self.baseline_pitch = None
        self.baseline_point = None
        self.pitch_samples = []
        self.point_samples = []
        self.candidate_since = None
        self.last_observation_time = None
        self.pitch_delta = 0.0
        self.height_drop = 0.0
        self.depth_shift = 0.0
        self.transition_progress = 0.0

    @property
    def calibrated(self) -> bool:
        return self.baseline_pitch is not None and self.baseline_point is not None

    def _hold_progress(self, timestamp: float, duration: float) -> float:
        if self.candidate_since is None:
            return 0.0
        return float(np.clip((timestamp - self.candidate_since) / duration, 0.0, 1.0))

    def update(self, pitch_deg: float, point_base: np.ndarray, timestamp: float) -> bool:
        """处理一帧观测；仅在模式实际改变时返回 True。"""
        point = np.asarray(point_base, dtype=np.float64).reshape(-1)
        if point.shape != (3,) or not np.all(np.isfinite(point)) or not np.isfinite(pitch_deg):
            self.candidate_since = None
            self.transition_progress = 0.0
            return False
        timestamp = float(timestamp)
        if (
            self.last_observation_time is not None
            and timestamp - self.last_observation_time > 0.25
        ):
            self.candidate_since = None
        self.last_observation_time = timestamp
        self.raw_pitch = float(pitch_deg)

        if self.filtered_pitch is None:
            self.filtered_pitch = float(pitch_deg)
        else:
            alpha = float(self.args.recline_pose_alpha)
            self.filtered_pitch = alpha * float(pitch_deg) + (1.0 - alpha) * self.filtered_pitch

        if not self.calibrated:
            self.pitch_samples.append(self.filtered_pitch)
            self.point_samples.append(point.copy())
            required = int(self.args.recline_baseline_frames)
            self.transition_progress = min(len(self.pitch_samples) / required, 1.0)
            if len(self.pitch_samples) >= required:
                self.baseline_pitch = float(np.median(np.asarray(self.pitch_samples)))
                self.baseline_point = np.median(np.asarray(self.point_samples), axis=0)
                self.pitch_samples.clear()
                self.point_samples.clear()
                self.transition_progress = 0.0
            return False

        self.pitch_delta = float(self.args.recline_pitch_sign) * (
            self.filtered_pitch - self.baseline_pitch
        )
        self.height_drop = float(self.baseline_point[2] - point[2])
        self.depth_shift = float(np.dot(point - self.baseline_point, self.forward_axis_base))

        changed = False
        if self.mode == self.NORMAL:
            pose_ready = self.pitch_delta >= float(self.args.recline_enter_pitch)
            evidence = (
                pose_ready
                and self.height_drop >= float(self.args.recline_height_drop)
            )
            hold_time = float(self.args.recline_enter_hold)
            next_mode = self.RECLINED
        else:
            # 退出只看双眼是否恢复到正常坐姿高度。PnP 俯仰在相机视角变化后
            # 仍可能带有固定偏差，不应阻止屏幕平滑回正。
            evidence = self.height_drop <= float(self.args.recline_exit_height)
            hold_time = float(self.args.recline_exit_hold)
            next_mode = self.NORMAL

        if evidence:
            if self.candidate_since is None:
                self.candidate_since = timestamp
            self.transition_progress = self._hold_progress(timestamp, hold_time)
            if timestamp - self.candidate_since >= hold_time:
                self.mode = next_mode
                self.candidate_since = None
                self.transition_progress = 0.0
                changed = True
        else:
            self.candidate_since = None
            self.transition_progress = 0.0
        return changed

    def snapshot(self) -> dict:
        return {
            "mode": self.mode,
            "calibrated": self.calibrated,
            "raw_pitch": self.raw_pitch,
            "pitch": self.filtered_pitch,
            "pitch_delta": self.pitch_delta,
            "height_drop": self.height_drop,
            "depth_shift": self.depth_shift,
            "progress": self.transition_progress,
        }

def detect_faces_eye_center(detector, color, depth_frame, intrinsics, args):
    """复用原检测流程，但将跟踪点改为双眼中心。

    SCRFD 的五个关键点顺序是左右眼、鼻尖、左右嘴角；原版使用五点平均，
    会让嘴部和鼻尖移动影响屏幕目标。动态试验版只用左右眼中点作为观看目标。
    """
    detections = ORIGINAL_DETECT_FACES(detector, color, depth_frame, intrinsics, args)
    for item in detections:
        points = item.get("keypoints")
        if points is None or np.asarray(points).shape[0] < 2:
            continue
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        pose_error, face_normal_camera = estimate_face_normal_5point(
            points,
            intrinsics,
            args.pose_max_reprojection_error,
        )
        eye_center = 0.5 * (points[0] + points[1])
        eye_points_camera = []
        eye_depths = []
        for eye in points[:2]:
            depth_m = sample_depth_m(
                depth_frame,
                eye[0],
                eye[1],
                args.depth_radius,
                args.min_depth,
                args.max_depth,
            )
            if depth_m is None:
                continue
            eye_points_camera.append(deproject_pixel(eye[0], eye[1], depth_m, intrinsics))
            eye_depths.append(depth_m)
        point_camera = None
        if len(eye_points_camera) == 2:
            # 先分别反投影，再取三维中点；这比在二维中点处只采一个深度更准确。
            point_camera = np.mean(np.asarray(eye_points_camera, dtype=np.float64), axis=0)
            depth_m = float(np.mean(eye_depths))
        else:
            # 双眼有一个深度无效时，回退到二维中点深度，避免整帧丢失目标。
            depth_m = sample_depth_m(
                depth_frame,
                eye_center[0],
                eye_center[1],
                args.depth_radius,
                args.min_depth,
                args.max_depth,
            )
            if depth_m is not None:
                point_camera = deproject_pixel(
                    eye_center[0], eye_center[1], depth_m, intrinsics
                )
        item["center"] = (float(eye_center[0]), float(eye_center[1]))
        item["depth_m"] = depth_m
        item["point_camera"] = point_camera
        item["pose_reprojection_error"] = pose_error
        item["face_normal_camera"] = face_normal_camera
    return detections


class DynamicPoseTrackingController(TrackingController):
    """实时生成“法向指向人脸、HOME 滚转”的屏幕姿态。"""

    def __init__(self, robot, home_position, camera_rotation, screen_rotation, args):
        super().__init__(robot, home_position, camera_rotation, screen_rotation, args)
        self.home_screen_rotation = np.asarray(screen_rotation, dtype=np.float64).copy()
        home_forward = screen_normal_in_base(self.home_screen_rotation, self.camera_rotation)
        self.recline_tracker = ReclineModeTracker(args, home_forward)
        self.mode_transition = AdaptiveModeTransition(args)
        self.task_hold = TaskSpaceHold(args)
        self.filtered_yaw = 0.0
        self.filtered_face_base = None
        self.transition_timeout_reported = False
        self.latest_control_dt = float(args.control_period)
        self.latest_face_normal_camera = None
        self.latest_face_normal_time = None
        self.last_recline_observation_time = None

    def set_tracking_enabled(self, enabled: bool) -> None:
        super().set_tracking_enabled(enabled)
        with self.lock:
            self.recline_tracker.reset()
            self.mode_transition.set_target(0.0)
            self.task_hold.reset()
            self.filtered_face_base = None
            self.transition_timeout_reported = False
            self.latest_face_normal_camera = None
            self.latest_face_normal_time = None
            self.last_recline_observation_time = None

    def update_face_observation(
        self, face_normal_camera: np.ndarray | None, timestamp: float | None
    ) -> None:
        with self.lock:
            if face_normal_camera is None or timestamp is None:
                self.latest_face_normal_camera = None
                self.latest_face_normal_time = None
                return
            normal = np.asarray(face_normal_camera, dtype=np.float64).reshape(-1)
            if normal.shape != (3,) or not np.all(np.isfinite(normal)):
                self.latest_face_normal_camera = None
                self.latest_face_normal_time = None
                return
            self.latest_face_normal_camera = normal.copy()
            self.latest_face_normal_time = float(timestamp)

    def recline_snapshot(self) -> dict:
        with self.lock:
            snapshot = self.recline_tracker.snapshot()
            snapshot.update(self.mode_transition.snapshot())
            snapshot.update(self.task_hold.snapshot())
            return snapshot

    def _face_target(self, point_camera: np.ndarray, fk: dict) -> np.ndarray:
        _, t_base_camera = make_transforms(fk, self.camera_rotation)
        observed_face_base = t_base_camera @ np.append(
            np.asarray(point_camera, dtype=np.float64), 1.0
        )
        with self.lock:
            if self.filtered_face_base is None:
                self.filtered_face_base = observed_face_base[:3].copy()
            else:
                self.filtered_face_base = (
                    FACE_POSITION_ALPHA * observed_face_base[:3]
                    + (1.0 - FACE_POSITION_ALPHA) * self.filtered_face_base
                )
            filtered_face_base = self.filtered_face_base.copy()
        filtered_face_camera = np.linalg.inv(t_base_camera) @ np.append(
            filtered_face_base, 1.0
        )
        target = face_to_screen_target(
            filtered_face_camera[:3],
            fk,
            self.camera_rotation,
            self.args.screen_distance,
            self.screen_rotation,
            self.args.camera_centering_gain,
        )
        if abs(self.args.screen_center_offset) > 0.0:
            screen_vertical = np.asarray(self.screen_rotation[:, 1], dtype=np.float64)
            target = target + self.args.screen_center_offset * screen_vertical
        return target

    def _smooth_face_target(self, raw_target: np.ndarray) -> np.ndarray:
        # 人脸位置已在不随相机运动的基坐标系滤波；模式位移由五次进度生成。
        # 此处不再叠加目标低通，避免位置和俯仰产生不同的相位延迟。
        return np.asarray(raw_target, dtype=np.float64).copy()

    def _apply_downward_pitch(
        self, rotation: np.ndarray, angle_deg: float | None = None
    ) -> np.ndarray:
        """绕屏幕水平轴选择真正使光轴向下的固定俯视方向。"""
        rotation = np.asarray(rotation, dtype=np.float64)
        camera_horizontal = np.asarray(self.camera_rotation[:, 0], dtype=np.float64)
        axis_base = rotation @ camera_horizontal
        axis_norm = float(np.linalg.norm(axis_base))
        if not np.isfinite(axis_norm) or axis_norm <= 1e-9:
            return rotation
        axis_base /= axis_norm
        angle = math.radians(float(
            self.args.recline_pitch_angle if angle_deg is None else angle_deg
        ))
        positive = Rotation.from_rotvec(axis_base * angle).as_matrix() @ rotation
        negative = Rotation.from_rotvec(-axis_base * angle).as_matrix() @ rotation
        positive_z = float(screen_normal_in_base(positive, self.camera_rotation)[2])
        negative_z = float(screen_normal_in_base(negative, self.camera_rotation)[2])
        return positive if positive_z < negative_z else negative

    def _screen_rotation_toward_face(self, fk: dict, point_camera: np.ndarray):
        """只根据水平投影调整偏航，俯仰/滚转保持 HOME。"""
        _, t_base_camera = make_transforms(fk, self.camera_rotation)
        face_h = t_base_camera @ np.append(np.asarray(point_camera, dtype=np.float64), 1.0)
        direction = face_h[:3] - np.asarray(fk["position"], dtype=np.float64)
        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        face_horizontal = direction - vertical * float(np.dot(direction, vertical))
        home_normal = self.home_screen_rotation @ np.asarray(self.camera_rotation[:, 2], dtype=np.float64)
        home_horizontal = home_normal - vertical * float(np.dot(home_normal, vertical))
        face_norm = float(np.linalg.norm(face_horizontal))
        home_norm = float(np.linalg.norm(home_horizontal))
        if not np.isfinite(face_norm) or not np.isfinite(home_norm) or face_norm < 1e-5 or home_norm < 1e-5:
            return None
        face_horizontal /= face_norm
        home_horizontal /= home_norm
        yaw = math.atan2(
            float(np.dot(vertical, np.cross(home_horizontal, face_horizontal))),
            float(np.clip(np.dot(home_horizontal, face_horizontal), -1.0, 1.0)),
        )
        yaw = float(np.clip(yaw, -MAX_YAW_OFFSET, MAX_YAW_OFFSET))
        with self.lock:
            yaw_delta = math.atan2(
                math.sin(yaw - self.filtered_yaw),
                math.cos(yaw - self.filtered_yaw),
            )
            self.filtered_yaw += POSE_FOLLOW_ALPHA * yaw_delta
            filtered_yaw = self.filtered_yaw
            blend = self.mode_transition.blend
        normal_rotation = (
            Rotation.from_rotvec(vertical * filtered_yaw).as_matrix()
            @ self.home_screen_rotation
        )
        return self._apply_downward_pitch(
            normal_rotation,
            blend * float(self.args.recline_pitch_angle),
        )

    def _handle_fresh_target(self, point_camera, previous_status, command_joint, command_velocity, dt):
        self.latest_control_dt = float(np.clip(dt, 1e-4, 0.05))
        actual_q = np.asarray(self.robot.get_current_pos(), dtype=np.float64)
        if actual_q.shape == (JOINT_COUNT,) and np.all(np.isfinite(actual_q)):
            fk = self.robot.forward_kinematics(actual_q)
            if fk is not None:
                with self.lock:
                    face_normal_camera = self.latest_face_normal_camera
                    pose_time = self.latest_face_normal_time
                if (
                    face_normal_camera is not None
                    and pose_time is not None
                    and pose_time != self.last_recline_observation_time
                ):
                    _, t_base_camera = make_transforms(fk, self.camera_rotation)
                    face_h = t_base_camera @ np.append(
                        np.asarray(point_camera, dtype=np.float64), 1.0
                    )
                    rotation_base_camera = (
                        np.asarray(fk["rotation"], dtype=np.float64) @ self.camera_rotation
                    )
                    face_normal_base = rotation_base_camera @ face_normal_camera
                    world_pitch_deg = math.degrees(math.atan2(
                        float(face_normal_base[2]),
                        float(np.hypot(face_normal_base[0], face_normal_base[1])),
                    ))
                    with self.lock:
                        changed = self.recline_tracker.update(
                            world_pitch_deg,
                            face_h[:3],
                            pose_time,
                        )
                        mode = self.recline_tracker.mode
                    self.last_recline_observation_time = pose_time
                    if changed:
                        target_pitch = (
                            self.args.recline_pitch_angle
                            if mode == ReclineModeTracker.RECLINED
                            else 0.0
                        )
                        print(
                            f"俯仰模式切换：{mode}；"
                            f"目标俯视角={target_pitch:.1f}°"
                        )
                with self.lock:
                    target_blend = (
                        1.0
                        if self.recline_tracker.mode == ReclineModeTracker.RECLINED
                        else 0.0
                    )
                    self.mode_transition.set_target(target_blend)
                    joint_execution_error = float(np.max(np.abs(actual_q - command_joint)))
                    self.mode_transition.update(dt, joint_execution_error)
                    if self.mode_transition.timed_out and not self.transition_timeout_reported:
                        print(
                            "模式过渡超时：保持当前安全过渡位置；"
                            "请检查关节限位或降低俯视角。",
                            file=sys.stderr,
                        )
                        self.transition_timeout_reported = True
                    elif not self.mode_transition.timed_out:
                        self.transition_timeout_reported = False
                dynamic_rotation = self._screen_rotation_toward_face(fk, point_camera)
                if dynamic_rotation is not None:
                    self.screen_rotation = dynamic_rotation
        return super()._handle_fresh_target(
            point_camera, previous_status, command_joint, command_velocity, dt
        )

    def _handle_lost_target(self, now: float) -> None:
        super()._handle_lost_target(now)
        with self.lock:
            if self.status == "RETURNING" and (
                self.recline_tracker.mode != ReclineModeTracker.NORMAL
                or self.last_recline_observation_time is not None
            ):
                # 丢脸返回 HOME 时退出半躺档，但保留本次 tracking 会话的
                # 正常坐姿基准，避免短暂遮挡后把半躺姿态重新标成 NORMAL。
                self.recline_tracker.mode = ReclineModeTracker.NORMAL
                self.recline_tracker.filtered_pitch = None
                self.recline_tracker.candidate_since = None
                self.recline_tracker.transition_progress = 0.0
                self.recline_tracker.last_observation_time = None
                self.mode_transition.reset(0.0)
                self.task_hold.reset()
                self.filtered_yaw = 0.0
                self.filtered_face_base = None
                self.transition_timeout_reported = False
                self.latest_face_normal_camera = None
                self.latest_face_normal_time = None
                self.last_recline_observation_time = None
                self.screen_rotation = self.home_screen_rotation.copy()
                print("目标持续丢失：半躺模式已重置为 NORMAL。")

    def _desired_twist(self, current_fk: dict, target_position: np.ndarray) -> np.ndarray:
        desired_twist = super()._desired_twist(current_fk, target_position)
        current_position = np.asarray(current_fk["position"], dtype=np.float64)
        current_rotation = np.asarray(current_fk["rotation"], dtype=np.float64)
        position_error_base = np.asarray(target_position, dtype=np.float64) - current_position
        rotation_base_camera = current_rotation @ self.camera_rotation
        position_error_camera = rotation_base_camera.T @ position_error_base
        rotation_error = Rotation.from_matrix(
            self.screen_rotation @ current_rotation.T
        ).as_rotvec()
        yaw_error = float(np.dot(rotation_error, np.array([0.0, 0.0, 1.0])))
        with self.lock:
            gain = self.task_hold.update(
                self.latest_control_dt,
                position_error_camera[0],
                position_error_camera[1],
                position_error_camera[2],
                yaw_error,
                self.mode_transition.active,
            )
        return gain * desired_twist

    def _solve_velocity_qp(
        self, q_command, qdot_previous, qdot_previous_previous,
        current_fk, desired_twist, dt,
    ):
        """动态姿态版本：位置优先，偏航/滚转较硬，俯仰较软，J4 延后。"""
        q_command = np.asarray(q_command, dtype=np.float64)
        qdot_previous = np.asarray(qdot_previous, dtype=np.float64)
        qdot_previous_previous = np.asarray(qdot_previous_previous, dtype=np.float64)
        with self.lock:
            follow_gain = self.task_hold.gain
            fully_holding = (
                not self.mode_transition.active
                and self.task_hold.state == TaskSpaceHold.HOLD
                and follow_gain <= 0.0
                and np.max(np.abs(qdot_previous)) <= 1e-3
                and np.max(np.abs(qdot_previous_previous)) <= 2e-3
            )
        if fully_holding:
            return (
                np.zeros(JOINT_COUNT, dtype=np.float64),
                "SOLVED",
                0,
                np.zeros(JOINT_COUNT, dtype=int),
            )
        jacobian = np.asarray(self.robot.get_jacobian(q_command), dtype=np.float64)
        if jacobian.shape != (6, JOINT_COUNT) or not np.all(np.isfinite(jacobian)):
            return np.zeros(JOINT_COUNT), "NUMERICAL_FAILURE", 0, np.zeros(JOINT_COUNT, dtype=int)

        # 角速度误差转到当前屏幕局部轴：x≈俯仰，y≈偏航，z≈滚转。
        current_rotation = np.asarray(current_fk["rotation"], dtype=np.float64)
        angular_local = current_rotation.T @ np.asarray(desired_twist[3:], dtype=np.float64)
        angular_scale_local = np.sqrt(np.array(
            [PITCH_TASK_WEIGHT, YAW_TASK_WEIGHT, ROLL_TASK_WEIGHT], dtype=np.float64
        ))
        angular_row = np.diag(angular_scale_local) @ current_rotation.T @ jacobian[3:, :]
        angular_target = angular_scale_local * angular_local
        position_scale = np.sqrt(np.full(3, 100.0, dtype=np.float64))
        rows = [
            np.diag(position_scale) @ jacobian[:3, :],
            angular_row,
            np.diag(np.sqrt(np.array([0.35, 0.6, 0.6, 1.5, 1.5, 1.5]))),
            np.diag(np.sqrt(np.array([0.18, 0.8, 0.8, 8.0, 6.0, 6.0]))),
            np.diag(np.sqrt(np.array([0.0, 1.0, 1.0, 8.0, 6.0, 6.0]))),
            np.diag(np.full(JOINT_COUNT, np.sqrt(self.args.qp_jerk_weight))),
        ]
        targets = [
            position_scale * np.asarray(desired_twist[:3], dtype=np.float64),
            angular_target,
            np.sqrt(np.array([0.35, 0.6, 0.6, 1.5, 1.5, 1.5])) * qdot_previous,
            np.zeros(JOINT_COUNT),
            np.sqrt(np.array([0.0, 1.0, 1.0, 8.0, 6.0, 6.0])) * np.clip(
                follow_gain
                * self.args.qp_home_gain
                * (self.home_position - q_command),
                -self.args.qp_home_speed,
                self.args.qp_home_speed,
            ),
            np.sqrt(self.args.qp_jerk_weight)
            * (2.0 * qdot_previous - qdot_previous_previous),
        ]
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
            upper[index] = min(
                upper[index],
                self.max_speed[index] * np.clip(
                    (self.safe_upper[index] - q_command[index]) / slowdown, 0.0, 1.0
                ),
            )
            lower[index] = max(
                lower[index],
                -self.max_speed[index] * np.clip(
                    (q_command[index] - self.safe_lower[index]) / slowdown, 0.0, 1.0
                ),
            )

        # J1 的 HOME 角度不再作为左右运动的回拉目标；仅保留有限工作范围。
        j1_delta = float(q_command[0] - self.home_position[0])
        if j1_delta >= self.args.j1_home_deviation:
            upper[0] = min(upper[0], 0.0)
        elif j1_delta <= -self.args.j1_home_deviation:
            lower[0] = max(lower[0], 0.0)
        if np.any(lower > upper + 1e-9):
            lower = np.minimum(lower, 0.0)
            upper = np.maximum(upper, 0.0)

        qdot, status, iterations, active = solve_bounded_qp(hessian, gradient, lower, upper)
        position_active = np.zeros(JOINT_COUNT, dtype=int)
        position_active[(active == -1) & (qdot <= position_lower + 1e-7)] = -1
        position_active[(active == 1) & (qdot >= position_upper - 1e-7)] = 1
        return qdot, status, iterations, position_active


# 独立动态姿态入口：不导入带手势的控制文件，也不加载旧辅助脚本。
detect_faces = detect_faces_eye_center
TrackingController = DynamicPoseTrackingController


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C。", file=sys.stderr)
        raise SystemExit(130)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
