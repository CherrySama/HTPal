#!/usr/bin/env python3
"""独立测试 RealSense 手势事件和 HOME 附近的 Panthera 动作。

默认只显示事件，不发送电机命令；加 ``--enable-motion`` 才会连接机械臂、
先移动到 HOME，并在检测到一次手势后执行一次固定动作。
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MODEL = REPO_ROOT / "core" / "handdetect" / "model" / "hand_landmarker.task"
DEFAULT_HOME_FILE = REPO_ROOT / "core" / "detect_test_pos.md"
DEFAULT_CONFIG = REPO_ROOT / "panthera_python" / "robot_param" / "Follower_tracking.yaml"
WINDOW = "HTPal hand gesture test"
JOINT_COUNT = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-motion", action="store_true", help="连接 Panthera 并执行实际动作")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--home-file", type=Path, default=DEFAULT_HOME_FILE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--num-hands", type=int, default=1)
    parser.add_argument("--min-detection", type=float, default=0.5)
    parser.add_argument("--min-presence", type=float, default=0.5)
    parser.add_argument("--min-tracking", type=float, default=0.5)
    parser.add_argument("--hand-score-min", type=float, default=0.75, help="手部分类最低置信度")
    parser.add_argument("--palm-depth-spread-max", type=float, default=0.08, help="掌部采样点允许的最大深度离散（米）")
    parser.add_argument("--motion-step", type=float, default=0.10, help="每个前后手势的 X 方向位移（米）")
    parser.add_argument("--motion-min", type=float, default=-0.10, help="HOME 相对 X 位移下限（米）")
    parser.add_argument("--motion-max", type=float, default=0.10, help="HOME 相对 X 位移上限（米）")
    parser.add_argument("--motion-duration", type=float, default=2.0, help="单次笛卡尔动作时长（秒）")
    parser.add_argument("--push-trigger", type=float, default=0.06, help="前后推动作触发的掌心深度位移（米）")
    parser.add_argument("--push-rearm", type=float, default=0.025, help="前后推动作重新布防的中立范围（米）")
    parser.add_argument("--smoothing-alpha", type=float, default=0.20)
    parser.add_argument("--open-required-frames", type=int, default=4, help="五指张开连续多少帧后进入 READY")
    parser.add_argument("--push-start", type=float, default=0.025, help="进入前后移动检测的深度位移（米）")
    parser.add_argument("--settle-frames", type=int, default=3, help="动作达到阈值后需稳定停止的连续帧数")
    parser.add_argument("--push-stop-delta", type=float, default=0.004, help="判定手掌停止的单帧深度变化（米）")
    parser.add_argument("--rotate-start", type=float, default=12.0, help="进入旋转检测的手腕角度（度）")
    parser.add_argument("--rotate-trigger", type=float, default=28.0)
    parser.add_argument("--rotate-rearm", type=float, default=10.0)
    parser.add_argument("--rotate-stop-delta", type=float, default=2.0, help="判定旋转停止的单帧角度变化（度）")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path, label in ((args.model, "模型"), (args.home_file, "HOME 文件")):
        if not path.is_file():
            raise FileNotFoundError(f"{label}不存在：{path}")
    if args.enable_motion:
        if not args.config.is_file():
            raise FileNotFoundError(f"机械臂配置不存在：{args.config}")
    if min(args.width, args.height, args.fps, args.num_hands) <= 0:
        raise ValueError("尺寸、帧率和手数必须大于 0")
    if args.num_hands > 2:
        raise ValueError("--num-hands 不能大于 2")
    for name in ("min_detection", "min_presence", "min_tracking"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} 必须位于 [0, 1]")
    if not 0.0 <= args.hand_score_min <= 1.0:
        raise ValueError("--hand-score-min 必须位于 [0, 1]")
    if args.palm_depth_spread_max <= 0.0:
        raise ValueError("--palm-depth-spread-max 必须大于 0")
    if args.motion_step <= 0.0 or args.motion_min >= args.motion_max:
        raise ValueError("动作步长必须大于 0，且 motion-min 必须小于 motion-max")
    if args.motion_min > 0.0 or args.motion_max < 0.0:
        raise ValueError("动作范围必须包含 HOME 的 0 位移")
    if args.motion_duration <= 0.0 or not 0.0 < args.smoothing_alpha <= 1.0:
        raise ValueError("动作时长必须大于 0，smoothing-alpha 必须位于 (0, 1]")
    if args.push_trigger <= args.push_rearm or args.push_rearm < 0.0:
        raise ValueError("push-trigger 必须大于 push-rearm")
    if args.open_required_frames <= 0:
        raise ValueError("open-required-frames 必须大于 0")
    if args.push_start <= 0.0 or args.push_start >= args.push_trigger:
        raise ValueError("push-start 必须大于 0 且小于 push-trigger")
    if args.settle_frames <= 0 or args.push_stop_delta <= 0.0:
        raise ValueError("settle-frames 和 push-stop-delta 必须大于 0")
    if args.rotate_trigger <= args.rotate_rearm or args.rotate_rearm < 0.0:
        raise ValueError("旋转触发角度必须大于回中立角度")
    if args.rotate_start <= 0.0 or args.rotate_start >= args.rotate_trigger:
        raise ValueError("rotate-start 必须大于 0 且小于 rotate-trigger")
    if args.rotate_stop_delta <= 0.0:
        raise ValueError("rotate-stop-delta 必须大于 0")


def wrap_degrees(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def load_home_position(path: Path) -> np.ndarray:
    values = []
    pattern = re.compile(r"位置=\s*([-+]?\d+(?:\.\d+)?)\s*rad")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            values.append(float(match.group(1)))
    if len(values) != JOINT_COUNT:
        raise ValueError(f"HOME 文件应包含 {JOINT_COUNT} 个关节位置，实际得到 {len(values)} 个")
    return np.asarray(values, dtype=np.float64)


class GestureState:
    """要求全程五指张开，并在动作完成后只产生一次事件。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.motion_level = 1
        self.motion_offsets = (-args.motion_step, 0.0, args.motion_step)
        self.orientation = "LANDSCAPE"
        self.depth_reference = None
        self.depth_filtered = None
        self.roll_reference = None
        self.roll_filtered = None
        self.phase = "WAIT_OPEN"
        self.open_frames = 0
        self.active_kind = None
        self.active_direction = None
        self.peak_displacement = 0.0
        self.peak_angle = 0.0
        self.settle_frames = 0
        self.neutral_frames = 0
        self.neutral_depth_last = None
        self.neutral_roll_last = None
        self.status = "NO_HAND"

    @property
    def x_offset(self) -> float:
        return self.motion_offsets[self.motion_level]

    def update(self, measurement: dict | None) -> list[str]:
        events: list[str] = []
        if measurement is None:
            self._reset_for_fresh_open("NO_HAND")
            return events

        depth = float(measurement["palm_depth"])
        roll = float(measurement["roll_deg"])
        if not (math.isfinite(depth) and math.isfinite(roll)):
            self._reset_for_fresh_open("INVALID_HAND")
            return events

        is_open = bool(measurement.get("open_palm", False))
        if not is_open and self.active_kind is None:
            self._reset_for_fresh_open("WAIT_OPEN_PALM")
            return events
        if not is_open:
            self._reset_for_fresh_open("CANCELLED_NOT_OPEN")
            return events

        self.open_frames += 1

        if self.active_kind is None and self.phase == "WAIT_NEUTRAL":
            # Re-arm only after the hand has returned near the pre-action
            # neutral pose. Merely holding still at the action endpoint must
            # not be mistaken for neutral, otherwise the next frame can
            # reuse the endpoint as a new baseline and retrigger.
            if (
                self.depth_reference is None
                or abs(depth - self.depth_reference) > self.args.push_rearm
                or self.roll_reference is None
                or abs(wrap_degrees(roll - self.roll_reference)) > self.args.rotate_rearm
            ):
                self.neutral_frames = 0
                self.neutral_depth_last = None
                self.neutral_roll_last = None
                self.status = "WAIT_NEUTRAL"
                return events
            if self.neutral_depth_last is None:
                self.neutral_depth_last = depth
                self.neutral_roll_last = roll
                self.neutral_frames = 1
            else:
                depth_step = abs(depth - self.neutral_depth_last)
                roll_step = abs(wrap_degrees(roll - self.neutral_roll_last))
                if depth_step <= self.args.push_stop_delta and roll_step <= self.args.rotate_stop_delta:
                    self.neutral_frames += 1
                else:
                    self.neutral_frames = 0
                self.neutral_depth_last = depth
                self.neutral_roll_last = roll
            if self.neutral_frames < self.args.open_required_frames:
                self.status = f"WAIT_NEUTRAL {self.neutral_frames}/{self.args.open_required_frames}"
                return events
            self.depth_reference = depth
            self.depth_filtered = depth
            self.roll_reference = roll
            self.roll_filtered = roll
            self.phase = "READY"
            self.status = "READY"
            return events

        # Do not accumulate any depth/angle history while the open-palm gate
        # is warming up. The frame that reaches the required count becomes a
        # fresh calibration frame and cannot itself trigger an action.
        if self.active_kind is None and self.open_frames < self.args.open_required_frames:
            self._clear_reference("WAIT_OPEN")
            self.phase = "OPENING"
            self.status = f"OPENING {self.open_frames}/{self.args.open_required_frames}"
            return events

        alpha = self.args.smoothing_alpha
        if self.depth_reference is None:
            self.depth_reference = depth
            self.depth_filtered = depth
            self.roll_reference = roll
            self.roll_filtered = roll
            self.phase = "READY"
            self.status = "READY"
            return events

        previous_depth = self.depth_filtered
        previous_roll = self.roll_filtered
        self.depth_filtered = alpha * depth + (1.0 - alpha) * self.depth_filtered
        roll_step = wrap_degrees(roll - self.roll_filtered)
        self.roll_filtered = wrap_degrees(self.roll_filtered + alpha * roll_step)
        depth_delta_frame = self.depth_filtered - previous_depth
        roll_delta_frame = wrap_degrees(self.roll_filtered - previous_roll)
        displacement = self.depth_reference - self.depth_filtered
        roll_delta = wrap_degrees(self.roll_filtered - self.roll_reference)

        if self.active_kind is None:
            if self.phase == "WAIT_NEUTRAL":
                if abs(displacement) <= self.args.push_rearm and abs(roll_delta) <= self.args.rotate_rearm:
                    self.depth_reference = self.depth_filtered
                    self.depth_filtered = self.depth_reference
                    self.roll_reference = self.roll_filtered
                    self.roll_filtered = self.roll_reference
                    self.phase = "READY"
                    self.status = "READY"
                else:
                    self.status = "WAIT_NEUTRAL"
                return events

            if self.phase != "READY":
                self.depth_reference = self.depth_filtered
                self.depth_filtered = self.depth_reference
                self.roll_reference = self.roll_filtered
                self.roll_filtered = self.roll_reference
                self.phase = "READY"
                self.status = "READY"
                return events

            if abs(displacement) >= self.args.push_start:
                self.active_kind = "PUSH"
                self.active_direction = "FORWARD" if displacement > 0 else "BACKWARD"
                self.peak_displacement = abs(displacement)
                self.settle_frames = 0
                self.phase = f"PUSHING_{self.active_direction}"
                self.status = self.phase
                return events
            if abs(roll_delta) >= self.args.rotate_start:
                self.active_kind = "ROTATE"
                self.active_direction = "LEFT" if roll_delta < 0 else "RIGHT"
                self.peak_angle = abs(roll_delta)
                self.settle_frames = 0
                self.phase = f"ROTATING_{self.active_direction}"
                self.status = self.phase
                return events
            self.status = "READY"
            return events

        # The following logic is reached only while the hand stayed open.
        if self.active_kind == "PUSH":
            same_direction = displacement > 0 if self.active_direction == "FORWARD" else displacement < 0
            if not same_direction:
                self._reset_motion("CANCELLED_REVERSED")
                return events
            self.peak_displacement = max(self.peak_displacement, abs(displacement))
            stopped = abs(depth_delta_frame) <= self.args.push_stop_delta
            if self.peak_displacement >= self.args.push_trigger and stopped:
                self.settle_frames += 1
            else:
                self.settle_frames = 0
            self.status = f"{self.phase} peak={self.peak_displacement:.3f} settle={self.settle_frames}/{self.args.settle_frames}"
            if self.settle_frames >= self.args.settle_frames:
                event = self._commit_push()
                if event is not None:
                    events.append(event)
                return events
            return events

        # ROTATE: the hand must remain open until the angle threshold is
        # reached and angular motion settles before emitting an event.
        same_direction = roll_delta < 0 if self.active_direction == "LEFT" else roll_delta > 0
        if not same_direction:
            self._reset_motion("CANCELLED_REVERSED")
            return events
        self.peak_angle = max(self.peak_angle, abs(roll_delta))
        stopped = abs(roll_delta_frame) <= self.args.rotate_stop_delta
        if self.peak_angle >= self.args.rotate_trigger and stopped:
            self.settle_frames += 1
        else:
            self.settle_frames = 0
        self.status = f"{self.phase} peak={self.peak_angle:.1f} settle={self.settle_frames}/{self.args.settle_frames}"
        if self.settle_frames >= self.args.settle_frames:
            event = self._commit_rotation()
            if event is not None:
                events.append(event)
        return events

    def _reset_motion(self, status: str) -> None:
        self.active_kind = None
        self.active_direction = None
        self.peak_displacement = 0.0
        self.peak_angle = 0.0
        self.settle_frames = 0
        self.neutral_frames = 0
        self.neutral_depth_last = None
        self.neutral_roll_last = None
        self.phase = "WAIT_OPEN" if "NOT_OPEN" in status else "WAIT_NEUTRAL"
        self.status = status
        if "NOT_OPEN" in status:
            self._clear_reference("WAIT_OPEN")

    def _reset_for_fresh_open(self, status: str) -> None:
        """Discard every prior sample so the next READY gets a fresh baseline."""
        self.active_kind = None
        self.active_direction = None
        self.peak_displacement = 0.0
        self.peak_angle = 0.0
        self.settle_frames = 0
        self.neutral_frames = 0
        self.neutral_depth_last = None
        self.neutral_roll_last = None
        self.open_frames = 0
        self._clear_reference("WAIT_OPEN")
        self.status = status

    def _clear_reference(self, phase: str = "WAIT_OPEN") -> None:
        self.depth_reference = None
        self.depth_filtered = None
        self.roll_reference = None
        self.roll_filtered = None
        self.phase = phase

    def _commit_push(self) -> str | None:
        event = self.active_direction
        # Interaction mapping: hand forward moves the screen one step farther
        # (0.4 -> 0.5 -> 0.6); hand backward moves it one step closer.
        if event == "FORWARD" and self.motion_level < 2 and self.motion_offsets[self.motion_level + 1] <= self.args.motion_max:
            self.motion_level += 1
        elif event == "BACKWARD" and self.motion_level > 0 and self.motion_offsets[self.motion_level - 1] >= self.args.motion_min:
            self.motion_level -= 1
        else:
            self._reset_motion("PUSH_LIMIT")
            return None
        self._reset_motion("PUSH_COMPLETED_WAIT_NEUTRAL")
        return event

    def _commit_rotation(self) -> str | None:
        event = f"ROTATE_{self.active_direction}"
        if self.orientation == "LANDSCAPE":
            self.orientation = "PORTRAIT_LEFT" if self.active_direction == "LEFT" else "PORTRAIT_RIGHT"
        elif self.orientation == "PORTRAIT_LEFT" and self.active_direction == "RIGHT":
            self.orientation = "LANDSCAPE"
        elif self.orientation == "PORTRAIT_RIGHT" and self.active_direction == "LEFT":
            self.orientation = "LANDSCAPE"
        else:
            self._reset_motion("ROTATE_LIMIT")
            return None
        self._reset_motion("ROTATE_COMPLETED_WAIT_NEUTRAL")
        return event


class MotionExecutor:
    """执行手势事件对应的 HOME 相对位移和姿态动作。"""

    def __init__(self, args: argparse.Namespace, home_position: np.ndarray):
        sdk_scripts = str((REPO_ROOT / "panthera_python" / "scripts").resolve())
        if sdk_scripts not in sys.path:
            sys.path.insert(0, sdk_scripts)
        from Panthera_lib import Panthera

        self.args = args
        self.home_position = home_position
        self.robot = Panthera(str(args.config.resolve()))
        if self.robot.motor_count != JOINT_COUNT or getattr(self.robot, "gripper_enabled", True):
            raise RuntimeError("手势动作测试要求无夹爪六轴配置")
        print(f"机械臂移动到 HOME：{home_position.tolist()}")
        reached = self.robot.Joint_Pos_Vel(
            home_position.tolist(),
            [0.20] * JOINT_COUNT,
            self.robot.max_torque.tolist(),
            iswait=True,
            tolerance=0.01,
            timeout=30.0,
        )
        if not reached:
            raise RuntimeError("机械臂未能到达 HOME")
        home_fk = self.robot.forward_kinematics(home_position)
        if home_fk is None:
            raise RuntimeError("无法计算 HOME 正运动学")
        self.home_position_xyz = np.asarray(home_fk["position"], dtype=np.float64)
        self.home_rotation = np.asarray(home_fk["rotation"], dtype=np.float64)
        self.home_joint6 = float(home_position[5])

    def target_position(self, state: GestureState) -> np.ndarray:
        """Return the HOME-relative Cartesian target for push/pull only."""
        position = self.home_position_xyz + np.array([state.x_offset, 0.0, 0.0])
        return position

    def execute(self, event: str, state: GestureState) -> bool:
        print(
            f"事件 {event}：目标 HOME 相对 X={state.x_offset:+.2f} m，"
            f"姿态={state.orientation}"
        )
        if event in ("ROTATE_LEFT", "ROTATE_RIGHT"):
            # 旋转事件严格走 joint6 关节空间：J1~J5 保持当前反馈值，
            # 只改变 J6。这里不调用 moveL/IK，避免其它关节联动。
            current_q = np.asarray(self.robot.get_current_pos(), dtype=np.float64)
            if current_q.shape != (JOINT_COUNT,) or not np.all(np.isfinite(current_q)):
                raise RuntimeError("无法读取当前六关节角度")
            target_q = current_q.copy()
            if state.orientation == "PORTRAIT_LEFT":
                target_q[5] = self.home_joint6 - 0.5 * math.pi
            elif state.orientation == "PORTRAIT_RIGHT":
                target_q[5] = self.home_joint6 + 0.5 * math.pi
            elif state.orientation == "LANDSCAPE":
                target_q[5] = self.home_joint6
            else:
                raise RuntimeError(f"未知屏幕姿态：{state.orientation}")
            velocity = [0.0] * JOINT_COUNT
            velocity[5] = min(0.20, float(getattr(self.robot, "velocity_limits", [1.0] * JOINT_COUNT)[5]))
            print(f"仅旋转 J6：{current_q[5]:+.3f} -> {target_q[5]:+.3f} rad；J1~J5 保持当前值")
            return bool(self.robot.Joint_Pos_Vel(
                target_q.tolist(), velocity, self.robot.max_torque.tolist(),
                iswait=True, tolerance=0.01, timeout=30.0,
            ))

        position = self.target_position(state)
        return bool(self.robot.moveL(
            target_position=position,
            target_rotation=self.home_rotation,
            duration=self.args.motion_duration,
            use_spline=True,
        ))

    def close(self) -> None:
        print("退出：返回 HOME...")
        try:
            self.robot.Joint_Pos_Vel(
                self.home_position.tolist(),
                [0.20] * JOINT_COUNT,
                self.robot.max_torque.tolist(),
                iswait=True,
                tolerance=0.01,
                timeout=30.0,
            )
        finally:
            self.robot.set_stop()


def create_landmarker(args: argparse.Namespace):
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model.resolve())),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_detection,
        min_hand_presence_confidence=args.min_presence,
        min_tracking_confidence=args.min_tracking,
    )
    return vision.HandLandmarker.create_from_options(options)


def angle_degrees(a, b, c) -> float:
    """Angle ABC in normalized landmark coordinates."""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z], dtype=np.float64)
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z], dtype=np.float64)
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator <= 1e-9:
        return 0.0
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def classify_open_palm(landmarks) -> tuple[bool, int]:
    """Classify an open palm from all five finger chains.

    This is deliberately conservative: all five fingers must pass the
    extension tests, so a fist or partially closed hand cannot arm a gesture.
    """
    wrist = landmarks[0]
    finger_chains = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
    extended = 0
    for mcp, pip, tip in finger_chains:
        pip_angle = angle_degrees(landmarks[mcp], landmarks[pip], landmarks[tip])
        tip_distance = np.linalg.norm(np.array([landmarks[tip].x - wrist.x, landmarks[tip].y - wrist.y]))
        pip_distance = np.linalg.norm(np.array([landmarks[pip].x - wrist.x, landmarks[pip].y - wrist.y]))
        if pip_angle >= 145.0 and tip_distance > pip_distance * 1.05:
            extended += 1

    # The thumb uses a different chain because its MCP points sideways.
    thumb_angle = angle_degrees(landmarks[1], landmarks[2], landmarks[4])
    thumb_tip_distance = np.linalg.norm(np.array([landmarks[4].x - wrist.x, landmarks[4].y - wrist.y]))
    thumb_ip_distance = np.linalg.norm(np.array([landmarks[3].x - wrist.x, landmarks[3].y - wrist.y]))
    thumb_extended = thumb_angle >= 125.0 and thumb_tip_distance > thumb_ip_distance * 1.05
    finger_count = extended + int(thumb_extended)
    return finger_count == 5, finger_count


def sample_hand(
    landmarker,
    color: np.ndarray,
    depth_frame,
    hand_score_min: float = 0.75,
    palm_depth_spread_max: float = 0.08,
):
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(color, cv2.COLOR_BGR2RGB),
    )
    result = landmarker.detect(image)
    if not result.hand_landmarks:
        return None
    hand_index = 0
    if result.handedness:
        hand_index = max(
            range(min(len(result.hand_landmarks), len(result.handedness))),
            key=lambda index: float(result.handedness[index][0].score),
        )
    landmarks = result.hand_landmarks[hand_index]
    height, width = color.shape[:2]
    depth_values = []
    for index in (0, 5, 9, 13, 17):
        x = min(width - 1, max(0, int(round(landmarks[index].x * width))))
        y = min(height - 1, max(0, int(round(landmarks[index].y * height))))
        distance = float(depth_frame.get_distance(x, y))
        if math.isfinite(distance) and distance > 0.0:
            depth_values.append(distance)
    if not depth_values:
        return None

    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    roll_deg = math.degrees(math.atan2(
        float(middle_mcp.x - wrist.x), float(-(middle_mcp.y - wrist.y))
    ))
    handedness, score = "Unknown", 0.0
    if result.handedness and hand_index < len(result.handedness):
        category = result.handedness[hand_index][0]
        handedness, score = category.category_name or "Unknown", float(category.score)
    if score < hand_score_min:
        return None
    points = [
        (
            min(width - 1, max(0, int(round(point.x * width)))),
            min(height - 1, max(0, int(round(point.y * height)))),
        )
        for point in landmarks
    ]
    open_palm, open_fingers = classify_open_palm(landmarks)
    depth_spread = max(depth_values) - min(depth_values)
    if depth_spread > palm_depth_spread_max:
        return None
    return {
        "palm_depth": float(np.median(depth_values)),
        "roll_deg": roll_deg,
        "handedness": handedness,
        "score": score,
        "points": points,
        "open_palm": open_palm,
        "open_fingers": open_fingers,
    }


def draw_hand(image: np.ndarray, measurement: dict | None) -> None:
    if measurement is None:
        return
    points = measurement["points"]
    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(image, points[connection.start], points[connection.end], (0, 220, 0), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, point, 4, (0, 128, 255), -1, cv2.LINE_AA)


def run(args: argparse.Namespace) -> None:
    state = GestureState(args)
    executor = None
    if args.enable_motion:
        executor = MotionExecutor(args, load_home_position(args.home_file))
        print("真实动作已启用：手势触发一次动作，完成后等待手回中立。")
    else:
        print("观察模式：不连接 Panthera，不发送电机命令。")
    landmarker = create_landmarker(args)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    align = rs.align(rs.stream.color)
    try:
        pipeline.start(config)
        print("独立手势测试已启动：按 Q 或 Esc 退出。")
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        previous_time = time.perf_counter()
        fps = 0.0
        while True:
            frames = align.process(pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            color = np.asanyarray(color_frame.get_data()).copy()
            measurement = sample_hand(landmarker, color, depth_frame)
            events = state.update(measurement)
            for event in events:
                if executor is None:
                    print(
                        f"模拟事件 {event}：目标 HOME 相对 X={state.x_offset:+.2f} m，"
                        f"姿态={state.orientation}"
                    )
                elif not executor.execute(event, state):
                    print(f"事件 {event} 执行失败。")
            draw_hand(color, measurement)
            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps
            hand_text = "NO_HAND" if measurement is None else (
                f"{measurement['handedness']} {measurement['score']:.2f} "
                f"fingers={measurement['open_fingers']}/5 "
                f"{'OPEN' if measurement['open_palm'] else 'CLOSED'} "
                f"depth={measurement['palm_depth']:.2f}m roll={measurement['roll_deg']:+.0f}deg"
            )
            cv2.putText(color, f"{hand_text}  FPS={fps:.1f}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(color, f"HOME X offset={state.x_offset:+.2f}m {state.orientation} | {state.status}", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(color, "forward/back: +/- X 0.10m | rotate: +/- 90deg | Q: quit", (16, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, color)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        landmarker.close()
        cv2.destroyAllWindows()
        pipeline.stop()
        if executor is not None:
            executor.close()


def main() -> int:
    args = parse_args()
    validate_args(args)
    run(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}")
