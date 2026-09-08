#!/usr/bin/env python3
"""RealSense + MediaPipe Hand Landmarker 预览。

本程序只读取相机并显示检测结果，不连接或驱动 Panthera 机械臂。
"""

from __future__ import annotations

import argparse
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
DEFAULT_WINDOW = "HTPal hand landmark preview"


def parse_args() -> argparse.Namespace:
    """Parse camera and detector settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Hand Landmarker .task 路径")
    parser.add_argument("--width", type=int, default=640, help="彩色/深度流宽度")
    parser.add_argument("--height", type=int, default=480, help="彩色/深度流高度")
    parser.add_argument("--fps", type=int, default=30, help="彩色/深度流帧率")
    parser.add_argument("--num-hands", type=int, default=1, help="最多检测的手数")
    parser.add_argument("--min-detection", type=float, default=0.5, help="手部检测置信度")
    parser.add_argument("--min-presence", type=float, default=0.5, help="手部存在置信度")
    parser.add_argument("--min-tracking", type=float, default=0.5, help="手部跟踪置信度")
    parser.add_argument("--no-depth", action="store_true", help="只显示关键点，不启动深度流")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate settings before opening the camera."""
    if not args.model.is_file():
        raise FileNotFoundError(f"Hand Landmarker 模型不存在：{args.model}")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("--width、--height 和 --fps 必须大于 0")
    if not 1 <= args.num_hands <= 2:
        raise ValueError("--num-hands 必须是 1 或 2")
    for name in ("min_detection", "min_presence", "min_tracking"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} 必须位于 [0, 1] 范围内")


def create_landmarker(args: argparse.Namespace) -> vision.HandLandmarker:
    """Create a MediaPipe landmarker in per-frame IMAGE mode."""
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model.resolve())),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_detection,
        min_hand_presence_confidence=args.min_presence,
        min_tracking_confidence=args.min_tracking,
    )
    return vision.HandLandmarker.create_from_options(options)


def palm_depth(depth_frame, landmarks, width: int, height: int) -> float | None:
    """Return the median depth of stable palm landmarks in metres."""
    if depth_frame is None:
        return None

    # Wrist plus four MCP landmarks; fingertips are deliberately excluded.
    indices = (0, 5, 9, 13, 17)
    distances = []
    for index in indices:
        landmark = landmarks[index]
        x = min(width - 1, max(0, int(round(landmark.x * width))))
        y = min(height - 1, max(0, int(round(landmark.y * height))))
        distance = float(depth_frame.get_distance(x, y))
        if distance > 0.0:
            distances.append(distance)
    return float(np.median(distances)) if distances else None


def draw_result(frame: np.ndarray, result, depth_frame) -> None:
    """Draw landmarks, connections, handedness, and palm depth."""
    height, width = frame.shape[:2]
    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    for hand_index, landmarks in enumerate(result.hand_landmarks):
        points = []
        for landmark in landmarks:
            point = (
                min(width - 1, max(0, int(round(landmark.x * width)))),
                min(height - 1, max(0, int(round(landmark.y * height)))),
            )
            points.append(point)

        for connection in connections:
            cv2.line(frame, points[connection.start], points[connection.end], (0, 220, 0), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 4, (0, 128, 255), -1, cv2.LINE_AA)

        handedness = "hand"
        score = 0.0
        if hand_index < len(result.handedness) and result.handedness[hand_index]:
            category = result.handedness[hand_index][0]
            handedness = category.category_name or "hand"
            score = float(category.score)
        depth = palm_depth(depth_frame, landmarks, width, height)
        depth_text = f"  palm={depth:.2f}m" if depth is not None else "  palm=--"
        label = f"{handedness} {score:.2f}{depth_text}"
        cv2.putText(
            frame,
            label,
            (points[0][0] + 8, max(22, points[0][1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def run_preview(args: argparse.Namespace) -> None:
    """Capture aligned RealSense frames and display hand landmarks."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if not args.no_depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    align = rs.align(rs.stream.color) if not args.no_depth else None
    landmarker = create_landmarker(args)
    try:
        profile = pipeline.start(config)
        depth_scale = None
        if not args.no_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = float(depth_sensor.get_depth_scale())
            print(f"RealSense depth scale: {depth_scale:g} m/unit")
        print("预览已启动：按 Q 或 Esc 退出。")
        cv2.namedWindow(DEFAULT_WINDOW, cv2.WINDOW_NORMAL)
        previous_time = time.perf_counter()
        fps = 0.0
        frame_index = 0
        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            depth_frame = frames.get_depth_frame() if align is not None else None
            color = np.asanyarray(color_frame.get_data()).copy()
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(image)
            draw_result(color, result, depth_frame)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps
            cv2.putText(
                color,
                f"hands: {len(result.hand_landmarks)}  FPS: {fps:.1f}  depth: {'on' if depth_scale else 'off'}",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(DEFAULT_WINDOW, color)
            frame_index += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        landmarker.close()
        cv2.destroyAllWindows()
        pipeline.stop()


def main() -> int:
    """Run the read-only hand landmark preview."""
    args = parse_args()
    validate_args(args)
    run_preview(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
