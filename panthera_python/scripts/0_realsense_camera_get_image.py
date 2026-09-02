import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pyrealsense2 as rs


SCRIPT_DIR = Path(__file__).resolve().parent


def _intrinsics_dict(intrinsics):
    return {
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.ppx),
        "cy": float(intrinsics.ppy),
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
    }


def save_fixture(
    out_dir: Path,
    color_frame,
    depth_frame,
    color_intr,
    depth_intr,
    depth_scale: float,
) -> None:
    """Save the currently displayed aligned RGB/depth pair."""
    out_dir.mkdir(parents=True, exist_ok=True)
    color = np.asanyarray(color_frame.get_data()).copy()
    depth = np.asanyarray(depth_frame.get_data()).copy()

    cv2.imwrite(str(out_dir / "color.png"), color)
    np.save(out_dir / "depth_raw.npy", depth)
    info = {
        "color_intrinsics": _intrinsics_dict(color_intr),
        "depth_intrinsics": _intrinsics_dict(depth_intr),
        "depth_scale_m_per_unit": float(depth_scale),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "color_frame_number": int(color_frame.get_frame_number()),
        "depth_frame_number": int(depth_frame.get_frame_number()),
        "color_device_timestamp_ms": float(color_frame.get_timestamp()),
        "depth_device_timestamp_ms": float(depth_frame.get_timestamp()),
        "depth_alignment": "depth_to_color",
    }
    with (out_dir / "camera_info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    print(f"Saved fixture: {out_dir}")
    print(json.dumps(info, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show a live aligned D435 stream and save a frame on request."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "realsense_fixture",
        help="fixture directory written by S/Space (default: SDK scripts/realsense_fixture)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()
    depth_intr = depth_stream.get_intrinsics()
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    align = rs.align(rs.stream.color)

    print("D435 live preview started.")
    print("S/Space: save current aligned RGB/depth fixture")
    print("Q/Esc: quit")

    try:
        for _ in range(max(0, args.warmup)):
            align.process(pipeline.wait_for_frames())

        window_name = "D435 live preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data()).copy()
            preview = color.copy()
            cv2.putText(
                preview,
                "S/Space: save  |  Q/Esc: quit",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S"), 32):
                save_fixture(
                    args.output_dir,
                    color_frame,
                    depth_frame,
                    color_intr,
                    depth_intr,
                    depth_scale,
                )
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
