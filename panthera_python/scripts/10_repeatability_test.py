#!/usr/bin/env python3
"""用算法驱动测量 Panthera 的重复定位误差。

每个目标关节姿态会被多次重复到达。脚本记录实际关节反馈、FK TCP 位姿，
并在相机可用时记录 ArUco 的 T_camera_target。该测试不修改手眼标定文件。

示例：
  python 10_repeatability_test.py --dry-run
  python 10_repeatability_test.py --pose 0,0.8,0.8,0.3,0,0 --cycles 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CALIBRATION_SCRIPT = SCRIPT_DIR / "8_eye_to_hand_calibration.py"
CONFIG_PATH = SCRIPT_DIR / "../robot_param/Follower.yaml"
MAX_TORQUE = [21.0, 36.0, 36.0, 21.0, 10.0, 10.0]
DEFAULT_POSES = [
    [0.0, 0.6, 0.6, 0.2, 0.0, 0.0],
    [0.0, 0.9, 0.8, 0.3, 0.0, 0.0],
]


def load_calibration_module():
    spec = importlib.util.spec_from_file_location(
        "panthera_calibration_for_repeatability", CALIBRATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {CALIBRATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_pose(value: str) -> list[float]:
    try:
        pose = [float(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"姿态必须是逗号分隔的数字：{value}") from error
    if len(pose) != 6 or not np.all(np.isfinite(pose)):
        raise argparse.ArgumentTypeError("每个姿态必须包含 6 个有限关节角")
    return pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose", action="append", type=parse_pose,
        help="六关节目标角度，格式 j1,j2,j3,j4,j5,j6；可重复指定，默认使用两个保守姿态",
    )
    parser.add_argument("--cycles", type=int, default=5, help="每个姿态重复次数，默认 5")
    parser.add_argument("--settle", type=float, default=1.0, help="到位后稳定等待秒数")
    parser.add_argument("--frames", type=int, default=10, help="每次到位采集相机帧数")
    parser.add_argument("--speed", type=float, default=0.3, help="关节到位控制速度 rad/s")
    parser.add_argument("--timeout", type=float, default=30.0, help="单次到位超时秒数")
    parser.add_argument("--dry-run", action="store_true", help="只检查参数，不连接机械臂和相机")
    parser.add_argument("--yes", action="store_true", help="跳过真机确认")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.cycles < 2:
        raise ValueError("cycles 至少为 2")
    if args.settle < 0 or args.frames < 1 or args.speed <= 0 or args.timeout <= 0:
        raise ValueError("settle、frames、speed、timeout 参数无效")


def transform_spread(transforms: list[np.ndarray], calibration_module) -> dict | None:
    if not transforms:
        return None
    reference = calibration_module.mean_transform(transforms)
    translation_errors = np.array([
        1000.0 * np.linalg.norm(
            (calibration_module.invert_transform(reference) @ transform)[:3, 3])
        for transform in transforms
    ])
    rotation_errors = np.array([
        calibration_module.rotation_angle_deg(
            (calibration_module.invert_transform(reference) @ transform)[:3, :3])
        for transform in transforms
    ])
    return {
        "sample_count": len(transforms),
        "reference_transform": reference.tolist(),
        "translation_error_mm": {
            "rms": float(np.sqrt(np.mean(translation_errors ** 2))),
            "max": float(np.max(translation_errors)),
        },
        "rotation_error_deg": {
            "rms": float(np.sqrt(np.mean(rotation_errors ** 2))),
            "max": float(np.max(rotation_errors)),
        },
    }


def summarize_pose(records: list[dict], target: np.ndarray, calibration_module) -> dict:
    actual_positions = np.asarray([record["actual_joint_positions_rad"] for record in records])
    joint_errors = actual_positions - target[None, :]
    fk_transforms = [np.asarray(record["T_base_tcp"], dtype=float) for record in records]
    camera_transforms = [
        np.asarray(record["T_camera_target"], dtype=float)
        for record in records if record.get("T_camera_target") is not None
    ]
    return {
        "target_joint_positions_rad": target.tolist(),
        "sample_count": len(records),
        "joint_error_rms_rad": np.sqrt(np.mean(joint_errors ** 2, axis=0)).tolist(),
        "joint_error_max_abs_rad": np.max(np.abs(joint_errors), axis=0).tolist(),
        "joint_error_overall_rms_rad": float(np.sqrt(np.mean(joint_errors ** 2))),
        "fk_tcp_spread": transform_spread(fk_transforms, calibration_module),
        "camera_target_spread": transform_spread(camera_transforms, calibration_module),
    }


def save_report(payload: dict) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = SCRIPT_DIR / f"repeatability_test_{timestamp}.json"
    suffix = 1
    while output.exists():
        output = SCRIPT_DIR / f"repeatability_test_{timestamp}_{suffix}.json"
        suffix += 1
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    temporary.replace(output)
    return output


def main() -> int:
    args = parse_args()
    validate_args(args)
    poses = args.pose if args.pose else DEFAULT_POSES
    print(f"测试姿态数={len(poses)}，每个姿态重复={args.cycles} 次")
    print("注意：脚本不会自动回零；中断时只发送停止命令。")
    if args.dry_run:
        print("dry-run 参数检查通过：")
        for index, pose in enumerate(poses):
            print(f"  姿态 {index + 1}: {np.array2string(np.asarray(pose), precision=4)}")
        return 0
    if not args.yes:
        confirmation = input(
            "确认机械臂工作空间安全，并输入 REPEATABILITY 开始："
        ).strip()
        if confirmation != "REPEATABILITY":
            print("未确认，已退出。")
            return 0

    calibration_module = load_calibration_module()
    from Panthera_lib import Panthera

    robot = None
    pipeline = None
    report_poses = []
    try:
        robot = Panthera(str(CONFIG_PATH.resolve()))
        if robot.motor_count != 6:
            raise RuntimeError(f"期望 6 个关节，实际为 {robot.motor_count}")
        lower = np.asarray(robot.joint_limits["lower"], dtype=float)
        upper = np.asarray(robot.joint_limits["upper"], dtype=float)
        for pose in poses:
            target = np.asarray(pose, dtype=float)
            if np.any(target < lower) or np.any(target > upper):
                raise ValueError(f"目标姿态超出关节限位：{target}")

        try:
            pipeline, camera_matrix, dist_coeffs = calibration_module.init_camera()
            print("相机已启动，将同时记录 ArUco 检测结果。")
        except Exception as error:
            print(f"相机启动失败，将仅测试关节和 FK：{error}")
            pipeline = None

        move_speed = [args.speed] * robot.motor_count
        staging = np.zeros(robot.motor_count, dtype=float)
        for pose_index, pose in enumerate(poses):
            target = np.asarray(pose, dtype=float)
            records = []
            print(f"\n开始测试姿态 {pose_index + 1}/{len(poses)}：{target}")
            for cycle in range(args.cycles):
                if not robot.Joint_Pos_Vel(
                    target, move_speed, MAX_TORQUE, iswait=True,
                    tolerance=0.03, timeout=args.timeout,
                ):
                    raise RuntimeError(f"姿态 {pose_index + 1} 第 {cycle + 1} 次未到位")
                time.sleep(args.settle)
                actual_q = np.asarray(robot.get_current_pos(), dtype=float)
                fk = robot.forward_kinematics(actual_q)
                if fk is None:
                    raise RuntimeError("正运动学计算失败")
                t_base_tcp = calibration_module.make_transform(
                    fk["rotation"], fk["position"])
                camera_target = None
                if pipeline is not None:
                    detections = []
                    for _ in range(args.frames):
                        image = calibration_module.get_color_frame(pipeline)
                        if image is None:
                            continue
                        detected, _, _ = calibration_module.detect_target(
                            image, camera_matrix, dist_coeffs)
                        if detected is not None:
                            detections.append(detected)
                    if detections:
                        camera_target = calibration_module.mean_transform(detections).tolist()
                records.append({
                    "cycle": cycle + 1,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "target_joint_positions_rad": target.tolist(),
                    "actual_joint_positions_rad": actual_q.tolist(),
                    "T_base_tcp": t_base_tcp.tolist(),
                    "T_camera_target": camera_target,
                })
                print(f"  第 {cycle + 1}/{args.cycles} 次：最大关节误差 "
                      f"{np.max(np.abs(actual_q - target)) * 1000:.2f} mrad")
                if cycle + 1 < args.cycles:
                    if not robot.Joint_Pos_Vel(
                        staging, move_speed, MAX_TORQUE, iswait=True,
                        tolerance=0.03, timeout=args.timeout,
                    ):
                        raise RuntimeError("移动到中间姿态失败")
                    time.sleep(args.settle)
            summary = summarize_pose(records, target, calibration_module)
            report_poses.append({"summary": summary, "samples": records})
            fk_spread = summary["fk_tcp_spread"]
            print(f"姿态 {pose_index + 1} FK TCP 平移 RMS："
                  f"{fk_spread['translation_error_mm']['rms']:.2f} mm，"
                  f"最大：{fk_spread['translation_error_mm']['max']:.2f} mm")

        payload = {
            "schema_version": "panthera-repeatability-v1",
            "test": "algorithmic_repeatability",
            "cycles_per_pose": args.cycles,
            "settle_seconds": args.settle,
            "poses": report_poses,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        output = save_report(payload)
        print(f"\n重复性报告已保存：{output}")
        print("判读：FK TCP 平移 RMS 若已超过 10 mm，优先修机器人本体/零位/结构；"
              "若 FK 很小而相机 spread 很大，优先修视觉和标定板安装。")
    except KeyboardInterrupt:
        if robot is not None:
            robot.set_stop()
        print("\n测试被中断，已发送停止命令。")
        return 130
    except Exception:
        if robot is not None:
            robot.set_stop()
        raise
    finally:
        if pipeline is not None:
            pipeline.stop()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, RuntimeError, ValueError, KeyError) as error:
        print(f"错误：{error}")
        raise SystemExit(1)
