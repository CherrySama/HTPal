#!/usr/bin/env python3
"""用未参与拟合的新姿态验证固定的 Eye-to-Hand 标定结果。

本脚本不会重新计算 T_base_camera，也不会主动驱动机械臂。操作者在重力
补偿下手动改变姿态，脚本使用训练阶段保存的 T_tcp_target_reference，
统计留出姿态上的空间一致性误差和像素重投影误差。

按键：SPACE/S 采集，C 计算并保存，R 清空，ESC/Q 退出。
"""

import importlib.util
import json
from pathlib import Path
import time

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CALIBRATION_SCRIPT = SCRIPT_DIR / "8_eye_to_hand_calibration.py"
CALIBRATION_FILE = SCRIPT_DIR / "eye_to_hand_calibration.json"
MIN_VALIDATION_SAMPLES = 10
MAX_CAPTURE_SPEED_RAD_S = 0.05


def load_calibration_module():
    """加载以数字开头的标定脚本，以复用相机、检测和运动学定义。"""
    spec = importlib.util.spec_from_file_location(
        "panthera_eye_to_hand_calibration", CALIBRATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载标定脚本：{CALIBRATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validated_transform(value, name):
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError(f"{name} 必须是有限的 4x4 矩阵")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise RuntimeError(f"{name} 的齐次矩阵末行无效")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
        raise RuntimeError(f"{name} 的旋转矩阵无效")
    return transform


def load_fixed_calibration(calibration_module):
    """读取并严格检查固定标定结果及训练阶段目标参考位姿。"""
    if not CALIBRATION_FILE.is_file():
        raise RuntimeError(f"找不到标定结果：{CALIBRATION_FILE}")
    with CALIBRATION_FILE.open("r", encoding="utf-8") as source:
        payload = json.load(source)

    if payload.get("calibration_type") != "eye_to_hand":
        raise RuntimeError("标定文件不是 eye_to_hand 类型")
    if calibration_module.CALIBRATION_TYPE != "eye_to_hand":
        raise RuntimeError("8_eye_to_hand_calibration.py 当前不是 eye_to_hand 配置")

    validation = payload.get("validation", {})
    if "T_tcp_target_reference" not in validation:
        raise RuntimeError(
            "当前标定文件缺少 validation.T_tcp_target_reference。请先使用更新后的 "
            "8_eye_to_hand_calibration.py 重新采集并标定；不能用验证姿态反算该参考值，"
            "否则会把验证集泄漏进基准。")

    raw_samples_name = validation.get("raw_samples_file")
    if not isinstance(raw_samples_name, str):
        raise RuntimeError("标定结果没有记录训练原始样本文件")
    raw_samples_path = SCRIPT_DIR / Path(raw_samples_name).name
    if not raw_samples_path.is_file():
        raise RuntimeError(f"找不到训练原始样本：{raw_samples_path}")
    with raw_samples_path.open("r", encoding="utf-8") as source:
        raw_payload = json.load(source)
    raw_samples = raw_payload.get("samples")
    if not isinstance(raw_samples, list):
        raise RuntimeError("训练原始样本文件中的 samples 无效")
    if len(raw_samples) != payload.get("num_samples"):
        raise RuntimeError(
            f"训练原始样本数量 {len(raw_samples)} 与标定结果记录的 "
            f"{payload.get('num_samples')} 不一致；请不要在标定完成后覆盖原始样本文件")
    training_base_to_tcp = [
        validated_transform(sample.get("T_base_tcp"), f"训练样本 {index} T_base_tcp")
        for index, sample in enumerate(raw_samples)
    ]
    training_camera = raw_payload.get("camera", {})
    training_camera_matrix = np.asarray(
        training_camera.get("matrix"), dtype=np.float64)
    training_dist_coeffs = np.asarray(
        training_camera.get("dist_coeffs"), dtype=np.float64).reshape(-1)
    if (training_camera_matrix.shape != (3, 3)
            or not np.all(np.isfinite(training_camera_matrix))
            or training_dist_coeffs.size == 0
            or not np.all(np.isfinite(training_dist_coeffs))):
        raise RuntimeError("训练原始样本中的相机内参矩阵无效")

    target = payload.get("target")
    if not isinstance(target, dict):
        raise RuntimeError("标定文件缺少 target 元数据")
    expected = calibration_module.target_metadata()
    if target.get("pattern") != expected["pattern"]:
        raise RuntimeError("标定结果与当前脚本的标定板类型不一致")
    if target["pattern"] == "aruco":
        if target.get("aruco_id") != expected["aruco_id"]:
            raise RuntimeError("标定结果与当前脚本的 ArUco ID 不一致")
        if not np.isclose(target.get("aruco_size_m", -1), expected["aruco_size_m"]):
            raise RuntimeError("标定结果与当前脚本的 ArUco 边长不一致")
    elif target["pattern"] == "chessboard":
        if target.get("chessboard_inner_corners") != expected["chessboard_inner_corners"]:
            raise RuntimeError("标定结果与当前脚本的棋盘格角点数不一致")
        if not np.isclose(target.get("square_size_m", -1), expected["square_size_m"]):
            raise RuntimeError("标定结果与当前脚本的棋盘格方格边长不一致")
    else:
        raise RuntimeError(f"不支持的标定板类型：{target['pattern']}")

    return {
        "source": payload,
        "T_base_camera": validated_transform(
            payload.get("T_base_camera"), "T_base_camera"),
        "T_tcp_target_reference": validated_transform(
            validation["T_tcp_target_reference"], "T_tcp_target_reference"),
        "target": target,
        "training_base_to_tcp": training_base_to_tcp,
        "training_camera_matrix": training_camera_matrix,
        "training_dist_coeffs": training_dist_coeffs,
    }


def pose_is_novel(candidate, reference_poses, calibration_module):
    """要求候选姿态与每个参考姿态至少有足够平移或旋转差异。"""
    for index, reference in enumerate(reference_poses):
        relative = calibration_module.invert_transform(reference) @ candidate
        translation_mm = 1000.0 * np.linalg.norm(relative[:3, 3])
        rotation_deg = calibration_module.rotation_angle_deg(relative[:3, :3])
        if translation_mm < 15.0 and rotation_deg < 8.0:
            return False, index, translation_mm, rotation_deg
    return True, None, None, None


def target_object_points(target):
    """按标定脚本采用的角点顺序生成目标三维点。"""
    if target["pattern"] == "aruco":
        half = float(target["aruco_size_m"]) / 2.0
        return np.array([
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)

    width, height = target["chessboard_inner_corners"]
    points = np.zeros((width * height, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    points *= float(target["square_size_m"])
    return points


def evaluate_sample(sample, fixed, camera_matrix, dist_coeffs, object_points,
                    calibration_module):
    """计算一个留出样本的空间残差和像素重投影残差。"""
    base_to_tcp = validated_transform(sample["T_base_tcp"], "T_base_tcp")
    camera_to_target = validated_transform(
        sample["T_camera_target"], "T_camera_target")
    image_points = np.asarray(sample["image_points_px"], dtype=np.float64)
    if image_points.shape != (len(object_points), 2):
        raise RuntimeError(
            f"图像角点形状应为 {(len(object_points), 2)}，实际为 {image_points.shape}")

    base_to_camera = fixed["T_base_camera"]
    tcp_to_target_reference = fixed["T_tcp_target_reference"]
    tcp_to_target_observed = (
        calibration_module.invert_transform(base_to_tcp)
        @ base_to_camera
        @ camera_to_target
    )
    target_delta = (
        calibration_module.invert_transform(tcp_to_target_reference)
        @ tcp_to_target_observed
    )

    base_to_camera_observed = (
        base_to_tcp
        @ tcp_to_target_reference
        @ calibration_module.invert_transform(camera_to_target)
    )
    camera_delta = (
        calibration_module.invert_transform(base_to_camera)
        @ base_to_camera_observed
    )

    camera_to_target_predicted = (
        calibration_module.invert_transform(base_to_camera)
        @ base_to_tcp
        @ tcp_to_target_reference
    )
    rotation_vector, _ = cv2.Rodrigues(camera_to_target_predicted[:3, :3])
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        camera_to_target_predicted[:3, 3],
        camera_matrix,
        dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    pixel_vectors = projected - image_points
    pixel_distances = np.linalg.norm(pixel_vectors, axis=1)

    evaluated = dict(sample)
    evaluated.update({
        "T_tcp_target_observed": tcp_to_target_observed.tolist(),
        "T_base_camera_observed": base_to_camera_observed.tolist(),
        "T_camera_target_predicted": camera_to_target_predicted.tolist(),
        "predicted_image_points_px": projected.tolist(),
        "target_translation_error_mm": float(
            1000.0 * np.linalg.norm(target_delta[:3, 3])),
        "target_rotation_error_deg": calibration_module.rotation_angle_deg(
            target_delta[:3, :3]),
        "camera_translation_error_mm": float(
            1000.0 * np.linalg.norm(camera_delta[:3, 3])),
        "camera_rotation_error_deg": calibration_module.rotation_angle_deg(
            camera_delta[:3, :3]),
        "corner_pixel_errors_px": pixel_distances.tolist(),
        "pixel_reprojection_rmse_px": float(
            np.sqrt(np.mean(np.sum(pixel_vectors ** 2, axis=1)))),
    })
    return evaluated


def error_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def summarize_samples(evaluated_samples):
    """汇总留出样本误差。"""
    if not evaluated_samples:
        raise RuntimeError("没有可汇总的验证样本")
    fields = (
        "target_translation_error_mm",
        "target_rotation_error_deg",
        "camera_translation_error_mm",
        "camera_rotation_error_deg",
        "pixel_reprojection_rmse_px",
    )
    summary = {field: error_statistics([sample[field] for sample in evaluated_samples])
               for field in fields}
    all_corner_errors = [error_value
                         for sample in evaluated_samples
                         for error_value in sample["corner_pixel_errors_px"]]
    summary["all_corner_pixel_error_px"] = error_statistics(all_corner_errors)
    summary["num_samples"] = len(evaluated_samples)
    return summary


def save_report(fixed, camera_matrix, dist_coeffs, evaluated_samples, summary):
    """保存完整验证样本与汇总指标，并返回输出路径。"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = SCRIPT_DIR / f"eye_to_hand_validation_{timestamp}.json"
    suffix = 1
    while output_path.exists():
        output_path = SCRIPT_DIR / f"eye_to_hand_validation_{timestamp}_{suffix}.json"
        suffix += 1
    payload = {
        "schema_version": "panthera-eye-to-hand-validation-v1",
        "validation_kind": "held_out_consistency_not_absolute_ground_truth",
        "calibration_file": CALIBRATION_FILE.name,
        "calibration_timestamp": fixed["source"].get("timestamp"),
        "T_base_camera_fixed": fixed["T_base_camera"].tolist(),
        "T_tcp_target_reference_from_training": (
            fixed["T_tcp_target_reference"].tolist()),
        "camera": {
            "matrix": np.asarray(camera_matrix, dtype=float).tolist(),
            "dist_coeffs": np.asarray(dist_coeffs, dtype=float).reshape(-1).tolist(),
        },
        "target": fixed["target"],
        "summary": summary,
        "samples": evaluated_samples,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
    temporary_path.replace(output_path)
    return output_path


def main():
    """运行交互式留出验证。"""
    calibration_module = load_calibration_module()
    try:
        fixed = load_fixed_calibration(calibration_module)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, RuntimeError) as error:
        print(f"无法开始验证：{error}")
        return 1

    print("Panthera Eye-to-Hand 独立留出姿态验证")
    print("SPACE/S 采集 | C 计算并保存 | R 清空验证样本 | ESC/Q 退出")
    print("本程序只保持重力补偿，不会执行位置轨迹，也不会重新拟合外参。")
    print("说明：结果是留出一致性误差，不是外部测量系统给出的绝对真值误差。")

    pipeline = None
    validation_samples = []
    try:
        pipeline, camera_matrix, dist_coeffs = calibration_module.init_camera()
        matrix_change = float(np.max(np.abs(
            camera_matrix - fixed["training_camera_matrix"])))
        old_distortion = fixed["training_dist_coeffs"]
        new_distortion = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        distortion_change = (float(np.max(np.abs(new_distortion - old_distortion)))
                             if new_distortion.shape == old_distortion.shape else np.inf)
        if matrix_change > 0.5 or distortion_change > 0.005:
            print("警告：本次相机内参与训练记录有明显差异，验证结果可能混入内参变化：")
            print(f"  内参矩阵最大差值={matrix_change:.6f}, "
                  f"畸变参数最大差值={distortion_change:.6f}")

        object_points = target_object_points(fixed["target"])
        robot = calibration_module.Panthera(
            config_path=str(calibration_module.SDK_CONFIG_FILE))
        window_name = "Eye-to-Hand Held-out Validation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            joint_positions_for_gravity = robot.get_current_pos()
            gravity = robot.get_Gravity(joint_positions_for_gravity)
            gravity[2] *= 1.05
            robot.pos_vel_tqe_kp_kd(
                [0.0] * robot.motor_count,
                [0.0] * robot.motor_count,
                gravity,
                [0.0] * robot.motor_count,
                [0.0] * robot.motor_count,
            )

            image = calibration_module.get_color_frame(pipeline)
            if image is None:
                continue
            camera_to_target, annotated, image_points = (
                calibration_module.detect_target(
                    image, camera_matrix, dist_coeffs))
            state = "TARGET OK" if camera_to_target is not None else "TARGET NOT FOUND"
            color = (0, 255, 0) if camera_to_target is not None else (0, 0, 255)
            cv2.putText(
                annotated,
                f"{state}  Held-out: {len(validation_samples)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                validation_samples.clear()
                print("已清空本次验证样本；训练数据和标定结果未改变")
                continue
            if key in (32, ord("s"), ord("S")):
                if camera_to_target is None or image_points is None:
                    print("采集失败：没有检测到标定板")
                    continue

                joint_positions = np.asarray(robot.get_current_pos(), dtype=np.float64)
                joint_velocities = np.asarray(robot.get_current_vel(), dtype=np.float64)
                expected_shape = (robot.motor_count,)
                if (joint_positions.shape != expected_shape
                        or joint_velocities.shape != expected_shape
                        or not np.all(np.isfinite(joint_positions))
                        or not np.all(np.isfinite(joint_velocities))):
                    print("采集失败：关节状态无效")
                    continue
                maximum_speed = float(np.max(np.abs(joint_velocities)))
                if maximum_speed > MAX_CAPTURE_SPEED_RAD_S:
                    print(f"采集失败：机械臂仍在运动，最大速度 {maximum_speed:.3f} rad/s，"
                          f"要求不超过 {MAX_CAPTURE_SPEED_RAD_S:.3f} rad/s")
                    continue

                base_to_tcp = calibration_module.get_base_to_tcp(
                    robot, joint_positions)
                novel, index, translation_mm, rotation_deg = pose_is_novel(
                    base_to_tcp,
                    fixed["training_base_to_tcp"],
                    calibration_module,
                )
                if not novel:
                    print(f"拒绝采集：与训练姿态 {index + 1} 过近 "
                          f"({translation_mm:.1f} mm, {rotation_deg:.1f} deg)")
                    continue
                prior_validation_poses = [
                    np.asarray(sample["T_base_tcp"], dtype=np.float64)
                    for sample in validation_samples
                ]
                novel, index, translation_mm, rotation_deg = pose_is_novel(
                    base_to_tcp,
                    prior_validation_poses,
                    calibration_module,
                )
                if not novel:
                    print(f"拒绝采集：与验证姿态 {index + 1} 过近 "
                          f"({translation_mm:.1f} mm, {rotation_deg:.1f} deg)")
                    continue

                raw_sample = {
                    "index": len(validation_samples),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "joint_positions_rad": joint_positions.tolist(),
                    "joint_velocities_rad_s": joint_velocities.tolist(),
                    "T_base_tcp": base_to_tcp.tolist(),
                    "T_camera_target": camera_to_target.tolist(),
                    "image_points_px": np.asarray(image_points, dtype=float).tolist(),
                }
                evaluated = evaluate_sample(
                    raw_sample,
                    fixed,
                    camera_matrix,
                    dist_coeffs,
                    object_points,
                    calibration_module,
                )
                validation_samples.append(evaluated)
                print(f"已采集留出姿态 {len(validation_samples)}："
                      f"空间={evaluated['target_translation_error_mm']:.2f} mm / "
                      f"{evaluated['target_rotation_error_deg']:.2f} deg，"
                      f"重投影={evaluated['pixel_reprojection_rmse_px']:.2f} px")
                continue

            if key in (ord("c"), ord("C")):
                if len(validation_samples) < MIN_VALIDATION_SAMPLES:
                    print(f"至少需要 {MIN_VALIDATION_SAMPLES} 个留出姿态，"
                          f"当前只有 {len(validation_samples)} 个")
                    continue
                summary = summarize_samples(validation_samples)
                report_path = save_report(
                    fixed,
                    camera_matrix,
                    dist_coeffs,
                    validation_samples,
                    summary,
                )
                target_translation = summary["target_translation_error_mm"]
                target_rotation = summary["target_rotation_error_deg"]
                pixel = summary["all_corner_pixel_error_px"]
                print("\n留出验证完成：")
                print(f"  固定目标关系平移 RMS / P95 / 最大值："
                      f"{target_translation['rms']:.2f} / "
                      f"{target_translation['p95']:.2f} / "
                      f"{target_translation['max']:.2f} mm")
                print(f"  固定目标关系旋转 RMS / P95 / 最大值："
                      f"{target_rotation['rms']:.2f} / "
                      f"{target_rotation['p95']:.2f} / "
                      f"{target_rotation['max']:.2f} deg")
                print(f"  全部角点像素误差 RMS / P95 / 最大值："
                      f"{pixel['rms']:.2f} / {pixel['p95']:.2f} / "
                      f"{pixel['max']:.2f} px")
                print(f"  报告已保存：{report_path}\n")
    finally:
        cv2.destroyAllWindows()
        if pipeline is not None:
            pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
