#!/usr/bin/env python3
"""Manual eye-to-hand calibration for a RealSense camera and Panthera.

Eye-in-hand: camera moves with TCP, target is fixed; result is T_tcp_camera.
Eye-to-hand: camera is fixed, target moves with TCP; result is T_base_camera.

The current calibration flow is intentionally manual: gravity compensation keeps
the six-axis follower compliant while the operator moves it and records samples.

Keys: SPACE/S collect, C calibrate, R reset, ESC/Q quit.
"""

import json
from pathlib import Path
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from Panthera_lib import Panthera


# This script lives inside the SDK's panthera_python/scripts directory, so the
# SDK root is its parent and the Follower config resolves the six-axis URDF.
SDK_PYTHON_ROOT = Path(__file__).resolve().parents[1]
SDK_CONFIG_FILE = SDK_PYTHON_ROOT / "robot_param" / "Follower.yaml"
SCRIPT_DIR = Path(__file__).resolve().parent


# -------- Change these values to match the physical target --------
CALIBRATION_TYPE = "eye_to_hand"  # "eye_in_hand" or "eye_to_hand"
PATTERN = "aruco"            # single marker shown in the current setup
CHESSBOARD_SIZE = (7, 7)     # inner corners: an 8x8-square board -> (7, 7)
SQUARE_SIZE_M = 0.020        # measured square edge length
ARUCO_DICT = cv2.aruco.DICT_7X7_100
ARUCO_ID = 59                # verified from the marker in the supplied camera image
ARUCO_SIZE_M = 0.102         # measured black marker edge length
OUTPUT_FILE = SCRIPT_DIR / ("eye_in_hand_calibration.json"
                            if CALIBRATION_TYPE == "eye_in_hand"
                            else "eye_to_hand_calibration.json")
SAMPLES_FILE = SCRIPT_DIR / ("eye_in_hand_calibration_samples.json"
                             if CALIBRATION_TYPE == "eye_in_hand"
                             else "eye_to_hand_calibration_samples.json")
MIN_SAMPLES = 10


def validate_calibration_dependencies():
    """Fail before hardware setup if the selected OpenCV build cannot solve hand-eye."""
    required = (
        "calibrateHandEye",
        "CALIB_HAND_EYE_TSAI",
        "CALIB_HAND_EYE_PARK",
        "CALIB_HAND_EYE_HORAUD",
        "CALIB_HAND_EYE_ANDREFF",
        "CALIB_HAND_EYE_DANIILIDIS",
    )
    missing = [name for name in required if not hasattr(cv2, name)]
    if missing:
        raise RuntimeError(
            "当前 OpenCV 不包含手眼标定接口: " + ", ".join(missing) +
            "; 请在 panthera 环境安装 OpenCV 4.x contrib 版本后重试")


def make_transform(rotation, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def rotation_angle_deg(rotation):
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def mean_transform(transforms):
    """用平移均值和投影到 SO(3) 的弦平均计算代表变换。"""
    if not transforms:
        raise ValueError("没有变换可供求平均")
    translations = np.asarray([transform[:3, 3] for transform in transforms])
    rotation_sum = np.sum([transform[:3, :3] for transform in transforms], axis=0)
    left, _, right = np.linalg.svd(rotation_sum)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    rotation = left @ correction @ right
    return make_transform(rotation, np.mean(translations, axis=0))


def target_metadata():
    return {
        "pattern": PATTERN,
        "chessboard_inner_corners": list(CHESSBOARD_SIZE),
        "square_size_m": SQUARE_SIZE_M,
        "aruco_id": ARUCO_ID,
        "aruco_size_m": ARUCO_SIZE_M,
    }


def save_raw_samples(samples, camera_matrix, dist_coeffs):
    """原子写入可供离线重算和留出验证的标定原始数据。"""
    payload = {
        "schema_version": "panthera-hand-eye-samples-v1",
        "calibration_type": CALIBRATION_TYPE,
        "transform_conventions": {
            "T_base_tcp": "p_base = T_base_tcp @ p_tcp",
            "T_camera_target": "p_camera = T_camera_target @ p_target",
        },
        "camera": {
            "matrix": np.asarray(camera_matrix, dtype=float).tolist(),
            "dist_coeffs": np.asarray(dist_coeffs, dtype=float).reshape(-1).tolist(),
            "stream": {"width": 640, "height": 480, "fps": 30},
        },
        "target": target_metadata(),
        "samples": samples,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    temporary_path = SAMPLES_FILE.with_suffix(SAMPLES_FILE.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
    temporary_path.replace(SAMPLES_FILE)


def snapshot_raw_samples(sample_count):
    """冻结本次参与求解的原始样本，避免后续采集覆盖验证依据。"""
    if not SAMPLES_FILE.is_file():
        raise RuntimeError(f"找不到原始样本文件：{SAMPLES_FILE}")
    with SAMPLES_FILE.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != sample_count:
        actual_count = len(samples) if isinstance(samples, list) else "无效"
        raise RuntimeError(
            f"原始样本数量 {actual_count} 与参与标定的 {sample_count} 不一致")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    snapshot_path = SAMPLES_FILE.with_name(
        f"{SAMPLES_FILE.stem}_{timestamp}{SAMPLES_FILE.suffix}")
    suffix = 1
    while snapshot_path.exists():
        snapshot_path = SAMPLES_FILE.with_name(
            f"{SAMPLES_FILE.stem}_{timestamp}_{suffix}{SAMPLES_FILE.suffix}")
        suffix += 1
    temporary_path = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
    temporary_path.replace(snapshot_path)
    return snapshot_path


def init_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = stream.get_intrinsics()
    camera_matrix = np.array(
        [[intrinsics.fx, 0.0, intrinsics.ppx],
         [0.0, intrinsics.fy, intrinsics.ppy],
         [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.asarray(intrinsics.coeffs, dtype=np.float64)
    for _ in range(30):
        pipeline.wait_for_frames()
    return pipeline, camera_matrix, dist_coeffs


def get_color_frame(pipeline):
    frame = pipeline.wait_for_frames().get_color_frame()
    return None if not frame else np.asanyarray(frame.get_data())


def detect_chessboard(image, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)
    if not found:
        return None, image.copy(), None

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    object_points = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    object_points *= SQUARE_SIZE_M
    ok, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
    annotated = image.copy()
    cv2.drawChessboardCorners(annotated, CHESSBOARD_SIZE, corners, found)
    if not ok:
        return None, annotated, None
    cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs, rvec, tvec, 3 * SQUARE_SIZE_M)
    rotation, _ = cv2.Rodrigues(rvec)
    return make_transform(rotation, tvec), annotated, corners.reshape(-1, 2)


def detect_aruco(image, camera_matrix, dist_coeffs):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    annotated = image.copy()
    if ids is None:
        return None, annotated, None
    flat_ids = ids.reshape(-1)
    matches = np.arange(len(flat_ids)) if ARUCO_ID is None else np.flatnonzero(flat_ids == ARUCO_ID)
    if len(matches) == 0:
        return None, annotated, None
    index = int(matches[0])
    selected_corner = [corners[index]]
    cv2.aruco.drawDetectedMarkers(annotated, selected_corner, ids[index:index + 1])
    half = ARUCO_SIZE_M / 2.0
    object_points = np.array([[-half, half, 0], [half, half, 0],
                              [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, selected_corner[0].reshape(4, 2), camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, annotated, None
    cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs, rvec, tvec, half)
    rotation, _ = cv2.Rodrigues(rvec)
    return (
        make_transform(rotation, tvec),
        annotated,
        selected_corner[0].reshape(4, 2).copy(),
    )


def detect_target(image, camera_matrix, dist_coeffs):
    if PATTERN == "chessboard":
        return detect_chessboard(image, camera_matrix, dist_coeffs)
    if PATTERN == "aruco":
        return detect_aruco(image, camera_matrix, dist_coeffs)
    raise ValueError(f"Unsupported PATTERN: {PATTERN}")


def get_base_to_tcp(robot, joint_positions=None):
    pose = robot.forward_kinematics(joint_positions)
    if pose is None:
        raise RuntimeError("机械臂正运动学计算失败")
    return make_transform(pose["rotation"], pose["position"])


def pose_diversity_ok(base_to_tcp_samples, candidate):
    if not base_to_tcp_samples:
        return True, ""
    relative = invert_transform(base_to_tcp_samples[-1]) @ candidate
    translation_mm = 1000.0 * np.linalg.norm(relative[:3, 3])
    rotation_deg = rotation_angle_deg(relative[:3, :3])
    if translation_mm < 15.0 and rotation_deg < 8.0:
        return False, f"姿态变化太小: {translation_mm:.1f} mm, {rotation_deg:.1f} deg"
    return True, ""


def rigidity_error(base_to_tcp_samples, camera_to_target_samples, result_transform):
    if CALIBRATION_TYPE == "eye_in_hand":
        # Board is fixed: T_base_board must be constant.
        rigid_poses = [
            base_to_tcp @ result_transform @ camera_to_target
            for base_to_tcp, camera_to_target
            in zip(base_to_tcp_samples, camera_to_target_samples)
        ]
    else:
        # Board is attached to TCP: T_tcp_board must be constant.
        rigid_poses = [
            invert_transform(base_to_tcp) @ result_transform @ camera_to_target
            for base_to_tcp, camera_to_target
            in zip(base_to_tcp_samples, camera_to_target_samples)
        ]
    translations = np.array([pose[:3, 3] for pose in rigid_poses])
    center = np.mean(translations, axis=0)
    translation_rms_mm = float(1000.0 * np.sqrt(np.mean(np.sum((translations - center) ** 2, axis=1))))
    reference_rotation = rigid_poses[0][:3, :3]
    rotation_rms_deg = float(np.sqrt(np.mean([
        rotation_angle_deg(reference_rotation.T @ pose[:3, :3]) ** 2
        for pose in rigid_poses
    ])))
    return translation_rms_mm, rotation_rms_deg, rigid_poses


def calibrate(base_to_tcp_samples, camera_to_target_samples):
    if len(base_to_tcp_samples) < MIN_SAMPLES:
        raise RuntimeError(f"至少需要 {MIN_SAMPLES} 组，当前只有 {len(base_to_tcp_samples)} 组")

    if CALIBRATION_TYPE == "eye_in_hand":
        robot_inputs = base_to_tcp_samples
    elif CALIBRATION_TYPE == "eye_to_hand":
        # Inverting robot poses makes OpenCV return T_base_camera.
        robot_inputs = [invert_transform(pose) for pose in base_to_tcp_samples]
    else:
        raise RuntimeError(f"未知标定类型: {CALIBRATION_TYPE}")
    robot_rotations = [pose[:3, :3] for pose in robot_inputs]
    robot_translations = [pose[:3, 3] for pose in robot_inputs]
    target_rotations = [pose[:3, :3] for pose in camera_to_target_samples]
    target_translations = [pose[:3, 3] for pose in camera_to_target_samples]

    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    candidates = []
    for name, method in methods.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                robot_rotations, robot_translations,
                target_rotations, target_translations, method=method)
            transform = make_transform(rotation, translation)
            trans_rms, rot_rms, rigid_poses = rigidity_error(
                base_to_tcp_samples, camera_to_target_samples, transform)
            if np.all(np.isfinite(transform)):
                candidates.append((trans_rms + rot_rms, name, transform,
                                   trans_rms, rot_rms, rigid_poses))
        except cv2.error as error:
            print(f"  {name} 求解失败: {error}")
    if not candidates:
        raise RuntimeError("所有手眼标定算法均求解失败；请增加姿态变化后重试")
    return min(candidates, key=lambda item: item[0])


def save_result(candidate, sample_count):
    _, method, result_transform, trans_rms, rot_rms, rigid_poses = candidate
    transform_key = ("T_tcp_camera" if CALIBRATION_TYPE == "eye_in_hand"
                     else "T_base_camera")
    convention = ("p_tcp = T_tcp_camera @ p_camera"
                  if CALIBRATION_TYPE == "eye_in_hand"
                  else "p_base = T_base_camera @ p_camera")
    rigid_translations = np.array([pose[:3, 3] for pose in rigid_poses])
    rigid_reference = mean_transform(rigid_poses)
    rigid_reference_key = ("T_base_target_reference"
                           if CALIBRATION_TYPE == "eye_in_hand"
                           else "T_tcp_target_reference")
    raw_samples_snapshot = snapshot_raw_samples(sample_count)
    result = {
        "calibration_type": CALIBRATION_TYPE,
        "transform_convention": convention,
        transform_key: result_transform.tolist(),
        "method": method,
        "num_samples": sample_count,
        "validation": {
            "rigid_pose_translation_rms_mm": trans_rms,
            "rigid_pose_rotation_rms_deg": rot_rms,
            "mean_rigid_pose_translation_m": np.mean(rigid_translations, axis=0).tolist(),
            rigid_reference_key: rigid_reference.tolist(),
            "raw_samples_file": raw_samples_snapshot.name,
        },
        "target": target_metadata(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    temporary_path = OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
    temporary_path.replace(OUTPUT_FILE)
    return result


def main():
    validate_calibration_dependencies()
    if CALIBRATION_TYPE == "eye_in_hand":
        print("Panthera Eye-in-Hand 标定：相机固定在TCP，标定板固定在桌面")
    else:
        print("Panthera Eye-to-Hand 标定：相机固定，标定板刚性固定在TCP")
    print("SPACE/S 手动采集 | C 计算 | R 清空 | ESC/Q 退出")
    pipeline = None
    base_to_tcp_samples = []
    camera_to_target_samples = []
    sample_records = []
    try:
        pipeline, camera_matrix, dist_coeffs = init_camera()
        robot = Panthera(config_path=str(SDK_CONFIG_FILE))
        cv2.namedWindow("Eye-to-Hand Calibration", cv2.WINDOW_NORMAL)
        while True:
            # Keep gravity compensation active so the arm can be moved by hand.
            joint_positions = robot.get_current_pos()
            gravity = robot.get_Gravity(joint_positions)
            gravity[2] *= 1.05
            robot.pos_vel_tqe_kp_kd(
                [0.0] * robot.motor_count, [0.0] * robot.motor_count,
                gravity, [0.0] * robot.motor_count, [0.0] * robot.motor_count)

            image = get_color_frame(pipeline)
            if image is None:
                continue
            camera_to_target, annotated, image_points = detect_target(
                image,
                camera_matrix,
                dist_coeffs,
            )
            state = "TARGET OK" if camera_to_target is not None else "TARGET NOT FOUND"
            color = (0, 255, 0) if camera_to_target is not None else (0, 0, 255)
            cv2.putText(annotated, f"{state}  Samples: {len(base_to_tcp_samples)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Eye-to-Hand Calibration", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                base_to_tcp_samples.clear()
                camera_to_target_samples.clear()
                sample_records.clear()
                save_raw_samples(sample_records, camera_matrix, dist_coeffs)
                print("已清空采样")
            elif key in (32, ord("s"), ord("S")):
                if camera_to_target is None:
                    print("采集失败：没有检测到标定板")
                    continue
                joint_positions = np.asarray(robot.get_current_pos(), dtype=float)
                joint_velocities = np.asarray(robot.get_current_vel(), dtype=float)
                if joint_positions.shape != (robot.motor_count,):
                    print(f"采集失败：关节位置形状无效 {joint_positions.shape}")
                    continue
                if joint_velocities.shape != (robot.motor_count,):
                    print(f"采集失败：关节速度形状无效 {joint_velocities.shape}")
                    continue
                base_to_tcp = get_base_to_tcp(robot, joint_positions)
                diverse, reason = pose_diversity_ok(base_to_tcp_samples, base_to_tcp)
                if not diverse:
                    print(f"拒绝重复采样：{reason}")
                    continue
                base_to_tcp_samples.append(base_to_tcp.copy())
                camera_to_target_samples.append(camera_to_target.copy())
                sample_records.append({
                    "index": len(sample_records),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "joint_positions_rad": joint_positions.tolist(),
                    "joint_velocities_rad_s": joint_velocities.tolist(),
                    "T_base_tcp": base_to_tcp.tolist(),
                    "T_camera_target": camera_to_target.tolist(),
                    "image_points_px": np.asarray(image_points, dtype=float).tolist(),
                })
                save_raw_samples(sample_records, camera_matrix, dist_coeffs)
                print(f"已采集第 {len(base_to_tcp_samples)} 组")
                print(f"原始样本已保存：{SAMPLES_FILE}")
            elif key in (ord("c"), ord("C")):
                try:
                    candidate = calibrate(base_to_tcp_samples, camera_to_target_samples)
                    _, method, transform, trans_rms, rot_rms, _ = candidate
                    saved_result = save_result(candidate, len(base_to_tcp_samples))
                    print(f"\n标定完成，方法: {method}")
                    result_name = ("T_tcp_camera" if CALIBRATION_TYPE == "eye_in_hand"
                                   else "T_base_camera")
                    print(f"{result_name} =\n", transform)
                    print(f"固定关系一致性 RMS: {trans_rms:.2f} mm, {rot_rms:.2f} deg")
                    print(f"结果已保存: {OUTPUT_FILE}\n")
                    print("训练原始样本快照: "
                          f"{SCRIPT_DIR / saved_result['validation']['raw_samples_file']}\n")
                except RuntimeError as error:
                    print(f"标定失败: {error}")
    finally:
        cv2.destroyAllWindows()
        if pipeline is not None:
            pipeline.stop()


if __name__ == "__main__":
    main()
