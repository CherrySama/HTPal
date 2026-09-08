#!/usr/bin/env python3
"""HTPal Active Tracking 动态屏幕姿态试验入口。

这是在 ``face_tracking_control.py`` 基础上的独立试验版本：原文件保持不变。
屏幕法向根据“屏幕中心到人脸”的方向动态更新，滚转参考 HOME，俯仰作为
较软的姿态任务；位置任务仍优先由 J2/J3 承担，J4/J5/J6 保持较高运动代价。
实时控制仍使用速度级 QP，不调用 moveL()。
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.spatial.transform import Rotation

import face_tracking_control as base
from tracking_qp import build_least_squares_qp, solve_bounded_qp


ORIGINAL_DETECT_FACES = base.detect_faces
ORIGINAL_FACE_TO_SCREEN_TARGET = base.face_to_screen_target


# 动态姿态滤波，避免人脸深度噪声直接变成屏幕角度抖动。
POSE_FOLLOW_ALPHA = 0.12
# 俯仰是软任务；偏航和滚转权重更高。
PITCH_TASK_WEIGHT = 20.0
YAW_TASK_WEIGHT = 50.0
ROLL_TASK_WEIGHT = 50.0


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
        eye_center = 0.5 * (points[0] + points[1])
        eye_points_camera = []
        eye_depths = []
        for eye in points[:2]:
            depth_m = base.sample_depth_m(
                depth_frame,
                eye[0],
                eye[1],
                args.depth_radius,
                args.min_depth,
                args.max_depth,
            )
            if depth_m is None:
                continue
            eye_points_camera.append(base.deproject_pixel(eye[0], eye[1], depth_m, intrinsics))
            eye_depths.append(depth_m)
        point_camera = None
        if len(eye_points_camera) == 2:
            # 先分别反投影，再取三维中点；这比在二维中点处只采一个深度更准确。
            point_camera = np.mean(np.asarray(eye_points_camera, dtype=np.float64), axis=0)
            depth_m = float(np.mean(eye_depths))
        else:
            # 双眼有一个深度无效时，回退到二维中点深度，避免整帧丢失目标。
            depth_m = base.sample_depth_m(
                depth_frame,
                eye_center[0],
                eye_center[1],
                args.depth_radius,
                args.min_depth,
                args.max_depth,
            )
            if depth_m is not None:
                point_camera = base.deproject_pixel(
                    eye_center[0], eye_center[1], depth_m, intrinsics
                )
        item["center"] = (float(eye_center[0]), float(eye_center[1]))
        item["depth_m"] = depth_m
        item["point_camera"] = point_camera
    return detections


def same_height_face_to_screen_target(
    point_camera, fk, camera_rotation, screen_distance, screen_rotation,
    camera_centering_gain=1.0,
):
    """复用原目标几何，但强制屏幕中心与双眼中心同高。"""
    target = ORIGINAL_FACE_TO_SCREEN_TARGET(
        point_camera,
        fk,
        camera_rotation,
        screen_distance,
        screen_rotation,
        camera_centering_gain,
    )
    _, t_base_camera = base.make_transforms(fk, camera_rotation)
    eye_h = t_base_camera @ np.append(np.asarray(point_camera, dtype=np.float64), 1.0)
    target[2] = eye_h[2]
    return target


class DynamicPoseTrackingController(base.TrackingController):
    """实时生成“法向指向人脸、HOME 滚转”的屏幕姿态。"""

    def __init__(self, robot, home_position, camera_rotation, screen_rotation, args):
        super().__init__(robot, home_position, camera_rotation, screen_rotation, args)
        self.home_screen_rotation = np.asarray(screen_rotation, dtype=np.float64).copy()

    def _screen_rotation_toward_face(self, fk: dict, point_camera: np.ndarray):
        """根据人脸位置构造动态 link6 旋转，并尽量保持 HOME 滚转。"""
        _, t_base_camera = base.make_transforms(fk, self.camera_rotation)
        face_h = t_base_camera @ np.append(np.asarray(point_camera, dtype=np.float64), 1.0)
        direction = face_h[:3] - np.asarray(fk["position"], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm < 1e-5:
            return None
        normal = direction / norm

        # camera_rotation 把相机坐标轴表达在 link6 中。先在 base 中构造
        # 目标相机坐标系，再变回 link6 旋转矩阵。
        camera_y_link6 = np.asarray(self.camera_rotation[:, 1], dtype=np.float64)
        home_y_base = self.home_screen_rotation @ camera_y_link6
        y_axis = home_y_base - normal * float(np.dot(home_y_base, normal))
        y_norm = float(np.linalg.norm(y_axis))
        if y_norm < 1e-5:
            home_x_base = self.home_screen_rotation @ np.asarray(self.camera_rotation[:, 0])
            y_axis = home_x_base - normal * float(np.dot(home_x_base, normal))
            y_norm = float(np.linalg.norm(y_axis))
        if y_norm < 1e-5:
            return None
        y_axis /= y_norm
        x_axis = np.cross(y_axis, normal)
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm < 1e-5:
            return None
        x_axis /= x_norm
        desired_camera_in_base = np.column_stack((x_axis, y_axis, normal))
        desired_rotation = desired_camera_in_base @ np.asarray(self.camera_rotation).T

        # 在 SO(3) 上做短弧滤波，而不是逐元素插值旋转矩阵。
        current = np.asarray(self.screen_rotation, dtype=np.float64)
        relative = Rotation.from_matrix(current.T @ desired_rotation).as_rotvec()
        return current @ Rotation.from_rotvec(POSE_FOLLOW_ALPHA * relative).as_matrix()

    def _handle_fresh_target(self, point_camera, previous_status, command_joint, command_velocity, dt):
        actual_q = np.asarray(self.robot.get_current_pos(), dtype=np.float64)
        if actual_q.shape == (base.JOINT_COUNT,) and np.all(np.isfinite(actual_q)):
            fk = self.robot.forward_kinematics(actual_q)
            if fk is not None:
                dynamic_rotation = self._screen_rotation_toward_face(fk, point_camera)
                if dynamic_rotation is not None:
                    self.screen_rotation = dynamic_rotation
        return super()._handle_fresh_target(
            point_camera, previous_status, command_joint, command_velocity, dt
        )

    def _solve_velocity_qp(self, q_command, qdot_previous, current_fk, desired_twist, dt):
        """动态姿态版本：位置优先，偏航/滚转较硬，俯仰较软，J4 延后。"""
        q_command = np.asarray(q_command, dtype=np.float64)
        qdot_previous = np.asarray(qdot_previous, dtype=np.float64)
        jacobian = np.asarray(self.robot.get_jacobian(q_command), dtype=np.float64)
        if jacobian.shape != (6, base.JOINT_COUNT) or not np.all(np.isfinite(jacobian)):
            return np.zeros(base.JOINT_COUNT), "NUMERICAL_FAILURE", 0, np.zeros(base.JOINT_COUNT, dtype=int)

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
        ]
        targets = [
            position_scale * np.asarray(desired_twist[:3], dtype=np.float64),
            angular_target,
            np.sqrt(np.array([0.35, 0.6, 0.6, 1.5, 1.5, 1.5])) * qdot_previous,
            np.zeros(base.JOINT_COUNT),
            np.sqrt(np.array([0.0, 1.0, 1.0, 8.0, 6.0, 6.0])) * np.clip(
                self.args.qp_home_gain * (self.home_position - q_command),
                -self.args.qp_home_speed,
                self.args.qp_home_speed,
            ),
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
        for index in range(base.JOINT_COUNT):
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
        position_active = np.zeros(base.JOINT_COUNT, dtype=int)
        position_active[(active == -1) & (qdot <= position_lower + 1e-7)] = -1
        position_active[(active == 1) & (qdot >= position_upper - 1e-7)] = 1
        return qdot, status, iterations, position_active


# 复用原入口的所有初始化、视觉循环、退出和安全处理，只替换控制器类。
base.TrackingController = DynamicPoseTrackingController
base.detect_faces = detect_faces_eye_center
base.face_to_screen_target = same_height_face_to_screen_target


if __name__ == "__main__":
    try:
        raise SystemExit(base.main())
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C。", file=sys.stderr)
        raise SystemExit(130)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)

