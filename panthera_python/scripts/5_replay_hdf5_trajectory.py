#!/usr/bin/env python3
"""按 HDF5 原始时序和配对 PKL 规划速度回放 RoboTwin Panthera 单臂轨迹。"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml


# ---------------- 参数区 ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAJECTORY_FILE = SCRIPT_DIR / "episode2.hdf5"
CONFIG_PATH = SCRIPT_DIR / "../robot_param/Follower.yaml"

SOURCE_DT = 0.06
PLANNER_DT = 1.0 / 250.0
POSITION_MATCH_TOLERANCE = 1e-4
EXACT_POSITION_MATCH_TOLERANCE = 1e-8
POSITIONING_SPEED = 0.5
RETURN_TO_ZERO_DURATION = 5.0
RETURN_CONTROL_DT = 0.01
RETURN_HOLD_DURATION = 0.2

MAX_TORQUE = np.array([21.0, 36.0, 36.0, 21.0, 10.0, 10.0])
GRIPPER_CLOSED_RAD = 0.0
GRIPPER_OPEN_RAD = 1.6
GRIPPER_MIT_VELOCITY = 0.0
GRIPPER_MIT_TORQUE = 0.0
GRIPPER_MIT_KP = 5.0
GRIPPER_MIT_KD = 0.5
GRIPPER_START_TOLERANCE = 0.05
GRIPPER_START_TIMEOUT = 10.0
GRIPPER_RETURN_TOLERANCE = 0.05
GRIPPER_RETURN_TIMEOUT = 10.0
# ---------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="?", type=Path, default=DEFAULT_TRAJECTORY_FILE)
    parser.add_argument(
        "--planner-trajectory",
        type=Path,
        help="对应的 RoboTwin 规划轨迹 PKL；默认使用 HDF5 同目录下的同名 .pkl",
    )
    parser.add_argument("--dry-run", action="store_true", help="只检查轨迹匹配和速度，不连接机械臂")
    parser.add_argument("--yes", action="store_true", help="跳过真机回放前的人工确认")
    return parser.parse_args()


def load_hdf5_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取并校验 HDF5 关节和夹爪轨迹。"""
    if not path.is_file():
        raise FileNotFoundError(f"轨迹文件不存在：{path}")

    with h5py.File(path, "r") as file:
        schema_version = file.attrs.get("schema_version")
        arm_mode = file.attrs.get("arm_mode")
        robot_type = file.attrs.get("robot_type")
        state_dim = file.attrs.get("state_dim")

        if schema_version != "panthera-single-v1":
            raise ValueError(f"不支持的 schema_version：{schema_version!r}")
        if arm_mode != "single" or robot_type != "panthera-6dof" or state_dim != 7:
            raise ValueError(
                "HDF5 不是预期的 Panthera 六轴单臂七维数据："
                f"arm_mode={arm_mode!r}, robot_type={robot_type!r}, state_dim={state_dim!r}"
            )

        joint_pos = np.asarray(file["joint_action/arm"][...], dtype=float)
        gripper = np.asarray(file["joint_action/gripper"][...], dtype=float)

    if joint_pos.ndim != 2 or joint_pos.shape[1] != 6:
        raise ValueError(f"关节轨迹形状必须是 (N, 6)，实际为 {joint_pos.shape}")
    if len(joint_pos) < 2:
        raise ValueError("轨迹至少需要两帧")
    if gripper.shape != (len(joint_pos),):
        raise ValueError(f"夹爪轨迹形状必须是 ({len(joint_pos)},)，实际为 {gripper.shape}")
    if not np.all(np.isfinite(joint_pos)) or not np.all(np.isfinite(gripper)):
        raise ValueError("轨迹包含 NaN 或无穷大")
    if np.any(gripper < -1e-6) or np.any(gripper > 1.0 + 1e-6):
        raise ValueError(
            f"夹爪数据必须在 [0, 1]，实际范围为 [{gripper.min():.6f}, {gripper.max():.6f}]"
        )

    return joint_pos, np.clip(gripper, 0.0, 1.0)


def load_planner_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 RoboTwin TOPP 规划位置与速度，并合并全部机械臂轨迹段。"""
    if not path.is_file():
        raise FileNotFoundError(f"规划轨迹文件不存在：{path}")

    with path.open("rb") as file:
        data = pickle.load(file)

    if not isinstance(data, dict):
        raise ValueError("PKL 根对象必须是字典")
    if data.get("schema_version") != "panthera-single-v1" or data.get("arm_mode") != "single":
        raise ValueError(
            "PKL 不是预期的 Panthera 单臂规划轨迹："
            f"schema_version={data.get('schema_version')!r}, arm_mode={data.get('arm_mode')!r}"
        )

    segments = data.get("arm_joint_path")
    if not isinstance(segments, list) or not segments:
        raise ValueError("PKL 缺少非空的 arm_joint_path")

    positions = []
    velocities = []
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict) or segment.get("status") != "Success":
            status = segment.get("status") if isinstance(segment, dict) else None
            raise ValueError(f"PKL 第 {segment_index} 段规划状态无效：{status!r}")
        position = np.asarray(segment.get("position"), dtype=float)
        velocity = np.asarray(segment.get("velocity"), dtype=float)
        if position.ndim != 2 or position.shape[1] != 6 or velocity.shape != position.shape:
            raise ValueError(
                f"PKL 第 {segment_index} 段 position/velocity 形状无效："
                f"{position.shape}/{velocity.shape}"
            )
        if len(position) == 0 or not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError(f"PKL 第 {segment_index} 段为空或包含 NaN/无穷大")
        positions.append(position)
        velocities.append(velocity)

    segment_offsets = np.cumsum([0] + [len(position) for position in positions])
    return np.concatenate(positions), np.concatenate(velocities), segment_offsets


def match_planner_velocities(
    hdf5_pos: np.ndarray,
    planner_pos: np.ndarray,
    planner_velocity: np.ndarray,
    segment_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按规划段和仿真采样步长匹配 HDF5；段间停留帧使用零速度。"""
    if planner_pos.shape != planner_velocity.shape or planner_pos.ndim != 2:
        raise ValueError("规划位置与速度数组形状不一致")
    if (
        segment_offsets.ndim != 1
        or len(segment_offsets) < 2
        or segment_offsets[0] != 0
        or segment_offsets[-1] != len(planner_pos)
        or np.any(np.diff(segment_offsets) <= 0)
    ):
        raise ValueError("PKL 规划段边界无效")

    planner_steps_per_frame = SOURCE_DT / PLANNER_DT
    planner_stride = int(round(planner_steps_per_frame))
    if not np.isclose(planner_steps_per_frame, planner_stride, atol=1e-12):
        raise ValueError(
            f"HDF5 周期 {SOURCE_DT:.6f} s 不是 PKL 周期 {PLANNER_DT:.6f} s 的整数倍"
        )

    frame_count = len(hdf5_pos)
    matched_indices = np.full(frame_count, -1, dtype=int)
    matched_errors = np.full(frame_count, np.inf, dtype=float)
    matched_velocity = np.zeros_like(hdf5_pos, dtype=float)
    hold_mask = np.ones(frame_count, dtype=bool)
    frame_cursor = 0

    for segment_index, (segment_start, segment_end) in enumerate(
        zip(segment_offsets[:-1], segment_offsets[1:])
    ):
        if frame_cursor >= frame_count:
            raise ValueError(f"HDF5 在匹配 PKL 第 {segment_index} 段前已没有剩余帧")

        start_errors = np.max(
            np.abs(hdf5_pos[frame_cursor:] - planner_pos[segment_start]),
            axis=1,
        )
        exact_candidates = np.flatnonzero(
            start_errors <= EXACT_POSITION_MATCH_TOLERANCE
        )
        if len(exact_candidates):
            segment_frame_start = frame_cursor + int(exact_candidates[0])
        else:
            near_candidates = np.flatnonzero(
                start_errors <= POSITION_MATCH_TOLERANCE
            )
            if not len(near_candidates):
                minimum_error = float(np.min(start_errors))
                raise ValueError(
                    f"HDF5 无法按顺序找到 PKL 第 {segment_index} 段起点："
                    f"最小位置误差 {minimum_error:.6g} rad，"
                    f"允许值 {POSITION_MATCH_TOLERANCE:.6g} rad"
                )
            segment_frame_start = frame_cursor + int(near_candidates[0])

        hold_planner_index = max(int(segment_start) - 1, 0)
        if segment_frame_start > frame_cursor:
            hold_slice = slice(frame_cursor, segment_frame_start)
            hold_errors = np.max(
                np.abs(hdf5_pos[hold_slice] - planner_pos[hold_planner_index]),
                axis=1,
            )
            if np.any(hold_errors > POSITION_MATCH_TOLERANCE):
                bad_offset = int(np.argmax(hold_errors))
                bad_frame = frame_cursor + bad_offset
                raise ValueError(
                    f"HDF5 第 {bad_frame} 帧既不属于规划段，也不是段间停留："
                    f"位置误差 {hold_errors[bad_offset]:.6g} rad"
                )
            matched_indices[hold_slice] = hold_planner_index
            matched_errors[hold_slice] = hold_errors

        planner_indices = np.arange(
            segment_start,
            segment_end,
            planner_stride,
            dtype=int,
        )
        if planner_indices[-1] != segment_end - 1:
            planner_indices = np.append(planner_indices, segment_end - 1)
        segment_frame_end = segment_frame_start + len(planner_indices)
        if segment_frame_end > frame_count:
            raise ValueError(
                f"HDF5 帧数不足，无法容纳 PKL 第 {segment_index} 段的采样点"
            )

        hdf5_slice = slice(segment_frame_start, segment_frame_end)
        segment_errors = np.max(
            np.abs(hdf5_pos[hdf5_slice] - planner_pos[planner_indices]),
            axis=1,
        )
        if np.any(segment_errors > POSITION_MATCH_TOLERANCE):
            bad_offset = int(np.argmax(segment_errors))
            bad_frame = segment_frame_start + bad_offset
            raise ValueError(
                f"HDF5 第 {bad_frame} 帧与 PKL 第 {segment_index} 段采样点不一致："
                f"位置误差 {segment_errors[bad_offset]:.6g} rad，"
                f"允许值 {POSITION_MATCH_TOLERANCE:.6g} rad"
            )

        matched_indices[hdf5_slice] = planner_indices
        matched_errors[hdf5_slice] = segment_errors
        matched_velocity[hdf5_slice] = planner_velocity[planner_indices]
        hold_mask[hdf5_slice] = False
        frame_cursor = segment_frame_end

    if frame_cursor < frame_count:
        trailing_slice = slice(frame_cursor, frame_count)
        trailing_errors = np.max(
            np.abs(hdf5_pos[trailing_slice] - planner_pos[-1]),
            axis=1,
        )
        if np.any(trailing_errors > POSITION_MATCH_TOLERANCE):
            bad_offset = int(np.argmax(trailing_errors))
            raise ValueError(
                f"HDF5 第 {frame_cursor + bad_offset} 帧不是末段后的有效停留帧："
                f"位置误差 {trailing_errors[bad_offset]:.6g} rad"
            )
        matched_indices[trailing_slice] = len(planner_pos) - 1
        matched_errors[trailing_slice] = trailing_errors

    return matched_velocity, matched_indices, matched_errors, hold_mask


def load_joint_limits(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 Follower.yaml 读取六个关节的位置和速度限位。"""
    if not path.is_file():
        raise FileNotFoundError(f"机器人配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    limits = config["robot"]["joint_limits"]
    lower = np.asarray(limits["lower"], dtype=float)
    upper = np.asarray(limits["upper"], dtype=float)
    velocity_limits = np.asarray(config["robot"]["velocity_limits"], dtype=float)
    if lower.shape != (6,) or upper.shape != (6,) or np.any(lower >= upper):
        raise ValueError("Follower.yaml 中的关节限位无效")
    if velocity_limits.shape != (6,) or np.any(velocity_limits <= 0):
        raise ValueError("Follower.yaml 中的关节速度限位无效")
    return lower, upper, velocity_limits


def validate_position_limits(
    joint_pos: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    out_of_range = (joint_pos < lower[None, :] - 1e-6) | (joint_pos > upper[None, :] + 1e-6)
    if not np.any(out_of_range):
        return
    frame, joint = np.argwhere(out_of_range)[0]
    raise ValueError(
        f"第 {frame} 帧关节{joint + 1}位置 {joint_pos[frame, joint]:.6f} rad "
        f"超出 [{lower[joint]:.6f}, {upper[joint]:.6f}] rad"
    )


def segment_dynamics(values: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """按相邻采样点计算每段速度以及相邻段之间的加速度。"""
    velocity = np.diff(values, axis=0) / dt
    if len(velocity) >= 2:
        acceleration = np.diff(velocity, axis=0) / dt
    else:
        acceleration = np.empty((0, values.shape[1]), dtype=float)
    return velocity, acceleration


def prepare_original_trajectory(
    joint_pos: np.ndarray,
    joint_velocity: np.ndarray,
    gripper: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """保持 HDF5 位置和时序，使用匹配得到的 TOPP 规划速度。"""
    validate_position_limits(joint_pos, lower, upper)
    if joint_velocity.shape != joint_pos.shape or not np.all(np.isfinite(joint_velocity)):
        raise ValueError(
            f"匹配后的规划速度形状必须是 {joint_pos.shape} 且全部有限，"
            f"实际为 {joint_velocity.shape}"
        )

    playback_dt = SOURCE_DT
    return joint_pos.copy(), gripper.copy(), joint_velocity.copy(), playback_dt, 1.0


def map_gripper_to_radians(gripper: np.ndarray) -> np.ndarray:
    """用端点线性近似将 RoboTwin [0, 1] 开度映射为真机电机角度。"""
    return GRIPPER_CLOSED_RAD + np.clip(gripper, 0.0, 1.0) * (
        GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD
    )


def command_gripper_mit(robot, target_pos: float) -> None:
    """使用仓库 JSON 回放参数发送夹爪 MIT 命令。"""
    accepted = robot.gripper_control_MIT(
        float(target_pos),
        GRIPPER_MIT_VELOCITY,
        GRIPPER_MIT_TORQUE,
        GRIPPER_MIT_KP,
        GRIPPER_MIT_KD,
    )
    if not accepted:
        raise RuntimeError(f"夹爪 MIT 目标 {target_pos:.3f} rad 被拒绝")


def move_gripper_to_start_mit(robot, target_pos: float) -> None:
    """在主轨迹开始前用 MIT 移动夹爪，并等待电机角度到位。"""
    print(f"夹爪使用 MIT 移动到起点：{target_pos:.3f} rad")
    deadline = time.perf_counter() + GRIPPER_START_TIMEOUT
    while True:
        command_gripper_mit(robot, target_pos)
        time.sleep(0.02)
        robot.send_get_motor_state_cmd()
        robot.motor_send_cmd()
        current_pos = float(robot.get_current_pos_gripper())
        if abs(current_pos - target_pos) <= GRIPPER_START_TOLERANCE:
            print(f"夹爪已到达起点，实际角度 {current_pos:.3f} rad。")
            return
        if time.perf_counter() >= deadline:
            raise RuntimeError(
                f"夹爪未在 {GRIPPER_START_TIMEOUT:.1f} 秒内到达起点；"
                f"目标 {target_pos:.3f} rad，实际 {current_pos:.3f} rad"
            )


def close_gripper_mit(robot) -> None:
    """持续用 MIT 将夹爪闭合到配置的闭合角度，并确认反馈到位。"""
    target_pos = float(GRIPPER_CLOSED_RAD)
    deadline = time.perf_counter() + GRIPPER_RETURN_TIMEOUT
    while True:
        command_gripper_mit(robot, target_pos)
        time.sleep(0.02)
        robot.send_get_motor_state_cmd()
        robot.motor_send_cmd()
        current_pos = float(robot.get_current_pos_gripper())
        if abs(current_pos - target_pos) <= GRIPPER_RETURN_TOLERANCE:
            print(f"夹爪已闭合，实际角度 {current_pos:.3f} rad。")
            return
        if time.perf_counter() >= deadline:
            raise RuntimeError(
                f"夹爪未在 {GRIPPER_RETURN_TIMEOUT:.1f} 秒内闭合；"
                f"目标 {target_pos:.3f} rad，实际 {current_pos:.3f} rad"
            )


def smoothstep_position(start: float, end: float, elapsed: float, duration: float) -> float:
    """五次 smoothstep，令夹爪目标从 start 平滑过渡到 end。"""
    if duration <= 0.0:
        return float(end)
    ratio = float(np.clip(elapsed / duration, 0.0, 1.0))
    blend = ratio ** 3 * (10.0 - 15.0 * ratio + 6.0 * ratio ** 2)
    return float(start + (end - start) * blend)


def next_tracking_plot_path() -> Path:
    """生成不覆盖旧结果的轨迹图路径。"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_path = SCRIPT_DIR / f"hdf5_joint_tracking_{timestamp}.png"
    if not base_path.exists():
        return base_path
    for suffix in range(1, 1000):
        candidate = base_path.with_name(f"{base_path.stem}_{suffix}.png")
        if not candidate.exists():
            return candidate
    raise RuntimeError("无法为关节轨迹图生成唯一文件名")


def save_joint_tracking_plot(
    expected_times: list[float],
    actual_times: list[float],
    desired_positions: list[np.ndarray],
    actual_positions: list[np.ndarray],
    output_path: Path | None = None,
) -> Path:
    """将 HDF5 期望关节位置与真机反馈保存为六子图 PNG。"""
    expected_time_array = np.asarray(expected_times, dtype=float)
    actual_time_array = np.asarray(actual_times, dtype=float)
    desired_array = np.asarray(desired_positions, dtype=float)
    actual_array = np.asarray(actual_positions, dtype=float)

    sample_count = len(expected_time_array)
    if sample_count == 0:
        raise ValueError("没有 HDF5 主轨迹样本可供绘图")
    if actual_time_array.shape != (sample_count,):
        raise ValueError("实际时间轴与期望时间轴长度不一致")
    if desired_array.shape != (sample_count, 6) or actual_array.shape != (sample_count, 6):
        raise ValueError("关节轨迹绘图数据必须是 (N, 6)")
    if not all(
        np.all(np.isfinite(values))
        for values in (expected_time_array, actual_time_array, desired_array, actual_array)
    ):
        raise ValueError("关节轨迹绘图数据包含 NaN 或无穷大")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    save_path = output_path if output_path is not None else next_tracking_plot_path()
    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for joint_index, axis in enumerate(axes.flat):
        axis.plot(
            expected_time_array,
            desired_array[:, joint_index],
            label="Expected",
            linewidth=2.0,
        )
        axis.plot(
            actual_time_array,
            actual_array[:, joint_index],
            label="Actual",
            linewidth=1.4,
            alpha=0.85,
        )
        axis.set_title(f"Joint {joint_index + 1}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Position (rad)")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle("HDF5 Joint Tracking: Expected vs Actual", fontsize=14)
    figure.tight_layout()
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"六关节期望/实际轨迹图已保存：{save_path}")
    return save_path


def build_smooth_return_trajectory(
    robot,
    start_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成固定 5 秒、100 Hz 的七次多项式回零轨迹。"""
    zero_pos = np.zeros(robot.motor_count, dtype=float)
    step_count = int(round(RETURN_TO_ZERO_DURATION / RETURN_CONTROL_DT))
    sample_times = np.linspace(0.0, RETURN_TO_ZERO_DURATION, step_count + 1)

    positions = []
    velocities = []
    accelerations = []
    for elapsed in sample_times:
        pos, vel, acc = robot.septic_interpolation(
            start_pos,
            zero_pos,
            RETURN_TO_ZERO_DURATION,
            float(elapsed),
        )
        positions.append(pos)
        velocities.append(vel)
        accelerations.append(acc)

    return (
        sample_times,
        np.asarray(positions, dtype=float),
        np.asarray(velocities, dtype=float),
        np.asarray(accelerations, dtype=float),
    )


def return_to_zero(robot, max_torque: list[float]) -> None:
    """从实际当前位置开始，六轴 5 秒平滑回零并闭合夹爪。"""
    robot.send_get_motor_state_cmd()
    robot.motor_send_cmd()
    time.sleep(0.05)
    start_pos = np.asarray(robot.get_current_pos(), dtype=float)
    start_gripper_pos = float(robot.get_current_pos_gripper())
    if start_pos.shape != (robot.motor_count,) or not np.all(np.isfinite(start_pos)):
        raise RuntimeError(f"无法获得有效的当前关节位置：{start_pos}")
    if not np.isfinite(start_gripper_pos):
        raise RuntimeError(f"无法获得有效的当前夹爪位置：{start_gripper_pos}")

    sample_times, positions, velocities, accelerations = build_smooth_return_trajectory(
        robot,
        start_pos,
    )
    velocity_limits = np.asarray(robot.velocity_limits, dtype=float)
    acceleration_limits = np.asarray(robot.acceleration_limits, dtype=float)
    if velocity_limits.shape != start_pos.shape or acceleration_limits.shape != start_pos.shape:
        raise RuntimeError("机器人速度或加速度限制与关节数不匹配")

    peak_velocity = np.max(np.abs(velocities), axis=0)
    peak_acceleration = np.max(np.abs(accelerations), axis=0)
    velocity_over_limit = peak_velocity > velocity_limits + 1e-9
    acceleration_over_limit = peak_acceleration > acceleration_limits + 1e-9
    if np.any(velocity_over_limit) or np.any(acceleration_over_limit):
        details = []
        for joint_index in np.where(velocity_over_limit | acceleration_over_limit)[0]:
            details.append(
                f"关节{joint_index + 1}: 速度 {peak_velocity[joint_index]:.3f}/"
                f"{velocity_limits[joint_index]:.3f} rad/s，加速度 "
                f"{peak_acceleration[joint_index]:.3f}/{acceleration_limits[joint_index]:.3f} rad/s^2"
            )
        raise RuntimeError(
            "5 秒回零轨迹超过机器人限制，为保证安全未执行；" + "；".join(details)
        )

    zero_velocity = np.zeros(robot.motor_count, dtype=float)
    if not robot.Joint_Pos_Vel(start_pos, zero_velocity, max_torque, iswait=False):
        raise RuntimeError("当前位置保持命令被拒绝")
    time.sleep(RETURN_HOLD_DURATION)

    print(
        f"机械臂开始 {RETURN_TO_ZERO_DURATION:.1f} 秒平滑回零，"
        f"预计峰值速度 {np.max(peak_velocity):.3f} rad/s，"
        f"夹爪从 {start_gripper_pos:.3f} rad 慢慢闭合..."
    )
    last_command_time = time.perf_counter()
    for sample_index in range(1, len(sample_times)):
        sample_dt = float(sample_times[sample_index] - sample_times[sample_index - 1])
        remaining = sample_dt - (time.perf_counter() - last_command_time)
        if remaining > 0:
            time.sleep(remaining)
        if not robot.Joint_Pos_Vel(
            positions[sample_index],
            velocities[sample_index],
            max_torque,
            iswait=False,
        ):
            raise RuntimeError(f"回零第 {sample_index} 个控制点被拒绝")
        # 与六轴回零同步发送五次 smoothstep 夹爪目标，避免目标角度瞬间跳变。
        gripper_target = smoothstep_position(
            start_gripper_pos,
            GRIPPER_CLOSED_RAD,
            float(sample_times[sample_index]),
            RETURN_TO_ZERO_DURATION,
        )
        command_gripper_mit(robot, gripper_target)
        last_command_time = time.perf_counter()

    zero_pos = np.zeros(robot.motor_count, dtype=float)
    if not robot.wait_for_position(zero_pos, tolerance=0.05, timeout=5.0):
        raise RuntimeError("平滑回零结束后，关节未在容差内到达零位")
    print("机械臂已平滑回到零位。")
    close_gripper_mit(robot)


def play_trajectory(
    joint_pos: np.ndarray,
    joint_velocity: np.ndarray,
    gripper_pos: np.ndarray,
    playback_dt: float,
    robot,
) -> None:
    """使用 Joint_Pos_Vel 连续回放，并在正常结束后回零。"""
    if robot.motor_count != 6:
        raise RuntimeError(f"期望发现 6 个机械臂电机，实际为 {robot.motor_count}")
    if joint_velocity.shape != joint_pos.shape:
        raise RuntimeError(
            f"关节位置与速度形状不一致：{joint_pos.shape}/{joint_velocity.shape}"
        )

    max_torque = MAX_TORQUE.tolist()
    positioning_speed = [POSITIONING_SPEED] * robot.motor_count
    returning_to_zero = False
    expected_times: list[float] = []
    actual_times: list[float] = []
    desired_positions: list[np.ndarray] = []
    actual_positions: list[np.ndarray] = []

    try:
        first_gripper_pos = float(gripper_pos[0])
        move_gripper_to_start_mit(robot, first_gripper_pos)

        print(f"机械臂以 {POSITIONING_SPEED:.1f} rad/s 移动到轨迹起点...")
        reached = robot.Joint_Pos_Vel(
            joint_pos[0],
            positioning_speed,
            max_torque,
            iswait=True,
            tolerance=0.05,
            timeout=30.0,
        )
        if not reached:
            raise RuntimeError("机械臂未在 30 秒内到达轨迹起点")

        print("已到达起点，开始完整回放...")
        start_time = time.perf_counter()
        expected_times.append(0.0)
        actual_times.append(time.perf_counter() - start_time)
        desired_positions.append(np.asarray(joint_pos[0], dtype=float).copy())
        actual_positions.append(np.asarray(robot.get_current_pos(), dtype=float).copy())

        for frame_index, target_pos in enumerate(joint_pos[1:], start=1):
            target_time = frame_index * playback_dt
            while True:
                remaining = target_time - (time.perf_counter() - start_time)
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.001))

            if not robot.Joint_Pos_Vel(
                target_pos,
                joint_velocity[frame_index],
                max_torque,
                iswait=False,
            ):
                raise RuntimeError(f"第 {frame_index} 帧关节控制命令被拒绝")
            try:
                command_gripper_mit(robot, float(gripper_pos[frame_index]))
            except RuntimeError as error:
                raise RuntimeError(f"第 {frame_index} 帧{error}") from error

            expected_times.append(target_time)
            actual_times.append(time.perf_counter() - start_time)
            desired_positions.append(np.asarray(target_pos, dtype=float).copy())
            actual_positions.append(np.asarray(robot.get_current_pos(), dtype=float).copy())

        final_target_time = (len(joint_pos) - 1) * playback_dt
        while True:
            remaining = final_target_time - (time.perf_counter() - start_time)
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.001))

        print("最后一帧已发送，等待机械臂实际到达轨迹终点...")
        reached = robot.Joint_Pos_Vel(
            joint_pos[-1],
            positioning_speed,
            max_torque,
            iswait=True,
            tolerance=0.05,
            timeout=30.0,
        )
        if not reached:
            raise RuntimeError("机械臂未在 30 秒内到达轨迹终点")

        print("轨迹完整回放结束。")
        returning_to_zero = True
        return_to_zero(robot, max_torque)
        print("夹爪已闭合。")
    except KeyboardInterrupt:
        if returning_to_zero:
            robot.set_stop()
            print("\n平滑回零被用户中断；已停止，不再自动回零。")
            raise
        print("\n检测到用户中断，将从当前实际位置开始 5 秒平滑回零。")
        try:
            return_to_zero(robot, max_torque)
        except KeyboardInterrupt:
            robot.set_stop()
            print("\n平滑回零被再次中断；已立即停止，不再重试。")
            raise
        except Exception as return_error:
            robot.set_stop()
            print(f"中断后回零失败：{return_error}", file=sys.stderr)
        raise
    except Exception:
        robot.set_stop()
        print("已发送电机停止命令；异常情况下未自动回零。", file=sys.stderr)
        raise
    finally:
        if expected_times:
            try:
                save_joint_tracking_plot(
                    expected_times,
                    actual_times,
                    desired_positions,
                    actual_positions,
                )
            except Exception as plot_error:
                print(f"关节轨迹图保存失败：{plot_error}", file=sys.stderr)


def print_preflight_report(
    planner_path: Path,
    planner_pos: np.ndarray,
    planner_segment_count: int,
    source_pos: np.ndarray,
    source_gripper: np.ndarray,
    playback_pos: np.ndarray,
    playback_velocity: np.ndarray,
    playback_gripper: np.ndarray,
    playback_dt: float,
    time_scale: float,
    matched_indices: np.ndarray,
    matched_errors: np.ndarray,
    hold_mask: np.ndarray,
    velocity_limits: np.ndarray,
) -> None:
    hdf5_diff_velocity, hdf5_diff_acceleration = segment_dynamics(playback_pos, playback_dt)
    planner_steps_per_frame = playback_dt / PLANNER_DT
    print(f"HDF5 轨迹帧数：{len(source_pos)}")
    print(f"原始推定时长：{(len(source_pos) - 1) * SOURCE_DT:.2f} s")
    print(f"原速回放帧数：{len(playback_pos)}")
    print(f"原速回放周期：{playback_dt:.5f} s")
    print(f"原速回放时长：{(len(playback_pos) - 1) * playback_dt:.2f} s")
    print(f"回放时间倍率：{time_scale:.3f}x")
    print(f"PKL 规划轨迹：{planner_path}")
    print(
        f"PKL 规划段/点数：{planner_segment_count}/{len(planner_pos)}，"
        f"规划周期 {PLANNER_DT:.5f} s"
    )
    print(f"每个 HDF5 周期对应仿真规划步数：{planner_steps_per_frame:.1f}")
    print(
        f"PKL 顺序匹配范围：[{matched_indices[0]}, {matched_indices[-1]}]，"
        f"最大位置误差 {np.max(matched_errors):.3g} rad"
    )
    print(f"停留帧：{np.count_nonzero(hold_mask)} 帧（关节速度已设为零）")
    print(
        "HDF5 相邻位置差分峰值："
        f"{np.max(np.abs(hdf5_diff_velocity)):.4f} rad/s"
    )
    playback_peak_velocity = np.max(np.abs(playback_velocity), axis=0)
    print(
        "实际发送的 PKL 规划速度峰值："
        f"{np.max(playback_peak_velocity):.4f} rad/s"
    )
    velocity_over_limit = playback_peak_velocity > velocity_limits + 1e-9
    if np.any(velocity_over_limit):
        details = "；".join(
            f"关节{joint_index + 1} {playback_peak_velocity[joint_index]:.4f}/"
            f"{velocity_limits[joint_index]:.4f} rad/s"
            for joint_index in np.where(velocity_over_limit)[0]
        )
        print(f"警告：PKL 规划速度超过 Follower.yaml 配置值：{details}")
    print(
        "HDF5 差分估计加速度峰值："
        f"{np.max(np.abs(hdf5_diff_acceleration)):.4f} rad/s^2"
    )
    print(
        "注意：主轨迹保持 HDF5 的 0.06 s/帧和全部位置点，"
        "关节速度改用对应 PKL 点的带符号 TOPP 规划速度，不进行重定时。"
    )
    print(f"起点关节位置：{np.array2string(playback_pos[0], precision=4)}")
    print(f"终点关节位置：{np.array2string(playback_pos[-1], precision=4)}")
    print(
        "夹爪归一化范围："
        f"[{source_gripper.min():.3f}, {source_gripper.max():.3f}]，"
        "真机映射范围："
        f"[{playback_gripper.min():.3f}, {playback_gripper.max():.3f}] rad"
    )
    print(
        "夹爪控制：MIT "
        f"(vel={GRIPPER_MIT_VELOCITY:.1f}, tqe={GRIPPER_MIT_TORQUE:.1f}, "
        f"kp={GRIPPER_MIT_KP:.1f}, kd={GRIPPER_MIT_KD:.1f})"
    )
    print("注意：夹爪 0～1.6 rad 为端点线性近似，真实开口宽度与电机角度可能非线性。")


def main() -> int:
    args = parse_args()
    trajectory_path = args.trajectory.expanduser().resolve()
    planner_path = (
        args.planner_trajectory.expanduser().resolve()
        if args.planner_trajectory is not None
        else trajectory_path.with_suffix(".pkl")
    )
    source_pos, source_gripper = load_hdf5_trajectory(trajectory_path)
    planner_pos, planner_velocity, segment_offsets = load_planner_trajectory(planner_path)
    planner_segment_count = len(segment_offsets) - 1
    (
        matched_velocity,
        matched_indices,
        matched_errors,
        hold_mask,
    ) = match_planner_velocities(
        source_pos,
        planner_pos,
        planner_velocity,
        segment_offsets,
    )
    lower, upper, velocity_limits = load_joint_limits(CONFIG_PATH.resolve())
    (
        playback_pos,
        playback_gripper_normalized,
        playback_velocity,
        playback_dt,
        time_scale,
    ) = prepare_original_trajectory(
        source_pos,
        matched_velocity,
        source_gripper,
        lower,
        upper,
    )
    playback_gripper = map_gripper_to_radians(playback_gripper_normalized)

    print(f"HDF5/PKL 轨迹检查与顺序匹配通过：{trajectory_path}")
    print_preflight_report(
        planner_path,
        planner_pos,
        planner_segment_count,
        source_pos,
        source_gripper,
        playback_pos,
        playback_velocity,
        playback_gripper,
        playback_dt,
        time_scale,
        matched_indices,
        matched_errors,
        hold_mask,
        velocity_limits,
    )

    if args.dry_run:
        print("dry-run 完成：未创建机器人实例，未发送任何电机命令。")
        return 0

    if not args.yes:
        confirmation = input("确认现场安全后输入 REPLAY 开始真机回放：").strip()
        if confirmation != "REPLAY":
            print("未收到 REPLAY，已取消；未连接机械臂。")
            return 0

    from Panthera_lib import Panthera

    robot = Panthera(str(CONFIG_PATH.resolve()))
    play_trajectory(playback_pos, playback_velocity, playback_gripper, playback_dt, robot)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n回放被用户中断；回零流程已结束，请确认机械臂实际状态。")
        raise SystemExit(130)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
