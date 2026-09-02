#!/usr/bin/env python3
"""
单臂重力补偿程序 + 实时六关节轨迹记录（位置+速度）
"""
import os
import time
import numpy as np
from Panthera_lib import Panthera, TrajectoryRecorder

# ---------------- 参数区 ----------------
DO_RECORD = True                 # True=记录  False=不记录
REC_FILE = None                  # None=自动生成文件名
USE_FRICTION_COMPENSATION = False
CONTROL_DT = 0.002
RECORD_DT = 0.01
PRINT_INTERVAL = 0.1
# ---------------------------------------

def main():
    # 获取主臂当前状态
    Leader_positions = Leader.get_current_pos()
    Leader_velocity = Leader.get_current_vel()

    # 计算重力补偿力矩
    Leader_gra = Leader.get_Gravity(Leader_positions)

    Leader_tor = np.array(Leader_gra)
    if USE_FRICTION_COMPENSATION:
        Leader_tor += Leader.get_friction_compensation(Leader_velocity, Fc, Fv, vel_threshold)

    # 力矩限幅（基于电机规格）
    tau_limit = np.array([15.0, 30.0, 30.0, 15.0, 5.0, 5.0])
    Leader_tor = np.clip(Leader_tor, -tau_limit, tau_limit)

    # 零刚度零阻尼控制（重力补偿模式，可自由拖动）
    Leader.pos_vel_tqe_kp_kd(zero_pos, zero_vel, Leader_tor, zero_kp, zero_kd)

    # 限频打印，避免终端输出拖慢控制循环
    global last_print_time
    now = time.perf_counter()
    if now - last_print_time >= PRINT_INTERVAL:
        print("\r", end="")
        for i in range(Leader.motor_count):
            print(f"J{i+1}: {Leader_positions[i]:6.3f}rad {Leader_velocity[i]:6.3f}rad/s | ", end="")
        last_print_time = now

    return Leader_positions, Leader_velocity

if __name__ == "__main__":
    # 创建机器人实例
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../robot_param/Follower.yaml")
    Leader = Panthera(config_path)

    # 创建零位置和零速度数组
    zero_pos = [0.0] * Leader.motor_count
    zero_vel = [0.0] * Leader.motor_count
    zero_kp = [0.0] * Leader.motor_count
    zero_kd = [0.0] * Leader.motor_count
    last_print_time = 0.0

    # 摩擦补偿参数
    Fc = np.array([0.15, 0.12, 0.12, 0.12, 0.04, 0.04])
    Fv = np.array([0.05, 0.05, 0.05, 0.03, 0.02, 0.02])
    vel_threshold = 0.02

    # 实例化记录器（如开启记录）
    if DO_RECORD:
        rec = TrajectoryRecorder(REC_FILE)
        print("开始记录轨迹（六关节位置+速度）...")

    try:
        # 记录轨迹之前先循环发送读取指令，避免未接收到关节状态导致关节角为999
        for i in range(10):
            Leader.send_get_motor_state_cmd()
            time.sleep(0.1)
        next_time = time.perf_counter()
        next_record_time = next_time
        while True:
            state = main()                    # 重力补偿控制循环
            now = time.perf_counter()
            if DO_RECORD and now >= next_record_time:
                # 记录六个关节的位置和速度
                rec.log(*state)
                next_record_time += RECORD_DT
            next_time += CONTROL_DT
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()
    except KeyboardInterrupt:
        if DO_RECORD:
            rec.close()
        print("\n程序停止" + ("，轨迹已保存" if DO_RECORD else ""))
