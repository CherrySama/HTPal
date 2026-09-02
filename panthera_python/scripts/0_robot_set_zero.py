#!/usr/bin/env python3
"""
获取并打印机械臂关节角度信息
设置零位后实时显示六个关节的当前状态（当前硬件无夹爪）
"""

import time
from Panthera_lib import Panthera

def print_robot_state(robot):
    """打印机器人状态信息"""
    # 获取关节角度
    # 利用运控模式发送控制帧以读取电机反馈状态
    robot.send_get_motor_state_cmd()
    robot.motor_send_cmd()
    positions = robot.get_current_pos()
    velocities = robot.get_current_vel()

    print("\n" + "="*50)
    print("机械臂状态信息")
    print("="*50)
    
    # 打印六个关节信息
    for i in range(robot.motor_count):
        print(f"关节{i+1}: 位置={positions[i]:7.3f} rad, 速度={velocities[i]:7.3f} rad/s")

def main():
    robot = Panthera()
    
    try:
        robot.set_reset_zero()
        robot.motor_send_cmd()
        time.sleep(1)  # 每0.5秒更新一次
        while True:
            print_robot_state(robot)
            time.sleep(0.5)  # 每0.5秒更新一次
            
    except KeyboardInterrupt:
        print("\n\n程序被中断")

if __name__ == "__main__":
    main()
