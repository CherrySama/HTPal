# HTPal 视觉伺服环境

`panthera_python/` 保留机械臂 SDK 与其高层封装；视觉伺服代码放在同级的
`core/` 下，并与机械臂 SDK 在同一个 Python 环境中运行。

## 环境

当前环境名为 `htpal_vision`，由已验证 YOLO/RealSense/CUDA 的 `charm_vision`
克隆而来。原有 `panthera` 环境不修改。

```bash
conda create -n htpal_vision --clone charm_vision
conda activate htpal_vision
python -m pip install -r core/requirements.txt
python -m pip install \
  panthera_python/motor_whl/hightorque_robot-1.2.0-cp312-cp312-linux_x86_64.whl
```

## 导入验证

```bash
PYTHONPATH=panthera_python/scripts python -c \
  "from Panthera_lib import Panthera; import insightface, pyrealsense2, onnxruntime as ort; print(ort.get_available_providers())"
```

预期 ONNX Runtime provider 包含：

```text
TensorrtExecutionProvider
CUDAExecutionProvider
CPUExecutionProvider
```

说明：InsightFace 的 PyPI 元数据依赖名是 `onnxruntime`，官方文档允许手动替换为
`onnxruntime-gpu`。因此在当前 GPU 环境中 `pip check` 可能提示缺少名为
`onnxruntime` 的 CPU 包；实际导入和 provider 检查应以 `onnxruntime-gpu` 为准，
不要为了消除这条提示再安装 CPU 版并覆盖 GPU 后端。

当前阶段只准备环境和目录，不自动下载 `buffalo_sc` 模型，不连接或驱动机械臂。

## 独立手势测试

手势识别暂时与正式人脸跟踪完全分离。默认只打开 RealSense 和 MediaPipe，
验证一次性前后/旋转事件以及目标 HOME 相对位姿：

```bash
conda activate htpal_vision
python core/handdetect/src/hand_gesture_test.py --fps 15
```

默认不会连接 Panthera，也不会发送电机命令。确认识别方向和单次触发行为后，
显式添加 `--enable-motion` 才会让机械臂先到 HOME，再执行 HOME 基坐标
`X=-0.10/0/+0.10 m` 三档位移和 joint6 `-90/0/+90°` 姿态动作：

```bash
python core/handdetect/src/hand_gesture_test.py --fps 15 --enable-motion
```

手势事件要求整个动作过程都保持五指张开。握拳、半握拳、丢手或关键点
置信度中断会取消当前动作；只有位移/旋转达到阈值并稳定停止后，才产生一次
事件。手向前会让屏幕距离增加一档（`0.4→0.5→0.6 m`），手向后会让屏幕
距离减少一档（`0.6→0.5→0.4 m`）。触发后必须回到中立并重新保持五指张开，才会再次布防。前后事件使用
`moveL` 做 HOME 基准的 X 位移；旋转事件只发送 joint6 的关节命令，J1~J5
保持当前反馈角度，不经过笛卡尔 IK。

## 人脸跟踪中的手势事件

手势事件已接入 `facedetect/src/face_tracking_control.py`。跟踪开启后，
程序同时检测人脸和手：前后手势切换 `0.4/0.5/0.6 m` 距离档位，左右旋转
切换 `LANDSCAPE/PORTRAIT_LEFT/PORTRAIT_RIGHT`。旋转阶段只推进 J6；完成后
读取实际 FK 姿态作为新的屏幕保持姿态，继续做人脸位置跟踪。

建议先观察模式验证画面和事件，不加 `--enable-motion`：

```bash
conda activate htpal_vision
python core/facedetect/src/face_tracking_control.py --fps 15
```

确认事件和方向后，再由现场操作者显式加 `--enable-motion`。


## 模型

当前已下载官方 `buffalo_sc` 模型包，实际目录为：

```text
core/facedetect/model/models/buffalo_sc/
├── det_500m.onnx
└── w600k_mbf.onnx
```

第一阶段只加载 `det_500m.onnx`；模型包中的识别权重不会被调用。
