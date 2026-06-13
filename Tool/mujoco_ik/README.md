# OpenDoge MuJoCo IK Gait Tools

基于 MuJoCo 的传统控制与参考运动录制工具。PD 位置控制 + 解析 IK trot 步态，键盘 / 手柄双输入，`--record` 导出 .npz 供 AMP / HIM 训练。

## Quick Start

```bash
# 首次：安装依赖
cd /home/lain/OpenDoge/OpenDoge_train/Tool/mujoco_ik
conda activate himloco
pip install -r requirements.txt
```

之后直接用 `./run.sh`，环境自动切换：

```bash
./run.sh pd stand                                    # PD 站立
./run.sh pd sine                                     # PD 正弦
./run.sh ik                                          # IK trot 步态（键盘）
./run.sh ik --input gamepad                          # IK trot 步态（手柄）
./run.sh ik --no-render --cmd-vx 0.8 --duration 20 --record trot.npz
```

## 子命令

```
./run.sh pd stand   [--duration N] [--no-render] [--print-rate N] [--record FILE]
./run.sh pd sine    [--duration N] [--no-render] [--print-rate N] [--record FILE]

./run.sh ik         [--input SRC] [--duration N] [--no-render] [--print-rate N]
                    [--cmd-vx V] [--cmd-vy V] [--cmd-yaw V]
                    [--c-style] [--record FILE]
```

| 通用选项 | 说明 |
|----------|------|
| `--duration N` | 仿真时长（秒），默认 20 |
| `--no-render` | 无渲染 headless 模式 |
| `--print-rate N` | 遥测打印频率（Hz），默认 2 |
| `--record FILE` | 导出参考运动到 .npz |

## 输入设备

`./run.sh ik --input <source>` 支持三种输入源：

| `--input` | 说明 |
|-----------|------|
| `x11` | X11 键盘直接轮询。真实按压/释放，支持多键同按。**默认** |
| `callback` | MuJoCo viewer 按键回调。X11 不可用时的自动回退 |
| `gamepad` | 自动扫描 `/dev/input/event*` 寻找 Xbox/PS 手柄 |
| `gamepad:<dev>` | 指定设备路径 |

### 键盘映射

| 按键 | 指令 |
|------|------|
| `↑` `↓` | 前进 / 后退 |
| `←` `→` | 左移 / 右移 |
| `Ctrl` + `←` `→` | 左转 / 右转 |
| `Space` | 停止（零指令） |
| `R` | 重置姿态 |
| `Esc` | 退出 |

### 手柄映射

| 操作 | 指令 |
|------|------|
| 左摇杆 ↑↓ | 前进 / 后退 |
| 左摇杆 ←→ | 左移 / 右移 |
| 右摇杆 ←→ | 左转 / 右转 |
| A | 停止 |
| B | 重置姿态 |
| Start | 退出 |

> 未检测到手柄时 `--input gamepad` 会报错退出。先用 `--input x11` 确认仿真正常，再插手柄测试。

## 步态遥测

IK 模式每条遥测行末尾输出实时步态状态：

```
cyc= 54.8% [ FL=S FR=W RL=W RR=S ] clr=[FL:-0.001 FR:-0.001 RL:-0.001 RR:-0.001]
```

| 字段 | 含义 |
|------|------|
| `cyc=54.8%` | 步态周期进度。0% = 支撑中期，以此为相位原点 |
| `FL=S` | 腿状态：`S` = Stance 支撑相，`W` = Swing 摆动相 |
| `clr=FL:-0.001` | 足端离地高度 [m]。负值 = 低于标称站立高度（触地）；正值 = 抬腿中 |

**正常 trot**：FL+RR 相位同步，FR+RL 为另一组对角——遥测中两组应分别为相同的 S/W。

## 步态参数

编辑 `configs/position_control.json` → `action_ik` 段：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cycle_time` | 0.26 | 步态周期 [s] |
| `duty_factor` | 0.58 | 支撑相占比。越大支撑越长、摆动越短 |
| `step_height` | 0.035 | 抬腿高度 [m] |
| `step_x` | 0.085 | 前后步幅 [m] |
| `step_y` | 0.050 | 侧移步幅 [m] |
| `step_yaw` | 0.060 | 转向步幅 [rad] |
| `rear_stance_height_offset` | 0.015 | 后腿站立高度偏移 [m] |
| `max_joint_speed_rad_s` | 18.0 | 关节速度限制 [rad/s] |
| `startup_blend_time` | 0.12 | 起步渐变时间 [s] |

IMU 反馈参数在 `imu_feedback` 段（heading hold、roll/pitch 补偿）。

## 参考运动录制

`--record <path.npz>` 逐帧记录仿真状态：

| 键 | 形状 | 说明 |
|----|------|------|
| `joint_positions` | (T, 12) | 关节位置 [rad] |
| `joint_velocities` | (T, 12) | 关节速度 [rad/s] |
| `joint_torques` | (T, 12) | 关节力矩 [Nm] |
| `base_position` | (T, 3) | 机身世界坐标 [m] |
| `base_quaternion` | (T, 4) | 机身朝向 w,x,y,z |
| `base_linear_vel` | (T, 3) | 机身线速度 [m/s] |
| `base_angular_vel` | (T, 3) | 机身角速度 [rad/s] |
| `feet_positions` | (T, 4, 3) | 足端世界坐标 [m] |
| `body_command` | (T, 3) | 速度指令 vx,vy,yaw |
| `timestep` | scalar | 仿真步长 [s] |
| `joint_names` | (12,) | 关节名 |
| `leg_names` | (4,) | 腿名 |

加载：

```python
import numpy as np
data = np.load("trot_forward.npz")
q   = data["joint_positions"]      # (T, 12)
dq  = data["joint_velocities"]     # (T, 12)
cmd = data["body_command"]         # (T, 3)
```

## 目录结构

```
mujoco_ik/
├── run.sh                         # 统一启动器（自动 conda 环境）
├── configs/
│   └── position_control.json      # 仿真 & 步态 & IMU 参数
├── opendoge_mujoco/
│   ├── sim_utils.py               # Config 加载 & 模型初始化
│   ├── position_controller.py     # PD 力矩控制
│   ├── leg_ik.py                  # 3-DoF 解析腿 IK
│   ├── action_gait.py             # TrotCycloidGait（摆线足轨）
│   ├── foot_track_gait.py         # C-style foot track
│   ├── imu_feedback.py            # IMU 反馈稳定（heading/roll/pitch）
│   ├── input_devices.py           # 输入层（X11 键盘 + evdev 手柄）
│   ├── gait_telemetry.py          # 步态遥测（相位/支撑/离地高度）
│   └── motion_recorder.py         # → .npz 参考运动录制
├── scripts/
│   ├── run_position_control.py    # PD 位置控制
│   ├── run_ik_control.py          # IK 步态控制
│   └── run_keyboard_ik_control.py # → run_ik_control.py（兼容链接）
└── requirements.txt
```

模型资产通过 config 相对路径解析至 `OpenDoge_description/URDF/`。
