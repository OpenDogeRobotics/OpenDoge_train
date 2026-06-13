# OpenDoge MuJoCo IK Gait Tools

基于 MuJoCo 的传统控制与运动录制工具。提供 PD 位置控制 + 解析 IK 步态生成，支持 `--record` 导出 .npz 参考运动供 AMP / HIM 训练。

## 启动

```bash
# 首次使用：安装依赖（只需一次）
cd /home/lain/OpenDoge/OpenDoge_train/Tool/mujoco_ik
conda activate himloco
pip install -r requirements.txt
```

之后直接用 `./run.sh`，环境由脚本自动检查并激活：

```bash
./run.sh pd stand                          # PD 站立保持
./run.sh pd sine                           # PD 正弦运动
./run.sh ik                                # 键盘 IK 步态（需 X11）

# 录制参考运动
./run.sh pd stand  --duration 5  --record stand.npz
./run.sh ik --no-render --cmd-vx 0.8 --duration 20 --record trot.npz
```

> `run.sh` 自动检测当前 conda 环境，非 `himloco` 时自动通过 `conda run` 注入，无需手动切换。

| 按键 | 功能 |
|------|------|
| `↑` / `↓` | 前进 / 后退 |
| `←` / `→` | 左移 / 右移 |
| `Ctrl` + `←` / `→` | 左转 / 右转 |
| `Space` | 停止 |
| `R` | 重置姿态 |
| `Esc` | 退出 |

## 子命令

```text
./run.sh pd stand   [--duration N] [--no-render] [--record FILE]
./run.sh pd sine    [--duration N] [--no-render] [--record FILE]
./run.sh ik         [--duration N] [--no-render] [--cmd-vx V] [--cmd-vy V] [--cmd-yaw V]
                    [--c-style] [--record FILE]
```

## 参考运动录制

`--record <path.npz>` 输出：

| 字段 | 形状 | 说明 |
|------|------|------|
| `joint_positions` | (T, 12) | 关节位置 [rad] |
| `joint_velocities` | (T, 12) | 关节速度 [rad/s] |
| `joint_torques` | (T, 12) | 关节力矩 [Nm] |
| `base_position` | (T, 3) | 机身世界坐标 [m] |
| `base_quaternion` | (T, 4) | 机身朝向 (w, x, y, z) |
| `base_linear_vel` | (T, 3) | 机身线速度 [m/s] |
| `base_angular_vel` | (T, 3) | 机身角速度 [rad/s] |
| `feet_positions` | (T, 4, 3) | 足端世界坐标 [m] |
| `body_command` | (T, 3) | 速度指令 (vx, vy, yaw) |
| `timestep` | scalar | 仿真步长 [s] |
| `joint_names` | (12,) | 关节名称列表 |
| `leg_names` | (4,) | 腿名称列表 |

```python
import numpy as np
data = np.load("trot_forward.npz")
ref_q  = data["joint_positions"]     # (T, 12)
ref_dq = data["joint_velocities"]    # (T, 12)
ref_feet = data["feet_positions"]    # (T, 4, 3)
```

## 模型资产

MJCF 与 URDF 从 config 相对路径解析至 `OpenDoge_description/`：

```text
../../../../OpenDoge_description/URDF/xml/scene.xml       # MJCF 物理模型
../../../../OpenDoge_description/URDF/urdf/Opendoge.urdf  # URDF（IK 几何）
```

## 目录结构

```text
mujoco_ik/
├── run.sh                         # 统一启动器
├── configs/
│   └── position_control.json      # PD / IK / IMU 参数
├── opendoge_mujoco/
│   ├── sim_utils.py               # config 加载 & 模型初始化
│   ├── position_controller.py     # PD 力矩控制器
│   ├── leg_ik.py                  # 解析 3-DoF 腿 IK
│   ├── action_gait.py             # TrotCycloidGait 步态规划器
│   ├── foot_track_gait.py         # C-style foot track 步态
│   ├── imu_feedback.py            # IMU 反馈稳定器
│   └── motion_recorder.py         # 参考运动录制 (→ .npz)
├── scripts/
│   ├── run_position_control.py    # PD 位置控制
│   └── run_keyboard_ik_control.py # 键盘 IK 步态控制
└── requirements.txt
```
