# OpenDoge 🤖

[![Hardware](https://img.shields.io/badge/Hardware-OpenDoge-blue)](https://github.com/OpenDogeRobotics/OpenDoge)

OpenDoge 四足机器人的强化学习训练框架，衍生自 [HIMLoco](https://github.com/InternRobotics/HIMLoco)。

![OpenDoge](./assets/image.png)

## 目录结构

```
OpenDoge_train/
├── legged_gym/                  # 核心训练框架
│   ├── envs/
│   │   ├── base/                # 基础类 (LeggedRobot, LeggedRobotCfg)
│   │   ├── opendoge/            # OpenDoge 训练配置
│   │   ├── a1/                  # A1 训练配置
│   │   └── go1/                 # Go1 训练配置
│   ├── scripts/
│   │   ├── train.py             # 训练入口
│   │   └── play.py              # 演示 & ONNX 导出
│   ├── utils/                   # 工具 (task_registry, logger, terrain 等)
│   └── sim2sim.py               # MuJoCo Sim2Sim 验证
├── rsl_rl/                      # HIMLoco PPO 算法实现
│   └── rsl_rl/
│       ├── algorithms/          # HIMPPO, PPO
│       ├── modules/             # HIMActorCritic, HIMEstimator
│       ├── runners/             # HIMOnPolicyRunner
│       └── storage/             # HIMRolloutStorage
├── resources/robots/
│   ├── Opendoge/                # OpenDoge URDF + MuJoCo XML + STL 网格
│   ├── g1_description/          # G1 描述
│   ├── go1/ go2/                # Go1 / Go2 模型
│   └── h1/ h1_2/                # H1 模型
├── deploy/
│   ├── deploy_mujoco/           # MuJoCo Sim2Sim 部署脚本
│   │   └── configs/             # opendoge.yaml, go2.yaml, g1.yaml 等
│   ├── deploy_real/             # 实机部署 (Go2/G1/H1)
│   │   └── configs/             # g1.yaml, h1.yaml, h1_2.yaml
│   └── pre_train/               # 预训练模型 (g1, h1, h1_2)
├── Tool/                        # 辅助工具
│   ├── check_urdf.py            # URDF 验证
│   └── simplify_mesh.py         # 网格减面
├── setup.py
└── LICENSE
```

## 快速启动

### 1. 创建 conda 环境

```bash
# Python 3.8 + PyTorch 2.3.1 + CUDA 12.1 + MuJoCo
conda create -n himloco python=3.8
conda activate himloco
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install mujoco==3.2.3
```

### 2. 安装 Isaac Gym

从 [NVIDIA 官网](https://developer.nvidia.com/isaac-gym) 下载 Isaac Gym Preview 4，解压后安装：

```bash
conda activate himloco
pip install -e <isaacgym_path>/isaacgym/python
```

### 3. 配置环境变量

Isaac Gym 需要 `libpython3.8.so` 和自身 lib 目录。创建 activate hook 使变量仅在 `himloco` 环境生效：

```bash
mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
cat > ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh << 'EOF'
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:<isaacgym_path>/isaacgym/lib:$LD_LIBRARY_PATH
EOF
```

> 替换 `<isaacgym_path>` 为 Isaac Gym 的实际路径，例如 `/home/lain/IsaacGym_Preview_4_Package`。

### 4. 安装项目依赖

```bash
conda activate himloco
cd OpenDoge_train
pip install -e .
```

## 环境检查

```bash
conda activate himloco
python -c "
import isaacgym
print(f'IsaacGym:               ok')

import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
print(f'PyTorch CUDA version:   {torch.version.cuda}')

import mujoco
print(f'MuJoCo:                 {mujoco.__version__}')

print('===== 环境就绪 =====')
"
```

预期输出：

```
IsaacGym:               ok
PyTorch CUDA available: True
PyTorch CUDA version:   12.1
MuJoCo:                 3.2.3
===== 环境就绪 =====
```

## 使用指南

### 1. Train

```bash
export PYTHONPATH=$PWD
python legged_gym/scripts/train.py --task=opendoge --headless
```

从 checkpoint 继续训练：

```bash
python legged_gym/scripts/train.py --task=opendoge --resume --load_run <run_name> --checkpoint <model_number>
```

#### 可用 task

| task | 机器人 |
|------|--------|
| `opendoge` | OpenDoge |
| `a1` | Unitree A1 |
| `go1` | Unitree Go1 |

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务名称 | 必选 |
| `--headless` | 不渲染图形界面 | false |
| `--num_envs` | 并行环境数 | 4096 |
| `--seed` | 随机种子 | 1 |
| `--max_iterations` | 最大迭代次数 | 300000 |
| `--resume` | 继续训练 | false |
| `--load_run` | 加载的运行名称 | -1 (最新) |
| `--checkpoint` | checkpoint 编号 | -1 (最新) |
| `--experiment_name` | 实验名称 | 覆盖默认值 |
| `--run_name` | 运行名称 | 覆盖默认值 |
| `--sim_device` | 仿真设备 | cuda:0 |
| `--rl_device` | 强化学习设备 | cuda:0 |

训练结果保存在 `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`。

### 2. Play

加载训练好的模型进行仿真演示并导出 ONNX：

```bash
export PYTHONPATH=.
python legged_gym/scripts/play.py --task=opendoge --load_run <run_name> --checkpoint <model_number>
```

导出的 ONNX 模型保存在 `onnx/` 目录，TorchScript 模型保存在 `logs/<experiment_name>/exported/policies/`。

查看训练曲线：

```bash
tensorboard --logdir=./logs/
```

### 3. Sim2Sim (MuJoCo)

```bash
# OpenDoge 仿真演示
python deploy/deploy_mujoco/deploy_opendoge.py

# 指定 ONNX 模型
python deploy/deploy_mujoco/deploy_opendoge.py --onnx onnx/flat_opendoge_xxx.onnx
```

#### 键盘操作

| 前进 | 后退 | 左转 | 右转 | 暂停 |
|------|------|------|------|------|
| ↑ | ↓ | ← | → | Space |

### 4. Sim2Real

目前 `deploy_real/` 中的实机部署脚本基于 Unitree SDK（Go2/G1/H1），OpenDoge 的实机部署通过 ROS2 固件仓库 [OpenDoge_firmware](https://github.com/OpenDogeRobotics/OpenDoge_firmware) 完成。

## 致谢

本项目衍生自 [HIMLoco](https://github.com/InternRobotics/HIMLoco)，同时参考了：

- [quadruped_rl](https://github.com/Benxiaogu/quadruped_rl)
- [himloco_lab](https://github.com/IsaacZH/himloco_lab)

## 许可证

BSD-3-Clause，详见 [LICENSE](LICENSE)。
