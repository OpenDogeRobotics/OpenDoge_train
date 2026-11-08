# OpenDoge Train（mjlab）

[![Hardware](https://img.shields.io/badge/Hardware-OpenDoge-blue)](https://github.com/OpenDogeRobotics/OpenDoge)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**OpenDoge** 是 OpenDogeRobotics 开源的四足机器人。本仓库 `OpenDoge_train` 负责它的强化学习训练、仿真验证与策略导出。

仓库有两条并行分支，训练栈不同、命令不通用：

- **`mjlab`（默认分支，本 README 描述的内容）**：基于 **mjlab 1.6.0**（MuJoCo-Warp）的当前主线，包含 OpenDoge 的速度/起身/倒立/跳跃/多技能任务、Viser 回放、独立工作台与浏览器试玩区。
- **`isaacgym`**：原 Isaac Gym 训练栈（Preview 4 + `legged_gym` + HIMLoco 衍生的 PPO/AMP 实现），保留 MuJoCo Sim2Sim 验证与 ONNX 导出。用于复现历史结果，以及和 mjlab 版本做对照。

```bash
git checkout isaacgym   # 切到 Isaac Gym 版本
git checkout mjlab      # 切回当前主线
```

## 分支对照

| 分支 | 训练栈 | 主要内容 | 定位 |
| --- | --- | --- | --- |
| `mjlab`（默认） | Python 3.10–3.13 + mjlab 1.6.0 + MuJoCo-Warp + RSL-RL + PyTorch | OpenDoge 与 Unitree 机型的任务注册、Viser 回放、独立工作台、多技能策略、ONNX 导出 | 当前开发主线 |
| `isaacgym` | Python 3.8 + Isaac Gym Preview 4 + `legged_gym` + HIMLoco `rsl_rl` | `OpenDoge`/`A1`/`Go1` 训练配置、MuJoCo Sim2Sim 验证、ONNX 导出 | 历史版本，维护性修复 |

## 为什么用 mjlab

- **不依赖 Isaac Gym**：mjlab 是纯 Python + MuJoCo-Warp 实现，CPU 单环境即可跑通环境构建、`reset` 与一步动作，开发机不需要 GPU 和 Isaac Gym 安装包。
- **资产与任务集中管理**：机器人资产在 `src/assets/`，任务差异在 `src/tasks/robots/`，训练、回放、工作台共用同一套注册表。
- **版本固定、适配层薄**：`mjlab==1.6.0` 作为固定依赖，本仓库只维护机器人资产、任务差异、验证契约和部署适配，不复制框架代码。
- **训练/验证/导出同一套 CLI**：`train`、`play`、`baseline-play`、`opendoge-eval`、`opendoge-export` 都在仓库内。

## 机器人资产与任务

**OpenDoge 是本仓库的主要机器人**：

- 速度跟踪：`LainLab-OpenDoge-Flat`、`LainLab-OpenDoge-Rough`
- 单技能：`LainLab-OpenDoge-Getup`（平地倒地自恢复）、`LainLab-OpenDoge-Handstand`（倒立）、`LainLab-OpenDoge-Jump`（跳跃）
- 单策略多技能：`LainLab-OpenDoge-Skills-Flat`，一个网络在 walk / handstand / getup / jump 之间切换，57 维观测、12 维动作

仓库同时保留 Unitree 机型的资产与任务（A2、As2、Go2、G1、G1-23Dof、H1_2、H2、R1，以及 Go2 的 DreamWaQ / AMP-DreamWaQ / CTS / TS Teacher 技能迁移），用于对照 mjlab 上游能力、验证共享 MDP 的通用性。新增机型沿用同一条路径：**资产 → profile/任务 → 验证 → 部署适配**。

> **任务 ID 前缀**：`LainLab-` 是历史命名，代码骨架沿用自 LainLab 项目，为兼容已经发布的 baseline 与 checkpoint 暂未改名。改成 `OpenDoge-` 前缀需要同步调整 `src/tasks/robots/opendoge/`、baseline 目录与测试，留待单独提交。

## 项目定位

| 维度 | 上游 `mjlab` | `unitree_rl_mjlab` 参考项目 | 本仓库 `isaacgym` 分支 | OpenDoge Train（`mjlab`） |
| --- | --- | --- | --- | --- |
| 角色 | 通用 MuJoCo/Warp 仿真与强化学习框架 | 以 Unitree 机型为中心的训练与实机部署项目 | HIMLoco 衍生的 Isaac Gym 训练栈 | OpenDoge 在 mjlab 上的训练、验证与导出层 |
| 机器人范围 | 通用框架能力 | Unitree 机型 | OpenDoge / A1 / Go1 | 以 OpenDoge 为主，Unitree 机型作为对照保留 |
| 框架维护 | 上游维护仿真、manager、MDP、runner 与 viewer | 项目自带机型任务与部署流程 | 自带 `legged_gym` 与 PPO/AMP 实现 | 固定依赖 `mjlab==1.6.0`，用薄适配层避免复制框架 |
| 任务组织 | 通用任务与扩展机制 | 速度跟踪、动作模仿、实机控制 | `legged_gym/envs` 下的机型配置 | 机器人 profile + 独立资产 + 可注册任务表达差异 |
| 可验证性 | 框架级能力 | 训练/部署工作流 | MuJoCo Sim2Sim + 真机 | 资产、配置、观测/奖励契约与 CPU smoke 测试，已知限制公开记录 |
| 动作数据 | 通用工具与接口 | 面向 Unitree 工作流 | AMP / 关键点数据集 | 本地 CSV → NPZ，按目标机器人校验关节列与自由度，不依赖 W&B |

## 目录结构

```text
OpenDoge_train/
├── src/                 # 源码根（Python 包为 src.*，发行包名 lainlab）
│   ├── assets/          # 各机器人 MJCF、URDF、网格与示例动作
│   ├── tasks/           # 任务类型与机器人组成
│   │   ├── velocity/    # 通用速度跟踪 MDP
│   │   ├── tracking/    # 通用动作跟踪 MDP
│   │   ├── getup/       # 起身任务类型
│   │   ├── handstand/   # 倒立任务类型
│   │   ├── jump/        # 跳跃任务类型
│   │   ├── multiskill/  # 单策略多技能任务类型
│   │   ├── amp/         # 预留通用 AMP core
│   │   └── robots/      # 机器人组成、profile 与任务注册
│   ├── toolchain/       # 离线工具链：标定 / 采集 / BC / DAgger / 导出
│   ├── workbench/       # 独立工作台（HTTP 服务与查看器）
│   ├── baseline/        # 版本化基线策略（随包分发）
│   ├── baselines.py     # 基线发现
│   ├── cli.py           # train / play / list-envs 入口
│   └── viewer.py        # Viser 回放扩展
├── tests/               # 兼容性测试
└── docs/                # 设计、验收与工作台文档
```

分层原则：

- `mjlab` 负责仿真、manager、通用 MDP、runner 和 viewer。
- `src.assets` 负责 OpenDoge 等机器人的模型描述与动作数据。
- `src.tasks` 只表达机器人名称、接触点、动作缩放等差异。
- CLI 先注册本仓库任务，再复用 mjlab 1.6 的训练与回放实现。

## 安装

需要 Python 3.10–3.13；训练需要 NVIDIA GPU，CPU 只适合配置检查与 smoke 测试。

```bash
uv sync --extra cu128
```

仅做 CPU 配置检查时：

```bash
uv sync --extra cpu
```

## 使用

```bash
# 查看任务
uv run list-envs

# 训练 OpenDoge
uv run train LainLab-OpenDoge-Flat --env.scene.num-envs 4096

# 直接播放内置 baseline
uv run baseline-play opendoge flat --device cuda:0 --num-envs 1
uv run baseline-play opendoge rough --device cuda:0 --num-envs 1 --no-terminations True
uv run baseline-play opendoge getup --device cuda:0 --num-envs 1

# 用随机动作做配置冒烟测试
uv run play LainLab-OpenDoge-Flat --agent random --num-envs 1

# 单策略多技能：walk / handstand / getup / jump
uv run train LainLab-OpenDoge-Handstand --env.scene.num-envs 4096
uv run train LainLab-OpenDoge-Jump --env.scene.num-envs 4096
uv run train LainLab-OpenDoge-Skills-Flat --env.scene.num-envs 4096
uv run baseline-play opendoge skills --num-envs 16
uv run opendoge-calibrate --write          # 倒立/跳跃目标标定
uv run opendoge-eval LainLab-OpenDoge-Skills-Flat --checkpoint <ckpt>
uv run opendoge-eval LainLab-OpenDoge-Skills-Flat --checkpoint <ckpt> --transitions

# 回放本地策略
uv run play LainLab-OpenDoge-Flat --checkpoint-file logs/.../model_1000.pt

# 导出 ONNX
uv run opendoge-export LainLab-OpenDoge-Flat --checkpoint <ckpt> --output policy.onnx

# 把 G1 CSV 动作转换为 mjlab 跟踪格式
uv run csv-to-npz --input-file src/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1-subject2 --robot g1
```

内置 baseline 位于 `src/baseline/<robot>/<terrain>/`，每个条目带 `baseline.json` 清单和 `model.pt` / `policy.onnx`，说明见 [`src/baseline/README.md`](src/baseline/README.md)。

> **续训提示**：Unitree 速度任务的实验名已由 `<robot>_velocity` 改为
> `<task_group>_<robot>_<terrain>_velocity`（Flat 与 Rough 不再共用同一目录）。
> 续训改名前的实验需显式指定旧目录名，例如：
> `uv run train Unitree-Go2-Flat --agent.resume --agent.load-run go2_velocity`。

### 独立工作台

```bash
# 在仓库根目录启动（webui / src-webui 兼容入口仍可用）
uv run --extra workbench src-workbench
```

浏览器打开 `http://127.0.0.1:7860`。工作台按固定顺序组织：

1. **项目内 URDF/MJCF 查看器**：加载项目模型，或导入包含网格的目录；调节关节，显示坐标系、关节轴、碰撞体和惯量。
2. **策略选择**：Velocity / Tracking 互斥展开；Tracking 内置 GMR 的 LAFAN1 BVH → G1 29-DoF 重定向，从统一动作资产库选择输入，也支持已有 GMR PKL → NPZ；实时查看重定向进度、人体骨架与 G1 姿态，完成后通过时间轴回放。
3. **算法选择**：Velocity / Tracking 分组；Go2 Skill 单独提供 DreamWaQ、AMP-DreamWaQ、CTS、TS Teacher 和动作模板。
4. **训练**：设置迭代轮次、保存间隔、并行环境数；查看实时日志、停止任务，在内嵌 TensorBoard 中查看奖励和训练指标。
5. **导出与验证**：浏览、选择、下载 checkpoint/ONNX，手动导出 ONNX，启动 CPU 单环境 Viser checkpoint 回放。

工作台源码在 `src/workbench/`，不改变 `tasks/` 中的训练算法与环境。查看器由项目自行实现：MuJoCo 解析模型，精简 Three.js 页面负责绘制；仅参考 `robot_viewer` 的功能思路，不复制其整套应用。查看器与 GMR 子集都在项目内运行。

模型导入用于检查；训练仍使用已注册任务的机器人配置。GMR 当前限 LAFAN1 骨架和 G1 29-DoF，23-DoF 任务需使用对应 NPZ。Viser 当前回放 checkpoint，ONNX 是独立导出产物。

动作集中在 `src/assets/motions/`。页面支持导入和选择文件，转换输出自动命名、入库，并选为训练动作。

前端开发、模块边界和验证说明见 [工作台文档](docs/workbench.md)。

### 在线仿真器

浏览器版 MuJoCo + ONNX 策略演示已迁出本仓库，独立部署在 <https://opendogerobotics.github.io/>（源码：[OpenDogeRobotics.github.io](https://github.com/OpenDogeRobotics/OpenDogeRobotics.github.io)）。训练、评估和工作台都不依赖任何前端资源。

### 动作数据

`csv-to-npz` 直接在本地生成 NPZ，不需要 Weights & Biases。`--output-name` 只传文件名时，输出会保存到输入 CSV 的同一目录；也可传入完整路径。G1-23DoF 动作使用 `--robot g1_23dof`。

任务命名统一使用分类前缀：`Unitree-<Robot>-Flat` / `Unitree-<Robot>-Rough` 和 `LainLab-<Robot>-Flat` / `LainLab-<Robot>-Rough`。速度任务支持 A2、As2、Go2、G1、G1-23Dof、H1_2、H2、R1 和 OpenDoge；动作跟踪任务支持 G1 与 G1-23Dof。OpenDoge 的 Flat、Rough、Getup、Handstand、Jump 与 `LainLab-OpenDoge-Skills-Flat` 均已注册，单策略多技能的完整设计与验收见 [docs/opendoge_multiskill.md](docs/opendoge_multiskill.md)。

## 运行验证与已知限制

截至 2026-09-09，项目的 CPU 单环境 smoke 验证覆盖了已注册任务的环境构建、`reset` 与一步零动作执行，并检查了观测和奖励是否为有限值。所有 Go2 任务（含 Go2 skill）、基础速度任务、Cartpole、Yam 操作任务和 29-DoF G1 跟踪任务均已通过；OpenDoge 的各任务另有 `opendoge-eval` 的可复现指标。

- 跟踪任务必须提供与机器人自由度匹配的 NPZ 动作文件。命令行训练/回放时请设置 `motion_file`；仓库本地的 29-DoF G1 示例不能用于 `G1-23Dof` 跟踪任务。
- `Unitree-H2-Rough` 和 `Unitree-R1-Rough` 当前的 MuJoCo-Warp 接触缓冲容量不足：分别至少需要 `nconmax=80` 和 `nconmax=141`。提高该配置值后，两项任务可完成 smoke；在修复默认配置前请勿直接用默认设置启动训练。
- Smoke 只验证初始化和单步数值稳定性，不代表策略已经收敛，也不替代长时间训练、checkpoint 回放或真机验收。

## 开发

```bash
make format
make check
```

机器人差异集中在 `src/tasks/robots/` 下的机器人包，通用 MDP 位于 `src/tasks/velocity/core.py`、`tracking/core.py` 等任务类型包。新增同类机器人时，通常在对应机器人包中增加 profile 和注册函数，无需复制整套 MDP。

## 部署与导出

本分支不包含实机部署源码：训练与验证通过 `play` 的 Viser 回放完成，导出的 ONNX 策略（`opendoge-export`）交给实机侧工程使用，例如 OpenDoge 固件仓库 `OpenDoge_firmware`。Isaac Gym 版本中的 `deploy/deploy_mujoco/`（配置驱动的 MuJoCo 验证与诊断）仅存在于 `isaacgym` 分支。

## 上游与许可

本分支是 [mjlab](https://github.com/mujocolab/mjlab) 1.6.0 的下游项目，项目组织参考 `unitree_rl_mjlab`，代码骨架与任务前缀沿袭自 LainLab。代码使用 Apache-2.0 许可；OpenDoge 资产由 OpenDogeRobotics 维护，第三方组件与机器人描述保留各自许可。

---

# OpenDoge Train (mjlab)

[![Hardware](https://img.shields.io/badge/Hardware-OpenDoge-blue)](https://github.com/OpenDogeRobotics/OpenDoge)

**OpenDoge** is an open-source quadruped robot from OpenDogeRobotics. This repository, `OpenDoge_train`, owns its reinforcement-learning training, simulation validation, and policy export.

The repository carries two parallel branches with different stacks and non-interchangeable commands:

- **`mjlab`** (default; described by this README): the current mainline, built on **mjlab 1.6.0** (MuJoCo-Warp), with OpenDoge velocity/get-up/handstand/jump/multi-skill tasks, Viser playback, a standalone workbench, and a separate online browser simulator.
- **`isaacgym`**: the original Isaac Gym stack (Preview 4 + `legged_gym` + HIMLoco-derived PPO/AMP), including MuJoCo Sim2Sim validation and ONNX export. Kept for reproducing earlier results and for comparison.

```bash
git checkout isaacgym   # the Isaac Gym version
git checkout mjlab      # back to the mainline
```

## Branch comparison

| Branch | Stack | Contents | Role |
| --- | --- | --- | --- |
| `mjlab` (default) | Python 3.10–3.13 + mjlab 1.6.0 + MuJoCo-Warp + RSL-RL + PyTorch | Task registration for OpenDoge and Unitree robots, Viser playback, workbench, multi-skill policies, ONNX export | Current development |
| `isaacgym` | Python 3.8 + Isaac Gym Preview 4 + `legged_gym` + HIMLoco `rsl_rl` | `OpenDoge`/`A1`/`Go1` configs, MuJoCo Sim2Sim, ONNX export | Historical version, maintenance fixes |

## Why mjlab

- **No Isaac Gym dependency.** mjlab is pure Python + MuJoCo-Warp, so a single CPU environment can exercise environment construction, `reset`, and one action step; development machines do not need a GPU or an Isaac Gym install.
- **Assets and tasks in one registry.** Robot assets live in `src/assets/`, task differences in `src/tasks/robots/`, and training, playback, and the workbench share the same task registry.
- **Pinned framework, thin adapter.** `mjlab==1.6.0` is a fixed dependency; this repository only owns robot assets, task differences, validation contracts, and deployment adapters.
- **One CLI for training, validation, and export.** `train`, `play`, `baseline-play`, `opendoge-eval`, and `opendoge-export` all ship in the repository.

## Robots and tasks

**OpenDoge is the primary robot of this repository:**

- Velocity tracking: `LainLab-OpenDoge-Flat`, `LainLab-OpenDoge-Rough`
- Single-skill tasks: `LainLab-OpenDoge-Getup` (fall recovery on flat ground), `LainLab-OpenDoge-Handstand`, `LainLab-OpenDoge-Jump`
- Single-policy multi-skill: `LainLab-OpenDoge-Skills-Flat`, one network switching between walk / handstand / getup / jump with a 57-D observation and 12-D action

Unitree assets and tasks (A2, As2, Go2, G1, G1-23Dof, H1_2, H2, R1, plus the Go2 DreamWaQ / AMP-DreamWaQ / CTS / TS Teacher skill migrations) remain as reference and as coverage for the shared MDPs. New robots follow the same path: **asset → profile/task → validation → deployment adapter**.

> **Task-ID prefix:** `LainLab-` is a historical name inherited from the LainLab codebase. It is kept for compatibility with already-published baselines and checkpoints; renaming to an `OpenDoge-` prefix requires updating `src/tasks/robots/opendoge/`, the baseline directories, and tests together, so it is left for a separate commit.

## Positioning

| Dimension | Upstream `mjlab` | `unitree_rl_mjlab` reference | `isaacgym` branch | OpenDoge Train (`mjlab`) |
| --- | --- | --- | --- | --- |
| Role | General MuJoCo/Warp simulation and RL framework | Unitree-centered training and real-robot deployment | HIMLoco-derived Isaac Gym stack | Training, validation, and export layer for OpenDoge on mjlab |
| Robot scope | General framework capabilities | Unitree robots | OpenDoge / A1 / Go1 | OpenDoge first; Unitree robots kept for comparison |
| Framework maintenance | Upstream owns simulation, managers, MDPs, runners, viewers | Carries robot tasks and deployment workflows | Carries `legged_gym` and PPO/AMP implementations | Pins `mjlab==1.6.0`; thin adapters avoid copying the framework |
| Task organization | General tasks and extension points | Velocity tracking, motion imitation, real-robot control | Per-robot configs under `legged_gym/envs` | Robot profiles, isolated assets, and registrable tasks express differences |
| Verifiability | Framework-level | Training/deployment workflow | MuJoCo Sim2Sim and real hardware | Asset, config, observation/reward contract, and CPU smoke tests; known limits documented |
| Motion data | General tooling | Unitree motion workflows | AMP and keypoint datasets | Local CSV → NPZ with joint-column and DoF validation per robot, no W&B |

## Layout

```text
OpenDoge_train/
├── src/                 # Source root (Python package: src.*, distribution name: lainlab)
│   ├── assets/          # Per-robot MJCF/URDF files, meshes, example motions
│   ├── tasks/           # Task types and robot composition
│   │   ├── velocity/    # Shared velocity-tracking MDP
│   │   ├── tracking/    # Shared motion-tracking MDP
│   │   ├── getup/       # Get-up task type
│   │   ├── handstand/   # Handstand task type
│   │   ├── jump/        # Jump task type
│   │   ├── multiskill/  # Single-policy multi-skill task type
│   │   ├── amp/         # Reserved generic AMP core
│   │   └── robots/      # Robot composition, profiles, registration
│   ├── toolchain/       # Offline toolchain: calibrate / collect / BC / DAgger / export
│   ├── workbench/       # Standalone workbench (HTTP server + viewer)
│   ├── baseline/        # Versioned baseline policies (shipped in the wheel)
│   ├── baselines.py     # Baseline discovery
│   ├── cli.py           # train / play / list-envs entry points
│   └── viewer.py        # Viser playback extensions
├── tests/               # Compatibility tests
└── docs/                # Design, acceptance, and workbench docs
```

Layering principles:

- `mjlab` provides simulation, managers, common MDPs, runners, and viewers.
- `src.assets` owns the OpenDoge and reference robot descriptions plus motion data.
- `src.tasks` expresses only task differences such as robot names, contacts, and action scales.
- The CLI registers this repository's tasks before reusing mjlab 1.6 training and playback.

## Installation

Python 3.10–3.13 is required. Training needs an NVIDIA GPU; CPU is only for configuration checks and smoke tests.

```bash
uv sync --extra cu128
```

For CPU-only configuration checks:

```bash
uv sync --extra cpu
```

## Usage

```bash
# List tasks
uv run list-envs

# Train OpenDoge
uv run train LainLab-OpenDoge-Flat --env.scene.num-envs 4096

# Play bundled baselines directly in Viser
uv run baseline-play opendoge flat --device cuda:0 --num-envs 1
uv run baseline-play opendoge rough --device cuda:0 --num-envs 1 --no-terminations True
uv run baseline-play opendoge getup --device cuda:0 --num-envs 1

# Random-action configuration smoke test
uv run play LainLab-OpenDoge-Flat --agent random --num-envs 1

# Single-policy multi-skill: walk / handstand / getup / jump
uv run train LainLab-OpenDoge-Handstand --env.scene.num-envs 4096
uv run train LainLab-OpenDoge-Jump --env.scene.num-envs 4096
uv run train LainLab-OpenDoge-Skills-Flat --env.scene.num-envs 4096
uv run baseline-play opendoge skills --num-envs 16
uv run opendoge-calibrate --write          # handstand / jump target calibration
uv run opendoge-eval LainLab-OpenDoge-Skills-Flat --checkpoint <ckpt>
uv run opendoge-eval LainLab-OpenDoge-Skills-Flat --checkpoint <ckpt> --transitions

# Replay a local policy
uv run play LainLab-OpenDoge-Flat --checkpoint-file logs/.../model_1000.pt

# Export ONNX
uv run opendoge-export LainLab-OpenDoge-Flat --checkpoint <ckpt> --output policy.onnx

# Convert a G1 CSV motion into mjlab tracking format
uv run csv-to-npz --input-file src/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1-subject2 --robot g1
```

Bundled baselines live in `src/baseline/<robot>/<terrain>/`; each entry has a `baseline.json` manifest plus `model.pt` / `policy.onnx`. See [`src/baseline/README.md`](src/baseline/README.md).

> **Resuming runs**: Unitree velocity experiment names changed from
> `<robot>_velocity` to `<task_group>_<robot>_<terrain>_velocity` (Flat and Rough
> no longer share a directory). To resume a pre-rename run, point at its old
> experiment directory explicitly, e.g.
> `uv run train Unitree-Go2-Flat --agent.resume --agent.load-run go2_velocity`.

### Standalone workbench

```bash
# Run from the repository root (webui / src-webui remain as compatible entry points)
uv run --extra workbench src-workbench
```

Open `http://127.0.0.1:7860`. The workbench is organized as a fixed five-step flow:

1. **In-project URDF/MJCF viewer**: load project models or import a directory with meshes; move joints, show frames, joint axes, collision bodies, and inertias.
2. **Policy selection**: Velocity and Tracking expand exclusively; Tracking bundles GMR's LAFAN1 BVH → G1 29-DoF retargeting from the shared motion library, also accepts existing GMR PKL → NPZ, and shows retargeting progress, the human skeleton, and the G1 pose before timeline playback.
3. **Algorithm selection**: Velocity / Tracking groups; Go2 skills expose DreamWaQ, AMP-DreamWaQ, CTS, TS Teacher, and motion templates separately.
4. **Training**: set iterations, save interval, and parallel environments; watch live logs, stop jobs, and inspect rewards and metrics in the embedded TensorBoard.
5. **Export and validation**: browse, select, and download checkpoints/ONNX, export ONNX manually, and launch CPU single-environment Viser checkpoint playback.

The workbench source lives in `src/workbench/` and does not change training algorithms or environments under `tasks/`. The viewer is implemented in-project: MuJoCo parses the model and a small Three.js page draws it, following `robot_viewer`'s feature outline without copying the application. The viewer and the GMR subset both run inside the project.

Model import is for inspection; training still uses the registered task's robot configuration. GMR currently supports the LAFAN1 skeleton and G1 29-DoF, so 23-DoF tasks need their own NPZ. Viser plays checkpoints; ONNX is a separate export artifact.

Motions live in `src/assets/motions/`. The page imports and selects files, names and stores converted outputs, and can pick them as training motions.

Front-end development, module boundaries, and validation notes are in the [workbench documentation](docs/workbench.md).

### Online simulator

The browser MuJoCo + ONNX demo now lives outside this repository at <https://opendogerobotics.github.io/> (source: [OpenDogeRobotics.github.io](https://github.com/OpenDogeRobotics/OpenDogeRobotics.github.io)). Training, evaluation, and the workbench have no front-end dependency.

### Motion data

`csv-to-npz` creates the NPZ locally and does not require Weights & Biases. If `--output-name` is only a filename, the output is written next to the input CSV; an absolute or relative output path may also be used. Use `--robot g1_23dof` for G1-23DoF motions.

Tasks use a category prefix: `Unitree-<Robot>-Flat` / `Unitree-<Robot>-Rough` and `LainLab-<Robot>-Flat` / `LainLab-<Robot>-Rough`. Velocity tasks cover A2, As2, Go2, G1, G1-23Dof, H1_2, H2, R1, and OpenDoge; motion tracking covers G1 and G1-23Dof. OpenDoge registers Flat, Rough, Getup, Handstand, Jump, and `LainLab-OpenDoge-Skills-Flat`; the multi-skill design and acceptance record is in [docs/opendoge_multiskill.md](docs/opendoge_multiskill.md).

## Validation status and known limitations

As of 2026-09-09, CPU single-environment smoke checks cover environment construction, `reset`, and one zero-action step for registered tasks, checking that observations and rewards stay finite. All Go2 tasks (including Go2 skills), baseline velocity tasks, Cartpole, Yam manipulation tasks, and 29-DoF G1 tracking tasks passed; OpenDoge tasks additionally have reproducible metrics from `opendoge-eval`.

- Tracking tasks need an NPZ motion file whose degrees of freedom match the robot. Set `motion_file` for training or playback. The local 29-DoF G1 example cannot be used for `G1-23Dof` tracking.
- The default MuJoCo-Warp contact-buffer capacity is insufficient for `Unitree-H2-Rough` and `Unitree-R1-Rough`: they need at least `nconmax=80` and `nconmax=141`, respectively. Both pass smoke checks after that adjustment; do not start training with their defaults until the configuration is fixed.
- Smoke tests validate initialization and single-step numerical stability only. They do not demonstrate convergence and do not replace long training, checkpoint playback, or real-robot validation.

## Development

```bash
make format
make check
```

Robot differences are centralized under `src/tasks/robots/`. Shared MDPs live in task-type packages such as `velocity/core.py` and `tracking/core.py`. Adding a similar robot is typically a profile plus a registration in its robot package, without copying the shared MDP implementation.

## Deployment and export

This branch does not ship real-robot deployment code: training and validation go through the `play` Viser viewer, and exported ONNX policies (`opendoge-export`) are handed to the robot-side project, e.g. the OpenDoge firmware repository `OpenDoge_firmware`. The Isaac Gym version's `deploy/deploy_mujoco/` (config-driven MuJoCo validation and diagnostics) exists only on the `isaacgym` branch.

## Upstream and license

This branch is a downstream project of [mjlab](https://github.com/mujocolab/mjlab) 1.6.0, organized after `unitree_rl_mjlab`; the code skeleton and task prefix are inherited from LainLab. The code is Apache-2.0 licensed; OpenDoge assets are maintained by OpenDogeRobotics, and third-party components and robot descriptions keep their own licenses.
