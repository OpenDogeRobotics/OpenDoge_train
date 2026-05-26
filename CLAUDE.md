# OpenDoge RL 训练调试 Skill

当用户调用此文件时，作为 Agent 执行以下循环工作流：

> **警告: 禁止执行任何 Git 操作。** 不要 `git commit`、`git push`、`git pull`、`git merge`、`git stash`、`git checkout`、`git reset` 或任何其他 Git 命令。本项目不需要 Agent 触碰版本控制。

---

## 前提

- **Conda 环境**: `himloco` (Python 3.8, PyTorch 2.3.1, Isaac Gym P4, MuJoCo 3.2.3)
- **项目根目录**: `/home/lain/OpenDoge/OpenDoge_train`
- **配置文件**: `legged_gym/envs/opendoge/opendoge_config.py`
- **训练入口**: `legged_gym/scripts/train.py --task=opendoge`
- **保存间隔**: 300 轮 (`save_interval=200`，但每 300 轮触发分析)
- **总迭代**: 3000 (`max_iterations=30000` 或用户指定)

---

## 工作流

### Step 1: 启动训练

如果训练未在运行：

```bash
cd /home/lain/OpenDoge/OpenDoge_train
export PYTHONPATH=$PWD
conda run -n himloco python legged_gym/scripts/train.py --task=opendoge --headless &
```

或者从 checkpoint 恢复：

```bash
export PYTHONPATH=$PWD
conda run -n himloco python legged_gym/scripts/train.py --task=opendoge --headless --resume --load_run <run_dir> --checkpoint <N> &
```

### Step 2: 每 300 轮读取并分析

```bash
conda run -n himloco python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

log_root = 'logs/flat_opendoge'
dirs = sorted(glob.glob(f'{log_root}/*/'))
if not dirs: exit('no log dir')
log_dir = dirs[-1]
ea = EventAccumulator(log_dir)
ea.Reload()
for s in sorted(ea.Tags()['scalars']):
    events = ea.Scalars(s)
    if events:
        e = events[-1]
        print(f'{s:45s} step={e.step:6d}  value={e.value:.6f}')
" 2>/dev/null | grep -v tensorflow
```

同时检查最新 checkpoint：

```bash
ls -t logs/flat_opendoge/*/model_*.pt 2>/dev/null | head -5
```

判断当前训练的 step 数，确认是否需要等待到下一个 300 轮 checkpoint。

### Step 3: 按课程阶段分析

四足步态训练的课程学习分为三个阶段，不同阶段的指标解读标准不同：

#### 阶段一: 生存期 (0 ~ 600 轮)

**目标**: 机器人学会站立不倒，开始尝试移动。

**合理期望**:
- `episode_length` 从 200~500 快速增长到 1500+
- `mean_reward` 快速上升
- 大部分惩罚项（collision、orientation 等）较负是正常的
- `tracking_lin_vel` 可能很低甚至为负

**此时不要过度干预**。只有以下情况才需要调整：
- `episode_length < 500` 持续 3 个 checkpoint → 降低 `action_scale` 到 0.15，机器人动作太大导致摔倒
- `collision < -3.0` → 检查 URDF 或 `penalize_contacts_on`
- 训练完全发散（reward 不增反降）→ 降低 `learning_rate` 到 5e-4

#### 阶段二: 步态涌现期 (600 ~ 1800 轮)

**目标**: 机器人开始形成清晰的 trot/bound 步态，能追踪速度指令。

**正常信号**:
- `tracking_lin_vel` 持续增长，达到 1.5~2.5
- `feet_air_time` 从负值逐步转正或接近 0
- `episode_length` 接近 2000 (20s 满时长)
- `smoothness`/`dof_acc` 仍然较负但趋势在改善

**调参窗口期**，需要关注：

| 症状 | 诊断 | 修复 |
|------|------|------|
| `feet_air_time` 长期 < -0.01 | 脚不离地，拖地行走 | `feet_air_time` 权重 ×2 (e.g. 0.05→0.1) |
| `smoothness` < -0.5 且无改善 | 动作抖动严重 | `smoothness` 权重 ×2 (e.g. -0.01→-0.02)，`dof_acc` 权重 ×2 |
| `tracking_lin_vel` 停滞在 < 1.0 | 跟不上速度指令 | 提升 `tracking_lin_vel` 权重到 4.0~5.0 |
| `lin_vel_z` < -0.15 | 上下弹跳 | 增加 `lin_vel_z` 惩罚到 -3.0 |
| `ang_vel_xy` < -0.2 | 机身摇晃 | 增加 `orientation` 惩罚到 -2.0 |
| `stand_still` < -0.1 | 零指令时静不下来 | 增加 `stand_still` 惩罚到 -3.0 |
| `base_height` < -0.05 | 高度保持不好 | 增加 `base_height` 惩罚到 -2.0 |

**步态节奏**: 如果 1200 轮后 `feet_air_time` 仍然是 < -0.005，考虑加 `clearance_height_target` 稍微抬高一点（-0.135 → -0.12，数字更大=脚抬更高）。

#### 阶段三: 精炼期 (1800 ~ 3000 轮)

**目标**: 步态稳定、低功耗、自然的行走风格。

**合理期望**:
- 各惩罚项趋近于 0
- `tracking_lin_vel` > 2.0
- `mean_reward` 增长趋缓（收敛）
- `noise_std` 下降到 0.5 以下

**精调方向**:
- 降低 `entropy_coef` (0.01 → 0.005) 减少探索
- 逐步提升 `max_curriculum` 让速度指令范围扩大
- 若 gait 过于僵硬，提高 `action_scale` (0.25 → 0.3)
- 若 gait 不够节能，提高 `joint_power` 惩罚

### Step 4: 输出分析报告

每次分析输出以下格式：

```
### 第 N 轮分析 | step=X | 阶段: [生存/步态/精炼]

**当前状态**:
- 生存: episode_length=Xs, 摔倒率=X%
- 跟踪: lin_vel=X, ang_vel=X
- 步态: smoothness=X, dof_acc=X, feet_air=X
- 稳定: orientation=X, ang_vel_xy=X, lin_vel_z=X

**趋势**: (与上一 checkpoint 对比)
- 改善: [列举]
- 恶化: [列举]
- 不变: [列举]

**诊断**: 一句话

**建议修改** (1-3 条):
| # | 修改项 | 从 | 到 | 原因 |
|---|--------|----|----|------|
| 1 | xxx    | X  | Y  | zzz  |
```

### Step 5: 应用修改

用户确认后，直接编辑 `legged_gym/envs/opendoge/opendoge_config.py`，然后：

- **小幅调整** (奖励权重变化 < 2x): 从当前 checkpoint 恢复训练：
  ```bash
  python legged_gym/scripts/train.py --task=opendoge --headless --resume --load_run <run> --checkpoint <N>
  ```

- **大幅调整** (参数变化 > 2x 或改了 PD/structure): 建议从头训练。

### Step 6: 循环

等待 300 轮后回到 Step 2。如果训练在后台运行，通过 `ps aux | grep train.py` 确认存活，通过 `ls -t logs/flat_opendoge/*/model_*.pt | head -1` 检查最新 checkpoint 时间戳。

---

## 首轮执行检查清单

首次被调用时，依次检查：

1. **环境**: `conda env list | grep himloco` 确认环境存在
2. **Isaac Gym**: 确认 `/home/lain/IsaacGym_Preview_4_Package/` 存在且 `torch_utils.py` 已修复 `np.float→np.float64`
3. **rsl_rl**: 确认已安装 `pip show rsl_rl 2>/dev/null || pip install -e rsl_rl/`
4. **URDF**: 确认 `resources/robots/Opendoge/urdf/Opendoge.urdf` 存在
5. **是否有运行中的训练**: `ps aux | grep train.py | grep -v grep`
6. **最新的日志目录**: `ls -lt logs/flat_opendoge/ | head -3`

---

## 快速诊断速查表

```
episode_length < 500     → 摔得太快，降低 action_scale 或检查 URDF
tracking_lin < 0.5       → 不会追速度，加大 tracking_lin_vel 权重
feet_air < -0.01         → 拖脚，加大 feet_air_time 权重
smoothness < -0.5        → 太抖，加大 smoothness + dof_acc 惩罚
lin_vel_z < -0.2         → 弹跳，加大 lin_vel_z 惩罚
orientation < -0.05      → 机身前倾，加大 orientation 惩罚
stand_still < -0.15      → 停不下来，加大 stand_still 惩罚
collision < -1.0         → 自碰严重，检查 penalize_contacts_on
noise_std > 0.9          → 探索太大(>1500轮时)，降低 entropy_coef
reward 持续下降           → 训练崩溃，降低 learning_rate
mono reward dominant     → <主 reward> 比其他大 10×+，重新平衡权重
stiffness > 20           → PD 过高，真机无法复现，降低 + 增加 dof_acc
```

---

## 增强型分析 (Step 3.5)

每轮分析时需额外执行以下深度诊断：

### A. Checkpoint 差值分析

读取最近两个 checkpoint 的值并计算 delta，判断趋势方向：

```bash
conda run -n himloco python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

log_root = 'logs/flat_opendoge'
dirs = sorted(glob.glob(f'{log_root}/*/'))
log_dir = dirs[-1]
ea = EventAccumulator(log_dir)
ea.Reload()

metrics = ['Train/mean_reward', 'Episode/rew_tracking_lin_vel',
           'Episode/rew_feet_air_time', 'Episode/rew_smoothness',
           'Episode/rew_base_height', 'Episode/rew_lin_vel_z']
for m in metrics:
    events = ea.Scalars(m)
    if len(events) >= 2:
        prev = events[-2].value
        curr = events[-1].value
        delta = curr - prev
        direction = 'improving' if delta > 0 else 'worsening'
        print(f'{m:45s}  prev={prev:8.4f}  curr={curr:8.4f}  delta={delta:+8.4f} {direction}')
" 2>/dev/null | grep -v tensorflow
```

### B. 步态模式检测

从指标组合推断当前步态类型：

| 模式 | 特征 | 工程直觉 |
|------|------|----------|
| **Trot (理想)** | `feet_air` 正值, `diagonal_sync` ≈ 0 | 对角交替支撑，高效节能 |
| **Bound (弹跳)** | `feet_air` 正值但 `lin_vel_z` < -0.15, `smoothness` 极负 | 四足同时离地，冲击大，Sim2Real 差 |
| **Shuffle (拖地)** | `feet_air` < -0.01, tracking 尚可 | 高频微滑代替抬腿，真机原地蹭 |
| **Static crawl** | tracking 好但 `action_rate` < -0.3 | 动作太慢太保守，速度上限低 |
| **Sitting (趴地)** | `base_height` < -0.08, tracking 好 | 降低身体省力，真机直接趴窝 |

### C. PD ↔ 真机可行性检查

| 参数 | 真机推荐上限 | 超过上限的后果 |
|------|-------------|--------------|
| `stiffness` | 10~20 | >30 真机伺服达不到，Sim2Real 断裂 |
| `damping` | 0.3~1.0 | >5 电机发热严重，电流飙升 |
| `action_scale` | 0.2~0.35 | >0.5 关节可能超出机械限位 |
| `decimation` | ≥2 | =1 需要 200Hz 控制，真实通信做不到 |
| `armature` | 0.003~0.01 | 超出电机转子惯量规格则仿真失真 |

> 当前 OpenDoge 配置 `stiffness=10, damping=0.3, armature=0.005` 均在合理范围内。如果后续调参导致这些值升高，需在报告中发出警告。

### D. 奖励平衡性分析

健康训练中各奖励项绝对值应不超过一个数量级的差距：

```
健康:     tracking=2.5 | smoothness=-0.3 | dof_acc=-0.3 | feet_air=0.01
不健康:   tracking=10.0| smoothness=-0.01| dof_acc=-0.01| feet_air=0.001
          ↑ tracking 主导梯度，其他奖励形同虚设
```

若检测到单一奖励主导，自动建议：将主导奖励权重减半，或放大最弱奖励权重 ×2。

---

## 典型失败模式诊断 (HIMLoco 经验)

### 模式 1: Shuffling (高频拖地)

**症状**: `smoothness` < -0.5, `dof_acc` < -0.4, `feet_air_time` ≈ -0.01, tracking 尚可
**根因**: 策略用高频微振替代抬腿。仿真中地面摩擦为零也能滑行，但真机有摩擦力会原地蹭。
**修复**:
1. `feet_air_time` 权重 ×3 (0.05 → 0.15)
2. `smoothness` 权重 ×2 (-0.01 → -0.02)
3. 将 `static_friction` / `dynamic_friction` 从 1.0 降到 0.6，逼策略找真实摩擦力下的步态
4. 阶段二中期可临时关闭 `randomize_friction` 让策略专注学抬脚

### 模式 2: Bounding (四足弹跳)

**症状**: `lin_vel_z` < -0.2, `feet_air_time` 突然转正, 生存率正常
**根因**: 四足同时蹬地弹跳能更快追踪速度指令（投机取巧）。
**修复**:
1. `lin_vel_z` 惩罚 ×2.5 (-2.0 → -5.0)
2. `ang_vel_xy` 惩罚 ×2 (-0.05 → -0.1)
3. `orientation` 惩罚 ×2 (-1.3 → -2.5)
4. 指令范围从 [-4,4] 降到 [-2,2]，移除高速诱因

### 模式 3: Sitting (降低身体)

**症状**: `base_height` < -0.08, 其他指标正常但机器人看起来"半蹲半蹭"
**根因**: 弯曲膝盖降低重心可省力（势能换低力矩）。
**修复**:
1. `base_height` 惩罚 ×3 (-1.0 → -3.0)
2. `init_state.pos[2]` 从 0.15 → 0.20
3. `default_joint_angles` 中 calf 角度从 -1.5 → -1.2（初始姿态膝盖更直）

### 模式 4: 不对称步态

**症状**: `hip_mirror_symmetry` < -0.1, `diagonal_sync` < -0.1
**根因**: 左右侧运动不对称，单侧负载过重。
**修复**:
1. `hip_mirror_symmetry` 权重 ×2
2. `diagonal_sync` 权重 ×2

### 模式 5: 训练崩溃

**症状**: `mean_reward` 单次 checkpoint 骤降 >30%
**根因**: PPO 遇到极端 batch 产生一次坏更新。
**修复**:
1. 回退上一 checkpoint 恢复训练
2. `learning_rate` 减半
3. 暂时关闭 `disturbance` + `delay`
4. 若反复出现，增大 `num_steps_per_env` (48 → 72)

---

## 域随机化调度

域随机化应**先学会再泛化**，不全程等强度开启：

| 域随机项 | 0~600 | 600~1200 | 1200~1800 | 1800+ |
|----------|-------|----------|-----------|-------|
| `randomize_friction` | ON (0.8~1.2) | ON (0.6~1.3) | ON (0.5~1.25) | ON |
| `randomize_motor_strength` | OFF | ON (0.95~1.05) | ON (0.9~1.1) | ON |
| `push_robots` | OFF | OFF | ON | ON |
| `disturbance` | OFF | OFF | ON (小) | ON |
| `delay` | OFF | OFF | OFF | ON |
| `randomize_kp/kd` | OFF | OFF | ON | ON |

> 当前配置中所有域随机化从 step 0 全部开启。如果步态涌现困难，优先关闭 `disturbance` + `delay` + `push_robots`，待 gait 形成后再逐步加回来。

---

## 实验日志

每次修改参数后在根目录 `tuning_log.json` 追加一条记录：

```json
{
  "timestamp": "2026-05-27T01:30:00",
  "checkpoint_step": 600,
  "run_dir": "May27_00-56-29_opendoge_himloco_v1.0",
  "diagnosis": "shuffling - feet not leaving ground",
  "changes": {
    "feet_air_time": "0.05 -> 0.15",
    "smoothness": "-0.01 -> -0.02"
  },
  "expected_effect": "encourage foot clearance, suppress vibration",
  "actual_effect": null
}
```

下次分析时读取 `tuning_log.json`，避免重复已证明无效的修改。效果验证后更新 `actual_effect` 字段。

---

## 地形课程规划

当前 `mesh_type='plane'`。平地步态收敛后可切换：

```python
# step 0~3000: mesh_type='plane'     (学步态)
# step 3000+:  mesh_type='trimesh'   (学地形适应)
```

切换要点：
- `terrain_proportions` 初始 `[0.3, 0.3, 0.15, 0.15, 0.1]` 多做缓坡
- `max_init_terrain_level` 从 0 开始
- `num_envs` 从 4096 降到 2048（地形渲染吃显存）
