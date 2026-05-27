# OpenDoge RL 训练调试 Skill

当用户调用此文件时，作为 Agent 执行以下循环工作流：

> **警告: 禁止执行任何 Git 操作。** 不要 `git commit`、`git push`、`git pull`、`git merge`、`git stash`、`git checkout`、`git reset` 或任何其他 Git 命令。本项目不需要 Agent 触碰版本控制。

---

## 核心原则: 参数灵活可调

**所有参数值、阈值、权重倍率均为参考建议，非硬性约束。** Agent 应根据实际训练曲线、指标组合、趋势方向自主判断调整幅度和时机。以下规则指南：

- 阈值和倍率给出的是**典型范围**，Agent 可在范围内外灵活选择
- 课程阶段边界是**近似值**，实际以指标表现为准
- 参数修改的"小幅/大幅"由 Agent 根据实际情况判断，无固定分界
- 如果某个修复方案连续 2 次无效，应尝试不同方向而非重复加倍
- 域随机化、PD 参数等硬约束同样可放宽，但需在报告中说明风险

---

## 前提

- **Conda 环境**: `himloco` (Python 3.8, PyTorch 2.3.1, Isaac Gym P4, MuJoCo 3.2.3)
- **项目根目录**: `/home/lain/OpenDoge/OpenDoge_train`
- **配置文件**: `legged_gym/envs/opendoge/opendoge_config.py`
- **训练入口**: `legged_gym/scripts/train.py --task=opendoge`
- **保存间隔**: 约 200~300 轮（由 `save_interval` 控制，Agent 可建议调整）
- **总迭代**: 默认 3000，用户可指定

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

### Step 2: 定期读取并分析

分析频率由 Agent 灵活决定：默认每 200~300 轮一次，但如果训练变化剧烈可缩短间隔，如果收敛平稳可拉长间隔。

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

### Step 3: 按课程阶段分析

四足步态训练的课程学习大致分为三个阶段，边界以实际指标表现为准，轮数仅为大致参考：

#### 阶段一: 生存期 (约 0 ~ 600 轮，以 episode_length 持续 > 1000 为出阶段标志)

**目标**: 机器人学会站立不倒，开始尝试移动。

**合理期望**:
- `episode_length` 从 200~500 快速增长
- `mean_reward` 快速上升
- 大部分惩罚项较负是正常的
- `tracking_lin_vel` 可能很低甚至为负

**此时不要过度干预**。只有以下情况才需要调整：
- `episode_length` 持续多轮无增长或下降 → 考虑降低 `action_scale`（如 0.25 → 0.15~0.2），机器人动作太大导致摔倒
- `collision` 持续很负 → 检查 URDF 或 `penalize_contacts_on`
- 训练完全发散（reward 不增反降） → 考虑降低 `learning_rate`（如 1e-3 → 5e-4 或更低）
- 调整幅度由 Agent 根据恶化速度判断，快速恶化用更大调整

#### 阶段二: 步态涌现期 (约 600 ~ 1800 轮，以 feet_air_time 转正或稳定为出阶段标志)

**目标**: 机器人开始形成清晰的 trot/bound 步态，能追踪速度指令。

**正常信号**:
- `tracking_lin_vel` 持续增长
- `feet_air_time` 从负值逐步改善
- `episode_length` 接近满时长
- 平滑性指标趋势改善（即使绝对值仍差）

**调参窗口期**，诊断和修复参考（倍率和目标值均为灵活范围）：

| 症状 | 诊断 | 修复方向（灵活选择倍率和目标） |
|------|------|------|
| `feet_air_time` 长期偏负 | 脚不离地，拖地行走 | 提高 `feet_air_time` 权重（如 ×1.5~3） |
| `smoothness` 或 `dof_acc` 很差且无改善 | 动作抖动严重 | 提高 `smoothness` 和 `dof_acc` 惩罚权重 |
| `tracking_lin_vel` 停滞在低值 | 跟不上速度指令 | 提高 `tracking_lin_vel` 权重 |
| `lin_vel_z` 偏负 | 上下弹跳 | 增加 `lin_vel_z` 惩罚 |
| `ang_vel_xy` 偏负 | 机身摇晃 | 增加 `orientation` 惩罚 |
| `stand_still` 偏负 | 零指令时静不下来 | 增加 `stand_still` 惩罚 |
| `base_height` 偏负 | 高度保持不好 | 增加 `base_height` 惩罚 |

**步态节奏**: 如果中期后 `feet_air_time` 仍无明显改善，可考虑调整 `clearance_height_target`（往更容易抬脚的方向调）。

#### 阶段三: 精炼期 (约 1800 轮+ ，以各惩罚项趋近 0 为标志)

**目标**: 步态稳定、低功耗、自然的行走风格。

**合理期望**:
- 各惩罚项趋近于 0
- `tracking_lin_vel` 达较高水平
- `mean_reward` 增长趋缓（收敛）
- 探索噪声自然下降

**精调方向**（Agent 根据实际情况选择幅度和时机）:
- 降低 `entropy_coef` 减少探索（如减半或逐步下调）
- 逐步提升 `max_curriculum` 扩大速度指令范围
- 若 gait 过于僵硬，提高 `action_scale`
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

**建议修改** (1-3 条，含范围和理由):
| # | 修改项 | 当前值 | 建议范围 | 原因 |
|---|--------|--------|---------|------|
| 1 | xxx    | X      | Y~Z     | zzz  |
```

### Step 5: 应用修改

用户确认后，直接编辑 `legged_gym/envs/opendoge/opendoge_config.py`，然后：

**修改记录**: 每次修改配置后，必须在 `docs/` 目录下创建以时间命名的文档（格式: `YYYY-MM-DD_HH-MM.md`），记录本次修改内容、原因和预期效果。模板：

```markdown
# 修改记录: YYYY-MM-DD HH:MM

## 诊断
(当前训练的问题描述)

## 修改内容
| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| xxx  | X    | Y    | zzz  |

## 预期效果
(期望训练朝哪个方向改善)
```

- **从 checkpoint 恢复训练**（多数情况适用）：
  ```bash
  python legged_gym/scripts/train.py --task=opendoge --headless --resume --load_run <run> --checkpoint <N>
  ```

- **从头训练**（仅当改动极大，如网络结构、PD 基础值、奖励结构重构时考虑）：
  ```bash
  python legged_gym/scripts/train.py --task=opendoge --headless
  ```

> Agent 需自行判断从 checkpoint 恢复还是从头训练。权重调整（无论倍率大小）通常可从 checkpoint 继续；结构性改动（增减奖励项、改网络、改 PD 基线）建议重训。不确定时列出利弊供用户选择。

### Step 6: 循环

等待合适间隔后回到 Step 2。Agent 根据训练进展速度自行决定等待时长。通过 `ps aux | grep train.py` 确认存活，通过 `ls -t logs/flat_opendoge/*/model_*.pt | head -1` 检查最新 checkpoint 时间戳。

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

以下阈值均为**典型参考值**，Agent 需结合趋势和阶段综合判断，不要机械套用：

```
episode_length 持续偏低        → 摔得太快，考虑降低 action_scale 或检查 URDF
tracking_lin   持续很低        → 不会追速度，考虑加大 tracking_lin_vel 权重
feet_air       长期为负        → 拖脚，考虑加大 feet_air_time 权重
smoothness     很差且无改善    → 太抖，考虑加大 smoothness + dof_acc 惩罚
lin_vel_z      持续很负        → 弹跳，考虑加大 lin_vel_z 惩罚
orientation    持续很负        → 机身前倾，考虑加大 orientation 惩罚
stand_still    持续很负        → 停不下来，考虑加大 stand_still 惩罚
collision      持续很负        → 自碰严重，检查 penalize_contacts_on
noise_std      居高不下(后期)  → 探索太大，考虑降低 entropy_coef
reward         持续下降        → 训练崩溃，考虑降低 learning_rate
mono reward    dominant        → 某奖励比其它大 10×+，重新平衡权重
stiffness      偏高            → PD 过高真机难复现，考虑降低 + 增加 dof_acc
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

从指标组合推断当前步态类型（特征值为典型表现，实际以组合判断为准）：

| 模式 | 特征 | 工程直觉 |
|------|------|----------|
| **Trot (理想)** | `feet_air` 正值, `diagonal_sync` ≈ 0 | 对角交替支撑，高效节能 |
| **Bound (弹跳)** | `feet_air` 正值但 `lin_vel_z` 很负, `smoothness` 极负 | 四足同时离地，冲击大，Sim2Real 差 |
| **Shuffle (拖地)** | `feet_air` 为负, tracking 尚可 | 高频微滑代替抬腿，真机原地蹭 |
| **Static crawl** | tracking 好但 `action_rate` 很负 | 动作太慢太保守，速度上限低 |
| **Sitting (趴地)** | `base_height` 很负, tracking 好 | 降低身体省力，真机直接趴窝 |

### C. PD ↔ 真机可行性检查

以下为**柔性参考范围**，超出时需在报告中说明 Sim2Real 风险，但不强制禁止：

| 参数 | 典型推荐范围 | 超出时的风险 |
|------|-------------|--------------|
| `stiffness` | 10~30 | 过高真机伺服达不到，Sim2Real 断裂 |
| `damping` | 0.3~5.0 | 过高电机发热严重，电流飙升 |
| `action_scale` | 0.15~0.5 | 过大关节可能超出机械限位 |
| `decimation` | ≥2 | =1 需要极高频控制，真实通信难实现 |
| `armature` | 0.003~0.01 | 超出电机转子惯量规格则仿真失真 |

> Agent 可根据训练需求灵活调整这些参数，只需在修改时标注 Sim2Real 风险等级（低/中/高）。

### D. 奖励平衡性分析

健康训练中各奖励项绝对值应在合理范围内，避免单一奖励主导梯度：

```
健康:     tracking=2.5 | smoothness=-0.3 | dof_acc=-0.3 | feet_air=0.01
不健康:   tracking=10.0| smoothness=-0.01| dof_acc=-0.01| feet_air=0.001
          ↑ tracking 主导梯度，其他奖励形同虚设
```

若检测到单一奖励主导（比其他奖励绝对值大一个数量级以上），Agent 应建议：降低主导奖励权重，或放大最弱奖励权重。具体倍率由 Agent 根据差距大小灵活选择。

---

## 典型失败模式诊断 (HIMLoco 经验)

以下修复方案中的倍率和目标值均为**灵活参考**，Agent 应根据严重程度和阶段选择合适的调整幅度：

### 模式 1: Shuffling (高频拖地)

**症状**: `smoothness` 很差, `dof_acc` 很差, `feet_air_time` 为负, tracking 尚可
**根因**: 策略用高频微振替代抬腿。仿真中地面摩擦为零也能滑行，但真机有摩擦力会原地蹭。
**修复方向**:
1. 提高 `feet_air_time` 权重（如 ×2~5）
2. 提高 `smoothness` 惩罚权重
3. 降低 `static_friction` / `dynamic_friction`（逼策略找真实摩擦力下的步态）
4. 阶段二中期可临时关闭 `randomize_friction` 让策略专注学抬脚

### 模式 2: Bounding (四足弹跳)

**症状**: `lin_vel_z` 很负, `feet_air_time` 突然转正, 生存率正常
**根因**: 四足同时蹬地弹跳能更快追踪速度指令（投机取巧）。
**修复方向**:
1. 加大 `lin_vel_z` 惩罚
2. 加大 `ang_vel_xy` 和 `orientation` 惩罚
3. 缩小指令范围移除高速诱因

### 模式 3: Sitting (降低身体)

**症状**: `base_height` 很负, 其他指标正常但机器人看起来"半蹲半蹭"
**根因**: 弯曲膝盖降低重心可省力（势能换低力矩）。
**修复方向**:
1. 加大 `base_height` 惩罚
2. 提高 `init_state.pos[2]`
3. 调整 `default_joint_angles` 中 calf 角度使初始姿态膝盖更直

### 模式 4: 不对称步态

**症状**: `hip_mirror_symmetry` 和 `diagonal_sync` 很差
**根因**: 左右侧运动不对称，单侧负载过重。
**修复方向**:
1. 提高 `hip_mirror_symmetry` 权重
2. 提高 `diagonal_sync` 权重

### 模式 5: 训练崩溃

**症状**: `mean_reward` 单次 checkpoint 骤降明显
**根因**: PPO 遇到极端 batch 产生一次坏更新。
**修复方向**:
1. 回退上一 checkpoint 恢复训练
2. 降低 `learning_rate`
3. 暂时关闭 `disturbance` + `delay`
4. 若反复出现，增大 `num_steps_per_env`

---

## 域随机化调度

域随机化应**先学会再泛化**，不全程等强度开启。以下为**柔性参考**，Agent 可根据实际训练进展调整开关时机和强度：

| 域随机项 | 早期 (生存期) | 中期 (步态涌现) | 后期 (精炼期) |
|----------|--------------|----------------|--------------|
| `randomize_friction` | 适度 ON | 适度 ON | ON 或加强 |
| `randomize_motor_strength` | OFF 或弱 | 适度 ON | ON 或加强 |
| `push_robots` | OFF | OFF 或弱 | ON |
| `disturbance` | OFF | OFF 或弱 | ON |
| `delay` | OFF | OFF | OFF 或弱 |
| `randomize_kp/kd` | OFF | OFF 或弱 | ON |

> 如果步态涌现困难，优先关闭 `disturbance` + `delay` + `push_robots`，待 gait 形成后再逐步加回来。Agent 可自由决定各项的开启时机和强度范围。

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
# 平地步态训练完成后: mesh_type='plane'
# 地形适应阶段:      mesh_type='trimesh'
```

切换要点：
- `terrain_proportions` 初始多做缓坡，逐步增加难度
- `max_init_terrain_level` 从 0 开始
- `num_envs` 地形渲染吃显存时可降低
- 切换时机由 Agent 根据步态收敛程度灵活判断
