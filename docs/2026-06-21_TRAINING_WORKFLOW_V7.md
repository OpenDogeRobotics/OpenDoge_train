# OpenDoge 训练工作流 V7 — 基于全量数据优化的训练方案

> 基于 30+ 次实验、6 轮消融(R1-R6)、Gen 1-5.2 全部历史数据的综合复盘

---

## 一、核心发现：Plan V6 的错误假设

Plan V6 基于以下假设构建三阶段计划，**但数据证明这些假设不成立**：

| V6 假设 | 实际数据 | 严重性 |
|----------|---------|--------|
| "R3 纯净版是稳定基底" | R3 自身在 step 3700+ 也开始退化 (rew 20→5, noise 上升) | **致命** |
| "push_robots 单项可控 (rew>5)" | R6 push-only: reward -32.78@step5837, smoothness -1.85 | **致命** |
| "从 R3 最优 checkpoint 恢复 + push" | 恢复后 600 轮内 reward 20→负值，不可逆退化 | **致命** |
| "disturbance 渐进可叠加" | R5 push+disturb 双开: reward -28@step6244 (即使从 R3 恢复) | **致命** |

**根因**: 问题不在 push_robots 或 disturbance 是否单独/组合开启，而是**策略在 step 3000+ 后天然退化**，任何域随机化只是加速了这一过程。

---

## 二、退化模式深度分析

### 2.1 时间线证据 (R3 纯净版)

```
Step 3000-3500:  mean_reward 15-25, tracking_lin 0.85-0.95, smoothness -0.10~-0.30
Step 3600-4200:  mean_reward 2-17 (大幅振荡), 负值初现
Step 4300-4800:  mean_reward -10~+10, smoothness 多次跳到 -1.0~-1.9
Step 4900-7000:  mean_reward 持续负值, noise_std 0.22→0.26 持续上升
```

**退化不是突然的，而是渐进的、不可逆的。**

### 2.2 退化特征

| 指标 | 退化方向 | 幅度 (3000→5800) |
|------|---------|-------------------|
| mean_reward | +24 → -32 | 下降 56 点 |
| tracking_lin_vel | 0.87 → 0.46 | **-47%** |
| smoothness | -0.12 → -1.85 | **15x 恶化** |
| base_height | -0.04 → -0.39 | **10x 恶化** (蹲伏) |
| collision | -0.01 → -0.58 | **58x 恶化** (自碰) |
| noise_std | 0.22 → ~0.28(est) | 持续上升 |

### 2.3 退化机制假说

1. **PPO 探索悬崖**: 策略在 step 3000-3500 到达 reward landscape 的局部最优峰顶。entropy_coef=0.003 的持续探索噪声偶尔将策略推入"悬崖区"(reward 极负的参数空间)，一次坏更新后 Adam 动量将其拉向错误方向。

2. **追踪-稳定性零和博弈**: 速度追踪和步态平滑是矛盾的优化目标。策略学到:为了追踪更高速度指令，可以牺牲 smoothness/base_height。一旦开始"暴力追踪"，所有稳定性指标连锁恶化。

3. **价值函数过拟合**: 训练后期价值函数对当前策略分布过拟合，对新区域估计不准，导致 PPO advantage 估计偏差增大，更新方向逐渐偏离。

---

## 三、优化后的训练工作流

### 核心原则

> **不追求"训练得更久"，而追求"在 peak 附近精确停止并导出"。**

### 3.1 单阶段训练 (替代三阶段)

```
从头训练 (0 → 4000 轮)
    │
    ├─ Step 0-1500:   生存期 — 不干预
    ├─ Step 1500-2500: 步态涌现 — 监控 feet_air, smoothness
    ├─ Step 2500-3500: 精炼期 — 这是 peak 区间
    ├─ Step 3000-4000: 紧盯着 — 一旦 noise_std 连续 300 轮上升, 停止
    │
    └─ 导出 peak checkpoint → ONNX
```

**关键变更**:
- `max_iterations` = **4000** (非 6000/7000) — 不再训练到退化区
- 在 step 2500-4000 之间找到最优 checkpoint (综合 smoothness + tracking + noise_std)
- **不从 checkpoint 恢复继续训练** — 每次恢复都加速退化

### 3.2 鲁棒性独立训练

如果需要抗推力/抗干扰能力，**单独训练鲁棒性策略**（非从纯净 checkpoint 恢复）:

```python
# 鲁棒性训练专用配置
max_iterations = 2500           # 更短的训练周期
push_robots = True
push_interval_s = 15            # 更稀疏 (原 10s)
max_push_vel_xy = 0.3           # 更弱 (原 0.5)
learning_rate = 3e-4            # 更低 LR (原 5e-4)
entropy_coef = 0.002            # 更低探索 (原 0.003)

# 课程式增加推力
# step 0-800:   max_push_vel_xy = 0.1
# step 800-1600: max_push_vel_xy = 0.2
# step 1600-2500: max_push_vel_xy = 0.3
```

### 3.3 最优导出策略

最终产出两个 ONNX:
1. **纯净步态 ONNX**: 从纯净训练的 step 3000-4000 最优 checkpoint
2. **鲁棒 ONNX** (可选): 从鲁棒独立训练的 step 1500-2500 最优 checkpoint

---

## 四、配置变更

### 4.1 纯净训练配置 (从当前配置修改)

```python
# opendoge_config.py 变更

# === PPO ===
entropy_coef = 0.002          # 0.003→0.002: R3证明0.003仍有late-stage noise上升
                              #                 R4证明0.001太低保peak
learning_rate = 5e-4          # 保持: 已验证最优
schedule = 'fixed'            # 保持: 根因修复
num_steps_per_env = 48        # 保持: 72验证过差

# === 域随机化 ===
push_robots = False           # 纯净训练关闭 (R6证明单项也崩溃)
disturbance = False           # 关闭 (R5证明双开崩溃)
randomize_kp = False          # 保持关闭 (R2证明破坏控制映射)
randomize_kd = False          # 保持关闭
randomize_friction = True     # 保持 (物理参数随机化安全)
randomize_base_mass = True    # 保持
randomize_motor_strength = True # 保持

# === 奖励权重 ===
tracking_lin_vel = 2.0        # 保持 (2.5验证破坏平衡)
tracking_ang_vel = 1.5        # 保持 V6 值
base_height = -1.5            # 保持 (-3.0反噬)
feet_air_time = 1.0           # 保持
smoothness = -0.04            # 保持
dof_acc = -2e-6               # 保持
collision = -1.0              # 保持

# === 训练 ===
max_iterations = 4000         # 6000→4000: 不训练到退化区
save_interval = 200           # 300→200: 更密集保存以便精确选peak
```

### 4.2 鲁棒训练配置 (独立运行)

```python
# 鲁棒训练专用 (从头训练，非resume)
max_iterations = 2500
push_robots = True
push_interval_s = 15
max_push_vel_xy = 0.3         # 弱推力起步
disturbance = False           # 先不加扰动
entropy_coef = 0.002
learning_rate = 3e-4          # 更低LR防崩溃
randomize_kp = False
randomize_kd = False
```

---

## 五、监控指标与自动化

### 5.1 退化预警 (新增)

在 Step 2 分析中增加自动退化检测:

```python
# 退化检测逻辑
if noise_std 连续 300 轮上升 and noise_std > 0.24:
    → 橙色预警: 策略开始漂移
if mean_reward 较 peak 下降 > 50%:
    → 红色预警: 可能已越过悬崖
if smoothness < -1.0 连续 200 轮:
    → 红色预警: 步态质量严重退化
```

### 5.2 最优 Checkpoint 选择算法

```python
def select_best_checkpoint(metrics_history):
    """
    综合评分 = smoothness * 0.4 + tracking_lin * 0.3 + mean_reward * 0.2 - noise_std * 0.1
    从 step >= 2500 的 checkpoint 中选择最高分
    """
```

### 5.3 自动停止条件

```
停止训练条件 (任一满足):
1. max_iterations 达到 (4000)
2. noise_std > 0.26 且连续 500 轮未下降
3. mean_reward < 0 连续 500 轮
4. smoothness < -2.0 连续 300 轮
```

---

## 六、为什么 V7 应该工作

| V6 失败原因 | V7 对策 |
|-------------|---------|
| 训练到退化区 (step 5000+) | max_iterations=4000, 在 peak 导出 |
| 从退化中的 checkpoint resume | 不再 resume, 每次独立从头训练 |
| 在脆弱策略上加 push/disturb | 鲁棒性单独训练, 降低 LR+推力 |
| 未检测到渐进退化 | 自动退化预警 + 自动停止 |
| entropy=0.003 仍有 late-stage 噪声 | entropy=0.002 (Goldilocks) |

---

## 七、执行计划

### 立即执行 (当前训练已崩溃, 需停止)

1. **停止当前训练** (PID 150266, reward -32.78)
2. **修改配置**: 
   - `max_iterations = 4000`
   - `push_robots = False`
   - `entropy_coef = 0.002`
   - `save_interval = 200`
3. **从头训练**: 不 resume 任何旧 checkpoint
4. **监控**: 设置退化预警自动检测

### 后续 (纯净训练成功后)

5. **导出纯净 ONNX**
6. **独立鲁棒训练**: 使用鲁棒配置从头训练
7. **导出鲁棒 ONNX**

---

## 八、已知风险

| 风险 | 缓解 |
|------|------|
| entropy=0.002 可能仍偏高 | 若 noise_std 仍上升, 降至 0.0015 |
| entropy=0.002 可能偏低 (peak 受限) | 若 tracking_lin 停滞 <0.7, 升至 0.0025 |
| 4000 轮可能不够收敛 | 若 smoothness 仍在改善趋势中, 延长至 5000 |
| 鲁棒独立训练可能也不稳定 | 进一步降低 push_vel_xy 至 0.2, LR 至 2e-4 |
