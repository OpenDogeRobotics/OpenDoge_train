# OpenDoge 从零训练计划 V6

> 基于 6 轮消融实验 (R1-R6) 的最佳实践总结

## 实验结论速览

| # | 实验 | 结论 |
|---|------|------|
| 1 | `schedule='fixed'` vs `'adaptive'` | **fixed** — adaptive 导致 LR 衰减至 1e-5 |
| 2 | `entropy=0.005/0.003/0.001` | **0.003** 最优 — 0.005 噪声发散, 0.001 探索不足 |
| 3 | `randomize_kp/kd` on/off | **off** — 改变 action→torque 映射, 破坏精确控制 |
| 4 | `base_height=-1.5/-3.0` | **-1.5** — -3.0 扭曲 reward landscape |
| 5 | `push_robots + disturbance` 双开 | **不可同时** — 双开导致崩溃 (rew -28.7) |
| 6 | `push_robots` 单项 | **可行** — 与纯净版持平 |
| 7 | 最优组合 | **schedule=fixed, ent=0.003, kp/kd=off, dist=off** |

---

## 固定配置 (全程不变)

```python
# === 训练稳定性 ===
schedule = 'fixed'          # 不衰减 LR
learning_rate = 5e-4        # 恒定
entropy_coef = 0.003        # 最优探索水平
num_steps_per_env = 48      # 已验证

# === 控制 ===
stiffness = {'joint': 12.0}
damping = {'joint': 0.5}
action_scale = 0.30
decimation = 2

# === 域随机化 (全程) ===
randomize_friction = True   # [0.3, 1.5]
randomize_base_mass = True  # [-0.15, 0.35]
randomize_motor_strength = True  # [0.85, 1.15]

# === 关闭 (不开启) ===
randomize_kp = False        # 破坏控制映射
randomize_kd = False        # 破坏控制映射
delay = False               # 步态未稳定时不加延迟

# === 奖励权重 ===
tracking_lin_vel = 2.0
tracking_ang_vel = 1.5
lin_vel_z = -2.5
ang_vel_xy = -0.10
orientation = -2.5
base_height = -1.5          # 不加大, 防反噬
feet_air_time = 1.0
smoothness = -0.04
dof_acc = -2e-6
collision = -1.0
stand_still = -2.0
diagonal_sync = -0.2
hip_mirror_symmetry = -0.1
action_rate = -0.02
joint_power = -2e-5
default_pos_linear = -0.05
```

---

## 三阶段训练计划

### 阶段一: 纯净步态 (0 → 3000 轮)

**目标**: 学习稳定、平滑的 trot 步态

| 参数 | 值 |
|------|-----|
| `push_robots` | **False** |
| `disturbance` | **False** |
| `max_iterations` | 3000 |
| `save_interval` | 300 |

**成功标准**:
- episode_length > 1900
- smoothness > -0.3
- tracking_lin > 0.75
- noise_std < 0.23 且趋势下降

**出口**: 达到标准后进入阶段二。若未达标，延长 1000 轮。

**期望效果**: 参考 R3 — peak mean_reward ~21, smoothness ~-0.15, noise_std ~0.22

---

### 阶段二: 推力鲁棒性 (3000 → 5000 轮)

**目标**: 在稳定步态基础上适应间歇性推力干扰

| 参数 | 值 |
|------|-----|
| `push_robots` | **True** (max_vel=0.5, interval=10s) |
| `disturbance` | **False** |
| `max_iterations` | 5000 |
| `save_interval` | 300 |

**从阶段一最优 checkpoint 恢复**:
```bash
python legged_gym/scripts/train.py --task=opendoge --headless \
    --resume --load_run <phase1_run> --checkpoint <best>
```

**成功标准**:
- mean_reward > 5 持续 500 轮以上
- noise_std 不持续上升
- smoothness > -0.5

**出口**: 达标后进入阶段三。若退化，回退检查。

**期望效果**: 参考 R6 — reward 维持在 5-10, noise 保持 <0.23

---

### 阶段三: 全面鲁棒性 (5000 → 7000 轮)

**目标**: 在推力基础上叠加外力干扰

| 参数 | 值 |
|------|-----|
| `push_robots` | **True** |
| `disturbance` | **True** (range=1.5, interval=6) |
| `max_iterations` | 7000 |
| `save_interval` | 300 |

**分步策略** (关键！):
1. 前 500 轮: disturbance_range = **[-0.5, 0.5]** (轻量)
2. 若 noise 不上升、reward > 0: 提升至 **[-1.0, 1.0]**
3. 再稳定 300 轮: 提升至 **[-1.5, 1.5]** (全量)

> R5 的教训: 直接全量开启导致崩溃。必须渐进。

**成功标准**:
- mean_reward > 0 持续
- smoothness > -0.6
- noise_std < 0.24

---

## 执行流程

```
从头训练 (阶段一)
    │
    ├─ step 0-3000: 纯净步态
    │    └─ 每 300 轮评估
    │
    ├─ 选择最优 checkpoint (smoothness 最高)
    │
    ├─ 恢复 → 阶段二 (push_robots)
    │    └─ step 3000-5000
    │    └─ 若退化, 回退纯净版
    │
    ├─ 选择稳定 checkpoint
    │
    └─ 恢复 → 阶段三 (push + disturb 渐进)
         └─ step 5000-7000
         └─ 轻→中→重 逐步加量
```

## 监控指标优先级

| 优先级 | 指标 | 告警阈值 |
|--------|------|---------|
| P0 | noise_std | > 0.25 持续上升 → 停止 |
| P1 | mean_reward | < -10 → 检查是否崩溃 |
| P2 | smoothness | < -1.0 → 步态质量差 |
| P3 | episode_length | < 1500 → 生存问题 |

## 预期最终产出

- 纯净 ONNX: 阶段一最优 (追求最平滑步态)
- 鲁棒 ONNX: 阶段三最优 (追求抗干扰)
- 两者可根据部署场景选择
