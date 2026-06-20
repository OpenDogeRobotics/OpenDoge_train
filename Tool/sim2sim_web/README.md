# OpenDoge Sim2Sim Web Dashboard

纯 MuJoCo + ONNX Runtime 的 Web 端 sim2sim 验证平台。一键切换模型、实时渲染、训练指标对比。

## 启动

```bash
# 进入项目目录
cd /home/lain/OpenDoge/OpenDoge_train

# 确保 PYTHONPATH 正确
export PYTHONPATH=$PWD

# 杀掉旧进程（如有），启动服务
fuser -k 8000/tcp 2>/dev/null
sleep 0.5
conda run -n himloco python Tool/sim2sim_web/server.py
```

浏览器打开 **http://localhost:8000**

> 关闭：`Ctrl+C` 停止服务

### 切换端口

```bash
conda run -n himloco python Tool/sim2sim_web/server.py --port 9000
```

### 允许局域网访问

```bash
conda run -n himloco python Tool/sim2sim_web/server.py --public
# 然后 http://你的IP:8000
```

## 界面操作

### 模型切换
| 操作 | 方式 |
|------|------|
| 切换模型 | 下拉菜单选择（自动加载） |
| 上一个/下一个 | 点击 ◀ ▶ 按钮，或键盘 `[` `]` |

### 速度指令
| 操作 | 滑块 | 键盘 |
|------|------|------|
| 前进/后退 | vx | `↑` `↓` |
| 左转/右转 | ω (yaw) | `←` `→` |

### 视角控制（同步 MuJoCo 操作逻辑）
| 操作 | 鼠标 |
|------|------|
| 旋转 | 左键拖拽 |
| 平移 | 右键拖拽 |
| 缩放 | 滚轮 |
| 重置视角 | 双击 |

### 仿真控制
| 操作 | 方式 |
|------|------|
| 启停 | 点击 ▶ 按钮，或键盘 `Space` |
| 重置 | 点击 ↺ 按钮，或键盘 `R` |

## 面板说明

```
┌──────────────────────────────────────────────────────────┐
│  左侧：控制面板     │  中间：3D 画面    │  右侧：指标    │
│                    │                   │               │
│  ONNX 模型选择     │  MuJoCo 实时渲染  │  关节角度柱图  │
│  速度滑块 (vx/vy/ω)│  鼠标拖拽旋转视角 │  机身高度/速度  │
│  启停 / 重置按钮   │                   │  脚底接触状态  │
│                    │                   │  历史曲线      │
├──────────────────────────────────────────────────────────┘
│  底部：训练指标对比 (多 run 横向对比表格)
└──────────────────────────────────────────────────────────┘
```

## ONNX 模型

自动扫描 `onnx/` 目录，按修改时间排序。文件名中的数字会被识别为训练步数。

## 文件结构

```
Tool/sim2sim_web/
├── engine.py               # 核心引擎 (MuJoCo + ONNX, 离屏渲染, 相机控制)
├── server.py               # FastAPI 服务 (REST API + WebSocket)
├── tensorboard_reader.py   # TensorBoard 日志解析
├── start.sh                # 启动脚本
├── static/
│   ├── index.html          # 仪表盘页面
│   └── js/
│       └── app.js          # 前端逻辑 (Canvas渲染, Chart.js, 鼠标交互)
└── README.md
```

## 依赖

- `himloco` conda 环境 (Python 3.8)
- MuJoCo 3.x, ONNX Runtime, PyTorch
- FastAPI, uvicorn, websockets, Pillow
- TensorBoard (仅读取事件文件)

## API 速查

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/models` | 模型列表 |
| POST | `/api/models/load` | 切换模型 `{path}` |
| POST | `/api/models/next` | 下一个模型 |
| POST | `/api/models/prev` | 上一个模型 |
| GET | `/api/camera` | 相机状态 |
| POST | `/api/camera` | 设置相机 `{azimuth, elevation, distance}` |
| POST | `/api/sim/cmd` | 速度指令 `{vx, vy, vyaw}` |
| POST | `/api/sim/reset` | 重置仿真 |
| GET | `/api/sim/state` | 完整状态 |
| GET | `/api/sim/render?w=640&h=360&q=60` | 渲染帧 JPEG |
| GET | `/api/monitor/runs` | 训练 run 列表 |
| GET | `/api/monitor/compare` | 多 run 指标对比 |
| WS | `/ws/stream` | 实时状态 (text) + 渲染帧 (binary) |

## 常见问题

**Q: `ModuleNotFoundError: No module named 'legged_gym'`**
```bash
export PYTHONPATH=/home/lain/OpenDoge/OpenDoge_train
```

**Q: 端口被占用**
```bash
fuser -k 8000/tcp
```

**Q: 画面不显示**
刷新页面 (`Ctrl+Shift+R`)，检查浏览器 F12 Console 是否有报错。

**Q: 模型不切换 / 仿真不动**
检查 WebSocket 连接状态 —— 页面顶部状态灯应为绿色 `● connected`。
