# OpenDoge SuperMimic Rough 策略

此目录保存 OpenDoge SuperMimic 粗糙地形策略第 20,000 轮的部署模型和训练检查点。

- `opendoge_supermimic_rough_20000.onnx`：部署/推理用 ONNX 策略，输入为 5 帧历史观测（235 维），输出为 12 维关节动作。
- `opendoge_supermimic_rough_20000.pt`：训练检查点，可用于续训或项目内回放。
- `task_config.json`：检查点对应的完整任务配置与模型元数据；项目内 SuperMimic 回放需要它与 `.pt` 放在同一目录。

## 回放检查点

从仓库根目录运行：

```bash
uv run --no-sync python -B scripts/play.py Opendoge-SuperMimic-Rough \
  --checkpoint-file policy/opendoge_supermimic_rough_20000.pt \
  --num-envs 1 --device cuda:0 --viewer viser
```

训练来源：8192 个并行环境、目标 20,000 轮，实验目录为 `logs/supermimic/opendoge_supermimic/2026-11-13_12-50-20_159537_rough_8192_20000/`。ONNX 元数据记录的训练轮次为 20,000。
