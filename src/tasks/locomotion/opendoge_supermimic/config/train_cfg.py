# Copyright (c) 2026-2027 zh
"""
内容：
    定义 OpenDoge SuperMimic 训练任务标识；启动参数复用 Go2 声明。
"""

from ...go2_supermimic.config.train_cfg import SuperMimicTrainConfig

TASK_MODES = {
    "Opendoge-SuperMimic-Flat": "flat",
    "Opendoge-SuperMimic-Rough": "rough",
}

__all__ = ["TASK_MODES", "SuperMimicTrainConfig"]
