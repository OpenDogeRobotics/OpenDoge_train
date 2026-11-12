# Copyright (c) 2026-2027 zh
"""
内容：
    装配 OpenDoge SuperMimic 训练；复用 Go2 训练流程，注入本任务身份。
"""

from ...go2_supermimic.rl.training import launch_training as _launch_training
from ...go2_supermimic.rl.runner import algorithm_config
from .checkpoints import configuration_for_settings
from ..config.train_cfg import SuperMimicTrainConfig, TASK_MODES


def launch_training(task_id: str, args: SuperMimicTrainConfig):
    """按OpenDoge任务身份启动训练；流程与Go2实现一致。"""
    return _launch_training(
        task_id, args, task_modes=TASK_MODES, configuration_factory=configuration_for_settings,
        runner_factory=None,
    )
