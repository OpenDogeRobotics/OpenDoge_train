# Copyright (c) 2026-2027 zh
"""
内容：
    OpenDoge 盲策略播放；复用 Go2 播放实现，注入本任务身份。
"""

from ...go2_supermimic.rl.playback import make_playback_env as _make_playback_env
from .checkpoints import configuration_for_settings
from ..config.train_cfg import TASK_MODES


def make_playback_env(task_id, **kwargs):
    """按OpenDoge任务身份装配播放环境与策略。"""
    return _make_playback_env(
        task_id, task_modes=TASK_MODES, configuration_factory=configuration_for_settings, **kwargs)
