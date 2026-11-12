# Copyright (c) 2026-2027 zh
"""
内容：
    OpenDoge 播放检查点装配；复用 Go2 实现，注入本任务身份。
"""

from ...go2_supermimic.rl.checkpoints import (
    PlaybackSource, build_playback_policy as _build_playback_policy,
    load_playback_source as _load_playback_source,
)
from ..config.loading import configuration_for_settings
from ..config.train_cfg import TASK_MODES


def load_playback_source(task_id: str, checkpoint_file):
    """按OpenDoge任务身份读取检查点与任务快照。"""
    return _load_playback_source(
        task_id, checkpoint_file, task_modes=TASK_MODES, configuration_factory=configuration_for_settings)


def build_playback_policy(source, adapter):
    """用原训练语义验证权重并装配策略；与Go2实现一致。"""
    return _build_playback_policy(source, adapter)


__all__ = ["PlaybackSource", "load_playback_source", "build_playback_policy",
           "configuration_for_settings"]
