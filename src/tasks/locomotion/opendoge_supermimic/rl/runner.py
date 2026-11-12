# Copyright (c) 2026-2027 zh
"""内容：装配 OpenDoge SuperMimic 训练器；复用 Go2 runner，仅替换默认配置。"""

from ...go2_supermimic.rl.runner import make_go2_supermimic_runner
from src.config.opendoge.opendoge_reference_supermimic_config import OpendogeReferenceCfg


def make_opendoge_supermimic_runner(cfg=None, **kwargs):
    """创建OpenDoge训练器；未提供配置时使用OpenDoge参考默认。"""
    return make_go2_supermimic_runner(cfg if cfg is not None else OpendogeReferenceCfg(), **kwargs)
