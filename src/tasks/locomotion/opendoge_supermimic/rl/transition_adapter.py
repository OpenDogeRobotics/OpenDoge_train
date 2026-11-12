# Copyright (c) 2026-2027 zh
"""
内容：
    OpenDoge 真实环境适配器；任务机制复用 Go2 实现，仅替换默认配置。
    环境工厂按 cfg.asset 分发机器人资产，适配器本体无需修改。
"""

from ...go2_supermimic.rl.transition_adapter import Go2TransitionAdapter, make_go2_supermimic_env
from src.config.opendoge.opendoge_reference_supermimic_config import OpendogeReferenceCfg


class OpendogeTransitionAdapter(Go2TransitionAdapter):
    """默认使用 OpenDoge 配置的转移适配器；行为与Go2实现一致。"""

    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg if cfg is not None else OpendogeReferenceCfg(), **kwargs)


def make_opendoge_supermimic_env(cfg=None, **kwargs):
    """创建可直接交给SuperMimicRunner的真实OpenDoge采集环境。"""
    return make_go2_supermimic_env(cfg if cfg is not None else OpendogeReferenceCfg(), **kwargs)
