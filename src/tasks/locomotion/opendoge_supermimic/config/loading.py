# Copyright (c) 2026-2027 zh
"""
内容：
    OpenDoge 任务配置读取；复用 Go2 的覆盖与校验逻辑，仅替换配置工厂。
"""

from ...go2_supermimic.config.loading import apply_settings, read_settings
from src.config.opendoge.opendoge_reference_supermimic_config import OpendogeReferenceCfg


def configuration_for_settings(settings: dict | None = None):
    """只接受混合专家历史策略，未声明算法时使用同一默认值。"""
    settings = settings or {}
    algorithm = settings.get("algorithm", "moe_history")
    if algorithm != "moe_history":
        raise ValueError(f"Unsupported SuperMimic algorithm: {algorithm}")
    cfg = OpendogeReferenceCfg()
    sections = set(cfg.to_dict()) - {"init_member_classes", "to_dict"}
    if sections.issubset(settings):
        # 旧完整快照使用Adam；局部覆盖不改变OpenDoge新训练的优化器默认。
        for name in ("actor_optimizer", "wm_optimizer"):
            setattr(cfg.algo, name, settings["algo"].get(name, "adam"))
    return cfg


__all__ = ["apply_settings", "read_settings", "configuration_for_settings"]
