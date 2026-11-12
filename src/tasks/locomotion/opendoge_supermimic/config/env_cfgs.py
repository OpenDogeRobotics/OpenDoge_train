# Copyright (c) 2026-2027 zh
"""内容：从集中参数创建独立的 OpenDoge 粗糙地形、平地及播放模式配置。"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from src.config.opendoge.opendoge_reference_supermimic_config import OpendogeReferenceCfg

from ...go2_supermimic.config.env_cfgs import _make_env_cfg
from ...go2_supermimic.go_supermimic_env_cfg import validate_config


def resolve_opendoge_supermimic_cfg(
    cfg: OpendogeReferenceCfg | None = None, *, play: bool = False, terrain_mode: str | None = None,
) -> OpendogeReferenceCfg:
    """覆盖完整任务参数；返回副本以便adapter与纯Tensor任务使用同一语义。"""
    cfg = deepcopy(cfg) if cfg is not None else OpendogeReferenceCfg()
    if terrain_mode is not None:
        cfg.terrain.mode = terrain_mode
    if play:
        cfg.noise.add_noise = False
        cfg.terrain.curriculum = False
    validate_config(cfg)
    return cfg


def opendoge_rough_env_cfg(
    play: bool = False, *, go2_cfg: OpendogeReferenceCfg | None = None, terrain_map=None,
) -> ManagerBasedRlEnvCfg:
    """创建保留完整固定连通地图的粗糙地形环境配置。"""
    return _make_env_cfg(play, go2_cfg, "rough", terrain_map)


def opendoge_flat_env_cfg(
    play: bool = False, *, go2_cfg: OpendogeReferenceCfg | None = None, terrain_map=None,
) -> ManagerBasedRlEnvCfg:
    """创建平地配置，沿用同一套任务协议与名义控制参数。"""
    return _make_env_cfg(play, go2_cfg, "flat", terrain_map)
