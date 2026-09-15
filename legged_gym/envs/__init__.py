from .base.legged_robot import LeggedRobot
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO
from .opendoge.opendoge_config import OpendogeCfg, OpendogeCfgPPO
from .opendoge_v1_1.opendoge_v1_1_config import OpenDogeV11Cfg, OpenDogeV11CfgPPO
from .opendoge_v1_1.opendoge_v1_1_terrain_config import (
    OpenDogeV11TerrainCfg,
    OpenDogeV11TerrainCfgPPO,
)
from .opendoge.opendoge_flat_config import OpendogeFlatCfg, OpendogeFlatCfgPPO
from .opendoge.opendoge_rough_config import OpendogeRoughCfg, OpendogeRoughCfgPPO
from .opendoge.opendoge_getup_config import OpendogeGetUpCfg, OpendogeGetUpCfgPPO
from .opendoge.opendoge_amp_config import (
    OpendogeAMPCfg,
    OpendogeAMPCfgPPO,
    OpendogeHIMAMPCfg,
    OpendogeHIMAMPCfgPPO,
)
from .opendoge.opendoge_getup import OpendogeGetUp
from .opendoge.opendoge_amp import OpendogeAMP

from legged_gym.utils.task_registry import task_registry

task_registry.register("a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO())
task_registry.register("go1", LeggedRobot, Go1RoughCfg(), Go1RoughCfgPPO())
task_registry.register("opendoge", LeggedRobot, OpendogeCfg(), OpendogeCfgPPO())
task_registry.register("opendoge_v1_1", LeggedRobot, OpenDogeV11Cfg(), OpenDogeV11CfgPPO())
task_registry.register("opendoge_v1_1_terrain", LeggedRobot, OpenDogeV11TerrainCfg(), OpenDogeV11TerrainCfgPPO())
task_registry.register("opendoge_flat", LeggedRobot, OpendogeFlatCfg(), OpendogeFlatCfgPPO())
task_registry.register("opendoge_rough", LeggedRobot, OpendogeRoughCfg(), OpendogeRoughCfgPPO())
task_registry.register("opendoge_getup", OpendogeGetUp, OpendogeGetUpCfg(), OpendogeGetUpCfgPPO())
task_registry.register("opendoge_amp", OpendogeAMP, OpendogeAMPCfg(), OpendogeAMPCfgPPO())
task_registry.register("opendoge_him_amp", OpendogeAMP, OpendogeHIMAMPCfg(), OpendogeHIMAMPCfgPPO())
