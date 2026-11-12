# Copyright (c) 2026-2027 zh
"""内容：导出 OpenDoge 混合专家历史策略训练器和真实环境适配器。"""

from .runner import make_opendoge_supermimic_runner
from .transition_adapter import make_opendoge_supermimic_env

__all__ = ["make_opendoge_supermimic_runner", "make_opendoge_supermimic_env"]
