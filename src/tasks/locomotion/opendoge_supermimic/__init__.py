# Copyright (c) 2026-2027 zh
"""
内容：
    提供独立 OpenDoge SuperMimic 任务工厂，复用 go2_supermimic 的任务机制。
"""


def make_opendoge_supermimic_env(*args, **kwargs):
    from ..go2_supermimic.rl import make_go2_supermimic_env as factory
    return factory(*args, **kwargs)


def make_opendoge_supermimic_runner(*args, **kwargs):
    from ..go2_supermimic.rl import make_go2_supermimic_runner as factory
    return factory(*args, **kwargs)


__all__ = ["make_opendoge_supermimic_env", "make_opendoge_supermimic_runner"]
