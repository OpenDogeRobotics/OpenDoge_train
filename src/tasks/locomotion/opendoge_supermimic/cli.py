# Copyright (c) 2026-2027 zh
"""
内容：
    把 OpenDoge SuperMimic 训练参数解析从统一入口收回本任务。

用法：
    python scripts/train.py Opendoge-SuperMimic-Rough --device cuda:0 --num-envs 4096
"""

from __future__ import annotations

from dataclasses import replace
import sys

import mjlab
import tyro

from .config.train_cfg import SuperMimicTrainConfig
from .rl import training


def train_main(task_id: str, arguments: list[str]) -> None:
    """按本任务的扁平参数解析，随后校验并启动单设备训练。"""
    args = tyro.cli(
        SuperMimicTrainConfig, args=arguments, prog=f"{sys.argv[0]} {task_id}", config=mjlab.TYRO_FLAGS,
    )
    if args.experiment_name == "go2_supermimic":
        # 复用Go2参数声明时未显式指定实验名，则落到本任务的默认目录。
        args = replace(args, experiment_name="opendoge_supermimic")
    training.launch_training(task_id, args)
