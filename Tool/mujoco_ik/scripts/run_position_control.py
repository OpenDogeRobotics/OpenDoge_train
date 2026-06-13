#!/usr/bin/env python3
"""PD position-control demo — stand or sine motion, render or headless, optional recording."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

MUJOCO_DIR = Path(__file__).resolve().parents[1]
if str(MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_DIR))

from opendoge_mujoco.motion_recorder import MotionRecorder
from opendoge_mujoco.position_controller import JointCommand, PDPositionController
from opendoge_mujoco.sim_utils import init_sim, joint_group


def sine_command(
    t: float,
    joint_names: list[str],
    default_q: np.ndarray,
    sine_config: dict,
) -> JointCommand:
    freq = float(sine_config["frequency_hz"])
    omega = 2.0 * math.pi * freq
    amp = sine_config["amplitude_rad"]
    q_des = default_q.copy()
    dq_des = np.zeros_like(default_q)
    for i, name in enumerate(joint_names):
        a = float(amp[joint_group(name)])
        phase = 0.0 if name.split("_", 1)[0] in {"FL", "RR"} else math.pi
        q_des[i] += a * math.sin(omega * t + phase)
        dq_des[i] = a * omega * math.cos(omega * t + phase)
    return JointCommand(q_des=q_des, dq_des=dq_des)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenDoge PD position control")
    p.add_argument("--config", type=Path, default=MUJOCO_DIR / "configs" / "position_control.json")
    p.add_argument("--mode", choices=("stand", "sine"), default="stand")
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--print-rate", type=float, default=2.0)
    p.add_argument("--record", type=Path, default=None, help="Save motion to .npz")
    return p


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    model, data, config, joint_names, controller, default_q = init_sim(
        config_path, model_path_override=args.model
    )

    recorder: MotionRecorder | None = None
    if args.record:
        recorder = MotionRecorder(joint_names, float(config["control_dt"]), args.record)
        print(f"Recording → {args.record.resolve()}")

    duration = float(config["duration"] if args.duration is None else args.duration)
    render = bool(config["render"]) and not args.no_render
    period = 1.0 / args.print_rate if args.print_rate > 0 else math.inf
    next_print = 0.0

    def step() -> None:
        nonlocal next_print
        cmd = (
            sine_command(data.time, joint_names, default_q, config["sine_motion"])
            if args.mode == "sine"
            else JointCommand(q_des=default_q, dq_des=np.zeros_like(default_q))
        )
        tau = controller.apply(data, cmd)
        mujoco.mj_step(model, data)

        if recorder is not None:
            recorder.record_frame(data=data, model=model, q_des=cmd.q_des, tau=tau)

        if data.time >= next_print:
            q_err = cmd.q_des - controller.joint_positions(data)
            print(f"t={data.time:7.3f}s  max_q_err={np.max(np.abs(q_err)):.4f}rad  max_tau={np.max(np.abs(tau)):.3f}Nm")
            next_print += period

    if render:
        from mujoco import viewer as v
        with v.launch_passive(model, data) as viewer:
            while viewer.is_running() and (duration <= 0 or data.time < duration):
                t0 = time.monotonic()
                step()
                viewer.sync()
                dt = model.opt.timestep - (time.monotonic() - t0)
                if dt > 0:
                    time.sleep(dt)
    else:
        while duration <= 0 or data.time < duration:
            step()

    if recorder is not None:
        p = recorder.save()
        print(f"Saved {recorder.num_frames} frames → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
