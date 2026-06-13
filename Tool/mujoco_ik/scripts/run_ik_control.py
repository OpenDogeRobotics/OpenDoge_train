#!/usr/bin/env python3
"""IK gait control — keyboard (X11) or gamepad, IMU feedback, gait telemetry, optional recording."""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

MUJOCO_DIR = Path(__file__).resolve().parents[1]
if str(MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_DIR))

from opendoge_mujoco.action_gait import BodyCommand, GaitConfig, TrotCycloidGait
from opendoge_mujoco.foot_track_gait import FootTrackGait
from opendoge_mujoco.gait_telemetry import GaitTelemetry
from opendoge_mujoco.imu_feedback import IMUFeedbackConfig, IMUReader, IMUStabilizer
from opendoge_mujoco.input_devices import create_input, MuJoCoKeyCallback
from opendoge_mujoco.leg_ik import OpenDogeLegIK, load_leg_geometries_from_urdf
from opendoge_mujoco.motion_recorder import MotionRecorder
from opendoge_mujoco.position_controller import JointCommand
from opendoge_mujoco.sim_utils import init_sim, resolve_from_config


def _cmd_text(cmd: BodyCommand, turn: bool) -> str:
    m = "turn" if turn else "strafe"
    return f"mode={m} vx={cmd.vx:+.1f} vy={cmd.vy:+.1f} yaw={cmd.yaw:+.1f}"


def _base_xyz(model, data) -> np.ndarray:
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "float_base")
    if fid < 0:
        return np.zeros(3)
    a = model.jnt_qposadr[fid]
    return data.qpos[a:a + 3].copy()


def _feet_world_z(model, data, legs) -> dict[str, float]:
    out = {}
    for leg in legs:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot")
        out[leg] = float(data.xpos[bid][2]) if bid >= 0 else 0.0
    return out


def _resolve_input(ks_or_gp, is_callback: bool, model, data):
    """Unified snapshot — handles both keyboard (callback/X11) and gamepad.

    For MuJoCoKeyCallback fallback, we need the viewer to call on_key() —
    the snapshot is polled independently. Gamepad reads directly.
    """
    return ks_or_gp.snapshot()


# ── CLI ────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenDoge IK gait control")
    p.add_argument("--config", type=Path, default=MUJOCO_DIR / "configs" / "position_control.json")
    p.add_argument("--input", type=str, default="x11",
                   help="Input source: x11, callback, gamepad, gamepad:/dev/input/event<N>")
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--urdf", type=Path, default=None)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--print-rate", type=float, default=2.0)
    p.add_argument("--cmd-vx", type=float, default=0.0)
    p.add_argument("--cmd-vy", type=float, default=0.0)
    p.add_argument("--cmd-yaw", type=float, default=0.0)
    p.add_argument("--c-style", action="store_true")
    p.add_argument("--record", type=Path, default=None, help="Save motion to .npz")
    return p


# ── main ───────────────────────────────────────────────────────
def main() -> int:
    # Line-buffered stdout so telemetry appears live even when piped (conda run).
    sys.stdout.reconfigure(line_buffering=True)

    args = build_parser().parse_args()
    config_path = args.config.resolve()
    model, data, config, joint_names, controller, default_q = init_sim(
        config_path, model_path_override=args.model
    )
    ik_cfg = config["action_ik"]
    imu_cfg = config["imu_feedback"]

    urdf_path = args.urdf.resolve() if args.urdf else resolve_from_config(config_path, ik_cfg["urdf_path"])
    imu_reader = IMUReader(model)
    imu_stab = IMUStabilizer(IMUFeedbackConfig(
        enabled=bool(imu_cfg["enabled"]),
        heading_kp=float(imu_cfg["heading_kp"]), yaw_rate_kd=float(imu_cfg["yaw_rate_kd"]),
        max_yaw_correction=float(imu_cfg["max_yaw_correction"]),
        roll_kp=float(imu_cfg["roll_kp"]), roll_kd=float(imu_cfg["roll_kd"]),
        pitch_kp=float(imu_cfg["pitch_kp"]), pitch_kd=float(imu_cfg["pitch_kd"]),
        max_foot_z_correction=float(imu_cfg["max_foot_z_correction"]),
    ))

    default_by_joint = dict(zip(joint_names, default_q.tolist()))
    legs = sorted({n.split("_", 1)[0] for n in joint_names})
    lg = load_leg_geometries_from_urdf(urdf_path, legs)
    ik = OpenDogeLegIK(lg, joint_names)
    nominal_feet = ik.nominal_feet(default_by_joint)

    cycle_time = float(ik_cfg["cycle_time"])
    duty_factor = float(ik_cfg["duty_factor"])

    if args.c_style:
        planner = FootTrackGait(nominal_feet, None)
        planner_label = f"C-style (h={planner.p.leg_high:.3f})"
        # FootTrackGait uses internal cycle (step_rate*2), expose cycle for telemetry
        gait_tel = GaitTelemetry(legs, cycle_time=planner._cycle_time, duty_factor=0.5)
    else:
        planner = TrotCycloidGait(nominal_feet, GaitConfig(
            cycle_time=cycle_time, duty_factor=duty_factor,
            step_height=float(ik_cfg["step_height"]), step_x=float(ik_cfg["step_x"]),
            step_y=float(ik_cfg["step_y"]), step_yaw=float(ik_cfg["step_yaw"]),
            rear_stance_height_offset=float(ik_cfg["rear_stance_height_offset"]),
        ))
        planner_label = "TrotCycloidGait"
        gait_tel = GaitTelemetry(legs, cycle_time=cycle_time, duty_factor=duty_factor)

    # Nominal foot Z for clearance calc
    nominal_z = {leg: float(nominal_feet[leg][2]) for leg in legs}
    gait_tel.nominal_feet_z = nominal_z

    imu_stab.reset(imu_reader.read(data))
    last_q = default_q.copy()

    recorder: MotionRecorder | None = None
    if args.record:
        recorder = MotionRecorder(joint_names, float(config["control_dt"]), args.record, leg_names=legs)
        print(f"Recording → {args.record.resolve()}")

    duration = float(config["duration"] if args.duration is None else args.duration)
    render = bool(config["render"]) and not args.no_render
    period = 1.0 / args.print_rate if args.print_rate > 0 else math.inf
    next_print = 0.0
    max_joint_speed = float(ik_cfg["max_joint_speed_rad_s"])
    blend_t = float(ik_cfg["startup_blend_time"])
    gait_t0, gait_on = 0.0, False

    # Input source
    input_src, input_label = create_input(args.input, float(ik_cfg.get("key_hold_timeout", 0.18)))
    is_callback = isinstance(input_src, MuJoCoKeyCallback)

    hcmd = BodyCommand(
        vx=max(-1.0, min(1.0, args.cmd_vx)),
        vy=max(-1.0, min(1.0, args.cmd_vy)),
        yaw=max(-1.0, min(1.0, args.cmd_yaw)),
    )

    def step(cmd: BodyCommand) -> np.ndarray:
        nonlocal gait_on, gait_t0, last_q, next_print
        imu_s = imu_reader.read(data)
        active = max(abs(cmd.vx), abs(cmd.vy), abs(cmd.yaw)) > 1e-6
        ccmd = imu_stab.command(cmd, imu_s) if active else BodyCommand()
        if not active:
            imu_stab.reset(imu_s)
        if active and not gait_on:
            gait_t0 = data.time
            imu_stab.reset(imu_s)
        gait_on = active
        gt = data.time - gait_t0 if active else 0.0

        seed = dict(zip(joint_names, last_q.tolist()))
        ft = planner.targets(gt, ccmd)
        b = 1.0 if blend_t <= 0 else min(1.0, gt / blend_t)
        fp = {leg: nominal_feet[leg] + b * (ft[leg].position - nominal_feet[leg]) for leg in ft}
        fp = imu_stab.feet(fp, imu_s)
        q_ik = controller.clamp_to_joint_limits(ik.inverse_feet(fp, seed))
        max_step = max_joint_speed * model.opt.timestep
        qs = np.clip(q_ik - last_q, -max_step, max_step)
        q_des = last_q + qs
        dq_des = qs / model.opt.timestep
        last_q = q_des

        tau = controller.apply(data, JointCommand(q_des=q_des, dq_des=dq_des))
        mujoco.mj_step(model, data)

        if recorder is not None:
            recorder.record_frame(data=data, model=model, q_des=q_des, tau=tau,
                                  body_command=np.array([cmd.vx, cmd.vy, cmd.yaw]))

        if data.time >= next_print:
            q_err = q_des - controller.joint_positions(data)
            bp = _base_xyz(model, data)
            fwz = _feet_world_z(model, data, legs)
            gait_info = gait_tel.summary_line(gt, active, fwz, base_z=float(bp[2]))
            print(f"t={data.time:7.3f}s  {_cmd_text(cmd, False)}  "
                  f"yaw_fb={ccmd.yaw - cmd.yaw:+.2f}  "
                  f"max_q_err={np.max(np.abs(q_err)):.4f}rad  "
                  f"max_dq={np.max(np.abs(dq_des)):.1f}rad/s  "
                  f"max_tau={np.max(np.abs(tau)):.3f}Nm  "
                  f"base=({bp[0]:+.3f},{bp[1]:+.3f})  "
                  f"{gait_info}  "
                  f"pitch={imu_s.pitch:+.3f}  yaw={imu_s.yaw:+.3f}")
            next_print += period
        return tau

    if render:
        from mujoco import viewer as v

        print(f"Input:   {input_label}")
        print(f"Planner: {planner_label}")
        if is_callback:
            print("Keys: ↑↓ fb  ←→ strafe  Ctrl+←→ turn  Space stop  R reset  Esc quit")
        else:
            print("Use controller / keyboard as configured")

        viewer = None
        try:
            key_cb = input_src.on_key if is_callback else None
            viewer = v.launch_passive(model, data, key_callback=key_cb)
            while viewer.is_running() and (duration <= 0 or data.time < duration):
                t0 = time.monotonic()
                cmd, turn, ex, rst = input_src.snapshot()
                if ex:
                    break
                if rst:
                    from opendoge_mujoco.sim_utils import _init_pose as _repose
                    _repose(model, data, controller, default_q, config["base_pose"])
                    imu_stab.reset(imu_reader.read(data))
                    last_q[:] = default_q
                step(cmd)
                try:
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                                      mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      f"OpenDoge IK [{planner_label}]  {input_label}",
                                      _cmd_text(cmd, turn)))
                except AttributeError:
                    pass
                viewer.sync()
                dt = model.opt.timestep - (time.monotonic() - t0)
                if dt > 0:
                    time.sleep(dt)
        finally:
            # MuJoCo 3.2.x cleanup may segfault — skip all explicit cleanup,
            # OS reclaims resources on process exit.
            pass
    else:
        while duration <= 0 or data.time < duration:
            step(hcmd)

    if recorder is not None:
        p = recorder.save()
        print(f"Saved {recorder.num_frames} frames → {p}")
    # MuJoCo 3.2.x C-library cleanup may segfault on exit.
    # Flush buffers then skip Python GC + C atexit destructors.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
