#!/usr/bin/env python3
"""Keyboard IK gait control — trot gait with X11 polling, IMU feedback, optional recording."""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import math
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

MUJOCO_DIR = Path(__file__).resolve().parents[1]
if str(MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_DIR))

from opendoge_mujoco.action_gait import BodyCommand, GaitConfig, TrotCycloidGait
from opendoge_mujoco.foot_track_gait import FootTrackGait
from opendoge_mujoco.imu_feedback import IMUFeedbackConfig, IMUReader, IMUStabilizer
from opendoge_mujoco.leg_ik import OpenDogeLegIK, load_leg_geometries_from_urdf
from opendoge_mujoco.motion_recorder import MotionRecorder
from opendoge_mujoco.position_controller import JointCommand, PDPositionController
from opendoge_mujoco.sim_utils import init_sim, joint_group, resolve_from_config

# ── key constants ──────────────────────────────────────────────
KEY_ESCAPE, KEY_SPACE = 256, 32
KEY_LEFT, KEY_RIGHT, KEY_DOWN, KEY_UP = 263, 262, 264, 265
KEY_LEFT_CTRL, KEY_RIGHT_CTRL = 341, 345

XK_ESCAPE, XK_SPACE = 0xFF1B, 0x0020
XK_R, XK_R_LOWER = 0x0052, 0x0072
XK_LEFT, XK_UP, XK_RIGHT, XK_DOWN = 0xFF51, 0xFF52, 0xFF53, 0xFF54
XK_CTRL_L, XK_CTRL_R = 0xFFE3, 0xFFE4


# ── keyboard helpers ───────────────────────────────────────────
def _cmd_from_keys(up: bool, down: bool, left: bool, right: bool, turn: bool) -> BodyCommand:
    c = BodyCommand()
    c.vx = float(up) - float(down)
    h = float(left) - float(right)
    if turn:
        c.yaw = h
    else:
        c.vy = h
    n = math.hypot(c.vx, c.vy)
    if n > 1.0:
        c.vx /= n; c.vy /= n
    return c


class KeyboardCommand:
    def __init__(self, timeout: float):
        self._lock = threading.Lock()
        self.timeout = timeout
        self._times: dict[int, float] = {}
        self.exit = False
        self.reset = False

    def on_key(self, key: int) -> None:
        now = time.monotonic()
        with self._lock:
            if key == KEY_ESCAPE:
                self.exit = True
            elif key == KEY_SPACE:
                self._times.clear()
            elif key == ord("R"):
                self._times.clear(); self.reset = True
            elif key in {KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_LEFT_CTRL, KEY_RIGHT_CTRL}:
                self._times[key] = now

    def snapshot(self) -> tuple[BodyCommand, bool, bool, bool]:
        now = time.monotonic()
        with self._lock:
            for k in list(self._times):
                if now - self._times[k] > self.timeout:
                    del self._times[k]
            turn = KEY_LEFT_CTRL in self._times or KEY_RIGHT_CTRL in self._times
            cmd = _cmd_from_keys(
                KEY_UP in self._times, KEY_DOWN in self._times,
                KEY_LEFT in self._times, KEY_RIGHT in self._times, turn,
            )
            ex, rst = self.exit, self.reset
            self.reset = False
        return cmd, turn, ex, rst


class X11KeyboardPoller:
    def __init__(self):
        lib = ctypes.util.find_library("X11")
        if not lib:
            raise RuntimeError("libX11 not found")
        self.x11 = ctypes.cdll.LoadLibrary(lib)
        self.x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.x11.XOpenDisplay.restype = ctypes.c_void_p
        self.x11.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.x11.XQueryKeymap.restype = ctypes.c_int
        self.x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.x11.XKeysymToKeycode.restype = ctypes.c_uint
        self.dpy = self.x11.XOpenDisplay(None)
        if not self.dpy:
            raise RuntimeError("cannot open X11 display")
        self._kc = {
            "esc": self._kc_(XK_ESCAPE), "space": self._kc_(XK_SPACE),
            "r": self._kc_(XK_R), "rl": self._kc_(XK_R_LOWER),
            "up": self._kc_(XK_UP), "down": self._kc_(XK_DOWN),
            "left": self._kc_(XK_LEFT), "right": self._kc_(XK_RIGHT),
            "cl": self._kc_(XK_CTRL_L), "cr": self._kc_(XK_CTRL_R),
        }
        self._last_r = False

    def close(self):
        if self.dpy:
            self.x11.XCloseDisplay(self.dpy); self.dpy = None

    def snapshot(self) -> tuple[BodyCommand, bool, bool, bool]:
        km = ctypes.create_string_buffer(32)
        if not self.x11.XQueryKeymap(self.dpy, km):
            return BodyCommand(), False, False, False
        p = {n: bool(km.raw[k >> 3] & (1 << (k & 7))) for n, k in self._kc.items() if k > 0}
        turn = p.get("cl", False) or p.get("cr", False)
        cmd = _cmd_from_keys(p.get("up", False), p.get("down", False),
                             p.get("left", False), p.get("right", False), turn)
        rst = (p.get("r", False) or p.get("rl", False)) and not self._last_r
        self._last_r = p.get("r", False) or p.get("rl", False)
        if p.get("space", False):
            cmd = BodyCommand()
        return cmd, turn, p.get("esc", False), rst

    def _kc_(self, ks): return int(self.x11.XKeysymToKeycode(self.dpy, ks))


# ── telemetry ──────────────────────────────────────────────────
def _cmd_text(cmd: BodyCommand, turn: bool) -> str:
    m = "turn" if turn else "strafe"
    return f"mode={m} vx={cmd.vx:+.1f} vy={cmd.vy:+.1f} yaw={cmd.yaw:+.1f}"

def _base_xy(model, data):
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "float_base")
    return data.qpos[model.jnt_qposadr[fid]:model.jnt_qposadr[fid] + 3].copy() if fid >= 0 else np.zeros(3)


# ── main ───────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenDoge keyboard IK gait control")
    p.add_argument("--config", type=Path, default=MUJOCO_DIR / "configs" / "position_control.json")
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


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    model, data, config, joint_names, controller, default_q = init_sim(
        config_path, model_path_override=args.model
    )
    ik_cfg = config["action_ik"]
    imu_cfg = config["imu_feedback"]

    # URDF for IK
    urdf_path = args.urdf.resolve() if args.urdf else resolve_from_config(config_path, ik_cfg["urdf_path"])

    # IMU
    imu_reader = IMUReader(model)
    imu_stab = IMUStabilizer(IMUFeedbackConfig(
        enabled=bool(imu_cfg["enabled"]),
        heading_kp=float(imu_cfg["heading_kp"]), yaw_rate_kd=float(imu_cfg["yaw_rate_kd"]),
        max_yaw_correction=float(imu_cfg["max_yaw_correction"]),
        roll_kp=float(imu_cfg["roll_kp"]), roll_kd=float(imu_cfg["roll_kd"]),
        pitch_kp=float(imu_cfg["pitch_kp"]), pitch_kd=float(imu_cfg["pitch_kd"]),
        max_foot_z_correction=float(imu_cfg["max_foot_z_correction"]),
    ))

    # IK
    default_by_joint = dict(zip(joint_names, default_q.tolist()))
    legs = sorted({n.split("_", 1)[0] for n in joint_names})
    lg = load_leg_geometries_from_urdf(urdf_path, legs)
    ik = OpenDogeLegIK(lg, joint_names)
    nominal_feet = ik.nominal_feet(default_by_joint)

    if args.c_style:
        planner = FootTrackGait(nominal_feet, None)
        planner_label = f"C-style (h={planner.p.leg_high:.3f})"
    else:
        planner = TrotCycloidGait(nominal_feet, GaitConfig(
            cycle_time=float(ik_cfg["cycle_time"]), duty_factor=float(ik_cfg["duty_factor"]),
            step_height=float(ik_cfg["step_height"]), step_x=float(ik_cfg["step_x"]),
            step_y=float(ik_cfg["step_y"]), step_yaw=float(ik_cfg["step_yaw"]),
            rear_stance_height_offset=float(ik_cfg["rear_stance_height_offset"]),
        ))
        planner_label = "TrotCycloidGait"

    imu_stab.reset(imu_reader.read(data))
    last_q = default_q.copy()

    # recorder
    recorder: MotionRecorder | None = None
    if args.record:
        recorder = MotionRecorder(joint_names, float(config["control_dt"]), args.record, leg_names=legs)
        print(f"Recording → {args.record.resolve()}")

    duration = float(config["duration"] if args.duration is None else args.duration)
    render = bool(config["render"]) and not args.no_render
    period = 1.0 / args.print_rate if args.print_rate > 0 else math.inf
    next_print = 0.0
    kbd = KeyboardCommand(float(ik_cfg["key_hold_timeout"]))
    max_joint_speed = float(ik_cfg["max_joint_speed_rad_s"])
    blend_t = float(ik_cfg["startup_blend_time"])
    gait_t0, gait_on = 0.0, False
    hcmd = BodyCommand(
        vx=max(-1.0, min(1.0, args.cmd_vx)),
        vy=max(-1.0, min(1.0, args.cmd_vy)),
        yaw=max(-1.0, min(1.0, args.cmd_yaw)),
    )

    def step(cmd: BodyCommand, turn: bool) -> np.ndarray:
        nonlocal gait_on, gait_t0, last_q, next_print
        imu_s = imu_reader.read(data)
        active = max(abs(cmd.vx), abs(cmd.vy), abs(cmd.yaw)) > 1e-6
        ccmd = imu_stab.command(cmd, imu_s) if active else BodyCommand()
        if not active:
            imu_stab.reset(imu_s)
        if active and not gait_on:
            gait_t0 = data.time; imu_stab.reset(imu_s)
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
            bp = _base_xy(model, data)
            print(f"t={data.time:7.3f}s  {_cmd_text(cmd, turn)}  "
                  f"yaw_fb={ccmd.yaw - cmd.yaw:+.2f}  "
                  f"max_q_err={np.max(np.abs(q_err)):.4f}rad  "
                  f"max_dq={np.max(np.abs(dq_des)):.1f}rad/s  "
                  f"max_tau={np.max(np.abs(tau)):.3f}Nm  "
                  f"base=({bp[0]:+.3f},{bp[1]:+.3f})  "
                  f"pitch={imu_s.pitch:+.3f}rad  yaw={imu_s.yaw:+.3f}rad")
            next_print += period
        return tau

    if render:
        from mujoco import viewer as v
        try:
            ks = X11KeyboardPoller()
            print("Keyboard: X11 polling")
        except RuntimeError as e:
            ks = kbd
            print(f"Keyboard: MuJoCo fallback ({e})")
        print(f"Planner: {planner_label}")
        print("Keys: ↑↓ forward/back  ←→ strafe  Ctrl+←→ turn  Space stop  R reset  Esc quit")
        viewer = None
        try:
            viewer = v.launch_passive(model, data, key_callback=kbd.on_key)
            while viewer.is_running() and (duration <= 0 or data.time < duration):
                t0 = time.monotonic()
                cmd, turn, ex, rst = ks.snapshot()
                if ex:
                    break
                if rst:
                    init_sim(config_path, model_path_override=args.model)
                    imu_stab.reset(imu_reader.read(data))
                    last_q = default_q.copy()
                step(cmd, turn)
                try:
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                                      mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      f"OpenDoge IK [{planner_label}]",
                                      _cmd_text(cmd, turn)))
                except AttributeError:
                    pass
                viewer.sync()
                dt = model.opt.timestep - (time.monotonic() - t0)
                if dt > 0:
                    time.sleep(dt)
        finally:
            if viewer is not None:
                viewer.close()
                dl = time.monotonic() + 2.0
                while viewer.is_running() and time.monotonic() < dl:
                    time.sleep(0.01)
            if isinstance(ks, X11KeyboardPoller):
                ks.close()
    else:
        while duration <= 0 or data.time < duration:
            step(hcmd, abs(hcmd.yaw) > 1e-6)

    if recorder is not None:
        p = recorder.save()
        print(f"Saved {recorder.num_frames} frames → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
