"""
OpenDoge MuJoCo Sim2Sim — keyboard control.

Arrow keys for locomotion commands, Ctrl for lateral movement, Space to pause.

Usage (from OpenDoge_train/):
    python sim2sim/sim2sim_keyboard.py
    python sim2sim/sim2sim_keyboard.py --onnx onnx/flat_opendoge_9000_omni.onnx
"""

import time
import os
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
import yaml
from collections import deque
from pynput import keyboard

from legged_gym import LEGGED_GYM_ROOT_DIR
from sim2sim.onnx_utils import resolve_onnx_path
from sim2sim._common import (
    quat_rotate_inverse,
    pd_control,
    build_policy_input,
    build_obs_raw,
)

# ==================== 1. Paths ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(SCRIPT_DIR, "configs", "opendoge.yaml")
DEFAULT_XML_PATH = os.path.join(LEGGED_GYM_ROOT_DIR, "resources", "robots", "Opendoge", "xml", "scene.xml")


def parse_args():
    parser = argparse.ArgumentParser(description="Sim2Sim keyboard control for OpenDoge policies.")
    parser.add_argument("--onnx", type=str, default=None, help="Path to ONNX policy model.")
    return parser.parse_args()


ARGS = parse_args()
ONNX_PATH = resolve_onnx_path(cli_onnx=ARGS.onnx)

print(f"YAML : {YAML_PATH}")
print(f"ONNX : {ONNX_PATH}")

# ==================== 2. Globals ====================
cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [vx, vy, omega]
paused = False
default_dof_pos = None


# ==================== 3. Keyboard handling ====================
_pressed = set()


def _update_cmd():
    global cmd
    ctrl = keyboard.Key.ctrl_l in _pressed or keyboard.Key.ctrl_r in _pressed
    vx = vy = omega = 0.0

    if keyboard.Key.up in _pressed:
        vx = 0.6
    elif keyboard.Key.down in _pressed:
        vx = -0.4

    if keyboard.Key.left in _pressed:
        if ctrl:
            vy = 0.4   # Ctrl+Left  → strafe left
        else:
            omega = 0.8  # Left       → turn left
    elif keyboard.Key.right in _pressed:
        if ctrl:
            vy = -0.4  # Ctrl+Right → strafe right
        else:
            omega = -0.8  # Right      → turn right

    cmd[0] = vx
    cmd[1] = vy
    cmd[2] = omega


def on_press(key):
    _pressed.add(key)
    _update_cmd()


def on_release(key):
    _pressed.discard(key)
    _update_cmd()


def key_callback(keycode):
    global paused
    if chr(keycode) == " ":
        paused = not paused
        print(f"Paused: {paused}")


# ==================== 4. Main ====================
def run_simulation():
    global cmd, default_dof_pos

    # --- Load YAML config ---
    if not os.path.exists(YAML_PATH):
        print(f"ERROR: config not found at {YAML_PATH}")
        return

    with open(YAML_PATH, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    sim_dt = float(config.get("simulation_dt", 0.005))
    control_decimation = int(config.get("control_decimation", 2))
    num_actions = int(config.get("num_actions", 12))
    num_obs = int(config.get("num_obs", 270))
    num_one_step_obs = int(config.get("num_one_step_obs", 45))
    init_base_height = float(config.get("init_base_height", 0.15))

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_dof_pos = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    cmd_init = np.array(config.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)

    if len(default_dof_pos) != num_actions or len(kps) != num_actions or len(kds) != num_actions:
        print("ERROR: YAML num_actions does not match kps/kds/default_angles dimensions.")
        return
    cmd[:] = cmd_init

    xml_path_cfg = config.get("xml_path", "")
    xml_path = (xml_path_cfg.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
                if xml_path_cfg else DEFAULT_XML_PATH)

    print(f"XML  : {xml_path}")

    # --- Load MuJoCo & ONNX ---
    if not os.path.exists(xml_path):
        print(f"ERROR: model file not found at {xml_path}")
        return

    print("Loading MuJoCo model …")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = sim_dt

    use_gyro_sensor = True
    try:
        _ = data.sensor("angular-velocity").data
    except KeyError:
        use_gyro_sensor = False
        print("WARNING: sensor 'angular-velocity' not found; falling back to data.qvel[3:6].")

    print(f"Loading ONNX: {ONNX_PATH}")
    ort_session = ort.InferenceSession(ONNX_PATH)
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    input_dim = int(input_shape[-1]) if isinstance(input_shape[-1], int) else num_obs
    print(f"ONNX Input Shape: {input_shape}")

    # --- Initialise state ---
    data.qpos[7:7 + num_actions] = default_dof_pos
    data.qpos[2] = init_base_height
    mujoco.mj_forward(model, data)

    target_dof_pos = default_dof_pos.copy()
    action = np.zeros(num_actions, dtype=np.float32)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("Simulation running!  ↑↓ fwd/back  ←→ turn  Ctrl+←→ strafe  Space pause")

    history_len = max(1, num_obs // num_one_step_obs)
    obs_dim = num_one_step_obs
    obs_history_buffer = deque(
        [np.zeros(obs_dim, dtype=np.float32) for _ in range(history_len)],
        maxlen=history_len,
    )

    # --- Simulation loop ---
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        step_counter = 0
        while viewer.is_running():
            step_start = time.time()

            if not paused:
                if step_counter % control_decimation == 0:
                    obs_raw = build_obs_raw(
                        data=data,
                        default_dof_pos=default_dof_pos,
                        cmd=cmd,
                        cmd_scale=cmd_scale,
                        ang_vel_scale=ang_vel_scale,
                        dof_pos_scale=dof_pos_scale,
                        dof_vel_scale=dof_vel_scale,
                        action=action,
                        num_actions=num_actions,
                        use_gyro_sensor=use_gyro_sensor,
                    )

                    policy_input = build_policy_input(
                        obs_raw=obs_raw,
                        history_buffer=obs_history_buffer,
                        input_dim=input_dim,
                        num_obs=num_obs,
                    )

                    outputs = ort_session.run(None, {input_name: policy_input})
                    raw_action = np.clip(outputs[0][0], -10.0, 10.0)
                    action = raw_action
                    target_dof_pos = raw_action * action_scale + default_dof_pos

                # PD torque control
                tau = pd_control(
                    target_dof_pos,
                    data.qpos[7:7 + num_actions],
                    kps,
                    np.zeros_like(kds),
                    data.qvel[6:6 + num_actions],
                    kds,
                )

                if model.nu < num_actions:
                    print(f"ERROR: MuJoCo actuator count ({model.nu}) < num_actions ({num_actions})")
                    return

                tau_limit = np.abs(model.actuator_ctrlrange[:num_actions, 1])
                tau = np.clip(tau, -tau_limit, tau_limit)
                data.ctrl[:num_actions] = tau

                mujoco.mj_step(model, data)
                step_counter += 1

            viewer.sync()

            # Frame-rate sync
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    run_simulation()
