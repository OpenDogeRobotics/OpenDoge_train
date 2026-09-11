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
    parser.add_argument("--config", type=str, default=YAML_PATH, help="Sim2Sim YAML configuration.")
    parser.add_argument("--xml", type=str, default=None, help="Override MuJoCo scene XML path.")
    return parser.parse_args()


ARGS = parse_args()
ONNX_PATH = resolve_onnx_path(cli_onnx=ARGS.onnx)

print(f"YAML : {os.path.abspath(ARGS.config)}")
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


def load_mujoco_model(xml_path, num_actions):
    """Load MJCF or compile the V1.1 URDF with a floor and actuators."""
    if not xml_path.lower().endswith(".urdf"):
        return mujoco.MjModel.from_xml_path(xml_path)

    spec = mujoco.MjSpec()
    spec.from_file(xml_path)
    # MuJoCo's URDF importer discards <visual> elements by default.  Keep the
    # primitive <collision> elements from the URDF, then add the STL visuals
    # explicitly below so they render without entering contact generation.
    spec.discardvisual = 0
    spec.meshdir = os.path.abspath(os.path.join(os.path.dirname(xml_path), "../meshes"))
    # URDF has an implicit floating base for Isaac Gym; add it explicitly for MuJoCo.
    robot_body = spec.worldbody.first_body()
    if robot_body is None:
        raise RuntimeError(f"No robot body found in URDF: {xml_path}")
    robot_body.add_freejoint()
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [2.5, 2.5, 0.05]
    floor.group = 2
    floor.rgba = [0.04, 0.20, 0.62, 1.0]
    floor.contype = 1
    floor.conaffinity = 1

    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    for name in joint_names[:num_actions]:
        actuator = spec.add_actuator()
        actuator.name = name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.target = name
        actuator.ctrlrange = [-9.0 if "calf" in name else -6.0,
                              9.0 if "calf" in name else 6.0]
        actuator.ctrllimited = True

    visual_meshes = [
        ("base_link", "base_link.STL"),
        ("FL_hip", "FL_hip.STL"), ("FL_thigh", "FL_thigh.STL"),
        ("FL_calf", "FL_calf.STL"),
        ("FR_hip", "FR_hip.STL"), ("FR_thigh", "FR_thigh.STL"),
        ("FR_calf", "FR_calf.STL"),
        ("RL_hip", "RL_hip.STL"), ("RL_thigh", "RL_thigh.STL"),
        ("RL_calf", "RL_calf.STL"),
        ("RR_hip", "RR_hip.STL"), ("RR_thigh", "RR_thigh.STL"),
        ("RR_calf", "RR_calf.STL"),
    ]
    for body_name, mesh_file in visual_meshes:
        mesh = spec.add_mesh()
        mesh.name = f"{body_name}_visual_mesh"
        mesh.file = mesh_file
        body = spec.find_body(body_name)
        if body is None:
            raise RuntimeError(f"Visual body not found in URDF: {body_name}")
        visual = body.add_geom()
        visual.name = f"{body_name}_visual"
        visual.type = mujoco.mjtGeom.mjGEOM_MESH
        visual.meshname = mesh.name
        visual.group = 1
        visual.rgba = [0.75294, 0.75294, 0.75294, 1.0]
        visual.contype = 0
        visual.conaffinity = 0
    model = spec.compile()
    # The URDF has near-zero distal-link inertia and no <dynamics> damping.
    # Isaac Gym's PD controller is numerically damped; mirror that behavior
    # here so the same policy does not excite the tiny calf inertia in MuJoCo.
    for joint_id in range(1, model.njnt):
        dof_id = model.jnt_dofadr[joint_id]
        model.dof_damping[dof_id] = 0.5
        model.dof_armature[dof_id] = 0.01
    # The URDF visual meshes are imported as regular geoms by MuJoCo.  V1.1
    # already has dedicated primitive collision geoms, so visual meshes must
    # not participate in contact generation.
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
    return model


def key_callback(keycode):
    global paused
    if chr(keycode) == " ":
        paused = not paused
        print(f"Paused: {paused}")


# ==================== 4. Main ====================
def run_simulation():
    global cmd, default_dof_pos

    # --- Load YAML config ---
    config_path = os.path.abspath(ARGS.config)
    if not os.path.exists(config_path):
        print(f"ERROR: config not found at {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
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

    xml_path_cfg = ARGS.xml or config.get("xml_path", "")
    xml_path = (xml_path_cfg.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
                if xml_path_cfg else DEFAULT_XML_PATH)

    print(f"XML  : {xml_path}")

    # --- Load MuJoCo & ONNX ---
    if not os.path.exists(xml_path):
        print(f"ERROR: model file not found at {xml_path}")
        return

    print("Loading MuJoCo model …")
    model = load_mujoco_model(xml_path, num_actions)
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
        # Show STL visuals and the blue floor, while keeping collision
        # primitives active for physics but hidden from the viewer.
        viewer.opt.geomgroup[0] = 0
        viewer.opt.geomgroup[1] = 1
        viewer.opt.geomgroup[2] = 1
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
