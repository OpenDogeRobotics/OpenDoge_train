"""MuJoCo sim2sim deployment and validation for an OpenDoge HIM policy.

Examples:
  python deploy/deploy_mujoco/deploy_opendoge.py opendoge.yaml --cmd_vx 1.0
  python deploy/deploy_mujoco/deploy_opendoge.py opendoge.yaml --keyboard
  python deploy/deploy_mujoco/deploy_opendoge.py opendoge.yaml \
      --headless --duration 20 --diagnostics --cmd_vx 1.0
  python deploy/deploy_mujoco/deploy_opendoge.py opendoge.yaml \
      --validate --duration 10
"""

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import time
import math

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Isaac Gym exposes the OpenDoge DOFs in this order. The policy observation and
# action vectors must use the same order, independent of the MJCF file order.
POLICY_JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

VALIDATION_CASES = (
    ("forward_1.0", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
    ("forward_0.5", np.array([0.5, 0.0, 0.0], dtype=np.float32)),
    ("lateral_0.5", np.array([0.0, 0.5, 0.0], dtype=np.float32)),
    ("yaw_1.0", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
)


def get_gravity_orientation(quat):
    """Return world gravity expressed in the base frame."""
    qw, qx, qy, qz = quat
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def pd_control(target_q, q, kp, dq, kd):
    return (target_q - q) * kp - dq * kd


def push_observation(obs_history, one_step):
    """Insert the newest observation without overlapping-copy corruption."""
    frame_size = one_step.shape[0]
    obs_history[:, frame_size:] = obs_history[:, :-frame_size].copy()
    obs_history[:, :frame_size] = one_step


def resolve_project_path(path):
    return Path(str(path).replace("{LEGGED_GYM_ROOT_DIR}", str(PROJECT_ROOT)))


def resolve_config_path(config_file):
    path = Path(config_file)
    if path.is_file():
        return path.resolve()

    path = PROJECT_ROOT / "deploy" / "deploy_mujoco" / "configs" / config_file
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return path


def reorder_config_values(values, config_joint_names, field_name):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(config_joint_names),):
        raise ValueError(
            f"{field_name} must contain {len(config_joint_names)} values, "
            f"got {values.shape}"
        )
    try:
        indices = [
            config_joint_names.index(name) for name in POLICY_JOINT_NAMES
        ]
    except ValueError as exc:
        raise ValueError(
            f"{field_name}: joint_names do not match the policy DOFs"
        ) from exc
    return values[indices]


def get_mujoco_joint_layout(model):
    qpos_indices = []
    qvel_indices = []
    actuator_indices = []

    for joint_name in POLICY_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint {joint_name}")

        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if matches.size != 1:
            raise ValueError(
                f"Expected one actuator for {joint_name}, found {matches.size}"
            )

        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        qvel_indices.append(int(model.jnt_dofadr[joint_id]))
        actuator_indices.append(int(matches[0]))

    return (
        np.asarray(qpos_indices, dtype=np.int32),
        np.asarray(qvel_indices, dtype=np.int32),
        np.asarray(actuator_indices, dtype=np.int32),
    )


@dataclass
class RuntimeDiagnostics:
    min_base_height: float = float("inf")
    max_tilt: float = 0.0
    max_abs_action: float = 0.0
    max_raw_torque: float = 0.0
    fall_time: float = None
    velocity_sum: np.ndarray = None
    angular_velocity_sum: np.ndarray = None
    velocity_samples: int = 0

    def __post_init__(self):
        if self.velocity_sum is None:
            self.velocity_sum = np.zeros(3, dtype=np.float64)
        if self.angular_velocity_sum is None:
            self.angular_velocity_sum = np.zeros(3, dtype=np.float64)

    @property
    def mean_body_velocity(self):
        if self.velocity_samples == 0:
            return np.zeros(3, dtype=np.float64)
        return self.velocity_sum / self.velocity_samples

    @property
    def mean_body_angular_velocity(self):
        if self.velocity_samples == 0:
            return np.zeros(3, dtype=np.float64)
        return self.angular_velocity_sum / self.velocity_samples


class OpenDogeSim2Sim:
    """Policy, MuJoCo state and policy-rate control contract."""

    def __init__(self, config_path, action_delay_steps=None, hfield_data=None):
        self.config_path = Path(config_path)
        with self.config_path.open("r") as stream:
            self.cfg = yaml.safe_load(stream)

        self.sim_dt = float(self.cfg["simulation_dt"])
        self.decimation = int(self.cfg["control_decimation"])
        self.policy_dt = self.sim_dt * self.decimation
        self.action_delay_steps = (
            int(action_delay_steps)
            if action_delay_steps is not None
            else int(self.cfg.get("action_delay_steps", 0))
        )
        if not 0 <= self.action_delay_steps < self.decimation:
            raise ValueError(
                "action_delay_steps must be in [0, control_decimation)"
            )

        config_joint_names = list(self.cfg["joint_names"])
        if len(config_joint_names) != len(POLICY_JOINT_NAMES):
            raise ValueError("joint_names must contain exactly 12 joints")

        self.default_angles = reorder_config_values(
            self.cfg["default_angles"], config_joint_names, "default_angles"
        )
        self.kps = reorder_config_values(
            self.cfg["kps"], config_joint_names, "kps"
        )
        self.kds = reorder_config_values(
            self.cfg["kds"], config_joint_names, "kds"
        )
        self.torque_limits = reorder_config_values(
            self.cfg["torque_limits"], config_joint_names, "torque_limits"
        )

        self.action_scale = float(self.cfg["action_scale"])
        self.cmd_scale = np.asarray(self.cfg["cmd_scale"], dtype=np.float32)
        self.ang_vel_scale = float(self.cfg["ang_vel_scale"])
        self.dof_pos_scale = float(self.cfg["dof_pos_scale"])
        self.dof_vel_scale = float(self.cfg["dof_vel_scale"])
        self.num_actions = int(self.cfg["num_actions"])
        self.num_obs = int(self.cfg["num_obs"])
        self.num_one_step = int(self.cfg["num_one_step_obs"])
        self._validate_dimensions()

        self.policy_path = resolve_project_path(self.cfg["policy_path"])
        self.xml_path = resolve_project_path(self.cfg["xml_path"])
        self.session = ort.InferenceSession(
            str(self.policy_path), providers=["CPUExecutionProvider"]
        )
        self.policy_input = self.session.get_inputs()[0]
        self.policy_output = self.session.get_outputs()[0]
        self._validate_policy_io()

        # ── Get-up policy (optional fallback) ──
        self.getup_session = None
        getup_path = self.cfg.get("getup_policy_path")
        if getup_path:
            getup_path = resolve_project_path(getup_path)
            self.getup_session = ort.InferenceSession(
                str(getup_path), providers=["CPUExecutionProvider"]
            )
            self.getup_input = self.getup_session.get_inputs()[0]
            self.getup_output = self.getup_session.get_outputs()[0]
            print(f"Get-up policy loaded: {getup_path}")
        self._in_recovery = False
        self._recovery_stable_steps = 0
        self._RECOVERY_STABLE_NEEDED = 50  # ~1s at 50Hz

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.model.opt.timestep = self.sim_dt
        if hfield_data is not None and self.model.nhfield == 0:
            raise ValueError("hfield_data given but model has no heightfield")
        if self.model.nhfield:
            # MJCF cannot embed heightfield data: the wave terrain writes a
            # companion .npy next to the scene XML (see make_terrain.py).
            # Data must be in place before MjData is built (collision uses
            # the copy captured at mj_makeData time).
            if hfield_data is None:
                hf_path = self.xml_path.with_name(
                    self.xml_path.stem + "_hfield.npy"
                )
                if hf_path.is_file():
                    hfield_data = np.load(hf_path)
                    print(f"Hfield data loaded: {hf_path}")
            if hfield_data is not None:
                hfield_data = np.asarray(hfield_data, dtype=np.float32).ravel()
                expected = int(
                    np.sum(self.model.hfield_nrow * self.model.hfield_ncol)
                )
                if hfield_data.size != expected:
                    raise ValueError(
                        f"hfield_data has {hfield_data.size} values, "
                        f"model expects {expected}"
                    )
                self.model.hfield_data[:] = hfield_data
        self.data = mujoco.MjData(self.model)
        (
            self.qpos_indices,
            self.qvel_indices,
            self.actuator_indices,
        ) = get_mujoco_joint_layout(self.model)
        base_body_name = self.cfg.get("base_body_name", "base_link")
        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name
        )
        if self.base_body_id < 0:
            raise ValueError(f"MuJoCo model is missing base body {base_body_name}")
        self._validate_model_contract()
        self.reset()

    def _validate_dimensions(self):
        if self.num_actions != len(POLICY_JOINT_NAMES):
            raise ValueError(f"Expected 12 actions, got {self.num_actions}")
        if self.num_obs % self.num_one_step != 0:
            raise ValueError(
                "num_obs must be an integer multiple of num_one_step_obs"
            )
        expected_one_step = 9 + 3 * self.num_actions
        if self.num_one_step != expected_one_step:
            raise ValueError(
                f"Expected {expected_one_step} one-step observations, "
                f"got {self.num_one_step}"
            )

    def _validate_policy_io(self):
        if self.policy_input.shape[-1] not in (None, "None", self.num_obs):
            raise ValueError(
                f"Policy input expects {self.policy_input.shape[-1]} values, "
                f"config has {self.num_obs}"
            )
        if self.policy_output.shape[-1] not in (
            None,
            "None",
            self.num_actions,
        ):
            raise ValueError(
                f"Policy output has {self.policy_output.shape[-1]} actions, "
                f"config has {self.num_actions}"
            )

    def _validate_model_contract(self):
        expected_mass = self.cfg.get("expected_total_mass")
        if expected_mass is not None:
            actual_mass = float(np.sum(self.model.body_mass))
            tolerance = float(self.cfg.get("mass_tolerance", 1e-6))
            if abs(actual_mass - float(expected_mass)) > tolerance:
                raise ValueError(
                    f"Model mass mismatch: expected {expected_mass}, "
                    f"got {actual_mass}"
                )

        expected_armature = self.cfg.get("expected_joint_armature")
        if expected_armature is not None:
            actual = self.model.dof_armature[self.qvel_indices]
            if not np.allclose(actual, float(expected_armature), atol=1e-9):
                raise ValueError(
                    f"Joint armature mismatch: expected {expected_armature}, "
                    f"got {actual.tolist()}"
                )

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_indices] = self.default_angles
        # optional spawn offset (e.g. on a competition field); defaults to origin
        start_xy = self.cfg.get("initial_base_pos", [0.0, 0.0])
        self.data.qpos[0] = float(start_xy[0])
        self.data.qpos[1] = float(start_xy[1])
        self.data.qpos[2] = float(self.cfg.get("initial_base_height", 0.35))
        mujoco.mj_forward(self.model, self.data)

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.previous_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_phase = 0
        self.counter = 0
        self.obs_history = np.zeros((1, self.num_obs), dtype=np.float32)

        settle_duration = float(self.cfg.get("settle_duration", 0.0))
        settle_steps = max(0, round(settle_duration / self.sim_dt))
        for t in range(settle_steps):
            torque = pd_control(
                self.default_angles,
                self.data.qpos[self.qpos_indices],
                self.kps,
                self.data.qvel[self.qvel_indices],
                self.kds,
            )
            torque = np.clip(
                torque, -self.torque_limits, self.torque_limits
            )
            self.data.ctrl[self.actuator_indices] = torque
            mujoco.mj_step(self.model, self.data)

            # Fill observation history during settle so HIM estimator sees
            # real data from the first policy step (not zeros).
            if (t + 1) % self.decimation == 0:
                qj = self.data.qpos[self.qpos_indices].copy()
                dqj = self.data.qvel[self.qvel_indices].copy()
                quat = self.data.qpos[3:7].copy()
                gravity = get_gravity_orientation(quat)
                cmd0 = np.zeros(3, dtype=np.float32)

                one_step = np.zeros(self.num_one_step, dtype=np.float32)
                one_step[0:3] = cmd0 * self.cmd_scale
                one_step[3:6] = self.data.qvel[3:6] * self.ang_vel_scale
                one_step[6:9] = gravity
                one_step[9:21] = (qj - self.default_angles) * self.dof_pos_scale
                one_step[21:33] = dqj * self.dof_vel_scale
                one_step[33:45] = self.action
                push_observation(self.obs_history, one_step)

        self.run_start_time = self.data.time
        self.diagnostics = RuntimeDiagnostics()

    @property
    def next_step_is_policy_tick(self):
        return (self.counter + 1) % self.decimation == 0

    def model_summary(self):
        return {
            "policy_joint_names": POLICY_JOINT_NAMES,
            "policy_path": str(self.policy_path),
            "xml_path": str(self.xml_path),
            "actuator_indices": self.actuator_indices.tolist(),
            "total_mass": float(np.sum(self.model.body_mass)),
            "max_joint_armature": float(
                np.max(self.model.dof_armature[self.qvel_indices])
            ),
            "simulation_dt": self.sim_dt,
            "policy_dt": self.policy_dt,
            "action_delay_steps": self.action_delay_steps,
        }

    def _check_fallen(self):
        """Return True if robot is in a fallen state and needs recovery."""
        quat = self.data.qpos[3:7].copy()
        gravity = get_gravity_orientation(quat)
        tilt = float(np.linalg.norm(gravity[:2]))
        base_z = float(self.data.qpos[2])
        ang_vel_norm = float(np.linalg.norm(self.data.qvel[3:6]))
        max_torque = float(np.max(np.abs(self.data.actuator_force)))
        fall_height = float(self.cfg.get("fall_base_height", 0.18))
        return (
            base_z < fall_height or tilt > 0.9 or
            ang_vel_norm > 8.0 or max_torque > 40.0
        )

    def _check_recovered(self):
        quat = self.data.qpos[3:7].copy()
        gravity = get_gravity_orientation(quat)
        tilt = float(np.linalg.norm(gravity[:2]))
        base_z = float(self.data.qpos[2])
        recovery_height = float(self.cfg.get("recovered_base_height", 0.25))
        return base_z > recovery_height and tilt < 0.35

    def _policy_step(self, command):
        # ── Fall detection + get-up activation ──
        if self.getup_session is not None:
            if self._check_fallen():
                if not self._in_recovery:
                    print("[get-up] fall detected — activating recovery")
                self._in_recovery = True
                self._recovery_stable_steps = 0
            elif self._in_recovery and self._check_recovered():
                self._recovery_stable_steps += 1
                if self._recovery_stable_steps >= self._RECOVERY_STABLE_NEEDED:
                    self._in_recovery = False
                    print("[get-up] recovered — switching to walking")

        # ── Build 45-dim one-step observation ──
        qj = self.data.qpos[self.qpos_indices].copy()
        dqj = self.data.qvel[self.qvel_indices].copy()
        quat = self.data.qpos[3:7].copy()
        gravity = get_gravity_orientation(quat)

        one_step = np.zeros(self.num_one_step, dtype=np.float32)

        # Command (zero during recovery)
        one_step[0:3] = 0.0 if (self._in_recovery and self.getup_session is not None) else command * self.cmd_scale
        one_step[3:6] = self.data.qvel[3:6] * self.ang_vel_scale
        one_step[6:9] = gravity
        one_step[9:21] = (qj - self.default_angles) * self.dof_pos_scale
        one_step[21:33] = dqj * self.dof_vel_scale
        one_step[33:45] = self.action
        push_observation(self.obs_history, one_step)

        self.previous_action = self.action.copy()

        if self._in_recovery and self.getup_session is not None:
            # Get-up: same 270-dim HIM input, smooth blend
            raw_action = self.getup_session.run(
                None,
                {self.getup_input.name: self.obs_history.astype(np.float32, copy=False)}
            )[0].squeeze().astype(np.float32)
            blend = 0.7
            self.action = (1 - blend) * self.action + blend * raw_action
        else:
            # Normal walking
            self.action = self.session.run(
                None,
                {self.policy_input.name: self.obs_history.astype(np.float32, copy=False)}
            )[0].squeeze().astype(np.float32, copy=False)

        self.action_phase = 0

        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        body_velocity = rotation.T @ self.data.qvel[:3]
        self.diagnostics.velocity_sum += body_velocity
        self.diagnostics.angular_velocity_sum += self.data.qvel[3:6]
        self.diagnostics.velocity_samples += 1
        self.diagnostics.max_abs_action = max(
            self.diagnostics.max_abs_action,
            float(np.max(np.abs(self.action))),
        )

    def step(self, command):
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (3,):
            raise ValueError(f"Command must have shape (3,), got {command.shape}")

        applied_action = (
            self.previous_action
            if self.action_phase < self.action_delay_steps
            else self.action
        )
        target = applied_action * self.action_scale + self.default_angles
        raw_torque = pd_control(
            target,
            self.data.qpos[self.qpos_indices],
            self.kps,
            self.data.qvel[self.qvel_indices],
            self.kds,
        )
        self.diagnostics.max_raw_torque = max(
            self.diagnostics.max_raw_torque,
            float(np.max(np.abs(raw_torque))),
        )
        torque = np.clip(
            raw_torque, -self.torque_limits, self.torque_limits
        )
        self.data.ctrl[self.actuator_indices] = torque
        mujoco.mj_step(self.model, self.data)

        self.action_phase += 1
        self.counter += 1
        if self.counter % self.decimation == 0:
            self._policy_step(command)

        gravity = get_gravity_orientation(self.data.qpos[3:7])
        current_tilt = float(np.linalg.norm(gravity[:2]))
        self.diagnostics.min_base_height = min(
            self.diagnostics.min_base_height, float(self.data.qpos[2])
        )
        self.diagnostics.max_tilt = max(
            self.diagnostics.max_tilt, current_tilt
        )
        if (
            self.diagnostics.fall_time is None
            and (
                self.data.qpos[2] < float(self.cfg.get("fall_base_height", 0.18))
                or current_tilt > 0.9
            )
        ):
            self.diagnostics.fall_time = (
                self.data.time - self.run_start_time
            )

    def run(
        self,
        duration,
        command,
        viewer=None,
        command_provider=None,
        realtime=False,
    ):
        command = np.asarray(command, dtype=np.float32).copy()
        self.run_start_time = self.data.time
        self.diagnostics = RuntimeDiagnostics()

        while (
            (viewer is None or viewer.is_running())
            and self.data.time - self.run_start_time < duration
        ):
            step_start = time.time()
            if (
                command_provider is not None
                and self.next_step_is_policy_tick
            ):
                command = np.asarray(
                    command_provider(command), dtype=np.float32
                )

            self.step(command)

            if viewer is not None:
                viewer.sync()
            if realtime:
                sleep_time = self.sim_dt - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        return self.diagnostics

    def diagnostics_summary(self):
        mean_velocity = self.diagnostics.mean_body_velocity
        mean_angular_velocity = (
            self.diagnostics.mean_body_angular_velocity
        )
        return (
            f"sim_time={self.data.time - self.run_start_time:.3f}s, "
            f"final_z={self.data.qpos[2]:.3f}m, "
            f"min_z={self.diagnostics.min_base_height:.3f}m, "
            f"max_tilt={self.diagnostics.max_tilt:.3f}, "
            f"max_abs_action={self.diagnostics.max_abs_action:.3f}, "
            f"max_raw_torque={self.diagnostics.max_raw_torque:.2f}Nm, "
            f"mean_body_v=({mean_velocity[0]:.3f},"
            f"{mean_velocity[1]:.3f})m/s, "
            f"mean_body_wz={mean_angular_velocity[2]:.3f}rad/s, "
            f"fall_time={self.diagnostics.fall_time}"
        )


class KeyboardCommand:
    """Policy-rate keyboard command provider (pynput, global — legacy)."""

    def __init__(self):
        from pynput import keyboard

        self.keyboard = keyboard
        self.key_state = set()
        self.linear_step = 0.05
        self.yaw_step = 0.1
        self.decay = 0.95
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.start()

    def _key_value(self, key):
        try:
            return key.char
        except AttributeError:
            return key

    def _on_press(self, key):
        self.key_state.add(self._key_value(key))

    def _on_release(self, key):
        self.key_state.discard(self._key_value(key))

    def update(self, command):
        command = command.copy()
        keyboard = self.keyboard

        if keyboard.Key.up in self.key_state:
            command[0] = min(command[0] + self.linear_step, 2.0)
        elif keyboard.Key.down in self.key_state:
            command[0] = max(command[0] - self.linear_step, -1.0)
        else:
            command[0] *= self.decay

        if keyboard.Key.left in self.key_state:
            command[1] = min(command[1] + self.linear_step, 1.0)
        elif keyboard.Key.right in self.key_state:
            command[1] = max(command[1] - self.linear_step, -1.0)
        else:
            command[1] *= self.decay

        if "w" in self.key_state:
            command[2] = min(command[2] + self.yaw_step, np.pi)
        elif "s" in self.key_state:
            command[2] = max(command[2] - self.yaw_step, -np.pi)
        else:
            command[2] *= self.decay

        if keyboard.Key.space in self.key_state:
            command[:] = 0.0
        return command


# ── MuJoCo viewer keyboard (only fires when viewer window is focused) ──

# GLFW key codes — only keys that do NOT conflict with MuJoCo viewer shortcuts
# (MuJoCo reserves: Space=pause, Tab=UI, Backspace=reset, Esc=exit, F1=help)
_GLFW_KEY_W = 87
_GLFW_KEY_A = 65
_GLFW_KEY_S = 83
_GLFW_KEY_D = 68
_GLFW_KEY_Q = 81
_GLFW_KEY_E = 69
_GLFW_KEY_UP = 265
_GLFW_KEY_DOWN = 264
_GLFW_KEY_MINUS = 45   # - key
_GLFW_KEY_EQUAL = 61   # = / + key
_GLFW_KEY_X = 88       # emergency stop
_GLFW_KEY_1 = 49       # speed preset
_GLFW_KEY_2 = 50
_GLFW_KEY_3 = 51
_GLFW_KEY_4 = 52
_GLFW_KEY_5 = 53

_SPEED_PRESETS = {_GLFW_KEY_1: 0.5, _GLFW_KEY_2: 1.0, _GLFW_KEY_3: 2.0,
                  _GLFW_KEY_4: 3.0, _GLFW_KEY_5: 5.0}


class MujocoViewerKeyboard:
    """Keyboard command provider via MuJoCo viewer's per-window key callback.

    Only captures keys when the viewer window has focus.
    All keys avoid MuJoCo built-in shortcuts (Space, Tab, Backspace, etc.).

    Movement (hold = go, release = decay to 0):
      W/S       — forward / backward
      A/D       — left / right strafe
      Q/E       — turn left / right

    Speed control:
      ↑ / ↓     — raise / lower base speed ±0.2
      - / +     — same (alternative)
      1–5       — speed presets: 0.2 / 0.5 / 1.0 / 1.5 / 2.0 m/s

    Stop:
      X         — emergency stop (cmd→0, base→0)
    """

    def __init__(self, base_vx=0.0):
        self._new_keys = set()
        self._lock = __import__('threading').Lock()
        self.vx_step = 0.08     # how fast vx/vy ramp per tick
        self.vyaw_step = 0.15   # how fast yaw ramps per tick
        self.decay = 0.90       # decay factor when keys released
        self.base_vx = base_vx
        self.base_vyaw = 1.5    # max yaw rate when Q/E held

    @property
    def key_callback(self):
        def _cb(keycode):
            with self._lock:
                self._new_keys.add(keycode)
        return _cb

    def update(self, command):
        """Compute new command from accumulated keys (called at policy rate, 50Hz).

        Holding a key → command ramps toward target at vx_step per tick.
        Releasing → command decays toward 0 at decay factor.
        Speed presets and ↑↓± adjust base_vx.
        """
        command = command.copy()
        with self._lock:
            keys = self._new_keys
            self._new_keys = set()

        # ── Stop (X) ──
        if _GLFW_KEY_X in keys:
            command[:] = 0.0
            self.base_vx = 0.0
            return command

        # ── Speed presets (1–5) ──
        for k, preset in _SPEED_PRESETS.items():
            if k in keys:
                self.base_vx = preset

        # ── Speed adjust (↑↓ or -+) ──
        if _GLFW_KEY_UP in keys or _GLFW_KEY_EQUAL in keys:
            self.base_vx = min(self.base_vx + 0.2, 5.0)
        if _GLFW_KEY_DOWN in keys or _GLFW_KEY_MINUS in keys:
            self.base_vx = max(self.base_vx - 0.2, -2.0)

        # Omnidirectional speed cap = base speed magnitude (0 ⇒ no motion
        # in any horizontal direction until a preset/↑↓ sets the speed).
        cap = abs(self.base_vx)

        # ── Forward / backward (W/S) ──
        if _GLFW_KEY_W in keys:
            command[0] = min(command[0] + self.vx_step, cap)
        elif _GLFW_KEY_S in keys:
            command[0] = max(command[0] - self.vx_step, -cap)
        else:
            command[0] *= self.decay

        # ── Lateral (A/D) ──
        if _GLFW_KEY_A in keys:
            command[1] = min(command[1] + self.vx_step, cap)
        elif _GLFW_KEY_D in keys:
            command[1] = max(command[1] - self.vx_step, -cap)
        else:
            command[1] *= self.decay

        # ── Yaw (Q/E) ──
        if _GLFW_KEY_Q in keys:
            command[2] = min(command[2] + self.vyaw_step, self.base_vyaw)
        elif _GLFW_KEY_E in keys:
            command[2] = max(command[2] - self.vyaw_step, -self.base_vyaw)
        else:
            command[2] *= self.decay

        # ── Omnidirectional speed limiting ──
        # Cap the horizontal velocity VECTOR, not each axis separately:
        # diagonal motion cannot exceed the base speed.
        if cap > 0.0:
            speed = math.hypot(command[0], command[1])
            if speed > cap and speed > 1e-6:
                scale = cap / speed
                command[0] *= scale
                command[1] *= scale
        else:
            command[0] = 0.0
            command[1] = 0.0

        return command


def print_model_summary(summary):
    print(f"Loading policy: {summary['policy_path']}")
    print(f"Loading model: {summary['xml_path']}")
    print("Policy joint order:", ", ".join(summary["policy_joint_names"]))
    print("MuJoCo actuator indices:", summary["actuator_indices"])
    print(
        f"Model mass={summary['total_mass']:.6f}kg, "
        f"max_joint_armature={summary['max_joint_armature']:.6f}, "
        f"sim_dt={summary['simulation_dt']:.4f}s, "
        f"policy_dt={summary['policy_dt']:.4f}s"
    )


def run_validation(runtime, duration):
    failures = []
    for name, command in VALIDATION_CASES:
        runtime.reset()
        runtime.run(duration, command)
        print(f"{name}: {runtime.diagnostics_summary()}")
        if runtime.diagnostics.fall_time is not None:
            failures.append(name)

    if failures:
        raise SystemExit(f"Unstable cases: {', '.join(failures)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenDoge MuJoCo sim2sim deployment and validation",
        epilog="Examples:\n"
               "  %(prog)s opendoge.yaml --cmd_vx 1.0\n"
               "  %(prog)s opendoge.yaml --keyboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config_file",
        nargs="?",
        default="opendoge.yaml",
        help="YAML file or name under deploy_mujoco/configs",
    )
    parser.add_argument("--cmd_vx", type=float, default=1.0,
                        help="Forward velocity command [m/s]")
    parser.add_argument("--cmd_vy", type=float, default=0.0,
                        help="Lateral velocity command [m/s]")
    parser.add_argument("--cmd_vyaw", type=float, default=0.0,
                        help="Yaw rate command [rad/s]")
    parser.add_argument("--keyboard", dest="keyboard", action="store_true",
                        default=True,
                        help="Use keyboard control (default)."
                             " Use --no-keyboard to disable.")
    parser.add_argument("--no-keyboard", dest="keyboard", action="store_false",
                        help="Disable keyboard, use fixed --cmd_vx/vy/vyaw")
    parser.add_argument("--headless", action="store_true",
                        help="Run without viewer window")
    parser.add_argument("--duration", type=float, default=None,
                        help="Simulation duration [s] (default: from config)")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Print diagnostics summary at end")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation cases (forward/lateral/yaw)")
    parser.add_argument("--action_delay_steps", type=int, default=None,
                        help="Action delay in sim substeps [0, decimation)")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = resolve_config_path(args.config_file)
    runtime = OpenDogeSim2Sim(
        config_path,
        action_delay_steps=args.action_delay_steps,
    )
    print_model_summary(runtime.model_summary())

    if args.validate:
        duration = float(args.duration) if args.duration is not None else 10.0
        run_validation(runtime, duration)
        return

    command = np.array(
        [args.cmd_vx, args.cmd_vy, args.cmd_vyaw],
        dtype=np.float32,
    )
    duration = (
        float(args.duration)
        if args.duration is not None
        else float(runtime.cfg["simulation_duration"])
    )

    if args.keyboard:
        # Keyboard mode: start stationary, user controls everything.
        # --cmd_vx is ignored (use presets 1-5 or ↑↓ to set speed).
        keyboard = MujocoViewerKeyboard(base_vx=0.0)
        command_provider = keyboard.update
        key_callback = keyboard.key_callback
        mode = "keyboard (viewer-focused, start stationary)"
    else:
        keyboard = None
        command_provider = None
        key_callback = None
        mode = command.tolist()
    print(
        f"Starting simulation. Mode={mode}, "
        f"action_delay={runtime.action_delay_steps} substeps"
    )

    viewer_context = (
        nullcontext(None)
        if args.headless
        else mujoco.viewer.launch_passive(
            runtime.model, runtime.data, key_callback=key_callback,
        )
    )
    with viewer_context as viewer:
        runtime.run(
            duration=duration,
            command=command,
            viewer=viewer,
            command_provider=command_provider,
            realtime=viewer is not None,
        )

    if args.diagnostics or args.headless:
        print("Diagnostics:", runtime.diagnostics_summary())


if __name__ == "__main__":
    main()
