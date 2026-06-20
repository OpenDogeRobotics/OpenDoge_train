"""
Sim2Sim engine — pure MuJoCo + ONNX Runtime, no Isaac Gym / ROS dependency.

Reuses sim2sim/_common.py for observation building, PD control, and policy input
packing.  Provides a step() API for external control loops (web server, etc.).
"""

from __future__ import annotations
import os
import glob
import time
import re
import numpy as np
import mujoco
import onnxruntime as ort
import yaml
from collections import deque
from io import BytesIO
from PIL import Image

from legged_gym import LEGGED_GYM_ROOT_DIR
from sim2sim._common import (
    quat_rotate_inverse,
    pd_control,
    build_policy_input,
    build_obs_raw,
)

# ── paths ────────────────────────────────────────────────────────────
_SIM2SIM_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "sim2sim")
_YAML_PATH = os.path.join(_SIM2SIM_DIR, "configs", "opendoge.yaml")
_DEFAULT_XML = os.path.join(
    LEGGED_GYM_ROOT_DIR, "resources", "robots", "Opendoge", "xml", "scene.xml"
)
_ONNX_DIRS = [
    os.path.join(LEGGED_GYM_ROOT_DIR, "onnx"),
]


class Sim2SimEngine:
    """MuJoCo + ONNX Runtime engine for OpenDoge sim2sim validation.

    Parameters
    ----------
    yaml_path : str
        Path to opendoge.yaml.
    xml_path : str or None
        MuJoCo scene XML.  Defaults to the project scene.xml.
    onnx_dirs : list[str] or None
        Directories scanned for .onnx files.
    """

    def __init__(self, yaml_path=None, xml_path=None, onnx_dirs=None):
        yaml_path = yaml_path or _YAML_PATH
        xml_path = xml_path or _DEFAULT_XML
        onnx_dirs = onnx_dirs or _ONNX_DIRS

        # ── load YAML ────────────────────────────────────────────────
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML config not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.sim_dt = float(cfg.get("simulation_dt", 0.005))
        self.control_decimation = int(cfg.get("control_decimation", 2))
        self.num_actions = int(cfg.get("num_actions", 12))
        self.num_obs = int(cfg.get("num_obs", 270))
        self.num_one_step_obs = int(cfg.get("num_one_step_obs", 45))
        self.init_base_height = float(cfg.get("init_base_height", 0.15))

        self.kps = np.array(cfg["kps"], dtype=np.float32)
        self.kds = np.array(cfg["kds"], dtype=np.float32)
        self.default_dof_pos = np.array(cfg["default_angles"], dtype=np.float32)
        self.ang_vel_scale = cfg["ang_vel_scale"]
        self.dof_pos_scale = cfg["dof_pos_scale"]
        self.dof_vel_scale = cfg["dof_vel_scale"]
        self.action_scale = cfg["action_scale"]
        self.cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)

        if len(self.default_dof_pos) != self.num_actions:
            raise ValueError("YAML default_angles length != num_actions")

        # ── resolve XML ──────────────────────────────────────────────
        self.xml_path = xml_path
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"MuJoCo XML not found: {self.xml_path}")

        # ── create MuJoCo model (offscreen) ──────────────────────────
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)

        # gyro sensor check
        self.use_gyro_sensor = True
        try:
            _ = self.data.sensor("angular-velocity").data
        except KeyError:
            self.use_gyro_sensor = False

        # ── renderer ─────────────────────────────────────────────────
        self.model.vis.global_.offwidth = 1280
        self.model.vis.global_.offheight = 720
        self._renderer = None
        self._render_w = 960
        self._render_h = 540
        self._jpeg_quality = 65

        # ── camera (mimics MuJoCo viewer controls) ────────────────────
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._cam.lookat[:] = [0.0, 0.0, 0.15]   # look at robot base
        self._cam.distance = 1.2
        self._cam.azimuth = 180.0                 # front view
        self._cam.elevation = -25.0               # slightly above
        self._has_viewer_clients = False          # lazy render flag

        # ── ONNX scanning ────────────────────────────────────────────
        self.onnx_dirs = onnx_dirs
        self._current_onnx = None
        self._ort_session = None
        self._input_name = None
        self._input_dim = None

        # ── runtime state ────────────────────────────────────────────
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_dof_pos.copy()
        self.step_counter = 0
        self._obs_history_buffer = None

        # initialise robot pose
        self._reset_pose()

        # load the first available model
        models = self.list_models()
        if models:
            self.load_model(models[0]["path"])

    # ═══════════════════════════════════════════════════════════════════
    # Model management
    # ═══════════════════════════════════════════════════════════════════

    def list_models(self) -> list[dict]:
        """Return available ONNX models sorted by mtime (newest first)."""
        results = []
        seen = set()
        for d in self.onnx_dirs:
            if not os.path.isdir(d):
                continue
            pattern = os.path.join(d, "*.onnx")
            for p in sorted(glob.glob(pattern)):
                abspath = os.path.abspath(p)
                if abspath in seen:
                    continue
                seen.add(abspath)
                st = os.stat(p)
                # try to extract training step from filename e.g. "model_4800"
                step_match = re.search(r"(\d{3,})", os.path.basename(p))
                results.append({
                    "name": os.path.basename(p).replace(".onnx", ""),
                    "path": abspath,
                    "size_kb": round(st.st_size / 1024, 1),
                    "mtime": st.st_mtime,
                    "step": int(step_match.group(1)) if step_match else None,
                })
        results.sort(key=lambda r: r["mtime"], reverse=True)
        return results

    def load_model(self, onnx_path: str) -> dict:
        """Hot-swap the ONNX policy.  Returns model info dict."""
        onnx_path = os.path.abspath(os.path.expanduser(onnx_path))
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")

        self._ort_session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        inp = self._ort_session.get_inputs()[0]
        self._input_name = inp.name
        self._input_dim = int(inp.shape[-1]) if isinstance(inp.shape[-1], int) else self.num_obs

        history_len = max(1, self.num_obs // self.num_one_step_obs)
        self._obs_history_buffer = deque(
            [np.zeros(self.num_one_step_obs, dtype=np.float32) for _ in range(history_len)],
            maxlen=history_len,
        )

        self._current_onnx = onnx_path
        self.cmd[:] = 0.0
        self.action[:] = 0.0

        return {
            "name": os.path.basename(onnx_path).replace(".onnx", ""),
            "path": onnx_path,
            "input_dim": self._input_dim,
            "num_actions": self.num_actions,
        }

    def get_current_model(self) -> dict | None:
        if self._current_onnx is None:
            return None
        return {
            "name": os.path.basename(self._current_onnx).replace(".onnx", ""),
            "path": self._current_onnx,
            "input_dim": self._input_dim,
            "num_actions": self.num_actions,
        }

    # ═══════════════════════════════════════════════════════════════════
    # Camera control  (mimics MuJoCo viewer interaction)
    # ═══════════════════════════════════════════════════════════════════

    def get_camera(self) -> dict:
        """Return current camera state."""
        return {
            "azimuth": round(float(self._cam.azimuth), 2),
            "elevation": round(float(self._cam.elevation), 2),
            "distance": round(float(self._cam.distance), 2),
            "lookat": [round(float(x), 3) for x in self._cam.lookat],
        }

    def set_camera(self, azimuth=None, elevation=None, distance=None,
                   lookat=None, lookat_delta=None,
                   delta_azimuth=0.0, delta_elevation=0.0,
                   delta_distance=0.0):
        """Update camera.

        Absolute values take priority over deltas.  Deltas are applied
        additively for mouse-drag style incremental updates.
        """
        if azimuth is not None:
            self._cam.azimuth = float(azimuth)
        else:
            self._cam.azimuth += float(delta_azimuth)

        if elevation is not None:
            self._cam.elevation = float(elevation)
        else:
            self._cam.elevation += float(delta_elevation)

        if distance is not None:
            self._cam.distance = float(distance)
        else:
            self._cam.distance += float(delta_distance)

        if lookat is not None and len(lookat) == 3:
            self._cam.lookat[:] = [float(x) for x in lookat]

        if lookat_delta is not None and len(lookat_delta) == 3:
            self._cam.lookat[0] += float(lookat_delta[0])
            self._cam.lookat[1] += float(lookat_delta[1])
            self._cam.lookat[2] += float(lookat_delta[2])

        # clamp
        self._cam.distance = max(0.2, min(10.0, self._cam.distance))
        self._cam.elevation = max(-89.0, min(89.0, self._cam.elevation))
        self._cam.azimuth %= 360.0

        return self.get_camera()

    def set_viewer_clients(self, has_clients: bool):
        """Tell the engine whether any web clients are viewing (for lazy render)."""
        self._has_viewer_clients = has_clients

    # ═══════════════════════════════════════════════════════════════════
    # Simulation control
    # ═══════════════════════════════════════════════════════════════════

    def reset(self):
        """Reset robot to the default standing pose."""
        self._reset_pose()
        self.cmd[:] = 0.0
        self.action[:] = 0.0
        self.target_dof_pos = self.default_dof_pos.copy()
        self.step_counter = 0
        if self._obs_history_buffer is not None:
            for i in range(len(self._obs_history_buffer)):
                self._obs_history_buffer[i][:] = 0.0

    def set_cmd(self, vx=0.0, vy=0.0, vyaw=0.0):
        """Update velocity command."""
        self.cmd[0] = float(vx)
        self.cmd[1] = float(vy)
        self.cmd[2] = float(vyaw)

    def step(self) -> dict:
        """Advance simulation by one control cycle."""
        if self._ort_session is None:
            return self.get_state()

        if self.step_counter % self.control_decimation == 0:
            obs_raw = build_obs_raw(
                data=self.data,
                default_dof_pos=self.default_dof_pos,
                cmd=self.cmd,
                cmd_scale=self.cmd_scale,
                ang_vel_scale=self.ang_vel_scale,
                dof_pos_scale=self.dof_pos_scale,
                dof_vel_scale=self.dof_vel_scale,
                action=self.action,
                num_actions=self.num_actions,
                use_gyro_sensor=self.use_gyro_sensor,
            )

            policy_input = build_policy_input(
                obs_raw=obs_raw,
                history_buffer=self._obs_history_buffer,
                input_dim=self._input_dim,
                num_obs=self.num_obs,
            )

            outputs = self._ort_session.run(
                None, {self._input_name: policy_input}
            )
            raw_action = np.clip(outputs[0][0], -10.0, 10.0)
            self.action = raw_action
            self.target_dof_pos = raw_action * self.action_scale + self.default_dof_pos

        tau = pd_control(
            self.target_dof_pos,
            self.data.qpos[7:7 + self.num_actions],
            self.kps,
            np.zeros_like(self.kds),
            self.data.qvel[6:6 + self.num_actions],
            self.kds,
        )

        if self.model.nu >= self.num_actions:
            tau_limit = np.abs(self.model.actuator_ctrlrange[:self.num_actions, 1])
            tau = np.clip(tau, -tau_limit, tau_limit)
        self.data.ctrl[:self.num_actions] = tau

        mujoco.mj_step(self.model, self.data)
        self.step_counter += 1

        return self.get_state()

    # ═══════════════════════════════════════════════════════════════════
    # State query
    # ═══════════════════════════════════════════════════════════════════

    def get_state(self) -> dict:
        """Return the full simulation state as a JSON-serialisable dict."""
        d = self.data

        qj = d.qpos[7:7 + self.num_actions].copy()
        dqj = d.qvel[6:6 + self.num_actions].copy()
        ctrl = d.ctrl[:self.num_actions].copy() if d.ctrl is not None else np.zeros(self.num_actions)

        base_pos = d.qpos[0:3].copy()
        base_quat = d.qpos[3:7].copy()
        base_lin_vel = d.qvel[0:3].copy()
        base_ang_vel = d.qvel[3:6].copy()

        feet_contact = np.zeros(4, dtype=bool)
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        for i, name in enumerate(foot_names):
            try:
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, name
                )
                if body_id >= 0:
                    for ci in range(d.ncon):
                        contact = d.contact[ci]
                        geom_ids = self.model.body_geomadr[body_id: body_id + 1]
                        if (contact.geom1 in geom_ids or contact.geom2 in geom_ids):
                            feet_contact[i] = True
                            break
            except Exception:
                pass

        return {
            "t": round(self.step_counter * self.sim_dt, 4),
            "step": self.step_counter,
            "model": os.path.basename(self._current_onnx or "").replace(".onnx", ""),
            "joint_pos": qj.tolist(),
            "joint_vel": dqj.tolist(),
            "joint_tau": ctrl.tolist(),
            "action": self.action.tolist(),
            "target_pos": self.target_dof_pos.tolist(),
            "base_pos": base_pos.tolist(),
            "base_quat": base_quat.tolist(),
            "base_lin_vel": base_lin_vel.tolist(),
            "base_ang_vel": base_ang_vel.tolist(),
            "cmd_vel": self.cmd.tolist(),
            "feet_contact": feet_contact.tolist(),
        }

    # ═══════════════════════════════════════════════════════════════════
    # Rendering
    # ═══════════════════════════════════════════════════════════════════

    def render(self, width=960, height=540, as_bytes=True, quality=None):
        """Offscreen render using the engine's camera, return JPEG bytes."""
        if quality is None:
            quality = self._jpeg_quality

        if self._renderer is None or width != self._render_w or height != self._render_h:
            self._renderer = mujoco.Renderer(self.model, height, width)
            self._render_w = width
            self._render_h = height

        self._renderer.update_scene(self.data, camera=self._cam)
        pixels = self._renderer.render()

        if not as_bytes:
            return pixels

        img = Image.fromarray(pixels)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    # ═══════════════════════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════════════════════

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ═══════════════════════════════════════════════════════════════════
    # Internals
    # ═══════════════════════════════════════════════════════════════════

    def _reset_pose(self):
        d = self.data
        d.qpos[7:7 + self.num_actions] = self.default_dof_pos
        d.qpos[2] = self.init_base_height
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)


# ── quick self-test ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    print("=== Sim2Sim Engine Self-Test ===\n")
    eng = Sim2SimEngine()

    print(f"Models: {len(eng.list_models())}")
    for m in eng.list_models():
        print(f"  {m['name']}  step={m.get('step')}")

    # camera test
    print(f"\nCamera: {eng.get_camera()}")
    eng.set_camera(delta_azimuth=45, delta_elevation=10)
    print(f"After rotate: {eng.get_camera()}")

    # run 200 steps
    eng.set_cmd(0.3, 0.0, 0.0)
    t0 = time.time()
    for _ in range(200):
        eng.step()
    print(f"\n200 steps in {time.time()-t0:.3f}s")

    # render test at multiple sizes
    for w, h in [(960, 540), (640, 360), (480, 270)]:
        jpeg = eng.render(w, h)
        print(f"Render {w}x{h}: {len(jpeg)} bytes")

    eng.close()
    print("\n=== Done ===")
