#!/usr/bin/env python3
"""PyQt5 sidebar controller with an embedded MuJoCo viewport for OpenDog V1.1.

The native GLFW viewer is NOT used. The simulator runs in a worker thread and
renders frames offscreen with ``mujoco.Renderer`` into a Qt canvas widget, so:

  * No MuJoCo built-in keyboard shortcuts exist anymore — every key is owned
    by this application (full isolation).
  * Camera (orbit / zoom / pan / track) and all commands are re-implemented
    here, cross-wired to the sidebar controls.
  * Telemetry is read directly from the live runtime in the same process.
  * The whole simulation can be hot-restarted without closing the panel.

Usage:
  python deploy/deploy_mujoco/opendoge_v1_1_panel.py
  python deploy/deploy_mujoco/opendoge_v1_1_panel.py \
      --onnx onnx/opendoge_v1_1_model_5200.onnx

Mouse (matches the native MuJoCo viewer):
  Drag left          orbit camera (Shift → rotate horizontal)
  Drag right         pan camera   (Shift → pan on horizontal plane)
  Middle drag / wheel  zoom camera (scroll up zooms OUT, as in MuJoCo)
  Double-click       track / untrack robot body

Keyboard:
  W/S/A/D/Q/E  drive        Space     pause sim
  X            emergency stop         R/Backspace  reset pose
  F            manual get-up

Forward speed is controlled by the sidebar slider only (0–4 m/s).
"""

import argparse
import math
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── PyQt5 ──

from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, QRectF
from PyQt5.QtGui import QImage, QPainter, QColor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QSlider, QLabel,
    QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox,
)


STYLE = """
QWidget {
    background-color: #f4f6f9;
    color: #26364a;
    font-size: 17px;
}
QGroupBox {
    border: 2px solid #c6d2de;
    border-radius: 10px;
    margin-top: 18px;
    padding-top: 18px;
    font-weight: bold;
    color: #26364a;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    color: #1f5f9e;
}
QPushButton {
    border: 2px solid #a9bccd;
    border-radius: 6px;
    background-color: #e7edf4;
    color: #26364a;
    font-weight: bold;
    min-height: 42px;
    font-size: 17px;
}
QPushButton:hover {
    background-color: #d5e2ef;
    border-color: #7f9db8;
}
QPushButton:pressed {
    background-color: #1f5f9e;
    border-color: #1f5f9e;
    color: #ffffff;
}
QPushButton#stopBtn {
    background-color: #2b5f8a;
    color: #ffffff;
    font-size: 20px;
    min-height: 52px;
}
QPushButton#stopBtn:hover { background-color: #1e4a6e; }
QSlider::groove:horizontal {
    border: 1px solid #b7c6d4; height: 12px;
    border-radius: 6px; background: #dfe6ee;
}
QSlider::handle:horizontal {
    background: #1f5f9e; width: 26px; height: 26px;
    margin: -8px 0; border-radius: 13px;
}
QSlider::sub-page:horizontal { background: #5b8fc0; border-radius: 6px; }
QLabel#speedLabel {
    color: #1f5f9e; font-size: 38px; font-weight: bold;
}
QLabel#statusLabel {
    color: #5b6b7c; font-size: 16px;
}
QLabel#teleKey { color: #5b6b7c; font-size: 16px; }
QLabel#teleVal { color: #26364a; font-size: 18px; font-weight: bold; }
"""


def _press_style():
    return ("background-color: #1f5f9e; color: #ffffff;"
            "border: 2px solid #1f5f9e; border-radius: 6px;"
            "font-weight: bold; min-height: 42px;")


class DPadButton(QPushButton):
    """Button that stays visually 'pressed' while the mouse is held down."""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        self.setStyleSheet(_press_style())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setStyleSheet("")
        super().mouseReleaseEvent(event)


# ── Qt ↔ worker bridge signals ──

class Bridge(QObject):
    frame = pyqtSignal(object)      # QImage
    telemetry = pyqtSignal(dict)    # snapshot dict for the sidebar
    sim_state = pyqtSignal(str)     # coarse status text
    error = pyqtSignal(str)


# ── Simulation worker thread ──

class SimWorker(threading.Thread):
    """Owns the simulator + offscreen renderer. Strictly one thread mutates
    the MjModel/MjData. Commands/camera come in from Qt through small locks."""

    SIM_RENDER_HZ = 60            # max frames/sec rendered to canvas (native viewer ≈ vsync 60 Hz)
    MAX_RENDER_W = 1920           # cap on offscreen render size (device px)
    MAX_RENDER_H = 1080

    def __init__(self, bridge, duration, action_delay_steps=0,
                 policy_path=None):
        super().__init__(daemon=True)
        self.bridge = bridge
        self.duration = duration   # None → run forever
        self.action_delay_steps = action_delay_steps
        self.policy_path = policy_path
        self.scene_mode = "flat"

        # command buffer (Qt writes, worker reads each sim step)
        self.cmd_lock = threading.Lock()
        self.cmd = np.zeros(3, dtype=np.float32)
        self.base_vx = 0.0

        # control flags (Qt writes atomically)
        self.ctrl_lock = threading.Lock()
        self.paused = False
        self.exit = False
        self.ctrl_pause_edge = False     # fire on press, not hold
        self.ctrl_getup_edge = False
        self.ctrl_reset_edge = False
        self.ctrl_stop_edge = False
        self.restart_req = False
        self.difficulty_now = None

        # camera deltas (Qt accumulates, worker applies+clears each frame).
        # dx/dy and pan_x/pan_y are device-pixel deltas (the canvas multiplies
        # Qt logical coords by the device pixel ratio, exactly like the native
        # viewer's buffer-to-window scaling); zoom_px is the device-pixel
        # delta for middle-drags; zoom_rel is the pre-normalized wheel delta.
        self.cam_lock = threading.Lock()
        self.cam_op = {"dx": 0.0, "dy": 0.0, "pan_x": 0.0, "pan_y": 0.0,
                       "zoom_px": 0.0, "zoom_rel": 0.0, "track_toggle": False,
                       "shift": False}

        # offscreen render target in device pixels (Qt tells us the canvas size)
        self._render_target_lock = threading.Lock()
        self._render_target = None     # (w, h) capped render buffer size
        self._canvas_dims = None       # (w, h) real canvas size in device px

        # telemetry (worker writes, Qt reads)
        self.tele = {}

        self.runtime = None
        self.renderer = None
        self.camera = None
        self._mvopt = None             # mujoco.MjvOption for camera scene
        self._mvscn = None             # mujoco.MjvScene used by mjv_moveCamera
        self._last_render = 0.0
        self._last_tele = 0.0

    # ── public control API (called from Qt thread) ──

    def set_command(self, vx, vy, vyaw, base_vx):
        with self.cmd_lock:
            self.cmd[0] = vx; self.cmd[1] = vy; self.cmd[2] = vyaw
        self.base_vx = base_vx

    def press_stop(self):
        with self.ctrl_lock:
            self.ctrl_stop_edge = True

    def toggle_pause(self):
        with self.ctrl_lock:
            self.ctrl_pause_edge = True

    def request_getup(self):
        with self.ctrl_lock:
            self.ctrl_getup_edge = True

    def request_reset(self):
        with self.ctrl_lock:
            self.ctrl_reset_edge = True

    def request_restart(self, difficulty):
        with self.ctrl_lock:
            self.restart_req = True

    def acc_cam(self, dx=0.0, dy=0.0, pan_x=0.0, pan_y=0.0,
                zoom_px=0.0, zoom_rel=0.0, track_toggle=False, shift=False):
        with self.cam_lock:
            c = self.cam_op
            c["dx"] += dx; c["dy"] += dy
            c["pan_x"] += pan_x; c["pan_y"] += pan_y
            c["zoom_px"] += zoom_px; c["zoom_rel"] += zoom_rel
            c["track_toggle"] = c["track_toggle"] or track_toggle
            c["shift"] = c["shift"] or shift

    def set_render_target(self, pw, ph):
        """Canvas size in device pixels; the renderer is re-sized to match so
        the displayed image is maximally crisp. Capped to keep fps sane. The
        uncapped canvas size is kept separately: the native viewer normalizes
        mouse deltas by the *actual* 3D viewport height, so camera math must
        use the real canvas height, never the capped render buffer height."""
        scale = min(self.MAX_RENDER_W / float(max(1, pw)),
                    self.MAX_RENDER_H / float(max(1, ph)), 1.0)
        w = int(round(pw * scale)); h = int(round(ph * scale))
        with self._render_target_lock:
            self._canvas_dims = (max(1, int(pw)), max(1, int(ph)))
            self._render_target = (max(160, w), max(120, h))

    def _current_render_size(self):
        with self._render_target_lock:
            return self._render_target

    def _canvas_height_px(self):
        """Real canvas height in device pixels (native viewer's r.height)."""
        with self._render_target_lock:
            if self._canvas_dims is not None:
                return float(self._canvas_dims[1])
        return float(self.renderer.height if self.renderer else 480)

    # ── internals ──

    def _read_command(self):
        with self.cmd_lock:
            return self.cmd.copy()

    def _make_runtime(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        from deploy.deploy_mujoco.deploy_mujoco import (
            OpenDogeSim2Sim, resolve_config_path,
        )
        self._safe_emit(self.bridge.sim_state, "加载策略与 V1.1 模型…")
        cfg = resolve_config_path("opendoge_v1_1.yaml")
        rt = OpenDogeSim2Sim(
            cfg,
            action_delay_steps=self.action_delay_steps,
            policy_path=self.policy_path,
        )
        self._safe_emit(self.bridge.sim_state, "模型已加载，创建 MuJoCo 渲染器…")
        rt.reset()
        return rt

    def _regenerate_terrain(self):
        self._safe_emit(self.bridge.sim_state, "加载 OpenDog V1.1 平地模型…")

    def _mj_camera_action(self, action, reldx, reldy):
        """Forward a mouse delta to MuJoCo's own camera mover — the exact
        function the native GLFW viewer calls — so rotation/pan/zoom feel
        identical to stock MuJoCo."""
        import mujoco
        try:
            mujoco.mjv_moveCamera(self.runtime.model, action, reldx, reldy,
                                  self._mvscn, self.camera)
        except Exception:
            pass

    def _refresh_camera_scene(self):
        """mjv_moveCamera's pan branch reads frustum data from an MjvScene;
        refresh it for the current camera pose (only needed when panning)."""
        import mujoco
        if self._mvscn is None or self._mvopt is None:
            return
        try:
            mujoco.mjv_updateScene(
                self.runtime.model, self.runtime.data, self._mvopt, None,
                self.camera, mujoco.mjtCatBit.mjCAT_ALL, self._mvscn)
        except Exception:
            pass

    def _apply_camera(self):
        import mujoco
        if self.camera is None:
            return
        with self.cam_lock:
            dx, dy = self.cam_op["dx"], self.cam_op["dy"]
            pan_x, pan_y = self.cam_op["pan_x"], self.cam_op["pan_y"]
            zoom_px, zoom_rel = self.cam_op["zoom_px"], self.cam_op["zoom_rel"]
            track_toggle = self.cam_op["track_toggle"]
            shift = self.cam_op["shift"]
            self.cam_op = {"dx": 0.0, "dy": 0.0, "pan_x": 0.0,
                           "pan_y": 0.0, "zoom_px": 0.0, "zoom_rel": 0.0,
                           "track_toggle": False, "shift": False}

        # viewport height in *device pixels of the real canvas*; deltas arrive
        # in the same units, so reldx = dx/height matches the native viewer
        # exactly (simulate.cc: mjv_moveCamera(..., state->dx / r.height, ...)).
        # Never use the (capped) render buffer height here, or drags would get
        # progressively faster than native as the window grows past the cap.
        vh = self._canvas_height_px()

        # orbit — native left-drag, Shift → ROTATE_H (identical behaviour to
        # ROTATE_V in MuJoCo 3.x, kept for mapping fidelity)
        if dx or dy:
            act = (mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift
                   else mujoco.mjtMouse.mjMOUSE_ROTATE_V)
            self._mj_camera_action(act, dx / vh, -dy / vh)

        # pan — native right-drag, Shift → MOVE_H (horizontal plane pan);
        # the lookat of a tracking camera is left alone.
        if (pan_x or pan_y) and self.camera.trackbodyid == -1:
            self._refresh_camera_scene()
            act = (mujoco.mjtMouse.mjMOUSE_MOVE_H if shift
                   else mujoco.mjtMouse.mjMOUSE_MOVE_V)
            self._mj_camera_action(act, pan_x / vh, -pan_y / vh)

        # zoom (native middle-drag & wheel → mjMOUSE_ZOOM). MuJoCo's wheel
        # convention is kept: scrolling up / dragging down zooms OUT.
        z = -zoom_px / vh + zoom_rel
        if z:
            self._mj_camera_action(mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, z)

        # track toggle — a camera only follows a body when its type is
        # mjCAMERA_TRACKING (like the native viewer's "Track" command)
        if track_toggle and self.camera.trackbodyid == -1:
            self.camera.trackbodyid = self.runtime.base_body_id
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        elif track_toggle:
            self.camera.trackbodyid = -1
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.lookat = self.runtime.data.qpos[:3].copy()

    def _render(self):
        import mujoco
        if self.renderer is None or self.runtime is None:
            return
        # re-size the offscreen framebuffer to track the canvas (crisp output).
        # Hysteresis avoids rebuild storms while the user drags the window edge.
        target = self._current_render_size()
        if target is not None:
            tw, th = target
            if (abs(self.renderer.width - tw) > 8
                    or abs(self.renderer.height - th) > 8):
                try:
                    self.renderer.close()
                except Exception:
                    pass
                self.renderer = mujoco.Renderer(self.runtime.model,
                                                height=th, width=tw)
        try:
            self.renderer.update_scene(
                self.runtime.data, self.camera, scene_option=self._mvopt
            )
            img = self.renderer.render()
            h, w = img.shape[:2]
            qimg = QImage(img.data, w, h, 3 * w,
                          QImage.Format_RGB888).copy()
            self._safe_emit(self.bridge.frame, qimg)
        except Exception:
            pass   # transient GL hiccups are not fatal

    def _telemetry(self):
        rt = self.runtime
        if rt is None:
            return
        try:
            import mujoco
            from deploy.deploy_mujoco.deploy_mujoco import (
                get_gravity_orientation,
            )
            sim_time = float(rt.data.time) - float(
                getattr(rt, "run_start_time", rt.data.time))
            base_z = float(rt.data.qpos[2])
            tilt = float(np.linalg.norm(
                get_gravity_orientation(rt.data.qpos[3:7].copy())[:2]))
        except Exception:
            return
        diag = getattr(rt, "diagnostics", None)
        fall = diag is not None and diag.fall_time is not None
        recovery = bool(getattr(rt, "_in_recovery", False))
        max_torque = float("nan")
        avg_vel = (float("nan"), float("nan"))
        if diag is not None:
            max_torque = diag.max_raw_torque
            if diag.velocity_samples > 0:
                avg_vel = (diag.mean_body_velocity[0],
                           diag.mean_body_velocity[1])
        with self.cmd_lock:
            vx, vy, vyaw = (float(x) for x in self.cmd)
        self.tele.update(
            sim_time=sim_time, z=base_z, tilt=tilt, fall=fall,
            recovery=recovery, torque=max_torque,
            vel=avg_vel, vx=vx, vy=vy, vyaw=vyaw,
            base=self.base_vx, paused=self.paused,
        )

    def _safe_emit(self, sig, *args):
        """Emit a queued signal from the worker thread without letting a
        transient Qt error (e.g. a slot being mid-teardown) kill the loop."""
        try:
            sig.emit(*args)
        except RuntimeError:
            pass

    def run(self):
        import mujoco   # created in this worker thread; owned here

        def build():
            self._regenerate_terrain()
            rt = self._make_runtime()
            # raise the offscreen framebuffer cap so we can render sharply
            rt.model.vis.global_.offwidth = self.MAX_RENDER_W
            rt.model.vis.global_.offheight = self.MAX_RENDER_H
            self._mvopt = mujoco.MjvOption()
            # Collision geoms are group 0, visual meshes group 1, and the
            # checker floor group 2. Keep physics unchanged while rendering
            # only the mesh and the environment.
            self._mvopt.geomgroup[:] = 0
            self._mvopt.geomgroup[1] = 1
            self._mvopt.geomgroup[2] = 1
            self._mvscn = mujoco.MjvScene(
                rt.model, maxgeom=max(1000, 2 * rt.model.ngeom))
            tw, th = self._current_render_size() or (1280, 800)
            renderer = mujoco.Renderer(rt.model, height=th, width=tw)
            cam = mujoco.MjvCamera()
            # Match the non-embedded MuJoCo viewer's initial orientation:
            # azimuth/elevation come from the model's <visual><global ...>
            # via mjv_defaultFreeCamera (the function the native viewer's
            # AlignAndScaleView calls), not from hand-picked angles.
            try:
                mujoco.mjv_defaultFreeCamera(rt.model, cam)
            except Exception:
                mujoco.mjv_defaultCamera(cam)
            # Keep the embedded panel's close framing on the robot start
            # (spawn x/y come from the scene config, e.g. Robocon right lane).
            try:
                sx, sy = (float(v) for v in rt.cfg.get("initial_base_pos", [0.0, 0.0]))
            except Exception:
                sx, sy = 0.0, 0.0
            cam.distance = 0.8
            cam.lookat = np.array([sx, sy, 0.12])
            self._safe_emit(self.bridge.sim_state, "仿真运行中")
            return rt, renderer, cam

        try:
            self.runtime, self.renderer, self.camera = build()
        except Exception as exc:
            traceback.print_exc()
            self._safe_emit(self.bridge.error,
                            f"仿真初始化失败: {type(exc).__name__}: {exc}")
            return

        self._last_render = time.perf_counter()

        acc = 0.0
        prev = time.perf_counter()
        render_interval = 1.0 / self.SIM_RENDER_HZ
        last_render = 0.0

        try:
            while not self.exit:
                # ── control edges ──
                with self.ctrl_lock:
                    pause_edge = self.ctrl_pause_edge; self.ctrl_pause_edge = False
                    getup_edge = self.ctrl_getup_edge; self.ctrl_getup_edge = False
                    reset_edge = self.ctrl_reset_edge; self.ctrl_reset_edge = False
                    stop_edge = self.ctrl_stop_edge; self.ctrl_stop_edge = False
                    restart = self.restart_req
                    self.restart_req = None
                if stop_edge:
                    with self.cmd_lock:
                        self.cmd[:] = 0.0
                    self.base_vx = 0.0
                if pause_edge:
                    self.paused = not self.paused
                    self._safe_emit(
                        self.bridge.sim_state,
                        "已暂停 ⏸" if self.paused else "仿真运行中")
                if getup_edge:
                    if self.runtime is not None and self.runtime.getup_session is not None:
                        self.runtime._in_recovery = True
                        self.runtime._recovery_stable_steps = 0
                        print("[panel] 手动触发起身")
                    else:
                        print("[panel] 无 getup 策略，忽略")
                if reset_edge and self.runtime is not None:
                    self.runtime.reset()
                    print("[panel] 已重置姿态")
                if restart:
                    print("[panel] 重启 OpenDog V1.1")
                    # tear down, rebuild
                    try:
                        del self.renderer
                    except Exception:
                        pass
                    self.renderer = None
                    self.runtime, self.renderer, self.camera = build()
                    self._safe_emit(self.bridge.sim_state, "V1.1 模型已重启")

                # ── camera ──
                self._apply_camera()

                # ── sim stepping (realtime-paced, bounded catch-up) ──
                now = time.perf_counter()
                wall = now - prev; prev = now
                if not self.paused:
                    acc += wall
                    rt_dt = float(self.runtime.sim_dt)
                    cmd = self._read_command()
                    max_steps = max(1, int(0.04 / rt_dt))
                    n = 0
                    while acc > rt_dt and n < max_steps:
                        self.runtime.step(cmd)
                        acc -= rt_dt
                        n += 1
                    if acc > 0.04:
                        acc = 0.0   # lost realtime budget; don't spiral
                    # enforce max duration
                    if self.duration is not None:
                        elapsed = (float(self.runtime.data.time)
                                   - float(getattr(self.runtime, "run_start_time",
                                                   self.runtime.data.time)))
                        if elapsed >= self.duration:
                            self.paused = True
                            self._safe_emit(
                                self.bridge.sim_state,
                                f"时间到 ({self.duration:.0f}s) 已暂停")

                # ── render (rate-limited) ──
                if now - last_render >= render_interval:
                    last_render = now
                    self._render()

                # telemetry @ ~10Hz
                if now - self._last_tele >= 0.1:
                    self._last_tele = now
                    self._telemetry()
                    self._safe_emit(self.bridge.telemetry,
                                    dict(self.tele))

                time.sleep(0.001)
        except Exception as exc:
            traceback.print_exc()
            self._safe_emit(self.bridge.error,
                            f"{type(exc).__name__}: {exc}")
        finally:
            self._safe_emit(self.bridge.sim_state, "仿真已停止")
            try:
                self.renderer.close()
            except Exception:
                pass


# ── Canvas widget ──

class SimCanvas(QWidget):
    """Renders the latest frame; owns all canvas mouse & keyboard events.

    Mouse mapping matches the native MuJoCo GLFW viewer (simulate.cc):
      left drag        orbit (Shift → rotate-horizontal)
      right drag       pan   (Shift → pan on the horizontal plane)
      middle drag/wheel zoom (2% of viewport height per notch)
      double-click     track / untrack the robot body
    Keyboard:
      W/S/A/D/Q/E  drive · Space pause · X stop · R reset · F getup
    Forward speed comes from the sidebar slider only.
    """

    drive_flag = pyqtSignal(str, bool)   # (direction, pressed)
    action = pyqtSignal(str)             # pause / stop / reset / getup
    cam = pyqtSignal(dict)
    resized = pyqtSignal(int, int)       # device-pixel canvas size (w, h)

    KEY_MAP = {
        Qt.Key_W: "fwd", Qt.Key_S: "back",
        Qt.Key_A: "left", Qt.Key_D: "right",
        Qt.Key_Q: "turn_l", Qt.Key_E: "turn_r",
    }

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(480, 360)
        self._frame = None
        self._drag = None   # (button, last_pos)

    def set_frame(self, qimg):
        self._frame = qimg
        self.update()

    def _dpr(self):
        try:
            return float(self.devicePixelRatio())
        except Exception:
            return 1.0

    # paint
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#dfe5ec"))
        if self._frame is not None:
            # keep aspect ratio
            dst = QRectF(self.rect()).adjusted(0, 0, -1, -1)
            src = self._frame.rect()
            scaled = src
            if (dst.width() / dst.height()) > (src.width() / src.height()):
                dst.setHeight(dst.height());
                nh = dst.width() * src.height() / src.width()
                dst = QRectF(dst.x(), dst.y() + (dst.height() - nh) / 2,
                             dst.width(), nh)
            else:
                nw = dst.height() * src.width() / src.height()
                dst = QRectF(dst.x() + (dst.width() - nw) / 2, dst.y(),
                             nw, dst.height())
            p.drawImage(dst, self._frame, QRectF(self._frame.rect()))

    # mouse — button mapping matches the native MuJoCo GLFW viewer:
    #   left = orbit, right = pan, middle = zoom, wheel = zoom.
    # Coordinates follow the native glfw adapter exactly: cursor x/y are
    # converted to the OpenGL convention (y up, i.e. y = height - qt_y)
    # *before* computing deltas, and deltas are scaled to device pixels
    # (× devicePixelRatio) so reldx = dx/height matches the native
    # buffer-to-window math. With y up, dragging DOWN gives negative dy and
    # reldy = -dy/height is positive → elevation decreases → view tilts DOWN,
    # exactly like the non-embedded viewer.
    def mousePressEvent(self, event):
        self.setFocus()
        pos = (event.x(), self.height() - event.y())   # OpenGL y (up positive)
        if event.button() == Qt.LeftButton:
            self._drag = ("rot", pos)
        elif event.button() == Qt.RightButton:
            self._drag = ("pan", pos)
        elif event.button() == Qt.MiddleButton:
            self._drag = ("zoom", pos)

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        mode, last = self._drag
        dpr = self._dpr()
        dx = (event.x() - last[0]) * dpr
        y_gl = self.height() - event.y()               # OpenGL y (up positive)
        dy = (y_gl - last[1]) * dpr
        self._drag = (mode, (event.x(), y_gl))
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        if mode == "rot":
            self.cam.emit({"dx": dx, "dy": dy, "shift": shift})
        elif mode == "pan":
            self.cam.emit({"pan_x": dx, "pan_y": dy, "shift": shift})
        else:   # middle-drag zoom: like the native viewer, only dy matters
            self.cam.emit({"zoom_px": dy})

    def mouseReleaseEvent(self, event):
        self._drag = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Native step = 2% of viewport height per notch. MuJoCo's wheel is
        # signed so that scrolling up zooms OUT. The native viewer also
        # scales the notch by the buffer-to-window ratio (devicePixelRatio).
        notches = event.angleDelta().y() / 120.0
        self.cam.emit({"zoom_rel": -0.02 * notches * self._dpr()})

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._emit_render_size()

    def showEvent(self, event):
        super().showEvent(event)
        self._emit_render_size()

    def _emit_render_size(self):
        dpr = 1.0
        try:
            dpr = self.devicePixelRatio()
        except Exception:
            pass
        self.resized.emit(int(self.width() * dpr), int(self.height() * dpr))

    def mouseDoubleClickEvent(self, event):
        self.cam.emit({"track_toggle": True})

    # keyboard — speed comes from the sidebar slider only; keys just steer
    def keyPressEvent(self, event):
        k = event.key()
        if k in self.KEY_MAP:
            self.drive_flag.emit(self.KEY_MAP[k], True)
        elif k == Qt.Key_Space:
            self.action.emit("pause")
        elif k == Qt.Key_X:
            self.action.emit("stop")
        elif k in (Qt.Key_R, Qt.Key_Backspace):
            self.action.emit("reset")
        elif k == Qt.Key_F:
            self.action.emit("getup")
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        k = event.key()
        if k in self.KEY_MAP:
            self.drive_flag.emit(self.KEY_MAP[k], False)
        else:
            super().keyReleaseEvent(event)


# ── Main window ──

class ControlPanel(QWidget):
    def __init__(self, worker: SimWorker, bridge: Bridge):
        super().__init__()
        self.w = worker
        self.bridge = bridge

        self.setWindowTitle("OpenDog V1.1 — MuJoCo（PyQt 内嵌）")
        self.resize(1280, 760)
        self.setMinimumSize(1024, 640)

        self._flags = {"fwd": False, "back": False, "left": False,
                       "right": False, "turn_l": False, "turn_r": False}
        self._vx = 0.0
        self._vy = 0.0
        self._vyaw = 0.0
        self._base_vx = 0.0

        self._init_ui()
        self._setup_timers()
        self._connect_bridge()

    # ── UI ──

    def _build_sidebar(self):
        side = QVBoxLayout()
        side.setSpacing(8)
        side.setContentsMargins(12, 10, 12, 10)

        # speed — the slider is the ONLY speed control (0 ~ 4.0 m/s)
        sg = QGroupBox("前进速度")
        sl = QVBoxLayout()
        self.speed_label = QLabel("0.00 m/s")
        self.speed_label.setObjectName("speedLabel")
        self.speed_label.setAlignment(Qt.AlignCenter)
        sl.addWidget(self.speed_label)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 80)          # 0 ~ 4.0 m/s
        self.speed_slider.valueChanged.connect(self._on_slider)
        sl.addWidget(self.speed_slider)
        sg.setLayout(sl)
        side.addWidget(sg)

        # telemetry
        tg = QGroupBox("遥测")
        te = QGridLayout()
        te.setHorizontalSpacing(8); te.setVerticalSpacing(3)
        self.status_lamp = QLabel("● 未连接")
        self.status_lamp.setAlignment(Qt.AlignCenter)
        self.status_lamp.setStyleSheet("color:#8a97a5;font-size:18px;font-weight:bold;")
        te.addWidget(self.status_lamp, 0, 0, 1, 3)
        self.t_sim = self._tele_pair(te, 1, "仿真时间", "–")
        self.t_z = self._tele_pair(te, 2, "机身高度", "–")
        self.t_tilt = self._tele_pair(te, 3, "倾角", "–")
        self.t_torque = self._tele_pair(te, 4, "最大扭矩", "–")
        self.t_vel = self._tele_pair(te, 5, "平均速度", "–")
        self.t_cmd = self._tele_pair(te, 6, "速度指令", "–")
        tg.setLayout(te)
        side.addWidget(tg)

        model_group = QGroupBox("模型")
        model_layout = QVBoxLayout()
        model_label = QLabel("OpenDog V1.1 · MuJoCo URDF")
        model_label.setAlignment(Qt.AlignCenter)
        model_label.setStyleSheet("color:#1f5f9e;font-weight:bold;")
        model_layout.addWidget(model_label)
        reload_btn = QPushButton("⟳ 重启 V1.1 仿真")
        reload_btn.clicked.connect(self._restart_sim)
        model_layout.addWidget(reload_btn)
        model_group.setLayout(model_layout)
        side.addWidget(model_group)

        # move
        mg = QGroupBox("移动")
        grid = QGridLayout(); grid.setSpacing(6)
        self.btn_fwd = DPadButton("▲ 前进")
        _press_link = lambda d, b_=self.btn_fwd: (
            b_.pressed.connect(lambda: self._set_flag(d, True)),
            b_.released.connect(lambda: self._set_flag(d, False)))
        _press_link("fwd")
        grid.addWidget(self.btn_fwd, 0, 1)
        self.btn_left = DPadButton("◀ 左移")
        _press_link = lambda d, b_=self.btn_left: (
            b_.pressed.connect(lambda: self._set_flag(d, True)),
            b_.released.connect(lambda: self._set_flag(d, False)))
        _press_link("left")
        grid.addWidget(self.btn_left, 1, 0)
        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.clicked.connect(self._stop)
        grid.addWidget(self.btn_stop, 1, 1)
        self.btn_right = DPadButton("右移 ▶")
        _press_link = lambda d, b_=self.btn_right: (
            b_.pressed.connect(lambda: self._set_flag(d, True)),
            b_.released.connect(lambda: self._set_flag(d, False)))
        _press_link("right")
        grid.addWidget(self.btn_right, 1, 2)
        self.btn_back = DPadButton("▼ 后退")
        _press_link = lambda d, b_=self.btn_back: (
            b_.pressed.connect(lambda: self._set_flag(d, True)),
            b_.released.connect(lambda: self._set_flag(d, False)))
        _press_link("back")
        grid.addWidget(self.btn_back, 2, 1)
        mg.setLayout(grid)
        side.addWidget(mg)

        # rotation + aux
        rg = QGroupBox("转向 / 辅助")
        rh = QHBoxLayout()
        self.btn_left_turn = DPadButton("↺ 左转")
        self.btn_left_turn.pressed.connect(lambda: self._set_flag("turn_l", True))
        self.btn_left_turn.released.connect(lambda: self._set_flag("turn_l", False))
        rh.addWidget(self.btn_left_turn)
        self.btn_right_turn = DPadButton("右转 ↻")
        self.btn_right_turn.pressed.connect(lambda: self._set_flag("turn_r", True))
        self.btn_right_turn.released.connect(lambda: self._set_flag("turn_r", False))
        rh.addWidget(self.btn_right_turn)
        rg.setLayout(rh)
        side.addWidget(rg)

        ah = QHBoxLayout()
        self.btn_getup = QPushButton("起身")
        self.btn_getup.clicked.connect(lambda: self.w.request_getup())
        ah.addWidget(self.btn_getup)
        self.btn_reset = QPushButton("重置姿态")
        self.btn_reset.clicked.connect(lambda: self.w.request_reset())
        ah.addWidget(self.btn_reset)
        self.btn_pause = QPushButton("暂停⏸")
        self.btn_pause.clicked.connect(lambda: self.w.toggle_pause())
        ah.addWidget(self.btn_pause)
        side.addLayout(ah)

        self.status_label = QLabel("就绪 — 先设速度，再按住▲ 前进")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        side.addWidget(self.status_label)
        side.addStretch()
        return side

    def _tele_pair(self, grid, row, key, value):
        k = QLabel(key); k.setObjectName("teleKey")
        v = QLabel(value); v.setObjectName("teleVal")
        grid.addWidget(k, row, 0)
        grid.addWidget(v, row, 1, 1, 2)
        return v

    def _init_ui(self):
        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        side = QVBoxLayout()
        side.addLayout(self._build_sidebar())
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(400)
        root.addWidget(side_widget)
        self.canvas = SimCanvas()
        root.addWidget(self.canvas, 1)
        self.setLayout(root)

        # canvas -> panel wiring
        self.canvas.drive_flag.connect(self._set_flag)
        self.canvas.action.connect(self._on_canvas_action)
        self.canvas.cam.connect(self._on_cam)
        self.canvas.resized.connect(self.w.set_render_target)

    def _connect_bridge(self):
        self.bridge.frame.connect(self.canvas.set_frame)
        self.bridge.telemetry.connect(self._update_telemetry)
        self.bridge.sim_state.connect(self._set_sim_state)
        self.bridge.error.connect(self._on_sim_error)

    def _setup_timers(self):
        self.cmd_timer = QTimer()
        self.cmd_timer.timeout.connect(self._tick)
        self.cmd_timer.start(20)          # 50 Hz control
        self.canvas.setFocus()

    # ── handlers ──

    def _on_canvas_action(self, action):
        if action == "pause":
            self.w.toggle_pause()
        elif action == "stop":
            self._stop()
        elif action == "reset":
            self.w.request_reset()
        elif action == "getup":
            self.w.request_getup()

    def _on_cam(self, op):
        self.w.acc_cam(**op)

    def _on_sim_error(self, msg):
        self.status_label.setText(f"仿真异常: {msg}")

    def _set_sim_state(self, text):
        self.status_lamp.setStyleSheet(
            "color:#8a97a5;font-size:17px;font-weight:bold;")
        self.status_lamp.setText(f"● {text}")

    def _on_slider(self, val):
        # the slider is the only speed control
        self._base_vx = val / 20.0
        self.speed_label.setText(f"{self._base_vx:.2f} m/s")

    def _restart_sim(self):
        self._stop()
        self.w.request_restart(None)
        self.status_label.setText("重启仿真 → OpenDog V1.1")

    def _set_flag(self, direction, pressed):
        self._flags[direction] = pressed

    def _stop(self):
        for k in self._flags:
            self._flags[k] = False
        self._vx = 0.0; self._vy = 0.0; self._vyaw = 0.0
        self._base_vx = 0.0
        self.speed_slider.blockSignals(True)
        self.speed_slider.setValue(0)
        self.speed_slider.blockSignals(False)
        self.speed_label.setText("0.00 m/s")
        self.w.press_stop()
        self.status_label.setText("已停止")

    def _tick(self):
        # Keep the command ramp aligned with the HIMloco PyQt controller:
        #   vx/vy ramp at 0.08 per 50 Hz tick, vyaw at 0.15, decay 0.90;
        #   OMNIDIRECTIONAL speed cap: ‖(vx,vy)‖ ≤ slider speed (0–4.0 m/s),
        #   yaw cap 1.5. Slider at 0 ⇒ no horizontal motion in any direction.
        accel = 0.08
        vyaw_accel = 0.15
        decay = 0.90
        max_speed = max(self._base_vx, 0.0)  # omnidirectional speed ceiling = slider
        max_vyaw = 1.5

        if self._flags["fwd"]:
            # forward speed is exactly the slider value — no other speed
            # factor applies (slider at 0 ⇒ no forward command)
            self._vx = min(self._vx + accel, max_speed)
        elif self._flags["back"]:
            self._vx = max(self._vx - accel, -max_speed)
        else:
            self._vx *= decay

        if self._flags["left"]:
            self._vy = min(self._vy + accel, max_speed)
        elif self._flags["right"]:
            self._vy = max(self._vy - accel, -max_speed)
        else:
            self._vy *= decay

        if self._flags["turn_l"]:
            self._vyaw = min(self._vyaw + vyaw_accel, max_vyaw)
        elif self._flags["turn_r"]:
            self._vyaw = max(self._vyaw - vyaw_accel, -max_vyaw)
        else:
            self._vyaw *= decay

        for n in ("_vx", "_vy", "_vyaw"):
            if abs(getattr(self, n)) < 0.01:
                setattr(self, n, 0.0)

        # Omnidirectional speed limiting: cap the horizontal velocity VECTOR,
        # not each axis separately — diagonal motion cannot exceed the slider.
        speed = math.hypot(self._vx, self._vy)
        if speed > max_speed and speed > 1e-6:
            scale = max_speed / speed
            self._vx *= scale
            self._vy *= scale

        now_base = self._base_vx
        self.w.set_command(self._vx, self._vy, self._vyaw, now_base)

        parts = []
        for ln, v in (("vx", self._vx), ("vy", self._vy), ("yaw", self._vyaw)):
            if abs(v) > 0.01:
                parts.append(f"{ln}={v:.2f}")
        self.status_label.setText(
            f"基速 {now_base:.2f} | " + (" ".join(parts) if parts else "待命"))

    def _update_telemetry(self, t):
        if t["fall"]:
            color, text = "#b03a2e", "● 已跌倒"
        elif t["recovery"]:
            color, text = "#b9770e", "● 起身中…"
        elif t["paused"]:
            color, text = "#8a97a5", "● 暂停"
        else:
            color, text = "#1e8449", "● 正常运行"
        self.status_lamp.setStyleSheet(
            f"color:{color};font-size:18px;font-weight:bold;")
        self.status_lamp.setText(text)
        self.t_sim.setText(f"{t['sim_time']:.1f} s")
        self.t_z.setText(f"{t['z']:.3f} m")
        self.t_tilt.setText(f"{t['tilt']:.3f}")
        self.t_torque.setText(f"{t['torque']:.1f} N·m"
                              if t["torque"] == t["torque"] else "–")
        v0, v1 = t["vel"]
        self.t_vel.setText(f"({v0:.2f}, {v1:.2f}) m/s" if v0 == v0 else "–")
        self.t_cmd.setText(f"vx={t['vx']:.2f} vy={t['vy']:.2f} yaw={t['vyaw']:.2f}")

    def closeEvent(self, event):
        self.w.exit = True
        # wait briefly for the worker to shut down while the Bridge is still
        # alive, so its final "仿真已停止" emit does not race object teardown
        try:
            self.w.join(timeout=1.0)
        except Exception:
            pass
        event.accept()


BANNER = """
╔══════════════════════════════════════════════════╗
║  OpenDog V1.1 MuJoCo — PyQt 侧边栏（内嵌）        ║
╠══════════════════════════════════════════════════╣
║  左栏控制 / 右侧实时仿真画面（无 MuJoCo 原生窗口） ║
║  鼠标规则与原生 MuJoCo 完全一致；渲染分辨率自适应  ║
╚══════════════════════════════════════════════════╝"""


def main():
    parser = argparse.ArgumentParser(description="OpenDog V1.1 PyQt 内嵌控制面板")
    parser.add_argument("--onnx", type=str, default=None,
                        help="Override the V1.1 ONNX policy path")
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    print(BANNER)

    bridge = Bridge()
    worker = SimWorker(bridge, duration=args.duration, policy_path=args.onnx)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)

    panel = ControlPanel(worker, bridge)
    panel.show()
    panel.raise_()
    panel.activateWindow()
    panel.canvas.setFocus()

    worker.start()

    print("  面板已启动，开始控制吧！\n")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
