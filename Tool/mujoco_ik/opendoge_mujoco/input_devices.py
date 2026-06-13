"""Unified input layer — keyboard (X11/MuJoCo callback) and gamepad (evdev).

All input sources expose a common interface:
    snapshot() -> tuple[BodyCommand, bool, bool, bool]
        returns (command, turn_mode_active, exit_requested, reset_requested)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import threading
import time
from typing import Optional

from opendoge_mujoco.action_gait import BodyCommand

# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def _clamp_cmd(cmd: BodyCommand) -> BodyCommand:
    n = math.hypot(cmd.vx, cmd.vy)
    if n > 1.0:
        cmd.vx /= n
        cmd.vy /= n
    cmd.vx = max(-1.0, min(1.0, cmd.vx))
    cmd.vy = max(-1.0, min(1.0, cmd.vy))
    cmd.yaw = max(-1.0, min(1.0, cmd.yaw))
    return cmd


# ═══════════════════════════════════════════════════════════════
# Keyboard — X11 direct polling (zero-dependency, real press/release)
# ═══════════════════════════════════════════════════════════════

KEY_ESCAPE, KEY_SPACE = 256, 32
KEY_LEFT, KEY_RIGHT, KEY_DOWN, KEY_UP = 263, 262, 264, 265
KEY_LEFT_CTRL, KEY_RIGHT_CTRL = 341, 345

XK_ESCAPE, XK_SPACE = 0xFF1B, 0x0020
XK_R, XK_R_L = 0x0052, 0x0072
XK_LEFT, XK_UP, XK_RIGHT, XK_DOWN = 0xFF51, 0xFF52, 0xFF53, 0xFF54
XK_CTRL_L, XK_CTRL_R = 0xFFE3, 0xFFE4


class X11Keyboard:
    """Polls physical key state via Xlib — real press/release, no key-repeat artifacts."""

    def __init__(self):
        lib = ctypes.util.find_library("X11")
        if not lib:
            raise RuntimeError("libX11 not found")
        self._x11 = ctypes.cdll.LoadLibrary(lib)
        self._x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._x11.XOpenDisplay.restype = ctypes.c_void_p
        self._x11.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._x11.XQueryKeymap.restype = ctypes.c_int
        self._x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self._x11.XKeysymToKeycode.restype = ctypes.c_uint
        self.dpy = self._x11.XOpenDisplay(None)
        if not self.dpy:
            raise RuntimeError("cannot open X11 display")
        self._kc = {n: int(self._x11.XKeysymToKeycode(self.dpy, ks))
                    for n, ks in [("esc", XK_ESCAPE), ("space", XK_SPACE),
                                  ("r", XK_R), ("rl", XK_R_L),
                                  ("up", XK_UP), ("down", XK_DOWN),
                                  ("left", XK_LEFT), ("right", XK_RIGHT),
                                  ("cl", XK_CTRL_L), ("cr", XK_CTRL_R)]}
        self._last_r = False

    def close(self):
        if self.dpy:
            self._x11.XCloseDisplay(self.dpy)
            self.dpy = None

    def snapshot(self) -> tuple[BodyCommand, bool, bool, bool]:
        km = ctypes.create_string_buffer(32)
        if not self._x11.XQueryKeymap(self.dpy, km):
            return BodyCommand(), False, False, False
        p = {n: bool(km.raw[k >> 3] & (1 << (k & 7))) for n, k in self._kc.items() if k > 0}
        cmd = BodyCommand()
        cmd.vx = float(p.get("up", False)) - float(p.get("down", False))
        h = float(p.get("left", False)) - float(p.get("right", False))
        turn = p.get("cl", False) or p.get("cr", False)
        if turn:
            cmd.yaw = h
        else:
            cmd.vy = h
        rst = (p.get("r", False) or p.get("rl", False)) and not self._last_r
        self._last_r = p.get("r", False) or p.get("rl", False)
        if p.get("space", False):
            cmd = BodyCommand()
        return _clamp_cmd(cmd), turn, p.get("esc", False), rst


class MuJoCoKeyCallback:
    """Fallback: receives MuJoCo viewer key events. Uses timeout for key release."""

    def __init__(self, timeout: float = 0.18):
        self._lock = threading.Lock()
        self._timeout = timeout
        self._times: dict[int, float] = {}
        self._exit = False
        self._reset = False

    def on_key(self, key: int) -> None:
        now = time.monotonic()
        with self._lock:
            if key == KEY_ESCAPE:
                self._exit = True
            elif key == KEY_SPACE:
                self._times.clear()
            elif key == ord("R"):
                self._times.clear()
                self._reset = True
            elif key in {KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_LEFT_CTRL, KEY_RIGHT_CTRL}:
                self._times[key] = now

    def snapshot(self) -> tuple[BodyCommand, bool, bool, bool]:
        now = time.monotonic()
        with self._lock:
            for k in list(self._times):
                if now - self._times[k] > self._timeout:
                    del self._times[k]
            turn = KEY_LEFT_CTRL in self._times or KEY_RIGHT_CTRL in self._times
            cmd = BodyCommand()
            cmd.vx = float(KEY_UP in self._times) - float(KEY_DOWN in self._times)
            h = float(KEY_LEFT in self._times) - float(KEY_RIGHT in self._times)
            if turn:
                cmd.yaw = h
            else:
                cmd.vy = h
            ex, rst = self._exit, self._reset
            self._reset = False
        return _clamp_cmd(cmd), turn, ex, rst


def create_keyboard(prefer_x11: bool = True, key_timeout: float = 0.18):
    """Factory: returns (input_source, label).

    Tries X11 first; falls back to MuJoCo callback if X11 unavailable.
    """
    if prefer_x11:
        try:
            return X11Keyboard(), "X11 keyboard"
        except RuntimeError:
            pass
    kb = MuJoCoKeyCallback(key_timeout)
    return kb, "MuJoCo callback (keyboard)"


# ═══════════════════════════════════════════════════════════════
# Gamepad — evdev (zero-dependency outside stdlib on Linux)
# ═══════════════════════════════════════════════════════════════

# Linux input event codes we care about
_EV_ABS = 0x03
_EV_KEY = 0x01
_ABS_X, _ABS_Y, _ABS_RX, _ABS_RY, _ABS_Z, _ABS_RZ = 0x00, 0x01, 0x03, 0x04, 0x02, 0x05
_ABS_HAT0X, _ABS_HAT0Y = 0x10, 0x11
_BTN_SOUTH, _BTN_EAST, _BTN_NORTH, _BTN_WEST = 0x130, 0x131, 0x133, 0x134
_BTN_START, _BTN_SELECT = 0x137, 0x136
_BTN_TL, _BTN_TR = 0x136, 0x137  # actually 0x136=TL, 0x137=TR — reuse BTN_START=0x137 conflict
# Correct mappings:
_BTN_TL2, _BTN_TR2 = 0x136, 0x137  # TL=0x136(select), TR=0x137(start)
# Re-define cleanly:
BTN_SOUTH, BTN_EAST, BTN_NORTH, BTN_WEST = 0x130, 0x131, 0x133, 0x134
BTN_START, BTN_SELECT = 0x137, 0x136
BTN_TL, BTN_TR = 0x136, 0x137  # these overlap — use BTN_SELECT for TL
# Actually let me just use the correct evdev codes:
BTN_A, BTN_B, BTN_X, BTN_Y = 0x130, 0x131, 0x133, 0x134
BTN_LB, BTN_RB = 0x136, 0x137
BTN_BACK, BTN_CANCEL = 0x139, 0x13A  # select, start alternate


def _find_gamepad() -> Optional[str]:
    """Scan /dev/input/event* for a gamepad/joystick device."""
    import os
    import struct
    import fcntl

    _EVIOCGNAME = 0x82006A13  # 256 bytes
    for i in range(32):
        path = f"/dev/input/event{i}"
        if not os.path.exists(path):
            break
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            name_buf = b"\x00" * 256
            try:
                name_buf = fcntl.ioctl(fd, _EVIOCGNAME, name_buf)
                name = name_buf.rstrip(b"\x00").decode("utf-8", errors="replace")
            except OSError:
                name = ""
            os.close(fd)
            low = name.lower()
            if any(kw in low for kw in ("gamepad", "joystick", "xbox", "playstation", "ps4", "ps5",
                                         "dualshock", "dualsense", "8bitdo", "shenghong", "controller")):
                return path
        except OSError:
            continue
    # fallback: try /dev/input/js0
    js = "/dev/input/js0"
    return js if os.path.exists(js) else None


class Gamepad:
    """Reads an evdev gamepad and maps axes/buttons to BodyCommand.

    Default mapping (Xbox layout):
        Left stick Y  → vx  (forward/back)
        Left stick X  → vy  (left/right strafe)
        Right stick X → yaw (turn)
        A button      → stop (zero command)
        B button      → reset
        Start button  → exit
    """

    def __init__(self, device: str | None = None):
        import struct as _struct
        self._struct = _struct
        self.device = device or _find_gamepad()
        if not self.device:
            raise RuntimeError("no gamepad found — connect a controller or specify --input gamepad:<device>")
        import os as _os
        self._fd = _os.open(self.device, _os.O_RDONLY | _os.O_NONBLOCK)
        self._axes: dict[int, float] = {}  # code → normalized [-1, 1]
        self._buttons: dict[int, bool] = {}
        self._exit = False
        self._reset = False
        self._stop = False
        # Deadzone
        self.deadzone = 0.08

    def close(self):
        import os as _os
        if hasattr(self, '_fd') and self._fd is not None:
            _os.close(self._fd)
            self._fd = None

    def _read_events(self) -> None:
        import os as _os
        try:
            data = _os.read(self._fd, 256)
        except (BlockingIOError, OSError):
            return
        s = self._struct
        for i in range(0, len(data), 16):
            if i + 16 > len(data):
                break
            _tv_sec, _tv_usec, etype, code, value = s.unpack("llHHi", data[i:i+16])
            if etype == _EV_ABS:
                self._axes[code] = float(value)
            elif etype == _EV_KEY:
                self._buttons[code] = bool(value)
                if value == 1:  # press
                    if code == BTN_A:  # A → stop
                        self._stop = True
                    elif code == BTN_B:  # B → reset
                        self._reset = True
                    elif code == BTN_START:  # Start → exit
                        self._exit = True

    def snapshot(self) -> tuple[BodyCommand, bool, bool, bool]:
        self._read_events()
        # Normalize axes to [-1, 1] with deadzone
        def _axis(code, invert=False):
            raw = self._axes.get(code, 0.0)
            # evdev ABS axes typically range 0..65535 or 0..255 for hats
            # Most gamepads use 0..65535 with center at ~32768
            # Normalize: assume center ≈ 32768, range ≈ 32768
            v = (raw - 32768.0) / 32768.0
            if abs(v) < self.deadzone:
                v = 0.0
            v = max(-1.0, min(1.0, v))
            return -v if invert else v

        def _hat_x():
            # D-pad hat: -1 left, 1 right, 0 center
            return _axis(_ABS_HAT0X)

        def _hat_y():
            return _axis(_ABS_HAT0Y, invert=True)

        cmd = BodyCommand()
        # Priority: sticks > d-pad
        ly = _axis(_ABS_Y, invert=True)  # up = negative in evdev
        lx = _axis(_ABS_X)
        rx = _axis(_ABS_RX) if _ABS_RX in self._axes else _axis(_ABS_Z)

        # If sticks are near zero, use d-pad
        if abs(ly) < 0.01 and abs(lx) < 0.01:
            ly = _hat_y()
            lx = _hat_x()

        cmd.vx = ly
        cmd.vy = lx
        cmd.yaw = rx

        if self._stop:
            cmd = BodyCommand()
            self._stop = False

        rst, ex = self._reset, self._exit
        self._reset = False
        return _clamp_cmd(cmd), False, ex, rst


def create_gamepad(device: str | None = None):
    g = Gamepad(device)
    label = f"Gamepad ({g.device})"
    return g, label


# ═══════════════════════════════════════════════════════════════
# Input factory
# ═══════════════════════════════════════════════════════════════

def create_input(source: str = "x11", key_timeout: float = 0.18):
    """Create an input source from a string spec.

    Args:
        source: "x11" | "callback" | "gamepad" | "gamepad:<device>"
        key_timeout: Key hold timeout for callback mode.

    Returns:
        (input_object, label_string)
    """
    if source == "x11":
        return create_keyboard(prefer_x11=True, key_timeout=key_timeout)
    elif source == "callback":
        kb = MuJoCoKeyCallback(key_timeout)
        return kb, "MuJoCo callback"
    elif source.startswith("gamepad"):
        dev = source.split(":", 1)[1] if ":" in source else None
        return create_gamepad(dev)
    else:
        raise ValueError(f"Unknown input source: {source}")
