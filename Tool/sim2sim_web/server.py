"""
FastAPI server for the OpenDoge Sim2Sim Web dashboard.

Start:
    cd /home/lain/OpenDoge/OpenDoge_train
    export PYTHONPATH=$PWD
    conda run -n himloco python Tool/sim2sim_web/server.py
    # open http://localhost:8000
"""

from __future__ import annotations
import os
import sys
import json
import asyncio
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Fast JSON encoder (orjson is ~3-5× faster than stdlib json)
try:
    import orjson

    def _json_dumps(obj) -> bytes:
        return orjson.dumps(obj)
except ImportError:
    def _json_dumps(obj) -> bytes:
        return json.dumps(obj).encode("utf-8")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

from Tool.sim2sim_web.engine import Sim2SimEngine
from Tool.sim2sim_web.tensorboard_reader import (
    list_runs,
    get_latest_metrics,
    get_metric_history,
    compare_runs,
    KEY_METRICS,
)

# ── globals ───────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

engine: Sim2SimEngine | None = None
_sim_running = False
_sim_task: asyncio.Task | None = None
_ws_clients: list[WebSocket] = []
_PUSH_INTERVAL = 0.02   # 50 Hz

# ── Lifecycle ─────────────────────────────────────────────────────────
from contextlib import asynccontextmanager

def _init_engine():
    global engine
    if engine is None:
        engine = Sim2SimEngine()
    return engine


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _init_engine()
    yield
    global engine, _sim_running, _sim_task
    _sim_running = False
    if _sim_task:
        _sim_task.cancel()
    if engine:
        engine.close()
        engine = None


# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(title="OpenDoge Sim2Sim Dashboard", lifespan=_lifespan)

# ── Static files + root ───────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r") as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════════════
# Model API
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/models")
async def api_list_models():
    return engine.list_models()


@app.post("/api/models/load")
async def api_load_model(data: dict):
    path = data.get("path", "")
    if not path:
        return {"ok": False, "error": "missing 'path'"}
    try:
        info = engine.load_model(path)
        return {"ok": True, "model": info}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/models/current")
async def api_current_model():
    m = engine.get_current_model()
    return m or {"name": None}


@app.post("/api/models/next")
async def api_next_model():
    """Switch to the next model in the list (cyclic)."""
    models = engine.list_models()
    cur = engine.get_current_model()
    if not models or not cur:
        return {"ok": False}
    for i, m in enumerate(models):
        if m["path"] == cur["path"]:
            nxt = models[(i + 1) % len(models)]
            engine.load_model(nxt["path"])
            return {"ok": True, "model": engine.get_current_model()}
    return {"ok": False}


@app.post("/api/models/prev")
async def api_prev_model():
    """Switch to the previous model in the list (cyclic)."""
    models = engine.list_models()
    cur = engine.get_current_model()
    if not models or not cur:
        return {"ok": False}
    for i, m in enumerate(models):
        if m["path"] == cur["path"]:
            prv = models[(i - 1) % len(models)]
            engine.load_model(prv["path"])
            return {"ok": True, "model": engine.get_current_model()}
    return {"ok": False}


# ═══════════════════════════════════════════════════════════════════════
# Camera API
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/camera")
async def api_get_camera():
    return engine.get_camera()


@app.post("/api/camera")
async def api_set_camera(data: dict):
    """Set camera. Accepts absolute values OR delta values for incremental moves."""
    kwargs = {}
    for key in ("azimuth", "elevation", "distance"):
        if key in data:
            kwargs[key] = float(data[key])
    for key in ("delta_azimuth", "delta_elevation", "delta_distance"):
        if key in data:
            kwargs[key] = float(data[key])
    if "lookat" in data and isinstance(data["lookat"], list):
        kwargs["lookat"] = data["lookat"]
    if "lookat_delta" in data and isinstance(data["lookat_delta"], list):
        kwargs["lookat_delta"] = data["lookat_delta"]
    return engine.set_camera(**kwargs)


# ═══════════════════════════════════════════════════════════════════════
# Simulation API
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/sim/cmd")
async def api_set_cmd(data: dict):
    vx = float(data.get("vx", 0.0))
    vy = float(data.get("vy", 0.0))
    vyaw = float(data.get("vyaw", 0.0))
    engine.set_cmd(vx, vy, vyaw)
    return {"ok": True, "cmd": [vx, vy, vyaw]}


@app.post("/api/sim/reset")
async def api_reset():
    engine.reset()
    return {"ok": True}


@app.get("/api/sim/state")
async def api_get_state():
    return engine.get_state()


@app.get("/api/sim/render")
async def api_render(w: int = 960, h: int = 540, q: int = 65):
    jpeg = engine.render(w, h, quality=q)
    return Response(content=jpeg, media_type="image/jpeg")


# ═══════════════════════════════════════════════════════════════════════
# Monitor API
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/monitor/runs")
async def api_list_runs():
    return list_runs()


@app.get("/api/monitor/metrics")
async def api_get_metrics(run: str = ""):
    if run:
        return get_latest_metrics(run_name=run)
    return get_latest_metrics()


@app.get("/api/monitor/history")
async def api_get_history(run: str = "", metric: str = "Train/mean_reward"):
    return get_metric_history(run_name=run or None, metric=metric)


@app.get("/api/monitor/compare")
async def api_compare(runs: str = ""):
    names = [r.strip() for r in runs.split(",") if r.strip()] if runs else None
    return compare_runs(run_names=names)


# ═══════════════════════════════════════════════════════════════════════
# WebSocket — real-time state + camera commands + binary render
# ═══════════════════════════════════════════════════════════════════════

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    engine.set_viewer_clients(True)

    # Start simulation broadcaster if not already running
    global _sim_running, _sim_task
    if not _sim_running:
        _sim_running = True
        _sim_task = asyncio.create_task(_broadcast_loop())

    try:
        while True:
            raw = await ws.receive()
            if "text" in raw:
                try:
                    msg = json.loads(raw["text"])
                except json.JSONDecodeError:
                    continue
                await _handle_ws_text(ws, msg)
            # binary messages are ignored (we send them, don't receive)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
        if not _ws_clients:
            engine.set_viewer_clients(False)


async def _handle_ws_text(ws: WebSocket, msg: dict):
    t = msg.get("type", "")
    if t == "cmd":
        engine.set_cmd(
            float(msg.get("vx", 0)),
            float(msg.get("vy", 0)),
            float(msg.get("vyaw", 0)),
        )
    elif t == "load_model":
        path = msg.get("path", "")
        if path:
            engine.load_model(path)
    elif t == "next_model":
        await _handle_next_model()
    elif t == "prev_model":
        await _handle_prev_model()
    elif t == "reset":
        engine.reset()
    elif t == "camera":
        await _handle_ws_camera(msg)
    elif t == "ping":
        await ws.send_text('{"type":"pong"}')


async def _handle_next_model():
    models = engine.list_models()
    cur = engine.get_current_model()
    if not models or not cur:
        return
    for i, m in enumerate(models):
        if m["path"] == cur["path"]:
            engine.load_model(models[(i + 1) % len(models)]["path"])
            return


async def _handle_prev_model():
    models = engine.list_models()
    cur = engine.get_current_model()
    if not models or not cur:
        return
    for i, m in enumerate(models):
        if m["path"] == cur["path"]:
            engine.load_model(models[(i - 1) % len(models)]["path"])
            return


async def _handle_ws_camera(msg: dict):
    kwargs = {}
    for key in ("azimuth", "elevation", "distance"):
        if key in msg:
            kwargs[key] = float(msg[key])
    for key in ("delta_azimuth", "delta_elevation", "delta_distance"):
        if key in msg:
            kwargs[key] = float(msg[key])
    if "lookat" in msg and isinstance(msg["lookat"], list):
        kwargs["lookat"] = msg["lookat"]
    if "lookat_delta" in msg and isinstance(msg["lookat_delta"], list):
        kwargs["lookat_delta"] = msg["lookat_delta"]
    engine.set_camera(**kwargs)


async def _broadcast_loop():
    """Continuously step engine and push state + render to all WS clients."""
    global _sim_running, engine
    render_interval = 10   # render every N steps (~10 fps at 100Hz control)

    while _sim_running:
        t0 = time.time()

        state = engine.step()
        step_no = state["step"]

        # JSON state every step (small: ~300 bytes)
        payload_text = _json_dumps(state)

        # Render frame periodically and push as binary
        payload_bin = None
        if engine._has_viewer_clients and step_no % render_interval == 0:
            jpeg = engine.render(width=640, height=360, quality=50)
            if jpeg:
                payload_bin = jpeg

        dead = []
        for ws in _ws_clients:
            try:
                await ws.send_bytes(payload_text)
                if payload_bin:
                    await ws.send_bytes(payload_bin)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in _ws_clients:
                _ws_clients.remove(ws)

        elapsed = time.time() - t0
        sleep_time = _PUSH_INTERVAL - elapsed
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys as _sys

    parser = argparse.ArgumentParser(description="OpenDoge Sim2Sim Web Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--public", action="store_true", help="Listen on 0.0.0.0")
    args = parser.parse_args()

    host = "0.0.0.0" if args.public else args.host

    eng = _init_engine()
    models = eng.list_models()
    cur = eng.get_current_model()
    _sys.stderr.write(f"[server] Listening  : http://localhost:{args.port}\n")
    _sys.stderr.write(f"[server] Models     : {len(models)} found\n")
    _sys.stderr.write(f"[server] Current    : {cur['name'] if cur else 'none'}\n")
    _sys.stderr.write(f"[server] Press Ctrl+C to stop\n")
    _sys.stderr.flush()

    uvicorn.run(app, host=host, port=args.port, log_level="warning")
