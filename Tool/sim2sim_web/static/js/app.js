// OpenDoge Sim2Sim Dashboard — app logic
// Features: camera drag/scroll, binary WS render, keyboard shortcuts, one-click model switch

const STATE = {
    ws: null,
    connected: false,
    simRunning: false,
    currentModel: null,
    cmd: { vx: 0, vy: 0, vyaw: 0 },
    history: { base_z: [], lin_vel_x: [], time: [] },
    maxHistory: 200,
    // camera state
    cam: { azimuth: 180, elevation: -25, distance: 1.2 },
    // mouse drag state
    mouse: { down: false, button: 0, x: 0, y: 0 },
};

const $ = (id) => document.getElementById(id);

// ── Charts ──────────────────────────────────────────────────────────
let chartJoints, chartHeight, chartVel;

function initCharts() {
    const jCtx = $("chart-joints").getContext("2d");
    chartJoints = new Chart(jCtx, {
        type: "bar",
        data: {
            labels: ["FL_h","FL_t","FL_c","FR_h","FR_t","FR_c","RL_h","RL_t","RL_c","RR_h","RR_t","RR_c"],
            datasets: [{ data: new Array(12).fill(0), backgroundColor: "#4caf50" }],
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            scales: {
                y: { min: -2, max: 2, ticks: { color: "#888", font: { size: 8 } } },
                x: { ticks: { color: "#888", font: { size: 7 } } },
            },
            plugins: { legend: { display: false } },
        },
    });

    chartHeight = new Chart($("chart-height").getContext("2d"), {
        type: "line",
        data: { labels: [], datasets: [{ data: [], borderColor: "#4caf50", borderWidth: 1.5, pointRadius: 0, tension: 0.1 }] },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            scales: {
                y: { min: 0.05, max: 0.3, ticks: { color: "#888", font: { size: 9 } } },
                x: { ticks: { color: "#888", font: { size: 8 }, maxTicksLimit: 6 } },
            },
            plugins: { legend: { display: false } },
        },
    });

    chartVel = new Chart($("chart-vel").getContext("2d"), {
        type: "line",
        data: {
            labels: [],
            datasets: [
                { data: [], borderColor: "#e94560", borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
                { data: [], borderColor: "#2196f3", borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
                { data: [], borderColor: "#ff9800", borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            scales: {
                y: { ticks: { color: "#888", font: { size: 9 } } },
                x: { ticks: { color: "#888", font: { size: 8 }, maxTicksLimit: 6 } },
            },
            plugins: { legend: { display: false } },
        },
    });
}

// ── UI helpers ───────────────────────────────────────────────────────

function setStatus(online, text) {
    $("status").textContent = text || (online ? "● connected" : "● disconnected");
    $("status").className = online ? "online" : "offline";
    STATE.connected = online;
}

function onSlider(axis) {
    STATE.cmd[axis] = parseFloat($("slider-" + axis).value);
    $("val-" + axis).textContent = STATE.cmd[axis].toFixed(2);
    sendCmd();
}

function sendCmd() {
    if (!STATE.connected) return;
    STATE.ws.send(json({ type: "cmd", vx: STATE.cmd.vx, vy: STATE.cmd.vy, vyaw: STATE.cmd.vyaw }));
}

// ── JSON helper (browser has native JSON) ────────────────────────────
function json(obj) { return JSON.stringify(obj); }

// ── Model switching ──────────────────────────────────────────────────

async function switchModel(path) {
    if (!path) return;
    STATE.ws.send(json({ type: "load_model", path }));
    $("model-name").textContent = path.split("/").pop().replace(".onnx", "");
}

async function nextModel() {
    STATE.ws.send(json({ type: "next_model" }));
}

async function prevModel() {
    STATE.ws.send(json({ type: "prev_model" }));
}

async function loadModelsList() {
    const resp = await fetch("/api/models");
    const models = await resp.json();
    const sel = $("model-select");
    sel.innerHTML = "";
    models.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.path;
        const stepStr = m.step ? ` [step ${m.step}]` : "";
        opt.textContent = `${m.name}${stepStr}`;
        sel.appendChild(opt);
    });
    // select current
    const curResp = await fetch("/api/models/current");
    const cur = await curResp.json();
    if (cur && cur.path) {
        sel.value = cur.path;
        $("model-name").textContent = cur.name;
        STATE.currentModel = cur;
    }
}

// ── Simulation control ───────────────────────────────────────────────

async function toggleSim() {
    STATE.simRunning = !STATE.simRunning;
    if (STATE.simRunning) {
        $("btn-start").textContent = "⏹ Stop";
        $("btn-start").className = "btn-stop";
    } else {
        $("btn-start").textContent = "▶ Start";
        $("btn-start").className = "btn-start";
    }
}

async function resetSim() {
    STATE.ws.send(json({ type: "reset" }));
    STATE.history = { base_z: [], lin_vel_x: [], time: [] };
}

// ── Camera mouse controls ───────────────────────────────────────────

function initViewerMouse() {
    const panel = $("viewer-panel");

    panel.addEventListener("mousedown", (e) => {
        e.preventDefault();
        STATE.mouse.down = true;
        STATE.mouse.button = e.button;
        STATE.mouse.x = e.clientX;
        STATE.mouse.y = e.clientY;
        if (e.button === 2) panel.classList.add("panning");
    });

    window.addEventListener("mousemove", (e) => {
        if (!STATE.mouse.down) return;
        const dx = e.clientX - STATE.mouse.x;
        const dy = e.clientY - STATE.mouse.y;
        STATE.mouse.x = e.clientX;
        STATE.mouse.y = e.clientY;

        if (dx === 0 && dy === 0) return;

        if (STATE.mouse.button === 0) {
            // Left drag → orbit (azimuth + elevation)
            sendCameraDelta(-dx * 0.3, dy * 0.3, 0);
        } else if (STATE.mouse.button === 2) {
            // Right drag → pan (move lookat in x/y plane)
            sendCameraDelta(0, 0, 0); // pan handled via lookat delta
            // Approximate: move lookat based on camera orientation
            const s = STATE.cam.distance * 0.002;
            sendLookatDelta(-dx * s, dy * s, 0);
        }
    });

    window.addEventListener("mouseup", () => {
        STATE.mouse.down = false;
        $("viewer-panel").classList.remove("panning");
    });

    panel.addEventListener("wheel", (e) => {
        e.preventDefault();
        // Scroll → zoom (distance)
        sendCameraDelta(0, 0, e.deltaY * 0.005);
    }, { passive: false });

    panel.addEventListener("dblclick", () => {
        // Double-click → reset camera
        STATE.ws.send(json({
            type: "camera",
            azimuth: 180, elevation: -25, distance: 1.2,
            lookat: [0, 0, 0.15],
        }));
    });

    panel.addEventListener("contextmenu", (e) => e.preventDefault());
}

function sendCameraDelta(dAz, dEl, dDist) {
    STATE.cam.azimuth += dAz;
    STATE.cam.elevation += dEl;
    STATE.cam.distance += dDist;
    STATE.cam.distance = Math.max(0.2, Math.min(10, STATE.cam.distance));
    STATE.cam.elevation = Math.max(-89, Math.min(89, STATE.cam.elevation));
    STATE.cam.azimuth = ((STATE.cam.azimuth % 360) + 360) % 360;
    STATE.ws.send(json({
        type: "camera",
        delta_azimuth: dAz,
        delta_elevation: dEl,
        delta_distance: dDist,
    }));
}

function sendLookatDelta(dx, dy, dz) {
    STATE.ws.send(json({
        type: "camera",
        lookat_delta: [dx, dy, dz],
    }));
}

// ── Keyboard shortcuts ───────────────────────────────────────────────

function initKeyboard() {
    window.addEventListener("keydown", (e) => {
        // Don't intercept when typing in inputs
        if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

        switch (e.key) {
            case " ":
                e.preventDefault();
                toggleSim();
                break;
            case "[":
                prevModel();
                break;
            case "]":
                nextModel();
                break;
            case "r":
                if (e.ctrlKey) break;
                resetSim();
                break;
            case "ArrowUp":
                STATE.cmd.vx = Math.min(1.5, STATE.cmd.vx + 0.05);
                $("slider-vx").value = STATE.cmd.vx;
                $("val-vx").textContent = STATE.cmd.vx.toFixed(2);
                sendCmd();
                break;
            case "ArrowDown":
                STATE.cmd.vx = Math.max(-1.5, STATE.cmd.vx - 0.05);
                $("slider-vx").value = STATE.cmd.vx;
                $("val-vx").textContent = STATE.cmd.vx.toFixed(2);
                sendCmd();
                break;
            case "ArrowLeft":
                STATE.cmd.vyaw = Math.min(2.0, STATE.cmd.vyaw + 0.1);
                $("slider-vyaw").value = STATE.cmd.vyaw;
                $("val-vyaw").textContent = STATE.cmd.vyaw.toFixed(2);
                sendCmd();
                break;
            case "ArrowRight":
                STATE.cmd.vyaw = Math.max(-2.0, STATE.cmd.vyaw - 0.1);
                $("slider-vyaw").value = STATE.cmd.vyaw;
                $("val-vyaw").textContent = STATE.cmd.vyaw.toFixed(2);
                sendCmd();
                break;
        }
    });
}

// ── Training comparison ──────────────────────────────────────────────

async function loadComparison() {
    const resp = await fetch("/api/monitor/compare");
    const data = await resp.json();
    const runs = Object.keys(data);
    if (runs.length === 0) return;

    const short = {
        "Train/mean_reward": "reward",
        "Episode/rew_tracking_lin_vel": "track_lin",
        "Episode/rew_feet_air_time": "feet_air",
        "Episode/rew_smoothness": "smooth",
        "Episode/rew_dof_acc": "dof_acc",
        "Episode/rew_base_height": "base_h",
        "Episode/rew_orientation": "orient",
    };

    const thead = document.querySelector("#comp-table thead");
    const tbody = document.querySelector("#comp-table tbody");
    thead.innerHTML = "<tr><th>Metric</th>" + runs.map(r => `<th>${r.slice(0,16)}</th>`).join("") + "</tr>";
    tbody.innerHTML = "";
    for (const [full, label] of Object.entries(short)) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${label}</td>` + runs.map(r => {
            const v = data[r]?.[full];
            return `<td>${v !== undefined ? v.toFixed(4) : "--"}</td>`;
        }).join("");
        tbody.appendChild(tr);
    }
}

// ── WebSocket ────────────────────────────────────────────────────────

function connectWS() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws/stream`;
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    STATE.ws = ws;

    ws.onopen = () => {
        setStatus(true);
        loadModelsList();
        loadComparison();
        // Auto-start: simulation begins on WS connect
        toggleSim();
    };

    ws.onclose = () => {
        setStatus(false);
        STATE.ws = null;
        setTimeout(connectWS, 2000);
    };

    ws.onerror = () => { ws.close(); };

    ws.onmessage = (evt) => {
        if (evt.data instanceof ArrayBuffer) {
            // Binary render frame → draw to canvas (no flicker)
            const blob = new Blob([evt.data], { type: "image/jpeg" });
            createImageBitmap(blob).then(bitmap => {
                const canvas = $("viewer-canvas");
                if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
                    canvas.width = bitmap.width;
                    canvas.height = bitmap.height;
                }
                const ctx = canvas.getContext("2d");
                ctx.drawImage(bitmap, 0, 0);
                bitmap.close();
            }).catch(() => {});
        } else {
            // Text state update
            try {
                const s = JSON.parse(evt.data);
                updateState(s);
            } catch (e) { /* ignore */ }
        }
    };
}

// ── Chart throttle ────────────────────────────────────────────────────
let _chartFrame = 0;
const CHART_EVERY = 5;  // update charts every Nth state frame (~10 Hz)

// ── State update ─────────────────────────────────────────────────────

function updateState(s) {
    if (s.type === "pong") return;
    _chartFrame++;

    // Lightweight text updates every frame (cheap)
    $("m-base-z").textContent = s.base_pos?.[2]?.toFixed(3) + " m" || "--";
    $("m-lin-vel").textContent = s.base_lin_vel
        ? `${s.base_lin_vel[0].toFixed(2)} / ${s.base_lin_vel[1].toFixed(2)} / ${s.base_lin_vel[2].toFixed(2)}`
        : "--";
    $("m-ang-vel").textContent = s.base_ang_vel
        ? `${s.base_ang_vel[0].toFixed(2)} / ${s.base_ang_vel[1].toFixed(2)} / ${s.base_ang_vel[2].toFixed(2)}`
        : "--";
    $("m-contact").textContent = s.feet_contact
        ? `FL:${s.feet_contact[0]?1:0} FR:${s.feet_contact[1]?1:0} RL:${s.feet_contact[2]?1:0} RR:${s.feet_contact[3]?1:0}`
        : "--";
    if (s.step !== undefined) $("m-step").textContent = `step ${s.step} / ${s.model}`;
    if (s.model && s.model !== STATE.currentModel?.name) {
        STATE.currentModel = { name: s.model };
        $("model-name").textContent = s.model;
        const sel = $("model-select");
        for (const opt of sel.options) {
            if (opt.textContent.startsWith(s.model)) { sel.value = opt.value; break; }
        }
    }

    // Rolling history (always updated)
    const t = s.t || 0;
    STATE.history.time.push(t);
    STATE.history.base_z.push(s.base_pos?.[2] || 0);
    STATE.history.lin_vel_x.push(s.base_lin_vel?.[0] || 0);
    if (STATE.history.time.length > STATE.maxHistory) {
        STATE.history.time.shift();
        STATE.history.base_z.shift();
        STATE.history.lin_vel_x.shift();
    }

    // Throttle heavy canvas chart updates
    if (_chartFrame % CHART_EVERY !== 0) return;

    if (chartJoints && s.joint_pos) {
        chartJoints.data.datasets[0].data = s.joint_pos;
        chartJoints.update("none");
    }
    if (chartHeight) {
        chartHeight.data.labels = STATE.history.time;
        chartHeight.data.datasets[0].data = STATE.history.base_z;
        chartHeight.update("none");
    }
    if (chartVel) {
        chartVel.data.labels = STATE.history.time;
        chartVel.data.datasets[0].data = STATE.history.lin_vel_x;
        chartVel.update("none");
    }
}

// ── Init ─────────────────────────────────────────────────────────────
initCharts();
initViewerMouse();
initKeyboard();

// Draw initial placeholder on canvas so it's never blank
(function() {
    const c = $("viewer-canvas");
    c.width = 640; c.height = 360;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.fillStyle = "#555";
    ctx.font = "16px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Connecting to server...", c.width/2, c.height/2);
})();

connectWS();
