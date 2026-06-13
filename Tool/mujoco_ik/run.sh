#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────
# OpenDoge MuJoCo IK Gait Tools — unified launcher
# Usage: ./run.sh <command> [args...]
# ───────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ensure conda env
if [[ "$CONDA_DEFAULT_ENV" != "himloco" ]]; then
    if command -v conda &>/dev/null; then
        exec conda run --no-capture-output -n himloco bash "$0" "$@"
    fi
    echo "WARNING: conda env 'himloco' not active and conda not found" >&2
fi

# ── help ──────────────────────────────────────────────────
usage() {
    cat << 'EOF'
OpenDoge MuJoCo IK Gait Tools

Usage:  ./run.sh <command> [options]

Commands:
  pd stand            PD position control — hold standing pose
  pd sine             PD position control — sinusoidal joint motion
  ik                  IK trot gait (keyboard X11 / gamepad)

Input sources (ik mode, --input flag):
  --input x11         X11 keyboard polling (default, real press/release)
  --input callback    MuJoCo viewer key callback (fallback)
  --input gamepad     Auto-detect gamepad (/dev/input/event*)
  --input gamepad:<N> Specific device, e.g. gamepad:/dev/input/event15

Recording (add --record to any command):
  ./run.sh pd stand  --duration 5  --record stand.npz
  ./run.sh pd sine   --duration 10 --record sine.npz
  ./run.sh ik --no-render --cmd-vx 0.8 --duration 20 --record trot.npz

Common options:
  --duration N         Simulation duration in seconds
  --no-render          Run headless (no viewer window)
  --record FILE        Save reference motion to .npz
  --print-rate N       Telemetry print rate in Hz (default 2.0)

IK options:
  --cmd-vx V           Forward velocity [-1, 1] (headless)
  --cmd-vy V           Lateral velocity [-1, 1] (headless)
  --cmd-yaw V          Yaw rate [-1, 1] (headless)
  --c-style            C-style foot track planner
EOF
    exit 0
}

# ── dispatch ───────────────────────────────────────────────
case "${1:-}" in
    pd)
        shift
        exec python3 scripts/run_position_control.py --mode "${1:-stand}" "${@:2}"
        ;;
    ik)
        shift
        exec python3 scripts/run_ik_control.py "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: ${1:-}"
        echo "Usage: ./run.sh {pd|ik} [options]"
        echo "       ./run.sh --help"
        exit 1
        ;;
esac
