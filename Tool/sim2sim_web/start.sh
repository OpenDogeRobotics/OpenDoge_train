#!/bin/bash
# OpenDoge Sim2Sim Web Dashboard launcher
# Usage: bash Tool/sim2sim_web/start.sh

set -e
cd "$(dirname "$0")/../.."   # → OpenDoge_train/
export PYTHONPATH="$PWD"
fuser -k 8000/tcp 2>/dev/null || true
sleep 0.5
echo "[start] dir=$PWD"
echo "[start] http://localhost:8000"
exec conda run -n himloco python Tool/sim2sim_web/server.py
