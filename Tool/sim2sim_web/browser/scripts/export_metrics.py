#!/usr/bin/env python3
"""
Export TensorBoard training metrics to static JSON for the browser dashboard.

Usage:
    cd /home/lain/OpenDoge/OpenDoge_train
    export PYTHONPATH=$PWD
    python Tool/sim2sim_web/browser/scripts/export_metrics.py

Output: browser/data/training-metrics/{run_name}.json + index.json
"""

from __future__ import annotations
import os
import sys
import json

# Allow importing from project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from legged_gym import LEGGED_GYM_ROOT_DIR

LOG_ROOT = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "flat_opendoge")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "training-metrics")
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_run(run_dir: str) -> dict | None:
    run_name = os.path.basename(run_dir.rstrip("/"))
    out_path = os.path.join(OUTPUT_DIR, f"{run_name}.json")

    try:
        ea = EventAccumulator(run_dir)
        ea.Reload()
    except Exception as e:
        print(f"  [skip] {run_name}: {e}")
        return None

    metrics: dict = {}
    for tag in sorted(ea.Tags().get("scalars", [])):
        events = ea.Scalars(tag)
        if events:
            metrics[tag] = {
                "values": [e.value for e in events],
                "steps": [e.step for e in events],
                "wall_times": [e.wall_time for e in events],
            }

    with open(out_path, "w") as f:
        json.dump(metrics, f)

    print(f"  [ ok ] {run_name} ({len(metrics)} metrics)")
    return run_name


def main():
    print(f"[export_metrics] Log root : {LOG_ROOT}")
    print(f"[export_metrics] Output   : {OUTPUT_DIR}")

    if not os.path.isdir(LOG_ROOT):
        print(f"[export_metrics] ERROR: Log root not found — {LOG_ROOT}")
        sys.exit(1)

    run_dirs = sorted(
        [d for d in os.listdir(LOG_ROOT) if os.path.isdir(os.path.join(LOG_ROOT, d))]
    )

    print(f"[export_metrics] Found {len(run_dirs)} runs")

    exported = []
    for name in run_dirs:
        run_dir = os.path.join(LOG_ROOT, name)
        result = export_run(run_dir)
        if result:
            exported.append(result)

    # Write index
    index_path = os.path.join(OUTPUT_DIR, "index.json")
    with open(index_path, "w") as f:
        json.dump({"runs": exported}, f)

    print(f"[export_metrics] Done. {len(exported)} runs exported to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
