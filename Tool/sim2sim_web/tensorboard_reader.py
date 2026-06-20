"""
TensorBoard log reader for OpenDoge training runs.

Reads scalar metrics from TensorFlow event files without needing tensorflow
installed — uses the lightweight tensorboard.backend.event_processing package.
"""

from __future__ import annotations
import os
import glob
from collections import defaultdict

from legged_gym import LEGGED_GYM_ROOT_DIR

DEFAULT_LOG_ROOT = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "flat_opendoge")

# metrics we care about for sim2sim evaluation
KEY_METRICS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode/rew_tracking_lin_vel",
    "Episode/rew_tracking_ang_vel",
    "Episode/rew_feet_air_time",
    "Episode/rew_smoothness",
    "Episode/rew_dof_acc",
    "Episode/rew_base_height",
    "Episode/rew_lin_vel_z",
    "Episode/rew_ang_vel_xy",
    "Episode/rew_orientation",
    "Episode/rew_collision",
    "Episode/rew_action_rate",
    "Episode/rew_stand_still",
]


def _get_ea(log_dir):
    """Get an EventAccumulator for a log directory."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    ea = EventAccumulator(log_dir)
    ea.Reload()
    return ea


def list_runs(log_root=None):
    """Return list of training run directories with basic info."""
    log_root = log_root or DEFAULT_LOG_ROOT
    if not os.path.isdir(log_root):
        return []

    runs = []
    for d in sorted(glob.glob(os.path.join(log_root, "*/")), reverse=True):
        run_name = os.path.basename(d.rstrip("/"))
        # count checkpoints
        ckpts = sorted(glob.glob(os.path.join(d, "model_*.pt")))
        runs.append({
            "name": run_name,
            "path": d,
            "num_checkpoints": len(ckpts),
            "latest_checkpoint": int(
                os.path.splitext(os.path.basename(ckpts[-1]))[0].replace("model_", "")
            )
            if ckpts
            else None,
        })
    return runs


def get_latest_metrics(run_name=None, run_dir=None, log_root=None):
    """Get the latest scalar values for a run.

    Specify either ``run_name`` (looked up under log_root) or ``run_dir`` directly.
    """
    log_root = log_root or DEFAULT_LOG_ROOT
    if run_dir is None:
        if run_name is None:
            runs = list_runs(log_root)
            if not runs:
                return None
            run_dir = runs[0]["path"]
        else:
            run_dir = os.path.join(log_root, run_name)
            if not os.path.isdir(run_dir):
                return None

    try:
        ea = _get_ea(run_dir)
    except Exception:
        return None

    metrics = {}
    for tag in sorted(ea.Tags().get("scalars", [])):
        events = ea.Scalars(tag)
        if events:
            metrics[tag] = {
                "step": events[-1].step,
                "value": round(events[-1].value, 6),
                "wall_time": events[-1].wall_time,
            }
    return metrics


def get_metric_history(run_name=None, run_dir=None, metric=None, log_root=None):
    """Return the full (step, value) history for a single metric."""
    log_root = log_root or DEFAULT_LOG_ROOT
    if run_dir is None:
        if run_name is None:
            return []
        run_dir = os.path.join(log_root, run_name)
        if not os.path.isdir(run_dir):
            return []

    try:
        ea = _get_ea(run_dir)
    except Exception:
        return []

    if metric not in ea.Tags().get("scalars", []):
        return []

    return [
        {"step": e.step, "value": round(e.value, 6), "wall_time": e.wall_time}
        for e in ea.Scalars(metric)
    ]


def compare_runs(run_names=None, log_root=None):
    """Compare the latest metrics across multiple runs.

    If ``run_names`` is None, compares all runs.
    """
    log_root = log_root or DEFAULT_LOG_ROOT
    runs = list_runs(log_root)

    if run_names:
        runs = [r for r in runs if r["name"] in run_names]

    comparison = {}
    for run in runs:
        metrics = get_latest_metrics(run_dir=run["path"], log_root=log_root)
        if metrics:
            comparison[run["name"]] = {
                m: metrics[m]["value"]
                for m in KEY_METRICS
                if m in metrics
            }
    return comparison


# ── quick self-test ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== TensorBoard Reader Self-Test ===\n")

    runs = list_runs()
    print(f"Found {len(runs)} runs:")
    for r in runs[:5]:
        print(f"  {r['name']}  (ckpts: {r['num_checkpoints']}, latest: {r['latest_checkpoint']})")

    if runs:
        print(f"\nLatest metrics for {runs[0]['name']}:")
        m = get_latest_metrics(run_dir=runs[0]["path"])
        if m:
            for tag in KEY_METRICS:
                if tag in m:
                    print(f"  {tag:45s}  step={m[tag]['step']:6d}  value={m[tag]['value']:.6f}")
        else:
            print("  (no metrics found)")

    print("\n=== Done ===")
