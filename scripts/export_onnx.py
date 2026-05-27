#!/usr/bin/env python3
"""
Export HIMLoco checkpoint to ONNX format for sim2sim deployment.

Usage:
    cd /home/lain/OpenDoge/OpenDoge_train
    export PYTHONPATH=$PWD
    python scripts/export_onnx.py \
        --checkpoint logs/flat_opendoge/May27_10-23-55_opendoge_himloco_v1.0/model_5700.pt \
        --output onnx/flat_opendoge_5700.onnx
"""

import os
import sys
import argparse

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import copy

from rsl_rl.modules import HIMActorCritic


class PolicyExporterHIMOnnx(torch.nn.Module):
    """Wraps HIMActorCritic for ONNX export with the HIM architecture."""

    def __init__(self, actor_critic: HIMActorCritic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)

    def forward(self, obs_history):
        # Estimator outputs [vel(3) + latent(16)] = 19 dims
        parts = self.estimator(obs_history)[:, 0:19]
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        # Actor input: one-step obs(45) + predicted vel(3) + latent z(16) = 64
        return self.actor(torch.cat((obs_history[:, 0:45], vel, z), dim=1))


def parse_args():
    p = argparse.ArgumentParser(description="Export HIMLoco checkpoint to ONNX")
    p.add_argument("--checkpoint", required=True, help="Path to model_N.pt checkpoint")
    p.add_argument("--output", required=True, help="Output .onnx path")
    p.add_argument("--num-obs", type=int, default=270, help="History input dim (6 frames x 45)")
    p.add_argument("--num-one-step-obs", type=int, default=45)
    p.add_argument("--num-actions", type=int, default=12)
    p.add_argument("--num-critic-obs", type=int, default=238, help="Privileged obs dim")
    p.add_argument("--actor-hidden", type=int, nargs="+", default=[512, 256, 128])
    p.add_argument("--critic-hidden", type=int, nargs="+", default=[512, 256, 128])
    p.add_argument("--activation", default="elu")
    p.add_argument("--opset", type=int, default=11)
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Create HIMActorCritic with same arch as training ──
    print("Creating HIMActorCritic with:")
    print(f"  num_actor_obs={args.num_obs}")
    print(f"  num_critic_obs={args.num_critic_obs}")
    print(f"  num_one_step_obs={args.num_one_step_obs}")
    print(f"  num_actions={args.num_actions}")
    print(f"  actor_hidden_dims={args.actor_hidden}")
    print(f"  critic_hidden_dims={args.critic_hidden}")
    print(f"  activation={args.activation}")

    actor_critic = HIMActorCritic(
        num_actor_obs=args.num_obs,
        num_critic_obs=args.num_critic_obs,
        num_one_step_obs=args.num_one_step_obs,
        num_actions=args.num_actions,
        actor_hidden_dims=args.actor_hidden,
        critic_hidden_dims=args.critic_hidden,
        activation=args.activation,
    )

    # ── 2. Load checkpoint ──
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    actor_critic.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded. iter={ckpt.get('iter', '?')}")

    actor_critic.eval()

    # ── 3. Wrap for ONNX export ──
    model = PolicyExporterHIMOnnx(actor_critic)
    model.eval()

    # ── 4. Export ──
    dummy_input = torch.zeros(1, args.num_obs, dtype=torch.float32)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"\nExporting ONNX to: {args.output}")
    print(f"  Input shape:  (batch, {args.num_obs})")
    print(f"  Output shape: (batch, {args.num_actions})")

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        opset_version=args.opset,
        verbose=False,
        input_names=["obs_history"],
        output_names=["actions"],
        dynamic_axes={
            "obs_history": {0: "batch"},
            "actions": {0: "batch"},
        },
    )

    # ── 5. Verify ──
    import onnxruntime as ort
    session = ort.InferenceSession(args.output)
    test_out = session.run(None, {"obs_history": dummy_input.numpy()})
    print(f"  Verified. Output shape: {test_out[0].shape}")
    print("Done.")


if __name__ == "__main__":
    main()
