# Legacy S-DOG2 AMP motion datasets

Place generated S-DOG2 AMP motion clips here.

Expected format is the same as `amp_go2` datasets:
- JSON with `"Frames"` and `"FrameDuration"`, or plain text AMP transition files accepted by `rsl_rl.datasets.motion_loader.AMPLoader`.

This directory is retained as the original S-DOG2 output for comparison. The
OpenDoge AMP configs use the retargeted clips in `datasets/opendoge_motion/`.

The old `sdog2_amp` config scans:

```text
datasets/sdog2_motion/*
```

## Generate from keypoint data

Run from the repository root:

```bash
python datasets/retarget_kp_motions.py
```

This retargets the dog keypoint clips from `datasets/keypoint_datasets/ai4animation/`
to the S-DOG2 URDF and writes the results into this directory.
