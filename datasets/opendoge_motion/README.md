# OpenDoge AMP motion dataset

These clips are generated from HIMloco's S-DOG2 AI4Animation keypoints and
retargeted to the OpenDoge URDF with PyBullet IK.

The files use the `AMPLoader` frame layout:

```text
root_pos(3), root_quat(4), joint_pos(12), feet_pos_local(12),
base_lin_vel(3), base_ang_vel(3), joint_vel(12), feet_vel_local(12)
```

Regenerate all clips from the repository root with:

```bash
python datasets/retarget_opendoge_motions.py
```

The original keypoint files are kept under
`datasets/keypoint_datasets/ai4animation/`; `datasets/sdog2_motion/` is the
legacy, non-OpenDoge-retargeted dataset and is not used by the OpenDoge AMP
config.
