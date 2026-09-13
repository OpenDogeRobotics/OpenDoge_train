# OpenDoge MuJoCo Sim2Sim

The deployment entry point is configuration-driven and validates the policy
shape, joint order, actuator mapping, torque limits, and optional model
contracts before simulation.

```bash
python deploy/deploy_mujoco/deploy_mujoco.py opendoge.yaml --no-keyboard --cmd_vx 1.0
python deploy/deploy_mujoco/deploy_mujoco.py opendoge.yaml --headless --duration 10 --diagnostics
python deploy/deploy_mujoco/deploy_mujoco.py opendoge.yaml --validate --duration 5
```

The YAML file selects the ONNX policy, MuJoCo scene, explicit Isaac Gym joint
order, PD gains, torque limits, observation dimensions, and optional get-up
policy. The runtime uses MuJoCo actuator indices rather than assuming XML
order, maintains the 270-dimensional HIM history without overlapping copies,
supports configurable action delay, and reports fall/tilt/torque/velocity
diagnostics.

Keyboard mode uses the MuJoCo viewer callback: `W/S` forward, `A/D` lateral,
`Q/E` yaw, number keys `1`-`5` for speed presets, and `X` for stop.
