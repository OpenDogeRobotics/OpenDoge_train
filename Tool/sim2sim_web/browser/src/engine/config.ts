/**
 * Configuration constants ported from sim2sim/configs/opendoge.yaml.
 * All values mirror the Python-side configuration exactly.
 */

// ── Simulation timing ─────────────────────────────────────────────────
export const SIM_DT = 0.005;            // 200 Hz physics
export const CONTROL_DECIMATION = 2;    // control at 100 Hz (SIM_DT * 2 = 0.01s)

// ── Dimensions ────────────────────────────────────────────────────────
export const NUM_ACTIONS = 12;
export const NUM_OBS = 270;
export const NUM_ONE_STEP_OBS = 45;

// ── Initial state ─────────────────────────────────────────────────────
export const INIT_BASE_HEIGHT = 0.15;

// ── PD gains (12 values, order: FL_hip/thigh/calf, FR_hip/thigh/calf, RL, RR) ─
export const KPS = new Float32Array([12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]);
export const KDS = new Float32Array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);

// ── Default joint angles (radians) ────────────────────────────────────
// hip=0.0 (centered), thigh=0.6 (forward), calf=-1.5 (bent)
export const DEFAULT_DOF_POS = new Float32Array([
  0.0, 0.6, -1.5,    // FL
  0.0, 0.6, -1.5,    // FR
  0.0, 0.6, -1.5,    // RL
  0.0, 0.6, -1.5,    // RR
]);

// ── Observation/action scaling ────────────────────────────────────────
export const LIN_VEL_SCALE = 2.0;
export const ANG_VEL_SCALE = 0.25;
export const DOF_POS_SCALE = 1.0;
export const DOF_VEL_SCALE = 0.05;
export const ACTION_SCALE = 0.30;
export const CMD_SCALE = new Float32Array([2.0, 2.0, 0.25]); // [vx, vy, vyaw]

// ── MuJoCo qpos/qvel offsets ─────────────────────────────────────────
export const QPOS_JOINT_OFFSET = 7;   // 3 pos + 4 quat base
export const QVEL_JOINT_OFFSET = 6;   // 3 lin vel + 3 ang vel base

// ── Torque limits (Nm) from XML actuator ctrlrange ────────────────────
// hip: ±6, thigh: ±6, calf: ±9
export const TORQUE_LIMITS = new Float32Array([
  6, 6, 9,    // FL
  6, 6, 9,    // FR
  6, 6, 9,    // RL
  6, 6, 9,    // RR
]);

// ── Foot names (for contact detection) ────────────────────────────────
// Foot collision spheres live inside the calf bodies (no separate foot body in XML)
export const FOOT_BODY_NAMES = ['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf'];

// ── Joint order (matches kps/kds/default_dof_pos arrays) ─────────────
export const JOINT_ORDER = [
  'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
  'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
  'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
  'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
];

// ── Joint axes (hip = X roll, thigh = Y pitch, calf = Y pitch) ───────
export const JOINT_AXIS: ('X' | 'Y')[] = [
  'X', 'Y', 'Y',    // FL
  'X', 'Y', 'Y',    // FR
  'X', 'Y', 'Y',    // RL
  'X', 'Y', 'Y',    // RR
];
