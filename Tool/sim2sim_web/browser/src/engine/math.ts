/**
 * Core math functions ported from sim2sim/_common.py.
 *
 * All functions operate on Float32Array / Float64Array to mirror MuJoCo's
 * internal representation and minimize GC pressure.
 */

import {
  NUM_ACTIONS,
  NUM_OBS,
  NUM_ONE_STEP_OBS,
  QPOS_JOINT_OFFSET,
  QVEL_JOINT_OFFSET,
} from './config';

// ═══════════════════════════════════════════════════════════════════════
// quat_rotate_inverse
// ═══════════════════════════════════════════════════════════════════════

/**
 * Rotate a world-frame vector `v` into body frame using the inverse of quaternion `q`.
 *
 * q is MuJoCo convention: [w, x, y, z].  v is [x, y, z].
 *
 * Math: R(q)^T * v = v + 2*w*(q_vec × v) + 2*(q_vec × (q_vec × v))
 * simplified to the form below.
 */
export function quatRotateInverse(
  q: Float64Array | Float32Array | number[],
  v: Float32Array
): Float32Array {
  const qw = q[0], qx = q[1], qy = q[2], qz = q[3];
  const vx = v[0], vy = v[1], vz = v[2];

  const twoWsqMinus1 = 2.0 * qw * qw - 1.0;

  // a = v * (2*w^2 - 1)
  const ax = vx * twoWsqMinus1;
  const ay = vy * twoWsqMinus1;
  const az = vz * twoWsqMinus1;

  // b = cross(q_vec, v) * w * 2
  const twoW = qw * 2.0;
  const bx = (qy * vz - qz * vy) * twoW;
  const by = (qz * vx - qx * vz) * twoW;
  const bz = (qx * vy - qy * vx) * twoW;

  // c = q_vec * dot(q_vec, v) * 2
  const dot = qx * vx + qy * vy + qz * vz;
  const twoDot = dot * 2.0;
  const cx = qx * twoDot;
  const cy = qy * twoDot;
  const cz = qz * twoDot;

  return new Float32Array([ax - bx + cx, ay - by + cy, az - bz + cz]);
}

// ═══════════════════════════════════════════════════════════════════════
// pd_control
// ═══════════════════════════════════════════════════════════════════════

/**
 * Proportional-derivative torque control.
 *
 * tau = kp * (target_q - q) + kd * (target_dq - dq)
 */
export function pdControl(
  targetQ: Float32Array,
  q: Float32Array,
  kp: Float32Array,
  targetDq: Float32Array,
  dq: Float32Array,
  kd: Float32Array
): Float32Array {
  const n = Math.min(targetQ.length, q.length, kp.length, targetDq.length, dq.length, kd.length);
  const tau = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    tau[i] = (targetQ[i] - q[i]) * kp[i] + (targetDq[i] - dq[i]) * kd[i];
  }
  return tau;
}

// ═══════════════════════════════════════════════════════════════════════
// build_obs_raw
// ═══════════════════════════════════════════════════════════════════════

export interface ObsRawParams {
  /** MuJoCo qpos (full array, length >= 7+NUM_ACTIONS) */
  qpos: Float64Array;
  /** MuJoCo qvel (full array, length >= 6+NUM_ACTIONS) */
  qvel: Float64Array;
  /** Gyro sensor data (body-frame angular velocity), 3 elements */
  gyroData: Float32Array | null;
  /** Scaled velocity command [vx, vy, vyaw], 3 elements */
  cmd: Float32Array;
  /** Command scale [vx_scale, vy_scale, vyaw_scale], 3 elements */
  cmdScale: Float32Array;
  /** Angular velocity scale factor */
  angVelScale: number;
  /** Joint position deviation scale factor */
  dofPosScale: number;
  /** Joint velocity scale factor */
  dofVelScale: number;
  /** Previous action (raw, unscaled), NUM_ACTIONS elements */
  action: Float32Array;
  /** Default joint positions, NUM_ACTIONS elements */
  defaultDofPos: Float32Array;
  /** Whether the gyro sensor is available */
  useGyroSensor: boolean;
  /** Number of actions (default 12) */
  numActions: number;
}

/**
 * Build a single-step observation vector (45 dimensions) from raw MuJoCo data.
 *
 * Layout (45 total):
 *   [0:3]   cmd_norm         — scaled velocity command
 *   [3:6]   omega_norm       — scaled body angular velocity
 *   [6:9]   proj_gravity     — gravity vector in body frame
 *   [9:21]  qj_norm          — normalized joint positions (12)
 *   [21:33] dqj_norm         — scaled joint velocities (12)
 *   [33:45] action           — previous action (12)
 */
export function buildObsRaw(params: ObsRawParams): Float32Array {
  const {
    qpos, qvel, gyroData, cmd, cmdScale,
    angVelScale, dofPosScale, dofVelScale, action,
    defaultDofPos, useGyroSensor, numActions,
  } = params;

  const obs = new Float32Array(NUM_ONE_STEP_OBS);

  // 1. Command: cmd * cmd_scale  (indices 0-2)
  obs[0] = cmd[0] * cmdScale[0];
  obs[1] = cmd[1] * cmdScale[1];
  obs[2] = cmd[2] * cmdScale[2];

  // 2. Angular velocity (body frame)  (indices 3-5)
  let omega: Float32Array | Float64Array;
  if (useGyroSensor && gyroData) {
    omega = gyroData;
  } else {
    // Fallback: world-frame angular velocity from qvel[3:6]
    omega = new Float32Array([qvel[3], qvel[4], qvel[5]]);
  }
  obs[3] = omega[0] * angVelScale;
  obs[4] = omega[1] * angVelScale;
  obs[5] = omega[2] * angVelScale;

  // 3. Projected gravity  (indices 6-8)
  const quat = new Float32Array([qpos[3], qpos[4], qpos[5], qpos[6]]);  // w,x,y,z
  const gravityWorld = new Float32Array([0, 0, -1]);
  const projGravity = quatRotateInverse(quat, gravityWorld);
  obs[6] = projGravity[0];
  obs[7] = projGravity[1];
  obs[8] = projGravity[2];

  // 4. Normalized joint positions  (indices 9-20)
  for (let i = 0; i < numActions; i++) {
    obs[9 + i] = (qpos[QPOS_JOINT_OFFSET + i] - defaultDofPos[i]) * dofPosScale;
  }

  // 5. Scaled joint velocities  (indices 21-32)
  for (let i = 0; i < numActions; i++) {
    obs[21 + i] = qvel[QVEL_JOINT_OFFSET + i] * dofVelScale;
  }

  // 6. Previous action (raw, unscaled)  (indices 33-44)
  for (let i = 0; i < numActions; i++) {
    obs[33 + i] = action[i];
  }

  return obs;
}

// ═══════════════════════════════════════════════════════════════════════
// build_policy_input
// ═══════════════════════════════════════════════════════════════════════

/**
 * Build the full policy input tensor from a single-step observation and history buffer.
 *
 * Supports three ONNX input shapes:
 *   - inputDim == numObs (270): 6-step history stacking (most common)
 *   - inputDim == 64: HIMLoco encoder (padded with zeros)
 *   - inputDim == 45: single step, no history
 */
export function buildPolicyInput(
  obsRaw: Float32Array,
  historyBuffer: Float32Array[],
  inputDim: number,
  numObs: number
): Float32Array {
  // Case 1: History stacking (inputDim == numObs, typically 270 = 45*6)
  if (inputDim === numObs) {
    // Shift history: push new obs to front, drop oldest
    historyBuffer.pop();
    historyBuffer.unshift(new Float32Array(obsRaw));

    // Concatenate all history steps
    const result = new Float32Array(inputDim);
    let offset = 0;
    for (const step of historyBuffer) {
      result.set(step, offset);
      offset += step.length;
    }
    return result;
  }

  // Case 2: HIMLoco 64-dim encoder (pad with zeros)
  if (inputDim === 64) {
    const result = new Float32Array(64);
    result.set(obsRaw, 0);  // obsRaw first 45 dims, rest stay zero
    return result;
  }

  // Case 3: Single step, no history (45-dim)
  if (inputDim === 45) {
    return new Float32Array(obsRaw);
  }

  throw new Error(`Unsupported ONNX input dim: ${inputDim}`);
}

/**
 * Create and initialize a history buffer with zeros.
 *
 * @param historyLen Number of history steps (typically 6 for numObs=270)
 * @param obsDim Single-step observation dimension (45)
 */
export function createHistoryBuffer(historyLen: number, obsDim: number): Float32Array[] {
  const buf: Float32Array[] = [];
  for (let i = 0; i < historyLen; i++) {
    buf.push(new Float32Array(obsDim));
  }
  return buf;
}
