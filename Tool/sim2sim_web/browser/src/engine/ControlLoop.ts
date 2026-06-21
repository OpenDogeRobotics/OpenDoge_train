/**
 * Control Loop — coordinates MuJoCo physics, ONNX policy inference, and Three.js rendering.
 *
 * Fixed-timestep accumulator:
 *   - Physics at 200 Hz  (SIM_DT = 0.005s)
 *   - Policy at 100 Hz   (every CONTROL_DECIMATION physics steps)
 *   - Rendering at rAF   (typically 60 fps)
 */

import { MujocoEngine } from './MujocoEngine';
import { OnnxPolicy } from './OnnxPolicy';
import { ThreeViewer } from '../rendering/ThreeViewer';
import { RobotModel } from '../rendering/RobotModel';
import { buildObsRaw, pdControl } from './math';
import {
  SIM_DT,
  CONTROL_DECIMATION,
  NUM_ACTIONS,
  ACTION_SCALE,
  CMD_SCALE,
  DEFAULT_DOF_POS,
  KPS,
  KDS,
  TORQUE_LIMITS,
  ANG_VEL_SCALE,
  DOF_POS_SCALE,
  DOF_VEL_SCALE,
} from './config';
import type { SimState, ModelInfo } from '../types';

export interface ControlLoopCallbacks {
  onStateUpdate?: (state: SimState) => void;
  onModelChange?: (model: ModelInfo) => void;
  onStatusChange?: (status: string) => void;
}

export class ControlLoop {
  private engine: MujocoEngine;
  private policy: OnnxPolicy;
  private viewer: ThreeViewer;
  private robotModel: RobotModel;

  private running = false;
  private cmd: Float32Array = new Float32Array(3);
  private action: Float32Array = new Float32Array(NUM_ACTIONS);
  private targetDofPos: Float32Array = new Float32Array(NUM_ACTIONS);
  private torqueSignal: Float32Array = new Float32Array(NUM_ACTIONS);

  private simAccumulator = 0;
  private lastFrameTime = 0;
  private stepCounter = 0;
  private frameCount = 0;

  // Inference state (fire-and-forget; reuse last action if busy)
  private inferencePending = false;

  callbacks: ControlLoopCallbacks = {};
  private modelList: ModelInfo[] = [];

  constructor(
    engine: MujocoEngine,
    policy: OnnxPolicy,
    viewer: ThreeViewer,
    robotModel: RobotModel
  ) {
    this.engine = engine;
    this.policy = policy;
    this.viewer = viewer;
    this.robotModel = robotModel;
  }

  // ═══════════════════════════════════════════════════════════════════
  // Model management
  // ═══════════════════════════════════════════════════════════════════

  setModelList(models: ModelInfo[]): void { this.modelList = models; }
  getModelList(): ModelInfo[] { return this.modelList; }
  getCurrentModel(): ModelInfo | null { return this.policy.currentModel; }

  async loadModel(path: string): Promise<void> {
    const wasRunning = this.running;
    if (wasRunning) this.pause();
    await this.policy.loadModel(path);
    this.callbacks.onModelChange?.(this.policy.currentModel!);
    if (wasRunning) this.start();
  }

  async nextModel(): Promise<void> {
    const cur = this.policy.currentModel;
    if (!cur || this.modelList.length === 0) return;
    for (let i = 0; i < this.modelList.length; i++) {
      if (this.modelList[i].path === cur.path) {
        await this.loadModel(this.modelList[(i + 1) % this.modelList.length].path);
        return;
      }
    }
  }

  async prevModel(): Promise<void> {
    const cur = this.policy.currentModel;
    if (!cur || this.modelList.length === 0) return;
    for (let i = 0; i < this.modelList.length; i++) {
      if (this.modelList[i].path === cur.path) {
        const idx = (i - 1 + this.modelList.length) % this.modelList.length;
        await this.loadModel(this.modelList[idx].path);
        return;
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // Control
  // ═══════════════════════════════════════════════════════════════════

  setCmd(vx: number, vy: number, vyaw: number): void {
    this.cmd[0] = vx; this.cmd[1] = vy; this.cmd[2] = vyaw;
  }

  reset(): void {
    this.engine.resetPose();
    this.cmd.fill(0);
    this.action.fill(0);
    this.targetDofPos.set(DEFAULT_DOF_POS);
    this.torqueSignal.fill(0);
    this.stepCounter = 0;
    this.simAccumulator = 0;
    this.policy.resetHistory();
    this.updateVisualization();
  }

  start(): void {
    if (this.running) return;
    if (!this.policy.isLoaded) return;
    this.running = true;
    this.lastFrameTime = performance.now() / 1000;
    this.callbacks.onStatusChange?.('running');
    requestAnimationFrame(this.frame.bind(this));
  }

  pause(): void {
    this.running = false;
    this.callbacks.onStatusChange?.('paused');
  }

  toggle(): void {
    if (this.running) this.pause(); else this.start();
  }

  get isRunning(): boolean { return this.running; }

  // ═══════════════════════════════════════════════════════════════════
  // Main loop
  // ═══════════════════════════════════════════════════════════════════

  private frame(timestamp: number): void {
    if (!this.running) return;

    const now = timestamp / 1000;
    let dt = now - this.lastFrameTime;
    this.lastFrameTime = now;

    if (dt > 0.05) dt = 0.05;
    if (dt <= 0) dt = SIM_DT;

    this.simAccumulator += dt;

    while (this.simAccumulator >= SIM_DT) {
      this.simAccumulator -= SIM_DT;

      if (this.stepCounter % CONTROL_DECIMATION === 0) {
        this.computeControl();
      }

      this.engine.step(this.torqueSignal);
      this.stepCounter++;
    }

    this.updateVisualization();

    this.frameCount++;
    if (this.callbacks.onStateUpdate && this.frameCount % 2 === 0) {
      const state = this.engine.getState(
        this.policy.currentModel?.name ?? '',
        this.cmd,
        this.action,
        this.targetDofPos
      );
      this.callbacks.onStateUpdate(state);
    }

    this.viewer.render();
    requestAnimationFrame(this.frame.bind(this));
  }

  // ═══════════════════════════════════════════════════════════════════
  // Policy inference
  // ═══════════════════════════════════════════════════════════════════

  private computeControl(): void {
    const qpos = this.engine.getQpos();
    const qvel = this.engine.getQvel();

    const obsRaw = buildObsRaw({
      qpos,
      qvel,
      gyroData: this.engine.getGyroData(),
      cmd: this.cmd,
      cmdScale: CMD_SCALE,
      angVelScale: ANG_VEL_SCALE,
      dofPosScale: DOF_POS_SCALE,
      dofVelScale: DOF_VEL_SCALE,
      action: this.action,
      defaultDofPos: DEFAULT_DOF_POS,
      useGyroSensor: this.engine.useGyroSensor,
      numActions: NUM_ACTIONS,
    });

    if (!this.inferencePending) {
      this.inferencePending = true;
      this.policy.infer(obsRaw).then((rawAction) => {
        this.action = rawAction;

        // Compute target positions: action * scale + default
        for (let i = 0; i < NUM_ACTIONS; i++) {
          this.targetDofPos[i] = rawAction[i] * ACTION_SCALE + DEFAULT_DOF_POS[i];
        }

        // PD torque computation
        const currentQ = this.engine.getJointPositions();
        const currentDq = this.engine.getJointVelocities();
        const zeroDq = new Float32Array(NUM_ACTIONS);
        const tau = pdControl(this.targetDofPos, currentQ, KPS, zeroDq, currentDq, KDS);

        for (let i = 0; i < NUM_ACTIONS; i++) {
          tau[i] = Math.max(-TORQUE_LIMITS[i], Math.min(TORQUE_LIMITS[i], tau[i]));
        }

        this.torqueSignal = tau;
        this.inferencePending = false;
      }).catch((err) => {
        console.error('[ControlLoop] Inference error:', err);
        this.inferencePending = false;
      });
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // Visualization
  // ═══════════════════════════════════════════════════════════════════

  private updateVisualization(): void {
    const qpos = this.engine.getQpos();
    this.robotModel.updateJointPositions(qpos);
    this.robotModel.updateBasePose(this.viewer.baseGroup, qpos);
  }
}
