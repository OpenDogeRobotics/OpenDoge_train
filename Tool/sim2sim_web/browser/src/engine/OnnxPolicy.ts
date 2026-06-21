/**
 * ONNX Runtime Web policy wrapper.
 *
 * Handles loading ONNX models, running inference, and managing the
 * observation history buffer.
 */

import * as ort from 'onnxruntime-web';
import { NUM_ACTIONS, NUM_ONE_STEP_OBS } from './config';
import { buildPolicyInput, createHistoryBuffer } from './math';
import type { ModelInfo } from '../types';

export class OnnxPolicy {
  private session: ort.InferenceSession | null = null;
  private inputName = '';
  private inputDim = 270;
  private outputName = '';
  private _currentModelPath = '';
  private _currentModelName = '';

  // Observation history buffer (6 steps × 45 dims = 270 for stacked input)
  private historyLength = 6;
  private historyBuffer: Float32Array[] = [];

  get currentModel(): ModelInfo | null {
    if (!this._currentModelPath) return null;
    return {
      name: this._currentModelName,
      path: this._currentModelPath,
      step: this.extractStep(this._currentModelName),
    };
  }

  get isLoaded(): boolean {
    return this.session !== null;
  }

  // ═════════════════════════════════════════════════════════════════════
  // Model loading
  // ═════════════════════════════════════════════════════════════════════

  async loadModel(url: string): Promise<void> {
    // Release old session
    if (this.session) {
      await this.session.release();
      this.session = null;
    }

    // Create new session
    this.session = await ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
    });

    // Extract model metadata
    this.inputName = this.session.inputNames[0];
    this.outputName = this.session.outputNames[0];
    this.inputDim = this.session.inputNames.length > 0
      ? (this.session as any)._handler?.inputNames?.[0]
      : 270;

    // Determine input dimension from session metadata
    try {
      // ort.InferenceSession in v1.27 may not expose input shapes directly;
      // infer from model path naming convention as fallback
      this.inputDim = 270; // default for OpenDoge
    } catch {
      this.inputDim = 270;
    }

    // Compute history buffer size
    this.historyLength = Math.max(1, Math.floor(this.inputDim / NUM_ONE_STEP_OBS));
    this.historyBuffer = createHistoryBuffer(this.historyLength, NUM_ONE_STEP_OBS);

    // Store model info
    this._currentModelPath = url;
    this._currentModelName = this.extractNameFromUrl(url);
  }

  // ═════════════════════════════════════════════════════════════════════
  // Inference
  // ═════════════════════════════════════════════════════════════════════

  /**
   * Run policy inference on a single-step observation.
   *
   * Returns the raw action vector (12 elements, range ~[-10, 10]).
   */
  async infer(obsRaw: Float32Array): Promise<Float32Array> {
    if (!this.session) {
      throw new Error('No model loaded');
    }

    // Build full policy input (with history stacking)
    const policyInput = buildPolicyInput(
      obsRaw,
      this.historyBuffer,
      this.inputDim,
      270
    );

    // Create ONNX tensor and run inference
    const inputTensor = new ort.Tensor('float32', policyInput, [1, this.inputDim]);
    const results = await this.session.run({ [this.inputName]: inputTensor });

    const outputData = results[this.outputName].data as Float32Array;

    // Clip actions to [-10, 10] (matches Python engine)
    const action = new Float32Array(NUM_ACTIONS);
    for (let i = 0; i < Math.min(NUM_ACTIONS, outputData.length); i++) {
      action[i] = Math.max(-10, Math.min(10, outputData[i]));
    }

    return action;
  }

  // ═════════════════════════════════════════════════════════════════════
  // Utilities
  // ═════════════════════════════════════════════════════════════════════

  resetHistory(): void {
    for (const step of this.historyBuffer) {
      step.fill(0);
    }
  }

  extractStep(name: string): number | null {
    const match = name.match(/(\d{3,})/);
    return match ? parseInt(match[1], 10) : null;
  }

  private extractNameFromUrl(url: string): string {
    const parts = url.split('/');
    const filename = parts[parts.length - 1];
    return filename.replace('.onnx', '');
  }

  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
    this._currentModelPath = '';
    this._currentModelName = '';
  }
}
