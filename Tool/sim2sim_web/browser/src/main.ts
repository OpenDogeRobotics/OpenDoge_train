/**
 * Main entry point for the OpenDoge Sim2Sim browser-native dashboard.
 *
 * Initialization order:
 *   1. MuJoCo WASM engine (load XML, compile model)
 *   2. Three.js viewer (scene, camera, lights)
 *   3. Robot model (load glTF meshes, build kinematic tree)
 *   4. ONNX policy (load first model)
 *   5. Control loop (wire everything together)
 *   6. UI panels (control panel, metrics, charts, training comparison)
 */

import { MujocoEngine } from './engine/MujocoEngine';
import { OnnxPolicy } from './engine/OnnxPolicy';
import { ControlLoop } from './engine/ControlLoop';
import { ThreeViewer } from './rendering/ThreeViewer';
import { RobotModel } from './rendering/RobotModel';
import { ControlPanel } from './ui/ControlPanel';
import { MetricsPanel } from './ui/MetricsPanel';
import { Charts } from './ui/Charts';
import { TrainingComparison } from './ui/TrainingComparison';

// ── DOM helpers ─────────────────────────────────────────────────────

const $ = (id: string): HTMLElement => document.getElementById(id)!;

// ═══════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  const metrics = new MetricsPanel();
  metrics.setStatus(false, 'Initializing...');
  metrics.setLoadingText('Loading MuJoCo WASM...');

  try {
    // ── 1. MuJoCo Engine ──────────────────────────────────────────
    metrics.setLoadingText('Loading MuJoCo WASM...');
    const engine = new MujocoEngine();
    await engine.initialize('/xml/scene.xml');
    console.log('[main] MuJoCo engine initialized');

    // ── 2. Three.js Viewer ────────────────────────────────────────
    const canvas = $('viewer-canvas') as HTMLCanvasElement;
    const viewer = new ThreeViewer(canvas);
    // Hide placeholder
    const placeholder = $('viewer-placeholder');
    if (placeholder) placeholder.style.display = 'none';
    const hint = $('viewer-hint');
    if (hint) hint.style.display = 'block';
    console.log('[main] Three.js viewer initialized');

    // ── 3. Robot Model (load meshes) ──────────────────────────────
    metrics.setLoadingText('Loading robot meshes...');
    const robotModel = new RobotModel();
    await robotModel.load('/meshes');
    robotModel.attachTo(viewer.baseGroup);
    // Initial pose
    const qpos = engine.getQpos();
    robotModel.updateJointPositions(qpos);
    robotModel.updateBasePose(viewer.baseGroup, qpos);
    console.log('[main] Robot model loaded');

    // ── 4. ONNX Policy ────────────────────────────────────────────
    metrics.setLoadingText('Loading ONNX policy...');
    const policy = new OnnxPolicy();

    // Scan for available models
    const modelPaths = await fetchModelList();
    if (modelPaths.length > 0) {
      await policy.loadModel(modelPaths[0].path);
      console.log(`[main] Policy loaded: ${policy.currentModel?.name}`);
    } else {
      console.warn('[main] No ONNX models found');
    }

    // ── 5. Control Loop ──────────────────────────────────────────
    metrics.setLoadingText('Starting control loop...');
    const loop = new ControlLoop(engine, policy, viewer, robotModel);
    loop.setModelList(modelPaths);

    // ── 6. UI Panels ─────────────────────────────────────────────
    const controlPanel = new ControlPanel(loop);
    controlPanel.init();
    controlPanel.setModelList(modelPaths);
    if (policy.currentModel) {
      controlPanel.setCurrentModel(policy.currentModel.path);
      controlPanel.updateModelName(policy.currentModel.name);
    }

    const charts = new Charts();
    charts.init();

    const trainingComp = new TrainingComparison();
    trainingComp.init();

    // Wire callbacks
    loop.callbacks = {
      onStateUpdate: (state) => {
        metrics.update(state);
        charts.feedState(state);
      },
      onModelChange: (model) => {
        controlPanel.setCurrentModel(model.path);
        controlPanel.updateModelName(model.name);
      },
      onStatusChange: (status) => {
        if (status === 'running') {
          metrics.setStatus(true, '● running');
        } else {
          metrics.setStatus(true, '● paused');
        }
      },
    };

    // ── 7. Ready ──────────────────────────────────────────────────
    metrics.setStatus(true, '● ready');
    metrics.setLoadingText('Ready — press ▶ Start');

    // Auto-start simulation
    loop.start();
    controlPanel.updateButtonState();

    console.log('[main] Dashboard ready');

  } catch (err) {
    console.error('[main] Initialization failed:', err);
    metrics.setStatus(false, '● error');
    metrics.setLoadingText(`Error: ${(err as Error).message}`);
  }
}

// ═══════════════════════════════════════════════════════════════════════

async function fetchModelList(): Promise<{ name: string; path: string; step: number | null }[]> {
  // Scan the /onnx/ directory for .onnx files
  // Since browser can't list directories, we hardcode known models
  // and verify their existence via fetch HEAD requests
  const knownModels = [
    'flat_opendoge_5700.onnx',
    'flat_opendoge_9000_omni.onnx',
    'flat_opendoge_fresh_6000.onnx',
    'gen52_model4800.onnx',
    'gen52_robust_3300.onnx',
  ];

  const results: { name: string; path: string; step: number | null }[] = [];

  for (const filename of knownModels) {
    const path = `/onnx/${filename}`;
    try {
      const resp = await fetch(path, { method: 'HEAD' });
      if (resp.ok) {
        const name = filename.replace('.onnx', '');
        const stepMatch = name.match(/(\d{3,})/);
        results.push({
          name,
          path,
          step: stepMatch ? parseInt(stepMatch[1], 10) : null,
        });
      }
    } catch {
      // file doesn't exist
    }
  }

  return results;
}

// ═══════════════════════════════════════════════════════════════════════

main();
