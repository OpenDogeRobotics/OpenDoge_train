/**
 * Three.js viewer — scene, camera, lighting, ground, OrbitControls.
 *
 * Uses ResizeObserver for accurate container-based sizing (handles CSS
 * Grid layout without relying on window resize events).
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export class ThreeViewer {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  readonly controls: OrbitControls;

  /** Top-level robot container in world space. */
  readonly robotGroup: THREE.Group;
  /** Converts MuJoCo coords (Z-up) → Three.js (Y-up): -90° around X. */
  readonly mj2three: THREE.Group;
  /** Base-pose group (translation + quaternion from MuJoCo qpos). */
  readonly baseGroup: THREE.Group;

  private _resizeObserver: ResizeObserver | null = null;

  constructor(canvas: HTMLCanvasElement) {
    // ── Renderer ──────────────────────────────────────────────────
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;

    // ── Scene ─────────────────────────────────────────────────────
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x151518);

    // ── Camera ────────────────────────────────────────────────────
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 50);
    this.camera.position.set(0.8, 0.6, 0.8);
    this.camera.lookAt(0, 0.15, 0);

    // ── OrbitControls ─────────────────────────────────────────────
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0.15, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.15;
    this.controls.minDistance = 0.3;
    this.controls.maxDistance = 5;
    this.controls.maxPolarAngle = Math.PI * 0.8;
    this.controls.update();

    // ── Lighting ──────────────────────────────────────────────────
    this.setupLighting();

    // ── Ground ────────────────────────────────────────────────────
    this.setupGround();

    // ── Robot hierarchy ───────────────────────────────────────────
    this.robotGroup = new THREE.Group();
    this.robotGroup.name = 'robot';

    this.mj2three = new THREE.Group();
    this.mj2three.name = 'mj2three';
    this.mj2three.rotation.x = -Math.PI / 2;  // MuJoCo Z↑ → Three.js Y↑
    this.robotGroup.add(this.mj2three);

    this.baseGroup = new THREE.Group();
    this.baseGroup.name = 'basePose';
    this.mj2three.add(this.baseGroup);

    this.scene.add(this.robotGroup);

    // ── ResizeObserver (container-based, handles Grid layout) ─────
    const parent = canvas.parentElement;
    if (parent) {
      this._resizeObserver = new ResizeObserver(() => this._resize());
      this._resizeObserver.observe(parent);
    }
    // Defer one frame to let CSS Grid layout settle
    requestAnimationFrame(() => this._resize());
  }

  // ═══════════════════════════════════════════════════════════════════
  // Lighting
  // ═══════════════════════════════════════════════════════════════════

  private setupLighting(): void {
    const ambient = new THREE.AmbientLight(0x5a5a68, 0.55);
    this.scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0xfff8ee, 2.2);
    keyLight.position.set(2.5, 3.5, 1.5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    keyLight.shadow.camera.near = 0.1;
    keyLight.shadow.camera.far = 20;
    keyLight.shadow.camera.left = -2;
    keyLight.shadow.camera.right = 2;
    keyLight.shadow.camera.top = 2;
    keyLight.shadow.camera.bottom = -2;
    keyLight.shadow.bias = -0.0001;
    this.scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x8890a8, 0.4);
    fillLight.position.set(-1.5, 0.8, -1.5);
    this.scene.add(fillLight);

    const hemi = new THREE.HemisphereLight(0x808098, 0x403830, 0.35);
    this.scene.add(hemi);
  }

  // ═══════════════════════════════════════════════════════════════════
  // Ground
  // ═══════════════════════════════════════════════════════════════════

  private setupGround(): void {
    const grid = new THREE.GridHelper(4, 20, 0x3a3a44, 0x262630);
    grid.position.y = 0.001;
    this.scene.add(grid);

    const groundGeo = new THREE.PlaneGeometry(10, 10);
    const groundMat = new THREE.ShadowMaterial({ opacity: 0.3 });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = 0.002;
    ground.receiveShadow = true;
    this.scene.add(ground);
  }

  // ═══════════════════════════════════════════════════════════════════
  // Resize
  // ═══════════════════════════════════════════════════════════════════

  private _resize(): void {
    const canvas = this.renderer.domElement;
    const parent = canvas.parentElement;
    if (!parent) return;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    if (w === 0 || h === 0) return;

    // Cap DPR at 2 for performance; allow higher on low-res screens
    const dpr = Math.min(window.devicePixelRatio, 2);
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(h, 1);
    this.camera.updateProjectionMatrix();
  }

  // ═══════════════════════════════════════════════════════════════════
  // Render / dispose
  // ═══════════════════════════════════════════════════════════════════

  render(): void {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this._resizeObserver?.disconnect();
    this._resizeObserver = null;
    this.renderer.dispose();
  }
}
