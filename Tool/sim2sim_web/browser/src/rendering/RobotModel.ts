/**
 * Three.js Robot Model — loads glTF meshes and builds the kinematic tree.
 *
 * The hierarchy mirrors the MuJoCo XML body/joint tree. Joint rotations
 * are applied to the child body's Group, matching MuJoCo's convention
 * where a joint connects a parent body to a child body.
 *
 * Joint axis mapping (from XML):
 *   hip:   X axis (roll / abduction-adduction)
 *   thigh: Y axis (pitch / flexion-extension)
 *   calf:  Y axis (pitch / flexion-extension)
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { JOINT_ORDER, JOINT_AXIS } from '../engine/config';

// ── Body hierarchy definition ────────────────────────────────────────
// Each entry: [bodyName, parentBodyName (or null for root child), jointIndex (or -1), localPos]
// Positions extracted from Opendoge.xml body definitions.

interface BodyDef {
  name: string;
  parent: string | null;
  jointIdx: number;   // index into JOINT_ORDER, or -1 if no joint
  pos: [number, number, number];
}

const BODY_DEFS: BodyDef[] = [
  // Front Left
  { name: 'FL_hip',   parent: null,    jointIdx: 0,  pos: [ 0.1425,  0.039989, 0] },
  { name: 'FL_thigh', parent: 'FL_hip',   jointIdx: 1,  pos: [0,  0.08615,  0] },
  { name: 'FL_calf',  parent: 'FL_thigh', jointIdx: 2,  pos: [0, 0,       -0.1] },
  { name: 'FL_foot',  parent: 'FL_calf',  jointIdx: -1, pos: [0, 0,        0] },

  // Front Right
  { name: 'FR_hip',   parent: null,    jointIdx: 3,  pos: [ 0.1425, -0.040075, 0] },
  { name: 'FR_thigh', parent: 'FR_hip',   jointIdx: 4,  pos: [0, -0.0861,   0] },
  { name: 'FR_calf',  parent: 'FR_thigh', jointIdx: 5,  pos: [0, 0,        -0.1] },
  { name: 'FR_foot',  parent: 'FR_calf',  jointIdx: -1, pos: [0, 0,         0] },

  // Rear Left
  { name: 'RL_hip',   parent: null,    jointIdx: 6,  pos: [-0.1425,  0.040025, 0] },
  { name: 'RL_thigh', parent: 'RL_hip',   jointIdx: 7,  pos: [0,  0.08615,  0] },
  { name: 'RL_calf',  parent: 'RL_thigh', jointIdx: 8,  pos: [0, 0,       -0.1] },
  { name: 'RL_foot',  parent: 'RL_calf',  jointIdx: -1, pos: [0, 0,        0] },

  // Rear Right
  { name: 'RR_hip',   parent: null,    jointIdx: 9,  pos: [-0.1425, -0.040025, 0] },
  { name: 'RR_thigh', parent: 'RR_hip',   jointIdx: 10, pos: [0, -0.08615,  0] },
  { name: 'RR_calf',  parent: 'RR_thigh', jointIdx: 11, pos: [0, 0,       -0.1] },
  { name: 'RR_foot',  parent: 'RR_calf',  jointIdx: -1, pos: [0, 0,        0] },
];

// ═══════════════════════════════════════════════════════════════════════

export class RobotModel {
  /** The root group containing all robot meshes. Attach to viewer.robotGroup. */
  readonly root: THREE.Group;

  /** Named groups for each body part (used for mesh loading). */
  private bodyGroups = new Map<string, THREE.Group>();

  /** Joint groups that get rotated each frame, keyed by body name. */
  private jointBones = new Map<string, THREE.Group>();

  private loader: GLTFLoader;

  constructor() {
    this.root = new THREE.Group();
    this.root.name = 'robot_body';
    this.loader = new GLTFLoader();
  }

  // ═══════════════════════════════════════════════════════════════════
  // Loading
  // ═══════════════════════════════════════════════════════════════════

  /**
   * Build the kinematic tree (Groups with correct positions) and load all meshes.
   */
  async load(meshBasePath: string): Promise<void> {
    // Create Groups for all bodies and arrange hierarchy
    for (const def of BODY_DEFS) {
      const group = new THREE.Group();
      group.name = def.name;
      group.position.set(def.pos[0], def.pos[1], def.pos[2]);
      this.bodyGroups.set(def.name, group);

      // Track joint bones (bodies that have an actuated joint)
      if (def.jointIdx >= 0) {
        this.jointBones.set(def.name, group);
      }
    }

    // Arrange hierarchy
    for (const def of BODY_DEFS) {
      const group = this.bodyGroups.get(def.name)!;
      if (def.parent) {
        const parentGroup = this.bodyGroups.get(def.parent);
        if (parentGroup) {
          parentGroup.add(group);
        } else {
          this.root.add(group);
        }
      } else {
        // Direct child of robot base
        this.root.add(group);
      }
    }

    // Load meshes into their body groups
    const loadPromises = BODY_DEFS.map(def => this.loadMesh(def.name, meshBasePath));
    await Promise.all(loadPromises);

    // Load base_link mesh directly into root
    await this.loadMesh('base_link', meshBasePath, this.root);
  }

  private async loadMesh(name: string, basePath: string, targetGroup?: THREE.Group): Promise<void> {
    const group = targetGroup || this.bodyGroups.get(name);
    if (!group) return;

    const url = `${basePath}/${name}.glb`;
    try {
      const gltf = await this.loader.loadAsync(url);

      // Traverse loaded scene and add meshes to our group
      gltf.scene.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.castShadow = true;
          child.receiveShadow = true;

          // Apply a metallic-ish material with slight color variation
          if (child.material instanceof THREE.MeshStandardMaterial) {
            child.material.metalness = 0.3;
            child.material.roughness = 0.7;
          }
        }
      });

      // Copy all children from loaded scene to our group
      while (gltf.scene.children.length > 0) {
        group.add(gltf.scene.children[0]);
      }
    } catch (err) {
      console.warn(`[RobotModel] Failed to load mesh "${name}":`, err);
      // Fallback: add a small box to visualize the body
      this.addFallbackBox(group, name);
    }
  }

  private addFallbackBox(group: THREE.Group, name: string): void {
    const size = name === 'base_link' ? [0.3, 0.05, 0.12] :
                 name.includes('hip') ? [0.04, 0.04, 0.04] :
                 name.includes('thigh') ? [0.03, 0.08, 0.03] :
                 name.includes('calf') ? [0.02, 0.1, 0.02] :
                 name.includes('foot') ? [0.03, 0.03, 0.03] : [0.05, 0.05, 0.05];
    const geo = new THREE.BoxGeometry(size[0], size[1], size[2]);
    const mat = new THREE.MeshStandardMaterial({ color: 0xe94560, metalness: 0.2, roughness: 0.8 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    group.add(mesh);
  }

  // ═══════════════════════════════════════════════════════════════════
  // Update
  // ═══════════════════════════════════════════════════════════════════

  /**
   * Update joint rotations from MuJoCo joint positions (qpos[7:19]).
   */
  updateJointPositions(qpos: Float64Array): void {
    for (let i = 0; i < JOINT_ORDER.length; i++) {
      const jointName = JOINT_ORDER[i];
      const angle = qpos[7 + i];
      const axis = JOINT_AXIS[i];

      // Find the child body of this joint
      // Joint name: "FL_hip_joint" → child body: "FL_hip"
      const bodyName = jointName.replace('_joint', '');
      const bone = this.jointBones.get(bodyName);
      if (!bone) continue;

      // Apply rotation
      if (axis === 'X') {
        bone.rotation.x = angle;
      } else {
        bone.rotation.y = angle;
      }
    }
  }

  /**
   * Update base pose from MuJoCo qpos.
   * Coords are in MuJoCo frame (Z-up); caller's parent group handles
   * the MuJoCo→Three.js rotation via mj2three group.
   *
   * @param baseGroup Group inside mj2three coordinate frame
   * @param qpos MuJoCo qpos [x, y, z, qw, qx, qy, qz, ...]
   */
  updateBasePose(baseGroup: THREE.Group, qpos: Float64Array): void {
    // Position: MuJoCo (X→, Y←, Z↑) → applied directly inside mj2three frame
    baseGroup.position.set(qpos[0], qpos[1], qpos[2]);

    // Quaternion: MuJoCo [w, x, y, z] → Three.js quaternion.set(x, y, z, w)
    baseGroup.quaternion.set(qpos[4], qpos[5], qpos[6], qpos[3]);
  }

  /**
   * Add the kinematic tree (root group) as child of the given parent.
   */
  attachTo(parent: THREE.Group): void {
    parent.add(this.root);
  }
}
