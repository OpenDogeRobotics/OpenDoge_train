/**
 * MuJoCo WASM engine — @mujoco/mujoco v3.9.0.
 *
 * RUNTIME BEHAVIOR (verified via Node.js test):
 *   - data.qpos / data.qvel / data.ctrl are Float64Array — use [i] NOT .get()/.set()
 *   - Enum values use .value (e.g. mjtObj.mjOBJ_BODY.value)
 *   - data.contact returns MjContactVec with .get(i) → MjContact
 */

import type { MjModel, MjData } from '@mujoco/mujoco';
import {
  SIM_DT, NUM_ACTIONS, INIT_BASE_HEIGHT, DEFAULT_DOF_POS,
  FOOT_BODY_NAMES, QPOS_JOINT_OFFSET,
} from './config';
import type { SimState } from '../types';

let _mujocoModule: any = null;
async function getMujoco(): Promise<any> {
  if (_mujocoModule) return _mujocoModule;
  _mujocoModule = await (await import('@mujoco/mujoco')).default();
  return _mujocoModule;
}

/** Merge scene.xml + Opendoge.xml, strip all mesh deps, return self-contained XML. */
function buildMergedXml(sceneXml: string, robotXml: string): string {
  let body = robotXml
    .replace(/<\?xml[^?]*\?>/gi, '')
    .replace(/<\/?mujoco[^>]*>/gi, '')
    .trim();
  body = body.replace(/\bmeshdir\s*=\s*"[^"]*"/gi, '');
  body = body.replace(/<mesh\b[^>]*\/>/gi, '');
  body = body.replace(/<geom\b[^>]*\btype\s*=\s*"mesh"[^>]*\/>/gi, '');
  body = body.replace(/<asset>\s*<\/asset>/gi, '');
  return sceneXml.replace(/<include\s+file\s*=\s*"Opendoge\.xml"\s*\/>/i, body);
}

/** Accessor helper: try .get(i) (MjContactVec etc), fall back to [i] (Float64Array). */
function idx(v: any, i: number): any {
  return typeof v?.get === 'function' ? v.get(i) : v?.[i];
}

export class MujocoEngine {
  private model!: MjModel;
  private data!: MjData;
  private mj!: any;
  private footBodyIds: number[] = [];
  private _useGyroSensor = true;
  private _stepCounter = 0;

  async initialize(sceneUrl: string): Promise<void> {
    this.mj = await getMujoco();

    const [sceneResp, robotResp] = await Promise.all([
      fetch(sceneUrl), fetch('/xml/Opendoge.xml'),
    ]);
    const mergedXml = buildMergedXml(
      await sceneResp.text(), await robotResp.text()
    );

    this.model = this.mj.from_xml_string(mergedXml);
    this.data = new this.mj.MjData(this.model);

    try { this.data.sensor('angular-velocity'); this._useGyroSensor = true; }
    catch { this._useGyroSensor = false; }

    const OBJ_BODY = this.mj.mjtObj.mjOBJ_BODY.value;
    for (const name of FOOT_BODY_NAMES) {
      const id = this.mj.mj_name2id(this.model, OBJ_BODY, name);
      this.footBodyIds.push(id >= 0 ? id : -1);
    }

    this.resetPose();
    console.log('[Mujoco] Ready nq=' + this.model.nq + ' nv=' + this.model.nv);
  }

  get useGyroSensor(): boolean { return this._useGyroSensor; }

  // ── Simulation ────────────────────────────────────────────────────

  resetPose(): void {
    const qp = this.data.qpos, qv = this.data.qvel;
    // Zero all
    for (let i = 0; i < qp.length; i++) qp[i] = 0;
    for (let i = 0; i < qv.length; i++) qv[i] = 0;
    // Base height + identity quat
    qp[2] = INIT_BASE_HEIGHT;
    qp[3] = 1; // qw
    // Default joint angles
    for (let i = 0; i < NUM_ACTIONS; i++) qp[QPOS_JOINT_OFFSET + i] = DEFAULT_DOF_POS[i];
    this.mj.mj_forward(this.model, this.data);
    this._stepCounter = 0;
  }

  step(ctrl: Float32Array): void {
    const c = this.data.ctrl;
    for (let i = 0; i < NUM_ACTIONS; i++) c[i] = ctrl[i];
    this.mj.mj_step(this.model, this.data);
    this._stepCounter++;
  }

  get stepCounter(): number { return this._stepCounter; }

  // ── State — Float64Array bracket access ───────────────────────────

  getQpos(): Float64Array { return this.data.qpos; }
  getQvel(): Float64Array { return this.data.qvel; }

  getGyroData(): Float32Array {
    if (this._useGyroSensor) {
      const d = this.data.sensor('angular-velocity').data;
      return new Float32Array([d[0], d[1], d[2]]);
    }
    const qv = this.data.qvel;
    return new Float32Array([qv[3], qv[4], qv[5]]);
  }

  getJointPositions(): Float32Array {
    const qp = this.data.qpos, out = new Float32Array(NUM_ACTIONS);
    for (let i = 0; i < NUM_ACTIONS; i++) out[i] = qp[QPOS_JOINT_OFFSET + i];
    return out;
  }
  getJointVelocities(): Float32Array {
    const qv = this.data.qvel, out = new Float32Array(NUM_ACTIONS);
    for (let i = 0; i < NUM_ACTIONS; i++) out[i] = qv[6 + i];
    return out;
  }
  getBasePosition(): Float32Array {
    const qp = this.data.qpos;
    return new Float32Array([qp[0], qp[1], qp[2]]);
  }
  getBaseQuaternion(): Float32Array {
    const qp = this.data.qpos;
    return new Float32Array([qp[3], qp[4], qp[5], qp[6]]);
  }
  getBaseLinearVelocity(): Float32Array {
    const qv = this.data.qvel;
    return new Float32Array([qv[0], qv[1], qv[2]]);
  }
  getBaseAngularVelocity(): Float32Array {
    const qv = this.data.qvel;
    return new Float32Array([qv[3], qv[4], qv[5]]);
  }

  getFootContacts(): boolean[] {
    const contacts = [false, false, false, false];
    for (let i = 0; i < 4; i++) {
      const bodyId = this.footBodyIds[i];
      if (bodyId < 0) continue;
      const geomAdr = idx(this.model.body_geomadr, bodyId) ?? 0;
      const geomNum = idx(this.model.body_geomnum, bodyId) ?? 0;
      for (let ci = 0; ci < this.data.ncon; ci++) {
        const c = this.data.contact.get(ci);
        if (!c) continue;
        for (let g = 0; g < geomNum; g++) {
          if (c.geom1 === geomAdr + g || c.geom2 === geomAdr + g) { contacts[i] = true; break; }
        }
        if (contacts[i]) break;
      }
    }
    return contacts;
  }

  getCtrl(): Float32Array {
    const c = this.data.ctrl, out = new Float32Array(NUM_ACTIONS);
    for (let i = 0; i < NUM_ACTIONS; i++) out[i] = c[i];
    return out;
  }

  getState(modelName: string, cmd: Float32Array, action: Float32Array, targetPos: Float32Array): SimState {
    return {
      t: +(this._stepCounter * SIM_DT).toFixed(4),
      step: this._stepCounter,
      model: modelName,
      joint_pos: Array.from(this.getJointPositions()),
      joint_vel: Array.from(this.getJointVelocities()),
      joint_tau: Array.from(this.getCtrl()),
      action: Array.from(action),
      target_pos: Array.from(targetPos),
      base_pos: Array.from(this.getBasePosition()),
      base_quat: Array.from(this.getBaseQuaternion()),
      base_lin_vel: Array.from(this.getBaseLinearVelocity()),
      base_ang_vel: Array.from(this.getBaseAngularVelocity()),
      cmd_vel: Array.from(cmd),
      feet_contact: this.getFootContacts(),
    };
  }

  close(): void {
    if (this.data) (this.data as any).delete();
    if (this.model) (this.model as any).delete();
  }
}
