/**
 * Metrics Panel — text-based numeric readouts for simulation state.
 *
 * Updates lightweight text fields every frame. Heavy chart updates are
 * handled separately by Charts.ts with throttling.
 */

import type { SimState } from '../types';

const $ = (id: string): HTMLElement => document.getElementById(id)!;

export class MetricsPanel {
  update(state: SimState): void {
    // Base height
    $('m-base-z').textContent = `${state.base_pos[2]?.toFixed(3)} m`;

    // Linear velocity
    $('m-lin-vel').textContent =
      `${state.base_lin_vel[0]?.toFixed(2)} / ${state.base_lin_vel[1]?.toFixed(2)} / ${state.base_lin_vel[2]?.toFixed(2)}`;

    // Angular velocity
    $('m-ang-vel').textContent =
      `${state.base_ang_vel[0]?.toFixed(2)} / ${state.base_ang_vel[1]?.toFixed(2)} / ${state.base_ang_vel[2]?.toFixed(2)}`;

    // Foot contacts
    const fc = state.feet_contact;
    $('m-contact').textContent =
      `FL:${fc[0] ? 1 : 0} FR:${fc[1] ? 1 : 0} RL:${fc[2] ? 1 : 0} RR:${fc[3] ? 1 : 0}`;

    // Step / model
    if (state.step !== undefined) {
      $('m-step').textContent = `step ${state.step} / ${state.model}`;
    }
  }

  setStatus(online: boolean, text?: string): void {
    const el = $('status');
    if (online) {
      el.textContent = text || '● ready';
      el.className = 'online';
    } else {
      el.textContent = text || '● loading';
      el.className = 'offline';
    }
  }

  setLoadingText(text: string): void {
    $('load-indicator').textContent = text;
  }
}
