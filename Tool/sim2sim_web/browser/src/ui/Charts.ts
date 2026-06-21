/**
 * Charts — Chart.js bar and line charts for joint positions, base height,
 * and velocity history.
 *
 * Chart updates are throttled (every ~5 state frames = ~10 Hz) to keep
 * rendering costs low while still providing smooth visual feedback.
 */

import { Chart, registerables } from 'chart.js';
import type { SimState } from '../types';

Chart.register(...registerables);

const HISTORY_MAX = 200;
const CHART_THROTTLE = 5;

export class Charts {
  private chartJoints: Chart<'bar', number[], string> | null = null;
  private chartHeight: Chart<'line', number[], string> | null = null;
  private chartVel: Chart<'line', number[], string> | null = null;

  // Rolling history
  private historyTime: number[] = [];
  private historyBaseZ: number[] = [];
  private historyLinVelX: number[] = [];
  private historyLinVelY: number[] = [];
  private historyLinVelZ: number[] = [];

  private frameCount = 0;
  private needsUpdate = false;

  init(): void {
    this.createCharts();
  }

  private createCharts(): void {
    // ── Joint positions bar chart ──────────────────────────────────
    const jCtx = (document.getElementById('chart-joints') as HTMLCanvasElement).getContext('2d')!;
    this.chartJoints = new Chart(jCtx, {
      type: 'bar',
      data: {
        labels: ['FL_h','FL_t','FL_c','FR_h','FR_t','FR_c','RL_h','RL_t','RL_c','RR_h','RR_t','RR_c'],
        datasets: [{
          data: new Array(12).fill(0),
          backgroundColor: '#4caf50',
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          y: { min: -2, max: 2, ticks: { color: '#888', font: { size: 8 } } },
          x: { ticks: { color: '#888', font: { size: 7 } } },
        },
        plugins: { legend: { display: false } },
      },
    });

    // ── Base height line chart ─────────────────────────────────────
    const hCtx = (document.getElementById('chart-height') as HTMLCanvasElement).getContext('2d')!;
    this.chartHeight = new Chart(hCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          data: [],
          borderColor: '#4caf50',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.1,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          y: { min: 0.05, max: 0.3, ticks: { color: '#888', font: { size: 9 } } },
          x: { ticks: { color: '#888', font: { size: 8 }, maxTicksLimit: 6 } },
        },
        plugins: { legend: { display: false } },
      },
    });

    // ── Velocity line chart ────────────────────────────────────────
    const vCtx = (document.getElementById('chart-vel') as HTMLCanvasElement).getContext('2d')!;
    this.chartVel = new Chart(vCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { data: [], borderColor: '#e94560', borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
          { data: [], borderColor: '#2196f3', borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
          { data: [], borderColor: '#ff9800', borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          y: { ticks: { color: '#888', font: { size: 9 } } },
          x: { ticks: { color: '#888', font: { size: 8 }, maxTicksLimit: 6 } },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  feedState(state: SimState): void {
    // Always record history
    const t = state.t;
    this.historyTime.push(t);
    this.historyBaseZ.push(state.base_pos[2] || 0);
    this.historyLinVelX.push(state.base_lin_vel[0] || 0);
    this.historyLinVelY.push(state.base_lin_vel[1] || 0);
    this.historyLinVelZ.push(state.base_lin_vel[2] || 0);

    // Trim history
    if (this.historyTime.length > HISTORY_MAX) {
      this.historyTime.shift();
      this.historyBaseZ.shift();
      this.historyLinVelX.shift();
      this.historyLinVelY.shift();
      this.historyLinVelZ.shift();
    }

    // Throttle chart updates
    this.frameCount++;
    this.needsUpdate = true;

    if (this.frameCount % CHART_THROTTLE === 0 && this.needsUpdate) {
      this.needsUpdate = false;

      // Joint bar chart
      if (this.chartJoints && state.joint_pos) {
        this.chartJoints.data.datasets[0].data = state.joint_pos;
        this.chartJoints.update('none');
      }

      // Base height
      if (this.chartHeight) {
        this.chartHeight.data.labels = this.historyTime.map(t => t.toFixed(2));
        this.chartHeight.data.datasets[0].data = this.historyBaseZ;
        this.chartHeight.update('none');
      }

      // Velocity
      if (this.chartVel) {
        this.chartVel.data.labels = this.historyTime.map(t => t.toFixed(2));
        this.chartVel.data.datasets[0].data = this.historyLinVelX;
        this.chartVel.data.datasets[1].data = this.historyLinVelY;
        this.chartVel.data.datasets[2].data = this.historyLinVelZ;
        this.chartVel.update('none');
      }
    }
  }

  resetHistory(): void {
    this.historyTime = [];
    this.historyBaseZ = [];
    this.historyLinVelX = [];
    this.historyLinVelY = [];
    this.historyLinVelZ = [];
    this.frameCount = 0;
  }
}
