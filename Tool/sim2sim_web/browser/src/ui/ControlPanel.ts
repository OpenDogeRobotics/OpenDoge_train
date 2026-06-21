/**
 * Control Panel — model selection, velocity sliders, start/stop/reset buttons,
 * and keyboard shortcuts.
 */

import type { ControlLoop } from '../engine/ControlLoop';
import type { ModelInfo } from '../types';

const $ = (id: string): HTMLElement => document.getElementById(id)!;

export class ControlPanel {
  private loop: ControlLoop;
  private modelList: ModelInfo[] = [];

  constructor(loop: ControlLoop) {
    this.loop = loop;
  }

  init(): void {
    this.bindModelSelect();
    this.bindSliders();
    this.bindButtons();
    this.bindKeyboard();
  }

  setModelList(models: ModelInfo[]): void {
    this.modelList = models;
    const sel = $('model-select') as HTMLSelectElement;
    sel.innerHTML = '';
    models.forEach((m) => {
      const opt = document.createElement('option');
      opt.value = m.path;
      const stepStr = m.step ? ` [step ${m.step}]` : '';
      opt.textContent = `${m.name}${stepStr}`;
      sel.appendChild(opt);
    });
  }

  setCurrentModel(path: string): void {
    ($('model-select') as HTMLSelectElement).value = path;
  }

  updateModelName(name: string): void {
    $('model-name').textContent = name;
  }

  // ── Model select ──────────────────────────────────────────────────

  private bindModelSelect(): void {
    $('model-select').addEventListener('change', async () => {
      const path = ($('model-select') as HTMLSelectElement).value;
      if (!path) return;
      await this.loop.loadModel(path);
    });

    $('btn-prev').addEventListener('click', () => this.loop.prevModel());
    $('btn-next').addEventListener('click', () => this.loop.nextModel());
  }

  // ── Velocity sliders ──────────────────────────────────────────────

  private bindSliders(): void {
    const sliderConfigs = [
      { id: 'vx', min: -1.5, max: 1.5, step: 0.01 },
      { id: 'vy', min: -1.0, max: 1.0, step: 0.01 },
      { id: 'vyaw', min: -2.0, max: 2.0, step: 0.01 },
    ];

    for (const cfg of sliderConfigs) {
      const slider = $(`slider-${cfg.id}`) as HTMLInputElement;
      slider.addEventListener('input', () => {
        const val = parseFloat(slider.value);
        $(`val-${cfg.id}`).textContent = val.toFixed(2);
        this.sendCmd();
      });
    }
  }

  private sendCmd(): void {
    const vx = parseFloat(($('slider-vx') as HTMLInputElement).value);
    const vy = parseFloat(($('slider-vy') as HTMLInputElement).value);
    const vyaw = parseFloat(($('slider-vyaw') as HTMLInputElement).value);
    this.loop.setCmd(vx, vy, vyaw);
  }

  get sliderValues(): { vx: number; vy: number; vyaw: number } {
    return {
      vx: parseFloat(($('slider-vx') as HTMLInputElement).value),
      vy: parseFloat(($('slider-vy') as HTMLInputElement).value),
      vyaw: parseFloat(($('slider-vyaw') as HTMLInputElement).value),
    };
  }

  setSliderValue(axis: 'vx' | 'vy' | 'vyaw', val: number): void {
    ($(`slider-${axis}`) as HTMLInputElement).value = String(val);
    $(`val-${axis}`).textContent = val.toFixed(2);
    this.sendCmd();
  }

  // ── Buttons ───────────────────────────────────────────────────────

  private bindButtons(): void {
    const btnStart = $('btn-start');
    btnStart.addEventListener('click', () => {
      this.loop.toggle();
      this.updateButtonState();
    });

    $('btn-reset').addEventListener('click', () => {
      this.loop.reset();
    });
  }

  updateButtonState(): void {
    const btn = $('btn-start');
    if (this.loop.isRunning) {
      btn.textContent = '⏹ Stop';
      btn.className = 'btn-stop';
    } else {
      btn.textContent = '▶ Start';
      btn.className = 'btn-start';
    }
  }

  // ── Keyboard shortcuts ────────────────────────────────────────────

  private bindKeyboard(): void {
    window.addEventListener('keydown', (e) => {
      // Don't intercept when typing in inputs
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

      switch (e.key) {
        case ' ':
          e.preventDefault();
          this.loop.toggle();
          this.updateButtonState();
          break;
        case '[':
          this.loop.prevModel();
          break;
        case ']':
          this.loop.nextModel();
          break;
        case 'r':
        case 'R':
          if (e.ctrlKey) break;
          this.loop.reset();
          break;
        case 'ArrowUp': {
          const vx = parseFloat(($('slider-vx') as HTMLInputElement).value) + 0.05;
          this.setSliderValue('vx', Math.min(1.5, vx));
          break;
        }
        case 'ArrowDown': {
          const vx = parseFloat(($('slider-vx') as HTMLInputElement).value) - 0.05;
          this.setSliderValue('vx', Math.max(-1.5, vx));
          break;
        }
        case 'ArrowLeft': {
          const vyaw = parseFloat(($('slider-vyaw') as HTMLInputElement).value) + 0.1;
          this.setSliderValue('vyaw', Math.min(2.0, vyaw));
          break;
        }
        case 'ArrowRight': {
          const vyaw = parseFloat(($('slider-vyaw') as HTMLInputElement).value) - 0.1;
          this.setSliderValue('vyaw', Math.max(-2.0, vyaw));
          break;
        }
      }
    });
  }
}
