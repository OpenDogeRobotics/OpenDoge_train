/** Shared type definitions for the sim2sim browser dashboard. */

export interface SimState {
  t: number;
  step: number;
  model: string;
  joint_pos: number[];
  joint_vel: number[];
  joint_tau: number[];
  action: number[];
  target_pos: number[];
  base_pos: number[];
  base_quat: number[];
  base_lin_vel: number[];
  base_ang_vel: number[];
  cmd_vel: number[];
  feet_contact: boolean[];
}

export interface ModelInfo {
  name: string;
  path: string;
  step: number | null;
}

export interface CameraState {
  azimuth: number;
  elevation: number;
  distance: number;
  lookat: number[];
}

export interface TrainingRun {
  name: string;
  path: string;
  num_checkpoints: number;
  latest_checkpoint: number | null;
}

export interface MetricPoint {
  step: number;
  value: number;
  wall_time: number;
}

export interface MetricSnapshot {
  [tag: string]: {
    step: number;
    value: number;
    wall_time: number;
  };
}

export interface RunComparison {
  [runName: string]: {
    [metricName: string]: number;
  };
}
