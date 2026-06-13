"""Gait stability telemetry — per-leg phase, swing/stance, foot clearance, cycle timing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass
class GaitTelemetry:
    """Computes gait diagnostics from planner state and MuJoCo data."""

    leg_names: Sequence[str] = ("FL", "FR", "RL", "RR")
    cycle_time: float = 0.26
    duty_factor: float = 0.58
    nominal_feet_z: dict[str, float] | None = None

    def __post_init__(self):
        if self.nominal_feet_z is None:
            self.nominal_feet_z = {leg: -0.18 for leg in self.leg_names}
        self.swing_ratio = 1.0 - self.duty_factor

    def leg_phase(self, gait_time: float, leg: str) -> float:
        """Phase in [0, 1) within the trot cycle for this leg."""
        if self.cycle_time <= 0:
            return 0.0
        mid_stance = self.swing_ratio + 0.5 * self.duty_factor
        base = (mid_stance + gait_time / self.cycle_time) % 1.0
        offset = 0.0 if leg in {"FL", "RR"} else 0.5
        return (base + offset) % 1.0

    def is_stance(self, gait_time: float, leg: str) -> bool:
        return self.leg_phase(gait_time, leg) >= self.swing_ratio

    def phase_label(self, gait_time: float, leg: str) -> str:
        """Single-char label: 'S' = stance, 'W' = swing."""
        return "S" if self.is_stance(gait_time, leg) else "W"

    def foot_clearance(self, feet_world_z: Mapping[str, float], base_z: float = 0.0) -> dict[str, float]:
        """Clearance above nominal stance height in world frame (positive = lifted)."""
        return {leg: float(feet_world_z.get(leg, 0.0)) - (base_z + self.nominal_feet_z.get(leg, -0.18))
                for leg in self.leg_names}

    def cycle_progress(self, gait_time: float) -> float:
        """Phase of the primary diagonal pair (FL+RR), 0..1."""
        return self.leg_phase(gait_time, "FL")

    def summary_line(self, gait_time: float, active: bool,
                     feet_world_z: Mapping[str, float], base_z: float = 0.0) -> str:
        """One-line gait status for terminal output."""
        if not active:
            phases = "".join(f"{leg}=·" for leg in self.leg_names)
            return f"gait=idle phases=[{phases}]"
        phases = "".join(f" {leg}={self.phase_label(gait_time, leg)}" for leg in self.leg_names)
        clearance = self.foot_clearance(feet_world_z, base_z)
        clr = " ".join(f"{leg}:{clearance[leg]:+.3f}" for leg in self.leg_names)
        cycle_pct = self.cycle_progress(gait_time) * 100
        return f"cyc={cycle_pct:5.1f}% [{phases} ] clr=[{clr}]"
