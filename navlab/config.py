from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    model_source: str = "Mujoco_0"
    sobol_power: int = 10
    inertia_min: float = 0.05       # kg m^2 about the HINGE, not the centre of mass
    inertia_max: float = 0.80
    friction_min: float = 0.01     # N m: MuJoCo hinge frictionloss
    friction_max: float = 8.00
    timestep: float = 0.002
    hinge_damping: float = 0.05    # fixed, not an additional hidden parameter
    probe_duration: float = 1.20
    probe_force_max: float = 2.20  # commanded horizontal force cap, N
    push_force_max: float = 2.20
    probe_velocity_damping: float = 1.2
    push_timeout: float = 10.0
    route_timeout: float = 85.0
    flight_speed: float = 0.65
    open_angle_deg: float = 75.0
    clearance: float = 0.29
    grid_resolution: float = 0.06
    merge_position_tolerance: float = 0.08
    merge_speed_tolerance: float = 0.06
    merge_yaw_tolerance: float = 0.10
    merge_gyro_tolerance: float = 0.15
    merge_dwell: float = 0.30
    force_scale: float = 0.30      # engineering consistency tolerance, N
    angle_scale: float = 0.035     # rad; not a Gaussian standard deviation
    alpha: float = 2.0
    geometry_threshold: float = 3.0  # m
    disagreement_threshold: float = 0.36  # corresponds to p_D in [0.1, 0.9]
    workers: int = 4

    def validate(self):
        if self.model_source not in ("Mujoco_0", "Mujoco_1", "Mujoco_2"):
            raise ValueError("model_source must be Mujoco_0, Mujoco_1 or Mujoco_2")
        if not isinstance(self.sobol_power, int) or not 1 <= self.sobol_power <= 20:
            raise ValueError("sobol_power must be an integer in [1,20]")
        if not 0 < self.inertia_min < self.inertia_max:
            raise ValueError("invalid inertia domain")
        if not 0 <= self.friction_min < self.friction_max:
            raise ValueError("invalid friction domain")
        if not 0 < self.disagreement_threshold <= 1:
            raise ValueError("disagreement_threshold must be in (0,1]")
        if not 0 < self.open_angle_deg < 88:
            raise ValueError("open_angle_deg must be in (0,88)")
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError("nonfinite config: " + name)
        for name in ("timestep", "probe_duration", "probe_force_max", "push_force_max",
                     "push_timeout", "route_timeout", "flight_speed", "clearance",
                     "grid_resolution", "merge_position_tolerance", "merge_speed_tolerance",
                     "merge_yaw_tolerance", "merge_gyro_tolerance", "merge_dwell",
                     "force_scale", "angle_scale", "alpha", "workers"):
            if getattr(self, name) <= 0:
                raise ValueError(name + " must be positive")
        if self.geometry_threshold < 0 or self.hinge_damping < 0:
            raise ValueError("negative geometry threshold or damping")
        return self


def load_config(path=None):
    path = Path(path) if path else ROOT / "experiment_config.json"
    return Config(**json.loads(path.read_text(encoding="utf-8"))).validate()
