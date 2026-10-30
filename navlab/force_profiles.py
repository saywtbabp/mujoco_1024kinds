"""Piecewise command-force caps applied to the existing Probe controller.

World.probe is intentionally unchanged. A temporary Config adapter replaces
only its probe_force_max scalar with a function of elapsed integration steps.
The original lateral feedback, velocity damping and motor controller still run.
"""
from dataclasses import dataclass
import math
from . import runtime
import numpy as np
from .physics import NumericalFailure
from .probe_stream import capture_probe
from .multi_probe import digest, exact_rows


@dataclass(frozen=True)
class ForceSegment:
    start_s: float
    end_s: float
    force_limit_N: float


@dataclass(frozen=True)
class ForceProfile:
    id: str
    label: str
    duration_s: float
    segments: tuple
    command_force_budget_Ns: float
    standard: bool = False
    matched_budget: bool = True

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value, segments=tuple(ForceSegment(**s) for s in value["segments"])))

    @property
    def peak_limit_N(self):
        return max(s.force_limit_N for s in self.segments)

    @property
    def switch_times(self):
        return [s.start_s for s in self.segments[1:]]

    def force_limit_at(self, elapsed_s):
        if not math.isfinite(elapsed_s) or elapsed_s < 0 or elapsed_s > self.duration_s + 1e-9:
            raise ValueError("elapsed Probe time outside profile")
        for s in self.segments:
            if elapsed_s < s.end_s:
                return s.force_limit_N
        return self.segments[-1].force_limit_N

    def budget_until(self, elapsed_s):
        t = min(max(float(elapsed_s), 0.), self.duration_s)
        return sum(max(0., min(t, s.end_s)-s.start_s)*s.force_limit_N for s in self.segments)

    def as_dict(self):
        from dataclasses import asdict
        return asdict(self)

    @property
    def signature(self):
        return digest(self.as_dict())

    def validate(self, timestep):
        if not self.segments or not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("empty/invalid force profile")
        end = 0.
        for s in self.segments:
            if not all(math.isfinite(v) for v in (s.start_s, s.end_s, s.force_limit_N)):
                raise ValueError("nonfinite profile segment")
            if s.start_s != end or s.end_s <= s.start_s or not 0 < s.force_limit_N <= 2.2:
                raise ValueError("profile segments must be contiguous, ordered and within (0,2.2] N")
            for t in (s.start_s, s.end_s):
                # Original motors update every five physical steps. Switch on that cadence.
                if not np.isclose(t/(5*timestep), round(t/(5*timestep)), atol=1e-9, rtol=0):
                    raise ValueError("profile switch must coincide with an existing controller tick")
            end = s.end_s
        if end != self.duration_s or not np.isclose(self.budget_until(end), self.command_force_budget_Ns, atol=1e-12, rtol=0):
            raise ValueError("profile duration/budget does not match its segments")
        return self


class ProfileControlConfig:
    """Read-only forwarding adapter: only the Probe force cap is time dependent.

The clock is elapsed integration steps * timestep, numerically equivalent to
data.time - Probe start. This avoids a floating-point late switch at 0.20/0.60 s.
It never reads door state, hidden parameters, observations or candidate labels.
"""
    def __init__(self, base, profile, world):
        self.base, self.profile, self.world = base, profile, world

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def probe_force_max(self):
        return self.profile.force_limit_at(self.world.step_count*self.base.timestep)


DIAGNOSTIC_COLUMNS = ["time_s", "door_omega_rad_s", "contact_force_N", "uav_tilt_rad", "uav_gyro_rad_s", "contact"]
COMMAND_COLUMNS = ["time_s", "force_limit_N", "horizontal_command_norm_N", "horizontal_command_x_N", "horizontal_command_y_N"]


def simulate_profile_response(world, approach, theta, times, profile, record_commands=False):
    """Run one full original Probe, recording genuine samples without interpolation."""
    world.restore(approach, theta)
    base, old_callback = world.cfg, world.callback
    if not np.isclose(profile.duration_s, base.probe_duration, atol=1e-12, rtol=0):
        raise ValueError("profile duration differs from the original Probe window")
    start = float(world.data.time)
    diagnostics, commands = [], []
    def observe(current):
        _, _, R, gyro = current.observed()
        diagnostics.append([float(current.data.time-start), float(current.data.qvel[current.da]),
                            float(current.force), float(np.arccos(np.clip(R[2, 2], -1, 1))),
                            float(np.linalg.norm(gyro)), float(current.force > 0)])
    observe(world)
    original_step = world.step
    had_step_override = "step" in world.__dict__
    def record_step(horizontal=None, yaw=None):
        t = world.step_count*base.timestep
        if horizontal is not None:
            commands.append([t, profile.force_limit_at(t), float(np.linalg.norm(horizontal)), *horizontal.tolist()])
        return original_step(horizontal, yaw)
    world.callback = observe
    # Standard retains even the original Config object and scalar access path.
    if not profile.standard:
        world.cfg = ProfileControlConfig(base, profile, world)
    if record_commands:
        world.step = record_step
    try:
        rows, z, wall = capture_probe(world)
    finally:
        world.cfg, world.callback = base, old_callback
        if record_commands:
            if had_step_override:
                world.step = original_step
            else:
                del world.step
    if (np.any(world.data.warning.number) or not np.isfinite(world.data.qpos).all()
            or not np.isfinite(world.data.qvel).all()):
        raise NumericalFailure("terminal MuJoCo warning/nonfinite mechanical state")
    response = exact_rows(rows, times)
    np.testing.assert_allclose(response[-1], z, atol=1e-9, rtol=0)
    diagnostics = np.asarray(diagnostics)
    indices = [int(np.argmin(abs(diagnostics[:, 0]-t))) for t in times]
    np.testing.assert_allclose(diagnostics[indices, 0], times, atol=1e-9, rtol=0)
    diagnostics = diagnostics[indices].copy(); diagnostics[:, 0] = times
    if not np.isfinite(diagnostics).all():
        raise NumericalFailure("nonfinite physical diagnostics")
    commands = np.asarray(commands).reshape((-1, len(COMMAND_COLUMNS)))
    if record_commands:
        expected_times = np.arange(round(profile.duration_s/base.timestep))*base.timestep
        np.testing.assert_array_equal(commands[:, 0], expected_times)
        np.testing.assert_array_equal(commands[:, 1], [profile.force_limit_at(t) for t in expected_times])
        if np.any(commands[:, 2] > commands[:, 1]+1e-12):
            raise RuntimeError("horizontal command exceeded its configured time-dependent cap")
    return response, wall, diagnostics, commands
