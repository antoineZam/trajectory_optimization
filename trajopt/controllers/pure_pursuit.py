"""Pure-pursuit reference controller with a curvature-based speed profile.

This is a scripted, non-learned baseline. Its purpose is diagnostic: if this
controller cannot complete a lap, the environment or the physics is broken and
no RL agent will succeed either. It is exercised by
`tests/test_reference_controller.py` as the project's integration test, and it
gives a lap time to compare learned policies against.

The controller is deliberately simple and geometric:

* Lateral -- pure pursuit on the centerline. A target point is taken a
  speed-dependent distance ahead; the bicycle-model steering angle that arcs
  onto it is `atan2(2 * L * sin(alpha), ld)`.
* Longitudinal -- an offline speed profile from track curvature
  (`v = sqrt(a_lat / kappa)`), made feasible by a backward pass that enforces
  the braking limit, then a proportional tracker on speed error.

Note that `physics_engine.get_max_steering_angle` clamps the applied steering
as a function of speed, so the speed profile must stay conservative enough that
the requested steering remains achievable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.physics_engine import VehicleSpec, VehicleState
from utils.track import Track

GRAVITY = 9.81


@dataclass
class PurePursuitConfig:
    """Tuning for the reference controller."""

    # --- Lateral ---
    lookahead_gain: float = 0.6      # seconds: base lookahead = gain * speed
    lookahead_min: float = 6.0       # meters
    lookahead_max: float = 30.0      # meters

    # --- Speed profile ---
    lat_accel_fraction: float = 0.45  # Fraction of mu*g usable laterally
    brake_decel: float = 6.0          # m/s^2 assumed for the backward pass
    max_speed: float = 32.0           # m/s ceiling
    min_speed: float = 8.0            # m/s floor (never crawl to a stop)
    curvature_smoothing: int = 41     # Centerline samples in the moving average

    # --- Longitudinal tracker ---
    throttle_gain: float = 0.35      # per m/s of speed deficit
    brake_gain: float = 0.25         # per m/s of speed excess
    brake_deadband: float = 1.0      # m/s of excess tolerated before braking


def _closed_curvature(points: np.ndarray, arc_step: float) -> np.ndarray:
    """Signed curvature of a closed polyline, via periodic central differences.

    Args:
        points: (N, 2) closed centerline samples, uniformly spaced.
        arc_step: Arc length between consecutive samples, in meters.

    Returns:
        (N,) array of curvature in 1/m.
    """
    # np.gradient is not periodic, so roll explicitly to wrap the seam.
    d1 = (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)) / (2.0 * arc_step)
    d2 = (
        np.roll(points, -1, axis=0) - 2.0 * points + np.roll(points, 1, axis=0)
    ) / (arc_step ** 2)

    numerator = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    denominator = np.power(np.sum(d1 ** 2, axis=1), 1.5)
    return numerator / np.maximum(denominator, 1e-12)


def _moving_average_periodic(values: np.ndarray, window: int) -> np.ndarray:
    """Circular moving average, used to damp spline curvature noise."""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([values[-window:], values, values[:window]])
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[window:-window]


def compute_speed_profile(
    track: Track,
    spec: VehicleSpec,
    cfg: PurePursuitConfig | None = None,
) -> np.ndarray:
    """Build a feasible target-speed profile along the interpolated centerline.

    Args:
        track: Track providing the interpolated centerline and arc step.
        spec: Vehicle specification (used for the friction coefficient).
        cfg: Controller tuning; defaults are used when omitted.

    Returns:
        (N,) array of target speeds in m/s, aligned with
        `track.interpolated_centerline`.
    """
    cfg = cfg or PurePursuitConfig()
    centerline = track.interpolated_centerline
    arc_step = track.arc_step

    curvature = _closed_curvature(centerline, arc_step)
    curvature = np.abs(_moving_average_periodic(curvature, cfg.curvature_smoothing))

    # Cornering limit: v = sqrt(a_lat / kappa)
    a_lat = cfg.lat_accel_fraction * spec.mu0 * GRAVITY
    with np.errstate(divide="ignore"):
        v_curve = np.sqrt(a_lat / np.maximum(curvature, 1e-9))
    profile = np.clip(v_curve, cfg.min_speed, cfg.max_speed)

    # Backward pass: enforce v[i]^2 <= v[i+1]^2 + 2*a*ds around the closed loop.
    # Two sweeps are enough for the seam to converge on a closed track.
    n = len(profile)
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            nxt = profile[(i + 1) % n]
            profile[i] = min(profile[i], np.sqrt(nxt ** 2 + 2.0 * cfg.brake_decel * arc_step))

    return profile


class PurePursuitController:
    """Geometric path-following controller producing `RacingEnv` actions."""

    def __init__(
        self,
        track: Track,
        spec: VehicleSpec,
        cfg: PurePursuitConfig | None = None,
    ):
        """Precompute the speed profile for a given track and vehicle.

        Args:
            track: Track to follow.
            spec: Vehicle specification.
            cfg: Controller tuning; defaults are used when omitted.
        """
        self.track = track
        self.spec = spec
        self.cfg = cfg or PurePursuitConfig()
        self.speed_profile = compute_speed_profile(track, spec, self.cfg)

    def _target_index(self, closest_idx: int, distance: float) -> int:
        offset = self.track.index_offset_for_distance(distance)
        return (closest_idx + offset) % len(self.track.interpolated_centerline)

    def act(self, state: VehicleState) -> np.ndarray:
        """Compute an action for the current vehicle state.

        Args:
            state: Current vehicle state.

        Returns:
            Action array `[throttle, brake, steer_command]` matching
            `RacingEnv.action_space`.
        """
        centerline = self.track.interpolated_centerline
        position = np.array([state.x, state.y])
        speed = float(np.hypot(state.vx, state.vy))

        # Nearest centerline sample
        closest_idx = int(np.argmin(np.sum((centerline - position) ** 2, axis=1)))

        # --- Lateral: pure pursuit -------------------------------------------
        lookahead = float(
            np.clip(
                self.cfg.lookahead_gain * speed,
                self.cfg.lookahead_min,
                self.cfg.lookahead_max,
            )
        )
        target = centerline[self._target_index(closest_idx, lookahead)]

        to_target = target - position
        ld = float(np.linalg.norm(to_target))
        # Heading error towards the target point, wrapped to [-pi, pi]
        alpha = np.arctan2(to_target[1], to_target[0]) - state.yaw
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        delta = np.arctan2(2.0 * self.spec.wheelbase * np.sin(alpha), max(ld, 1e-3))
        steer_command = float(np.clip(delta / self.spec.max_steering_angle, -1.0, 1.0))

        # --- Longitudinal: track the profile a little way ahead --------------
        # Look ahead in the profile so braking starts before the corner, not in it.
        preview_idx = self._target_index(closest_idx, max(lookahead, 2.0 * speed))
        target_speed = float(
            min(
                self.speed_profile[closest_idx],
                self.speed_profile[preview_idx],
            )
        )

        error = target_speed - speed
        if error > 0.0:
            throttle = float(np.clip(self.cfg.throttle_gain * error, 0.0, 1.0))
            brake = 0.0
        elif error < -self.cfg.brake_deadband:
            throttle = 0.0
            brake = float(np.clip(self.cfg.brake_gain * (-error), 0.0, 1.0))
        else:
            throttle = 0.0
            brake = 0.0

        return np.array([throttle, brake, steer_command], dtype=np.float32)


def drive(
    env,
    controller: PurePursuitController,
    max_steps: int = 5_000,
    laps: float = 1.0,
) -> dict:
    """Drive `env` with `controller` until the requested distance is covered.

    Lap completion is measured here from cumulative *unwrapped* arc length
    rather than from `env.lap_completed`, so this function stays a valid
    reference even while the environment's own lap detection is broken
    (checkpoint 4 is unreachable -- see the Phase 1 fixes).

    Args:
        env: A `RacingEnv` instance (unwrapped).
        controller: Controller to drive with.
        max_steps: Hard step budget, on top of the environment's own
            max_steps. A lap is ~540 steps at the reference pace.
        laps: Number of laps to cover before stopping.

    Note that a completed lap now terminates the episode, so `laps` above 1.0
    only reaches further if the environment is configured not to end there.
    To cover several laps, call this once per lap -- it resets on entry.

    Returns:
        Dict with `laps_completed`, `distance_m`, `steps`, `lap_time_s`,
        `terminated`, `truncated`, `lap_completed`, `left_track`,
        `min_wheels_inside` and the `xs`/`ys` path.
    """
    env.reset()

    lap_length = env.track.lap_length
    target_distance = laps * lap_length

    cumulative = 0.0  # unwrapped progress, in laps
    previous = env.track.get_track_progress(np.array([env.state.x, env.state.y]))

    xs, ys, speeds = [], [], []
    min_wheels = 4
    terminated = truncated = False
    steps = 0

    while steps < max_steps:
        action = controller.act(env.state)
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1

        xs.append(env.state.x)
        ys.append(env.state.y)
        speeds.append(float(np.hypot(env.state.vx, env.state.vy)))
        min_wheels = min(min_wheels, env._count_wheels_inside_track())

        # Unwrap the [0, 1) progress signal across the start/finish seam
        current = env.track.get_track_progress(np.array([env.state.x, env.state.y]))
        delta = current - previous
        if delta > 0.5:
            delta -= 1.0
        elif delta < -0.5:
            delta += 1.0
        cumulative += delta
        previous = current

        if terminated or truncated:
            break
        if cumulative * lap_length >= target_distance:
            break

    return {
        "laps_completed": cumulative,
        "distance_m": cumulative * lap_length,
        "steps": steps,
        "lap_time_s": steps * env.cfg.dt,
        "terminated": terminated,
        "truncated": truncated,
        "lap_completed": env.lap_completed,
        # Terminating on a finished lap is success; terminating otherwise
        # means the vehicle left the track.
        "left_track": terminated and not env.lap_completed,
        "min_wheels_inside": min_wheels,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "xs": np.array(xs),
        "ys": np.array(ys),
    }
