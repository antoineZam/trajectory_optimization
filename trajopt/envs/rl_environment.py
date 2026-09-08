"""
Racing Environment for Trajectory Optimization using Reinforcement Learning.

This environment implements a racing simulation where an RL agent learns to
drive a vehicle around a track as fast as possible while staying within boundaries.

Observation Space (21D):
    - Vehicle Dynamics (5D): speed, lateral velocity, yaw rate, slip angle, steering
    - Track Position (5D): lateral offset, heading error, progress, boundary distances
    - Lookahead (8D): 4 points ahead in vehicle-relative polar coordinates
    - Control Context (3D): previous throttle, brake, steer

Action Space (3D):
    - Throttle: [0, 1]
    - Brake: [0, 1]
    - Steering command: [-1, 1] (maps to physical steering angle)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, fields

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from physics.physics_engine import (
    VehicleSpec,
    VehicleState,
    get_max_steering_angle,
    get_wheel_positions,
    step_dynamics,
)
from utils.telemetry import RacingTelemetry, TelemetryFrame
from utils.track import Track

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RLConfig:
    """Configuration for the racing environment."""

    # Simulation parameters
    dt: float = 0.05  # Timestep in seconds
    # A lap of the sample track is ~540 steps (27 s) at the reference pace.
    # 1500 leaves room for a slow lap without burning a whole rollout on one
    # episode; the old 8000 assumed a lap took 8000 steps, a 15x error.
    max_steps: int = 1500  # Maximum steps per episode

    # Normalization constants (for observation scaling)
    max_speed: float = 50.0  # m/s (~180 km/h)
    max_lateral_velocity: float = 10.0  # m/s
    max_yaw_rate: float = 3.0  # rad/s
    max_lookahead_distance: float = 100.0  # meters

    # Target speeds for reward shaping
    target_speed: float = 25.0  # m/s (~90 km/h) - optimal racing speed
    min_speed: float = 5.0  # m/s - minimum acceptable speed

    # Reward weights, in O(1) units.
    #
    # The reward field is kept small and in physical units so that
    # VecNormalize's running return scale stays sane. Previously the return
    # standard deviation was 126 (ret_rms.var 15859), which made the critic
    # fit targets with a squared loss around 15900; at vf_coef 0.5 the value
    # gradient outweighed the policy gradient by ~1e6, and max_grad_norm 0.5
    # then spent the entire gradient budget on the critic. The policy's action
    # std was still 1.09 after 2M steps -- its initialisation.
    #
    # Reference lap (607.3 m, ~540 steps): progress 60.7, checkpoints 20,
    # lap bonus 15 -> ~96 total, with milestones about 37% of the return.
    progress_reward_scale: float = 0.1  # Reward per METER of forward progress
    speed_reward_scale: float = 0.03  # Reward for maintaining good speed
    centerline_reward_scale: float = 0.02  # Edge-proximity penalty scale

    # Milestone rewards: ~37% of a completed lap's return
    checkpoint_bonus: float = 5.0  # Per checkpoint (4 x 5 = 20 per lap)
    lap_completion_bonus: float = 15.0  # For completing a lap

    # Penalties
    off_track_penalty_scale: float = 0.2  # Per-step penalty when off track
    termination_penalty: float = -10.0  # One-time penalty on termination

    # Checkpoint system
    num_checkpoints: int = 4

    # Wheels that must stay inside the track; below this the episode ends.
    wheels_required_inside: int = 2

    # Initial-state randomization. Off gives a deterministic start on the
    # centerline at the start line, which is what the export needs for a
    # comparable lap; on spreads episodes over the whole track.
    randomize_reset: bool = True
    reset_lateral_fraction: float = 0.6  # of the half-width, either side
    reset_max_speed: float = 30.0  # m/s; lower bound is min_speed
    reset_heading_error: float = 0.2  # rad, either side of the track tangent

    @classmethod
    def from_dict(cls, config: dict | None) -> RLConfig:
        """Build an RLConfig from a plain dict (e.g. the Hydra `env` group).

        Unknown keys are ignored so that the config file can carry documentation
        entries without breaking instantiation.

        Args:
            config: Mapping of field name to value, or None for all defaults.

        Returns:
            RLConfig instance.
        """
        if not config:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in known})


# =============================================================================
# Main Environment Class
# =============================================================================


class RacingEnv(gym.Env):
    """
    Racing environment for reinforcement learning.

    The agent controls a vehicle around a track, receiving observations about
    the vehicle state and track geometry, and must learn to drive fast while
    staying on track.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    # Observation space dimensions
    VEHICLE_DYNAMICS_DIM = 5
    TRACK_POSITION_DIM = 5
    LOOKAHEAD_DIM = 8  # 4 points × 2 (distance, angle)
    CONTROL_CONTEXT_DIM = 3
    OBS_DIM = VEHICLE_DYNAMICS_DIM + TRACK_POSITION_DIM + LOOKAHEAD_DIM + CONTROL_CONTEXT_DIM  # 21

    # Number of lookahead points
    NUM_LOOKAHEAD_POINTS = 4

    # Distances ahead, in METERS of arc length along the centerline
    LOOKAHEAD_SPACING_M = (20.0, 50.0, 80.0, 100.0)

    # Half-width, in centerline indices, of the local closest-point search.
    # 64 indices is ~19 m at the default arc step, against a worst-case 2.5 m
    # of travel per step -- wide enough that the global fallback is rare.
    SEARCH_WINDOW = 64

    def __init__(
        self,
        track: Track,
        veh_spec: VehicleSpec,
        cfg: RLConfig | None = None,
        enable_telemetry: bool = True,
    ):
        """
        Initialize the racing environment.

        Args:
            track: Track object containing centerline and boundaries.
            veh_spec: Vehicle specification with physical parameters.
            cfg: Environment configuration.
            enable_telemetry: Whether to collect detailed telemetry data.
        """
        super().__init__()

        self.track = track
        self.spec = veh_spec
        self.cfg = cfg or RLConfig()

        self.telemetry = RacingTelemetry() if enable_telemetry else None

        # Define observation space with proper bounds
        # All observations are normalized to roughly [-1, 1] or [0, 1]
        self.observation_space = self._create_observation_space()

        # Action space: [throttle, brake, steer_command]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Episode state
        self.state: VehicleState | None = None
        self.step_count = 0
        self.track_progress = 0.0
        self.last_track_progress = 0.0  # For computing progress delta
        # Unwrapped lap counter: laps completed since reset. Unlike
        # track_progress this is not confined to [0, 1), so it can express
        # "a lap is finished" at all.
        self.cumulative_progress = 0.0
        self.lap_completed = False
        self.total_distance_traveled = 0.0  # Track total distance for metrics

        # Checkpoint tracking
        self.checkpoints_hit: set = set()
        self.current_checkpoint = 0

        # Control history (for observation)
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.prev_steer = 0.0

        # Previous position for distance calculation
        self._prev_position: np.ndarray | None = None

        # Track state caching (computed once per step for efficiency)
        self._cached_track_state: dict | None = None
        # Seed for the local closest-point search; None forces a global one.
        self._last_closest_idx: int | None = None

        # Statistics
        self.termination_stats = {
            "wheel_violations": 0,
            "off_track": 0,
            "max_steps": 0,
            "lap_completed": 0,
            "total_episodes": 0,
        }


    def _create_observation_space(self) -> spaces.Box:
        """
        Create the observation space with properly defined bounds.

        Returns:
            Box observation space with 21 dimensions.
        """
        # Define bounds for each observation component
        obs_low = np.array([
            # Vehicle Dynamics (5D)
            0.0,    # speed_normalized [0, 1+]
            -1.0,   # lateral_velocity_normalized [-1, 1]
            -1.0,   # yaw_rate_normalized [-1, 1]
            -1.0,   # slip_angle_normalized [-1, 1]
            -1.0,   # steering_angle_normalized [-1, 1]

            # Track Position (5D)
            -2.0,   # lateral_offset_normalized (can exceed track width)
            -1.0,   # heading_error_normalized [-1, 1] (cos of angle)
            0.0,    # track_progress [0, 1]
            0.0,    # left_boundary_distance_normalized [0, inf)
            0.0,    # right_boundary_distance_normalized [0, inf)

            # Lookahead (8D) - 4 points × (distance, angle)
            0.0, -1.0,   # Point 1: distance [0,1], angle [-1,1]
            0.0, -1.0,   # Point 2
            0.0, -1.0,   # Point 3
            0.0, -1.0,   # Point 4

            # Control Context (3D)
            0.0,    # prev_throttle [0, 1]
            0.0,    # prev_brake [0, 1]
            -1.0,   # prev_steer [-1, 1]
        ], dtype=np.float32)

        obs_high = np.array([
            # Vehicle Dynamics (5D)
            2.0,    # speed_normalized (can exceed max temporarily)
            1.0,    # lateral_velocity_normalized
            1.0,    # yaw_rate_normalized
            1.0,    # slip_angle_normalized
            1.0,    # steering_angle_normalized

            # Track Position (5D)
            2.0,    # lateral_offset_normalized
            1.0,    # heading_error_normalized
            1.0,    # track_progress
            3.0,    # left_boundary_distance_normalized
            3.0,    # right_boundary_distance_normalized

            # Lookahead (8D)
            1.0, 1.0,
            1.0, 1.0,
            1.0, 1.0,
            1.0, 1.0,

            # Control Context (3D)
            1.0,    # prev_throttle
            1.0,    # prev_brake
            1.0,    # prev_steer
        ], dtype=np.float32)

        return spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    # =========================================================================
    # Track State Computation
    # =========================================================================

    def _closest_centerline_index(self, position: np.ndarray) -> int:
        """Index of the closest interpolated centerline sample to `position`.

        The vehicle moves at most `max_speed * dt` per step -- 2.5 m at the
        default 50 m/s and dt 0.05, about 8 samples at the default arc step --
        so the answer is almost always a few indices from the previous one.
        Searching a window around it is O(1) instead of a scan over all 2000
        samples, which cost ~14% of the step budget.

        The window is a shortcut, never a different answer. Two conditions
        must both hold for its result to be accepted:

        * the minimum is interior, not on an edge -- an edge minimum means the
          true closest sample is probably just outside the window;
        * the point is within one track width of the sample found. Distance to
          a closed centerline has a local minimum against *every* branch of
          the track, so a point far inside the infield of this oval sits in a
          basin belonging to the far side. Such a minimum is interior and
          looks perfectly valid, which is why the edge check alone is not
          enough. On track the nearest branch is the only one within reach.

        Anything else -- a reset teleport, a run off track, a diagnostic query
        from anywhere -- falls back to the exact global search.

        Args:
            position: Vehicle position [x, y].

        Returns:
            Index into `track.interpolated_centerline`.
        """
        centerline = self.track.interpolated_centerline
        n = len(centerline)
        closest_idx = None

        if self._last_closest_idx is not None and 2 * self.SEARCH_WINDOW + 1 < n:
            offsets = (
                np.arange(
                    self._last_closest_idx - self.SEARCH_WINDOW,
                    self._last_closest_idx + self.SEARCH_WINDOW + 1,
                )
                % n
            )
            distances_sq = np.sum((centerline[offsets] - position) ** 2, axis=1)
            local = int(np.argmin(distances_sq))
            interior = 0 < local < len(offsets) - 1
            if interior and distances_sq[local] <= self.track.width ** 2:
                closest_idx = int(offsets[local])

        if closest_idx is None:
            closest_idx = self.track.closest_index(position)

        self._last_closest_idx = closest_idx
        return closest_idx

    def _compute_track_state(self, position: np.ndarray) -> dict:
        """
        Compute all track-relative state information for the current position.

        This is computed once per step and cached for efficiency.

        Args:
            position: Current vehicle position [x, y].

        Returns:
            Dictionary containing track state information.
        """
        centerline = self.track.interpolated_centerline

        # Find closest point on centerline
        closest_idx = self._closest_centerline_index(position)
        closest_point = centerline[closest_idx]
        center_dist = float(np.linalg.norm(position - closest_point))

        # Track progress (0 to 1)
        track_progress = closest_idx / len(centerline)

        # Compute track tangent vector at closest point
        if closest_idx < len(centerline) - 1:
            tangent = centerline[closest_idx + 1] - closest_point
        else:
            tangent = closest_point - centerline[closest_idx - 1]

        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-6:
            tangent = tangent / tangent_norm
        else:
            tangent = np.array([1.0, 0.0])

        # Track heading
        track_heading = np.arctan2(tangent[1], tangent[0])

        # Compute signed lateral offset (positive = right, negative = left)
        to_vehicle = position - closest_point
        # 2D cross product (right-hand rule). np.cross on 2-vectors is
        # deprecated in NumPy 2.0 and is also ~20x slower than the scalar form.
        signed_offset = tangent[0] * to_vehicle[1] - tangent[1] * to_vehicle[0]

        # Normal vector (pointing right from track direction)
        normal = np.array([-tangent[1], tangent[0]])

        return {
            "position": position.copy(),  # cache key, see _track_state_for
            "closest_idx": closest_idx,
            "closest_point": closest_point,
            "center_dist": center_dist,
            "track_progress": track_progress,
            "tangent": tangent,
            "normal": normal,
            "track_heading": track_heading,
            "signed_offset": signed_offset,
        }

    def _track_state_for(self, position: np.ndarray) -> dict:
        """Track state at `position`, reusing the cache when it already holds it.

        `step()` and `_get_obs()` both need the track state at the same
        position, and computing it is the single most expensive part of a step
        (two closest-point searches per step, ~14% of the step budget). The
        cached entry carries the position it was computed for, so reuse is
        only ever a hit on the same point.

        Args:
            position: Vehicle position [x, y].

        Returns:
            The track state dict, from cache or freshly computed.
        """
        cached = self._cached_track_state
        if cached is not None and np.array_equal(cached["position"], position):
            return cached

        state = self._compute_track_state(position)
        self._cached_track_state = state
        return state

    def _compute_lookahead_points(self, track_state: dict) -> np.ndarray:
        """
        Compute lookahead points on the track ahead of the vehicle.

        Args:
            track_state: Cached track state from _compute_track_state.

        Returns:
            Array of shape (NUM_LOOKAHEAD_POINTS, 2) with (distance, angle) for each point.
        """
        centerline = self.track.interpolated_centerline
        num_points = len(centerline)
        closest_idx = track_state["closest_idx"]
        vehicle_pos = np.array([self.state.x, self.state.y])
        vehicle_heading = self.state.yaw

        lookahead_data = np.zeros((self.NUM_LOOKAHEAD_POINTS, 2), dtype=np.float32)

        # Sample points at increasing distances ahead, in meters along the track
        lookahead_spacing = self.LOOKAHEAD_SPACING_M

        for i, spacing in enumerate(lookahead_spacing):
            # Get point ahead on track (wrap around for closed track).
            # interpolation_resolution is a POINT COUNT, so dividing a distance
            # by it yielded 0 for every spacing here: all four lookahead points
            # collapsed onto the vehicle's own position and the effective
            # horizon was 0 m. arc_step is the real meters-per-index factor.
            index_offset = self.track.index_offset_for_distance(spacing)
            ahead_idx = (closest_idx + index_offset) % num_points
            ahead_point = centerline[ahead_idx]

            # Vector from vehicle to lookahead point
            to_point = ahead_point - vehicle_pos
            distance = np.linalg.norm(to_point)

            # Angle relative to vehicle heading
            point_angle = np.arctan2(to_point[1], to_point[0])
            relative_angle = point_angle - vehicle_heading
            # Normalize to [-π, π]
            relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))

            # Normalize values
            distance_normalized = np.clip(distance / self.cfg.max_lookahead_distance, 0.0, 1.0)
            angle_normalized = relative_angle / np.pi  # [-1, 1]

            lookahead_data[i] = [distance_normalized, angle_normalized]

        return lookahead_data

    # =========================================================================
    # Observation Computation
    # =========================================================================

    def _get_obs(self) -> np.ndarray:
        """
        Compute the observation vector.

        Returns:
            Numpy array of shape (21,) containing the observation.
        """
        s = self.state
        current_pos = np.array([s.x, s.y])

        # Track state for this position. In step() this is a cache hit: the
        # state was already computed there for the very same position.
        track_state = self._track_state_for(current_pos)

        # =====================================================================
        # Vehicle Dynamics (5D)
        # =====================================================================

        # Speed (normalized by max expected speed)
        speed = np.hypot(s.vx, s.vy)
        speed_normalized = speed / self.cfg.max_speed

        # Lateral velocity (normalized)
        lateral_velocity_normalized = np.clip(
            s.vy / self.cfg.max_lateral_velocity, -1.0, 1.0
        )

        # Yaw rate (normalized)
        yaw_rate_normalized = np.clip(
            s.yaw_rate / self.cfg.max_yaw_rate, -1.0, 1.0
        )

        # Slip angle (vehicle body angle relative to velocity direction)
        if speed > 0.5:  # Avoid division by near-zero
            slip_angle = np.arctan2(s.vy, max(s.vx, 0.1))
        else:
            slip_angle = 0.0
        slip_angle_normalized = np.clip(slip_angle / (np.pi / 4), -1.0, 1.0)  # ±45° range

        # Current steering angle (from previous command, normalized)
        steering_angle_normalized = self.prev_steer  # Already [-1, 1]

        # =====================================================================
        # Track Position (5D)
        # =====================================================================

        # Lateral offset normalized by half track width
        # -1 = left edge, 0 = center, +1 = right edge
        half_width = self.track.width / 2.0
        lateral_offset_normalized = np.clip(
            track_state["signed_offset"] / half_width, -2.0, 2.0
        )

        # Heading error (as cosine: 1 = aligned, -1 = opposite direction)
        heading_diff = s.yaw - track_state["track_heading"]
        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
        heading_error_normalized = heading_diff / np.pi  # [−1, 1]

        # Track progress
        track_progress = track_state["track_progress"]

        # Boundary distances (normalized)
        # Using signed offset to compute approximate boundary distances
        left_boundary_dist = half_width - track_state["signed_offset"]
        right_boundary_dist = half_width + track_state["signed_offset"]
        left_boundary_normalized = np.clip(left_boundary_dist / half_width, 0.0, 3.0)
        right_boundary_normalized = np.clip(right_boundary_dist / half_width, 0.0, 3.0)

        # =====================================================================
        # Lookahead (8D)
        # =====================================================================

        lookahead_data = self._compute_lookahead_points(track_state)
        lookahead_flat = lookahead_data.flatten()  # 8 values

        # =====================================================================
        # Control Context (3D)
        # =====================================================================

        prev_throttle = self.prev_throttle
        prev_brake = self.prev_brake
        prev_steer = self.prev_steer

        actual_steering_angle = prev_steer * self.spec.max_steering_angle
        max_steer_allowed = get_max_steering_angle(self.spec, speed)
        applied_steer = np.clip(actual_steering_angle, -max_steer_allowed, max_steer_allowed)

        steering_angle_normalized = applied_steer / self.spec.max_steering_angle
        # =====================================================================
        # Assemble Observation
        # =====================================================================

        obs = np.array([
            # Vehicle Dynamics (5D)
            speed_normalized,
            lateral_velocity_normalized,
            yaw_rate_normalized,
            slip_angle_normalized,
            steering_angle_normalized,

            # Track Position (5D)
            lateral_offset_normalized,
            heading_error_normalized,
            track_progress,
            left_boundary_normalized,
            right_boundary_normalized,

            # Lookahead (8D)
            *lookahead_flat,

            # Control Context (3D)
            prev_throttle,
            prev_brake,
            prev_steer,
        ], dtype=np.float32)

        # Safety check: replace any NaN/Inf with zeros
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs

    # =========================================================================
    # Environment Interface
    # =========================================================================

    def _sample_initial_state(self) -> VehicleState:
        """Build the starting vehicle state for an episode.

        With `cfg.randomize_reset` the start is drawn from the whole track:
        any point around the lap, offset laterally, at any racing speed, with
        a small heading error. Without it the vehicle is placed on the
        centerline at the start line at `cfg.min_speed`, which is what the
        export needs to produce a comparable lap.

        Randomisation matters more than it looks. reset() previously always
        placed the vehicle at centerline[0] with vx=5.0 and no noise at all,
        so all four parallel workers produced identical rollouts: an 8192-step
        batch had the effective diversity of 2048 samples, the value function
        was only ever trained inside one narrow tube of state space, and 99
        recorded episodes agreed to 16 significant figures.

        Returns:
            The initial VehicleState.
        """
        centerline = self.track.interpolated_centerline

        if not self.cfg.randomize_reset:
            start_idx = 0
            lateral = 0.0
            speed = self.cfg.min_speed
            heading_error = 0.0
        else:
            start_idx = int(self.np_random.integers(len(centerline)))
            lateral = float(
                self.np_random.uniform(-self.cfg.reset_lateral_fraction,
                                       self.cfg.reset_lateral_fraction)
            ) * (self.track.width / 2.0)
            speed = float(
                self.np_random.uniform(self.cfg.min_speed, self.cfg.reset_max_speed)
            )
            heading_error = float(
                self.np_random.uniform(-self.cfg.reset_heading_error,
                                       self.cfg.reset_heading_error)
            )

        # Track tangent at the start point, using periodic differences so the
        # seam is not a special case.
        nxt = centerline[(start_idx + 1) % len(centerline)]
        prv = centerline[(start_idx - 1) % len(centerline)]
        tangent = nxt - prv
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-9)
        normal = np.array([-tangent[1], tangent[0]])

        position = centerline[start_idx] + normal * lateral
        yaw = float(np.arctan2(tangent[1], tangent[0])) + heading_error

        return VehicleState(
            x=float(position[0]),
            y=float(position[1]),
            yaw=yaw,
            vx=speed,
            vy=0.0,
            yaw_rate=0.0,
            gear=2,
            rpm=2000.0,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)

        # Update statistics
        self.termination_stats["total_episodes"] += 1

        # Print stats every 100 episodes
        if self.termination_stats["total_episodes"] % 100 == 0:
            self._print_termination_stats()
            if self.telemetry:
                self.telemetry.save_summaries()

        # Reset episode state
        self.step_count = 0
        self.track_progress = 0.0
        self.last_track_progress = 0.0
        self.cumulative_progress = 0.0
        self.total_distance_traveled = 0.0
        self.lap_completed = False
        self.checkpoints_hit = set()
        self.current_checkpoint = 0

        # Reset control history
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.prev_steer = 0.0

        # Drop the previous episode's track state: the new position is
        # unrelated to it, and a stale cache entry would be read as a hit.
        # The search seed goes with it -- a randomised start is nowhere near
        # where the last episode ended.
        self._cached_track_state = None
        self._last_closest_idx = None

        # Initialize vehicle
        self.state = self._sample_initial_state()

        # Initialize previous position for distance tracking
        self._prev_position = np.array([self.state.x, self.state.y])

        # Anchor the progress baseline to where the vehicle actually is, so
        # cumulative_progress counts from the start position rather than from
        # index 0. This matters as soon as the start state is randomised.
        self.last_track_progress = self.track.get_track_progress(self._prev_position)
        self.track_progress = self.last_track_progress

        # Start telemetry
        if self.telemetry:
            self.telemetry.start_episode(self.termination_stats["total_episodes"])

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        # Parse action
        throttle = float(np.clip(action[0], 0.0, 1.0))
        brake = float(np.clip(action[1], 0.0, 1.0))
        steer_command = float(np.clip(action[2], -1.0, 1.0))

        # Convert steering command to angle
        steer_angle = steer_command * self.spec.max_steering_angle

        # Update physics
        self.state = step_dynamics(
            self.spec, self.state, self.cfg.dt, throttle, brake, steer_angle
        )
        self.step_count += 1

        # Store control history for next observation
        self.prev_throttle = throttle
        self.prev_brake = brake
        self.prev_steer = steer_command

        # Compute track state. _get_obs() below reads it back from the cache
        # rather than recomputing it for the same position.
        current_pos = np.array([self.state.x, self.state.y])
        track_state = self._track_state_for(current_pos)

        # Progress accounting happens exactly once per step, here, so the
        # reward and the checkpoint logic read the same numbers.
        progress_delta = self._advance_progress(track_state["track_progress"])

        # Check checkpoint progress
        checkpoint_hit = self._check_checkpoint()

        # Compute reward
        reward = self._compute_reward(track_state, checkpoint_hit, progress_delta)

        # Check termination conditions. The wheel count is four
        # point-in-polygon tests, so it is done once here and shared with
        # telemetry rather than computed by each.
        wheels_inside = self._count_wheels_inside_track()
        terminated, truncated = self._check_termination(track_state, wheels_inside)

        # Apply termination penalty (only for bad terminations, not lap completion)
        if terminated and not self.lap_completed:
            reward += self.cfg.termination_penalty

        # Log telemetry
        if self.telemetry:
            self._log_telemetry(
                track_state, throttle, brake,
                steer_command, steer_angle,
                reward, checkpoint_hit, wheels_inside,
            )
            if terminated or truncated:
                self._end_episode_telemetry()

        # Pass episode results to the callback via info dict
        info = {}
        if terminated or truncated:
            info["checkpoints_hit"] = len(self.checkpoints_hit)
            info["lap_completed"] = self.lap_completed

        return self._get_obs(), reward, terminated, truncated, info

    @staticmethod
    def _unwrap_progress_delta(current: float, previous: float) -> float:
        """Shortest signed step between two [0, 1) progress values.

        Args:
            current: Progress this step.
            previous: Progress last step.

        Returns:
            Signed delta in (-0.5, 0.5], so crossing the start/finish line
            forwards reads as a small positive step rather than -0.99.
        """
        delta = current - previous
        if delta > 0.5:
            delta -= 1.0
        elif delta < -0.5:
            delta += 1.0
        return delta

    def _advance_progress(self, raw_progress: float) -> float:
        """Advance the unwrapped lap counter and return this step's delta.

        `raw_progress` is closest_idx / n_points: it lives in [0, 1) and wraps
        at the start/finish line. `cumulative_progress` is the same signal
        unwrapped, which is what checkpoint and lap detection need.

        Args:
            raw_progress: Progress reported by _compute_track_state.

        Returns:
            This step's signed progress delta, in laps.
        """
        delta = self._unwrap_progress_delta(raw_progress, self.last_track_progress)
        self.cumulative_progress += delta
        self.track_progress = raw_progress
        self.last_track_progress = raw_progress
        return delta

    def _check_checkpoint(self) -> bool:
        """Award the next checkpoint if cumulative progress has reached it.

        Checkpoints are at 25%, 50%, 75% and 100% of a lap (1-indexed).

        This previously divided the RAW progress signal by 1/num_checkpoints.
        Raw progress is in [0, 1), so int() capped the id at num_checkpoints-1
        and checkpoint 4 was unreachable. That made lap_completed permanently
        false, and with it the lap bonus, the lap termination and every
        stage-graduation criterion -- all dead code. Worse, once progress
        wrapped back towards 0 the id dropped while current_checkpoint stayed
        at 3, so no further checkpoint was ever awarded on later laps either.
        """
        reached = int(self.cumulative_progress * self.cfg.num_checkpoints)
        checkpoint_id = min(reached, self.cfg.num_checkpoints)

        if (
            checkpoint_id > 0
            and checkpoint_id == self.current_checkpoint + 1
            and checkpoint_id not in self.checkpoints_hit
        ):
            self.checkpoints_hit.add(checkpoint_id)
            self.current_checkpoint = checkpoint_id
            return True
        return False

    def _compute_reward(
        self, track_state: dict, checkpoint_hit: bool, progress_delta: float
    ) -> float:
        """
        Compute the reward for the current step.

        Reward Design Principles:
        - Dense positive rewards for good behavior (progress, speed, centerline)
        - Sparse large bonuses for milestones (checkpoints, lap completion)
        - Small penalties that don't dominate the positive signal
        - Net positive reward during normal on-track driving

        Measured magnitudes per step at ~23 m/s on the centerline:
        - Progress: ~0.11 (main signal, = meters travelled x 0.1)
        - Speed:    ~0.015 (bonus for being at or above target speed)
        - Edge:      0.0 while away from the boundary
        Over a reference lap (607.3 m, ~540 steps): progress 60.7,
        checkpoints 20, lap bonus 15, total ~96.
        """
        reward = 0.0
        current_pos = np.array([self.state.x, self.state.y])
        center_dist = track_state["center_dist"]
        half_width = self.track.width / 2.0
        on_track = center_dist <= half_width

        # =====================================================================
        # 1. PROGRESS REWARD (Primary learning signal)
        # =====================================================================
        # progress_delta is the unwrapped, signed step computed once per step
        # by _advance_progress. It is added SIGNED and unclipped: going
        # backwards must cost exactly what going forwards pays.
        #
        # Clipping the negative half made the reward field non-conservative,
        # so a round trip was paid for the outbound leg and not charged for
        # the return. Oscillating in place earned ~300 points for zero net
        # displacement -- 20% of a full lap's progress reward -- and that is
        # the exploit the agent actually found: 3002 points per episode with
        # zero checkpoints reached.
        #
        # Expressed in meters travelled along the track, so the weight is a
        # reward-per-meter and does not silently depend on track length or on
        # max_steps the way the old fixed 1500.0 gain did.
        reward += progress_delta * self.track.lap_length * self.cfg.progress_reward_scale

        # Also track distance for telemetry
        if self._prev_position is not None:
            distance = np.linalg.norm(current_pos - self._prev_position)
            self.total_distance_traveled += distance
        self._prev_position = current_pos.copy()

        # =====================================================================
        # 2. SPEED REWARD (Encourage maintaining good racing speed)
        # =====================================================================
        speed = np.hypot(self.state.vx, self.state.vy)

        if on_track:
            # Reward for being at or above target speed
            if speed >= self.cfg.target_speed:
                # Bonus for being fast (capped to avoid rewarding reckless speed)
                speed_bonus = min(speed / self.cfg.target_speed, 1.5) - 1.0
                speed_reward = speed_bonus * self.cfg.speed_reward_scale
            elif speed >= self.cfg.min_speed:
                # Proportional reward for acceptable speed range
                speed_ratio = (
                    (speed - self.cfg.min_speed)
                    / (self.cfg.target_speed - self.cfg.min_speed)
                )
                speed_reward = speed_ratio * self.cfg.speed_reward_scale * 0.5
            else:
                # Small penalty for being too slow (but not harsh)
                speed_reward = -0.02 * (self.cfg.min_speed - speed) / self.cfg.min_speed

            reward += speed_reward

        # =====================================================================
        # 3. BOUNDARY PENALTY (penalise being near or beyond the track edge,
        #    but don't reward the centerline -- let the agent find its own racing line)
        # =====================================================================
        if not on_track:
            off_track_amount = (center_dist - half_width) / half_width
            reward -= self.cfg.off_track_penalty_scale * min(off_track_amount, 2.0)
        elif center_dist > half_width * 0.8:
            # Gentle ramp starting at 80% of half-width (near the edge)
            edge_proximity = (center_dist - half_width * 0.8) / (half_width * 0.2)
            reward -= self.cfg.centerline_reward_scale * edge_proximity

        # =====================================================================
        # 4. CHECKPOINT MILESTONE BONUS
        # =====================================================================
        if checkpoint_hit:
            checkpoint_reward = self.cfg.checkpoint_bonus
            reward += checkpoint_reward
            # Only the telemetry-enabled worker prints. With laps now
            # completable this fires several times per lap; from every worker
            # it would flood console.log and interleave unreadably.
            if self.telemetry:
                print(
                    f"CHECKPOINT {len(self.checkpoints_hit)}"
                    f"/{self.cfg.num_checkpoints}"
                    f" HIT! (+{checkpoint_reward:.0f})"
                )

        # =====================================================================
        # 5. LAP COMPLETION BONUS
        # =====================================================================
        if self.cumulative_progress >= 1.0 and not self.lap_completed:
            self.lap_completed = True
            reward += self.cfg.lap_completion_bonus
            if self.telemetry:
                avg_speed = (
                    self.total_distance_traveled
                    / (self.step_count * self.cfg.dt)
                    if self.step_count > 0
                    else 0
                )
                print(f"LAP COMPLETED! Steps: {self.step_count}, Avg Speed: {avg_speed:.1f} m/s")

        return reward

    def _check_termination(
        self, track_state: dict, wheels_inside: int | None = None
    ) -> tuple[bool, bool]:
        """
        Check if the episode should end.

        Args:
            track_state: Track state for the current position.
            wheels_inside: Wheels inside the track, if already counted this
                step. Counting them is four point-in-polygon tests, so step()
                does it once and passes the result here and to telemetry.

        Returns:
            Tuple of (terminated, truncated).
        """
        terminated = False
        truncated = False

        center_dist = track_state["center_dist"]
        half_width = self.track.width / 2.0

        if wheels_inside is None:
            wheels_inside = self._count_wheels_inside_track()

        # Leaving the track ends the episode
        if wheels_inside < self.cfg.wheels_required_inside:
            terminated = True
            self.termination_stats["wheel_violations"] += 1

        # Safety check: very far from track
        if center_dist > half_width + 20.0:
            terminated = True
            self.termination_stats["off_track"] += 1

        # Episode limits.
        # A finished lap is a task-terminal state, not a time-limit
        # truncation: reporting it as truncated makes SB3 bootstrap V(s')
        # past the finish line, so the lap bonus gets counted twice -- once
        # as the bonus and again as the estimated value of the hereafter.
        if self.lap_completed:
            terminated = True
            self.termination_stats["lap_completed"] += 1
        elif self.step_count >= self.cfg.max_steps:
            truncated = True
            self.termination_stats["max_steps"] += 1

        return terminated, truncated

    def _count_wheels_inside_track(self) -> int:
        """Count how many wheels are inside the track boundaries."""
        wheel_positions = get_wheel_positions(self.spec, self.state)
        wheels_inside = 0

        for wheel_pos in wheel_positions:
            if self.track.is_point_inside_track(wheel_pos):
                wheels_inside += 1

        return wheels_inside

    # =========================================================================
    # Telemetry and Logging
    # =========================================================================

    def _log_telemetry(
        self,
        track_state: dict,
        throttle: float,
        brake: float,
        steer_command: float,
        steer_angle: float,
        reward: float,
        checkpoint_hit: bool,
        wheels_inside: int | None = None,
    ) -> None:
        """Log telemetry data for the current step."""
        if not self.telemetry:
            return

        if wheels_inside is None:
            wheels_inside = self._count_wheels_inside_track()

        speed = np.hypot(self.state.vx, self.state.vy)
        max_steer_allowed = get_max_steering_angle(self.spec, speed)

        frame = TelemetryFrame(
            timestamp=time.time(),
            step=self.step_count,
            x=self.state.x,
            y=self.state.y,
            yaw=self.state.yaw,
            speed=speed,
            vx=self.state.vx,
            vy=self.state.vy,
            yaw_rate=self.state.yaw_rate,
            throttle=throttle,
            brake=brake,
            steer_command=steer_command,
            steer_angle=steer_angle,
            max_steer_allowed=max_steer_allowed,
            track_progress=track_state["track_progress"],
            center_distance=track_state["center_dist"],
            racing_line_distance=track_state["center_dist"],
            wheels_inside=wheels_inside,
            current_checkpoint=len(self.checkpoints_hit),
            total_reward=reward,
            track_reward=0.0,  # Simplified reward structure
            speed_reward=0.0,
            checkpoint_reward=self.cfg.checkpoint_bonus if checkpoint_hit else 0.0,
            racing_line_reward=0.0,
            magnetism_reward=0.0,
            steering_penalty=0.0,
            smooth_steering_penalty=0.0,
        )

        self.telemetry.log_frame(frame)

    def _end_episode_telemetry(self) -> None:
        """End episode telemetry recording."""
        if not self.telemetry:
            return

        if self.lap_completed:
            reason = "lap_completed"
        elif self.step_count >= self.cfg.max_steps:
            reason = "max_steps"
        else:
            reason = "wheel_violation"

        self.telemetry.end_episode(self.lap_completed, reason, "standard")

        if self.termination_stats["total_episodes"] % 100 == 0:
            self.telemetry.print_training_progress(self.termination_stats["total_episodes"])

    def _print_termination_stats(self) -> None:
        """Print episode termination statistics."""
        total = self.termination_stats["total_episodes"]
        if total == 0:
            return

        recent = min(100, total)
        print(f"\n{'='*50}")
        print(f"EPISODE STATS (last {recent} episodes)")
        print(f"{'='*50}")
        print(f"Wheel violations: {self.termination_stats['wheel_violations']}")
        print(f"Off track: {self.termination_stats['off_track']}")
        print(f"Max steps: {self.termination_stats['max_steps']}")
        print(f"Lap completed: {self.termination_stats['lap_completed']}")
        print(f"{'='*50}")

        # Reset counters
        for key in self.termination_stats:
            if key != "total_episodes":
                self.termination_stats[key] = 0

    def render(self) -> None:
        """Render the environment (not implemented)."""
        pass
