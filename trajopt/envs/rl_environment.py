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
from dataclasses import dataclass
from typing import Tuple

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
from utils.curriculum import CurriculumLearning
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
    max_steps: int = 8000  # Maximum steps per episode
    
    # Normalization constants (for observation scaling)
    max_speed: float = 50.0  # m/s (~180 km/h)
    max_lateral_velocity: float = 10.0  # m/s
    max_yaw_rate: float = 3.0  # rad/s
    max_lookahead_distance: float = 100.0  # meters
    
    # Target speeds for reward shaping
    target_speed: float = 25.0  # m/s (~90 km/h) - optimal racing speed
    min_speed: float = 5.0  # m/s - minimum acceptable speed
    
    # Reward weights (tuned for stable learning)
    # Dense rewards (per-step, should be small but positive for good behavior)
    progress_reward_scale: float = 1.0  # Scale for distance-based progress
    speed_reward_scale: float = 0.1  # Reward for maintaining good speed
    centerline_reward_scale: float = 0.05  # Reward for staying centered
    
    # Milestone rewards (large, infrequent)
    checkpoint_bonus: float = 100.0  # Per checkpoint
    lap_completion_bonus: float = 1000.0  # For completing a lap
    
    # Penalties (should be smaller than positive rewards during normal operation)
    off_track_penalty_scale: float = 0.2  # Per-step penalty when off track
    termination_penalty: float = -50.0  # One-time penalty on termination
    
    # Checkpoint system
    num_checkpoints: int = 4


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
    
    def __init__(
        self,
        track: Track,
        veh_spec: VehicleSpec,
        cfg: RLConfig | None = None,
        enable_telemetry: bool = True,
        enable_curriculum: bool = True,
    ):
        """
        Initialize the racing environment.
        
        Args:
            track: Track object containing centerline and boundaries.
            veh_spec: Vehicle specification with physical parameters.
            cfg: Environment configuration.
            enable_telemetry: Whether to collect detailed telemetry data.
            enable_curriculum: Whether to use curriculum learning.
        """
        super().__init__()
        
        self.base_track = track
        self.track = track
        self.spec = veh_spec
        self.cfg = cfg or RLConfig()
        
        # Initialize optional systems
        self.curriculum = CurriculumLearning() if enable_curriculum else None
        self.telemetry = RacingTelemetry() if enable_telemetry else None
        
        # Define observation space with proper bounds
        # All observations are normalized to roughly [-1, 1] or [0, 1]
        self.observation_space = self._create_observation_space()
        
        # Action space: [throttle, brake, steer_command]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Episode state
        self.state: VehicleState | None = None
        self.step_count = 0
        self.track_progress = 0.0
        self.last_track_progress = 0.0  # For computing progress delta
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
        
        # Statistics
        self.termination_stats = {
            "wheel_violations": 0,
            "off_track": 0,
            "max_steps": 0,
            "lap_completed": 0,
            "total_episodes": 0,
        }
        
        # Curriculum settings
        self._curriculum_termination_mode = "normal"
        self._curriculum_wheels_required = 2
        self._off_track_count = 0
        
        # Apply initial curriculum settings
        self._update_curriculum_settings()
    
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
    
    def _update_curriculum_settings(self) -> None:
        """Update environment parameters based on curriculum stage."""
        if not self.curriculum:
            return
        
        track_params = self.curriculum.get_track_parameters()
        episode_params = self.curriculum.get_episode_parameters()
        
        # Create modified track with wider boundaries
        self._create_curriculum_track(track_params["width_multiplier"])
        
        # Update episode parameters
        self.cfg.max_steps = episode_params["max_steps"]
        self._curriculum_termination_mode = episode_params["termination_mode"]
        self._curriculum_wheels_required = episode_params["wheels_required_inside"]
    
    def _create_curriculum_track(self, width_multiplier: float) -> None:
        """Create a modified track with adjusted width for curriculum learning."""
        if width_multiplier == 1.0:
            self.track = self.base_track
            return
        
        self.track = Track(
            name=f"{self.base_track.name}_curriculum_{width_multiplier:.1f}x",
            centerline=self.base_track.centerline.copy(),
            width=self.base_track.width * width_multiplier,
            interpolation_resolution=self.base_track.interpolation_resolution,
        )
    
    # =========================================================================
    # Track State Computation
    # =========================================================================
    
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
        diff = centerline - position
        distances_sq = np.sum(diff ** 2, axis=1)
        closest_idx = np.argmin(distances_sq)
        closest_point = centerline[closest_idx]
        center_dist = np.sqrt(distances_sq[closest_idx])
        
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
        signed_offset = np.cross(tangent, to_vehicle)  # Right-hand rule
        
        # Normal vector (pointing right from track direction)
        normal = np.array([-tangent[1], tangent[0]])
        
        return {
            "closest_idx": closest_idx,
            "closest_point": closest_point,
            "center_dist": center_dist,
            "track_progress": track_progress,
            "tangent": tangent,
            "normal": normal,
            "track_heading": track_heading,
            "signed_offset": signed_offset,
        }
    
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
        
        # Sample points at increasing distances ahead
        # Spacing: 10, 25, 50, 100 meters ahead (approximately)
        lookahead_spacing = [20, 50, 100, 200]  # indices on interpolated track
        
        for i, spacing in enumerate(lookahead_spacing):
            # Get point ahead on track (wrap around for closed track)
            ahead_idx = (closest_idx + spacing) % num_points
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
        
        # Compute track state (cached for this step)
        track_state = self._compute_track_state(current_pos)
        self._cached_track_state = track_state
        
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
        heading_error_normalized = np.cos(heading_diff)  # [−1, 1]
        
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
    
    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        
        # Update statistics
        self.termination_stats["total_episodes"] += 1
        
        # Print stats every 100 episodes
        if self.termination_stats["total_episodes"] % 100 == 0:
            self._print_termination_stats()
            if self.telemetry:
                self.telemetry.save_summaries()
        
        # Handle curriculum progression
        if self.curriculum and hasattr(self, "_last_checkpoints_hit"):
            graduated = self.curriculum.record_episode_result(
                checkpoints_hit=getattr(self, "_last_checkpoints_hit", 0),
                lap_completed=getattr(self, "_last_lap_completed", False),
                episode_num=self.termination_stats["total_episodes"],
            )
            if graduated:
                self._update_curriculum_settings()
                self.curriculum.print_curriculum_status()
        
        # Reset episode state
        self.step_count = 0
        self.track_progress = 0.0
        self.last_track_progress = 0.0
        self.total_distance_traveled = 0.0
        self.lap_completed = False
        self.checkpoints_hit = set()
        self.current_checkpoint = 0
        self._off_track_count = 0
        
        # Reset control history
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.prev_steer = 0.0
        
        # Initialize vehicle at start of track
        x0, y0 = self.track.centerline[0]
        
        # Compute initial heading from first two track points
        if len(self.track.centerline) > 1:
            dx = self.track.centerline[1][0] - x0
            dy = self.track.centerline[1][1] - y0
            initial_yaw = np.arctan2(dy, dx)
        else:
            initial_yaw = 0.0
        
        self.state = VehicleState(
            x=x0,
            y=y0,
            yaw=initial_yaw,
            vx=5.0,  # Start with some forward velocity
            vy=0.0,
            yaw_rate=0.0,
            gear=2,
            rpm=2000.0,
        )
        
        # Initialize previous position for distance tracking
        self._prev_position = np.array([x0, y0])
        
        # Start telemetry
        if self.telemetry:
            self.telemetry.start_episode(self.termination_stats["total_episodes"])
        
        return self._get_obs(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
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
        
        # Compute track state
        current_pos = np.array([self.state.x, self.state.y])
        track_state = self._compute_track_state(current_pos)
        self._cached_track_state = track_state
        
        # Check checkpoint progress
        checkpoint_hit = self._check_checkpoint(track_state["track_progress"])
        
        # Compute reward
        reward = self._compute_reward(track_state, checkpoint_hit)
        
        # Check termination conditions
        terminated, truncated = self._check_termination(track_state)
        
        # Apply termination penalty (only for bad terminations, not lap completion)
        if terminated and not self.lap_completed:
            reward += self.cfg.termination_penalty
        
        # Log telemetry
        if self.telemetry:
            self._log_telemetry(track_state, throttle, brake, steer_command, steer_angle, reward, checkpoint_hit)
            if terminated or truncated:
                self._end_episode_telemetry()
        
        # Store for curriculum
        self._last_checkpoints_hit = len(self.checkpoints_hit)
        self._last_lap_completed = self.lap_completed
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _check_checkpoint(self, current_progress: float) -> bool:
        """Check if we've reached the next checkpoint."""
        checkpoint_spacing = 1.0 / self.cfg.num_checkpoints
        checkpoint_id = int(current_progress / checkpoint_spacing)
        checkpoint_id = min(checkpoint_id, self.cfg.num_checkpoints - 1)
        
        if checkpoint_id == self.current_checkpoint and checkpoint_id not in self.checkpoints_hit:
            self.checkpoints_hit.add(checkpoint_id)
            self.current_checkpoint = (checkpoint_id + 1) % self.cfg.num_checkpoints
            return True
        return False
    
    def _compute_reward(self, track_state: dict, checkpoint_hit: bool) -> float:
        """
        Compute the reward for the current step.
        
        Reward Design Principles:
        - Dense positive rewards for good behavior (progress, speed, centerline)
        - Sparse large bonuses for milestones (checkpoints, lap completion)
        - Small penalties that don't dominate the positive signal
        - Net positive reward during normal on-track driving
        
        Expected reward magnitudes per step at 25 m/s on centerline:
        - Progress: ~0.5 (main signal)
        - Speed: ~0.1 (bonus for good speed)  
        - Centerline: ~0.05 (bonus for being centered)
        - Total: ~0.65 per step (positive reinforcement)
        """
        reward = 0.0
        current_pos = np.array([self.state.x, self.state.y])
        center_dist = track_state["center_dist"]
        half_width = self.track.width / 2.0
        on_track = center_dist <= half_width
        
        # =====================================================================
        # 1. PROGRESS REWARD (Primary learning signal)
        # =====================================================================
        # Reward based on actual track progress (more robust than velocity)
        current_progress = track_state["track_progress"]
        
        # Handle wraparound at lap completion
        progress_delta = current_progress - self.last_track_progress
        if progress_delta < -0.5:  # Wrapped around (e.g., 0.99 -> 0.01)
            progress_delta += 1.0
        elif progress_delta > 0.5:  # Went backwards past start
            progress_delta -= 1.0
        
        # Scale progress to reward (~0.5 per step at good speed)
        # Full lap = 1.0, typical episode = 8000 steps, so per-step progress ~ 0.0001
        # We scale up significantly to make it the dominant signal
        progress_reward = progress_delta * 5000.0 * self.cfg.progress_reward_scale
        
        # Only reward forward progress, don't penalize backwards (let termination handle that)
        if progress_reward > 0:
            reward += progress_reward
        
        # Update for next step
        self.last_track_progress = current_progress
        
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
                speed_ratio = (speed - self.cfg.min_speed) / (self.cfg.target_speed - self.cfg.min_speed)
                speed_reward = speed_ratio * self.cfg.speed_reward_scale * 0.5
            else:
                # Small penalty for being too slow (but not harsh)
                speed_reward = -0.02 * (self.cfg.min_speed - speed) / self.cfg.min_speed
            
            reward += speed_reward
        
        # =====================================================================
        # 3. CENTERLINE REWARD (Encourage racing line adherence)
        # =====================================================================
        if on_track:
            # Reward inversely proportional to distance from centerline
            # Max reward at center, zero reward at edge
            centerline_bonus = (1.0 - center_dist / half_width) * self.cfg.centerline_reward_scale
            reward += centerline_bonus
        else:
            # Gentle penalty for being off track (termination handles severe cases)
            off_track_amount = (center_dist - half_width) / half_width
            off_track_penalty = -self.cfg.off_track_penalty_scale * min(off_track_amount, 2.0)
            reward += off_track_penalty
        
        # =====================================================================
        # 4. CHECKPOINT MILESTONE BONUS
        # =====================================================================
        if checkpoint_hit:
            checkpoint_reward = self.cfg.checkpoint_bonus
            if self.curriculum:
                reward_params = self.curriculum.get_reward_parameters()
                checkpoint_reward *= reward_params["checkpoint_multiplier"]
            reward += checkpoint_reward
            print(f"CHECKPOINT {len(self.checkpoints_hit)}/{self.cfg.num_checkpoints} HIT! (+{checkpoint_reward:.0f})")
        
        # =====================================================================
        # 5. LAP COMPLETION BONUS
        # =====================================================================
        if len(self.checkpoints_hit) >= self.cfg.num_checkpoints and not self.lap_completed:
            self.lap_completed = True
            reward += self.cfg.lap_completion_bonus
            avg_speed = self.total_distance_traveled / (self.step_count * self.cfg.dt) if self.step_count > 0 else 0
            print(f"LAP COMPLETED! Steps: {self.step_count}, Avg Speed: {avg_speed:.1f} m/s")
        
        # =====================================================================
        # 6. CURRICULUM PROGRESS BONUS (Extra encouragement in early stages)
        # =====================================================================
        if self.curriculum and on_track:
            reward_params = self.curriculum.get_reward_parameters()
            progress_bonus = reward_params.get("progress_bonus", 0.0)
            if progress_bonus > 0:
                reward += progress_bonus
        
        return reward
    
    def _check_termination(self, track_state: dict) -> Tuple[bool, bool]:
        """
        Check if the episode should end.
        
        Returns:
            Tuple of (terminated, truncated).
        """
        terminated = False
        truncated = False
        
        center_dist = track_state["center_dist"]
        half_width = self.track.width / 2.0
        
        # Count wheels inside track
        wheels_inside = self._count_wheels_inside_track()
        
        # Update off-track counter
        if wheels_inside == 0:
            self._off_track_count += 1
        else:
            self._off_track_count = 0
        
        # Check curriculum-aware termination
        if self._should_terminate_curriculum(wheels_inside):
            terminated = True
            self.termination_stats["wheel_violations"] += 1
        
        # Safety check: very far from track
        if center_dist > half_width + 20.0:
            terminated = True
            self.termination_stats["off_track"] += 1
        
        # Episode limits
        if self.lap_completed:
            truncated = True
            self.termination_stats["lap_completed"] += 1
        elif self.step_count >= self.cfg.max_steps:
            truncated = True
            self.termination_stats["max_steps"] += 1
        
        return terminated, truncated
    
    def _should_terminate_curriculum(self, wheels_inside: int) -> bool:
        """Check termination based on curriculum rules."""
        if not self.curriculum:
            return wheels_inside < 2
        
        mode = self._curriculum_termination_mode
        required = self._curriculum_wheels_required
        
        if mode == "never":
            return False
        elif mode == "soft":
            return wheels_inside == 0 and self._off_track_count > 20
        else:  # "normal" or "strict"
            return wheels_inside < required
    
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
    ) -> None:
        """Log telemetry data for the current step."""
        if not self.telemetry:
            return
        
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
            wheels_inside=self._count_wheels_inside_track(),
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
        
        stage = self.curriculum.get_stage_name() if self.curriculum else "standard"
        self.telemetry.end_episode(self.lap_completed, reason, stage)
        
        if self.termination_stats["total_episodes"] % 100 == 0:
            self.telemetry.print_training_progress(self.termination_stats["total_episodes"])
            if self.curriculum:
                self.curriculum.print_curriculum_status()
    
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
