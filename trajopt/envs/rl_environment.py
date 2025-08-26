from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from scipy.spatial.distance import cdist

from utils.track import Track
from utils.map_processing import sample_next_centerline_points, speed_along_tangent
from physics.physics_engine import VehicleSpec, VehicleState, step_dynamics, get_wheel_positions, get_max_steering_angle
from utils.telemetry import RacingTelemetry, TelemetryFrame
from utils.curriculum import CurriculumLearning
import time




@dataclass
class RLConfig:
    dt: float = 0.05
    max_steps: int = 8000  # Increased to allow full lap completion
    offtrack_penalty: float = -10.0
    progress_reward_scale: float = 1.0
    time_penalty: float = -0.001
    max_steer_rate: float = 0.25

    lap_completion_bonus: float = 500.0  # MASSIVE bonus for completing a lap
    min_progress_threshold: float = 0.8  # Must complete 80% of track before terminating
    track_violation_penalty: float = -20.0   # Harsh penalty for track limit violations
    illegal_position_penalty: float = -100.0  # Very harsh penalty for < 2 wheels inside


class RacingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}
    
    def __init__(self, track: Track, veh_spec: VehicleSpec, cfg: RLConfig | None = None, 
                 enable_telemetry: bool = True, enable_curriculum: bool = True):
        super().__init__()
        self.base_track = track  # Store original track
        self.track = track  # Current track (may be modified by curriculum)
        self.spec = veh_spec
        self.cfg = cfg or RLConfig()
        
        # Initialize curriculum learning system
        self.curriculum = CurriculumLearning() if enable_curriculum else None
        
        # Initialize telemetry system
        self.telemetry = RacingTelemetry() if enable_telemetry else None

        # Observation: [norm_x, norm_y, yaw, vx, vy, yaw_rate] + next 5 relative points (x,y)
        self.k = 5
        self.obs_dim = 6 + 2*self.k
        
        # Define reasonable observation bounds matching our clipping
        obs_low = np.array([-1000.0, -1000.0, -np.pi, -100.0, -100.0, -10.0] + [-100.0]*10)  # 6 + 2*5
        obs_high = np.array([1000.0, 1000.0, np.pi, 100.0, 100.0, 10.0] + [100.0]*10)  # 6 + 2*5
        
        # Action space: [throttle, brake, steer_command]
        # steer_command ∈ [-1, 1] maps to [-max_steering_angle, +max_steering_angle]
        # Physics engine applies additional speed-dependent limitations
        self.action_space = spaces.Box(low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.state = None
        self.step_count = 0
        self.track_progress = 0.0  # Progress around track [0-1]
        self.last_track_progress = 0.0 # For continuous reward
        self.lap_completed = False
        self.max_track_idx_reached = 0  # Furthest point reached on track
        
        # Checkpoint system for full lap completion
        self.num_checkpoints = 4  # Divide track into 4 larger segments - easier to complete
        self.checkpoints_hit = set()  # Track which checkpoints have been reached
        self.current_checkpoint = 0  # Next checkpoint to reach
        
        # Logging for episode termination analysis
        self.termination_stats = {
            "wheel_violations": 0,
            "off_track": 0, 
            "max_steps": 0,
            "lap_completed": 0,
            "total_episodes": 0
        }
        
        # Progressive training: start lenient, gradually get stricter
        self.training_phase = "learning"  # "learning" -> "intermediate" -> "strict"
        self.phase_episode_threshold = 500  # Switch phases every 500 episodes
        
        # Update track and parameters based on curriculum
        self._update_curriculum_settings()

    def _update_curriculum_settings(self):
        """Update track and episode parameters based on current curriculum stage."""
        if not self.curriculum:
            return
            
        # Get current curriculum parameters
        track_params = self.curriculum.get_track_parameters()
        episode_params = self.curriculum.get_episode_parameters()
        
        # Create modified track with wider boundaries
        self._create_curriculum_track(track_params["width_multiplier"])
        
        # Update episode parameters
        self.cfg.max_steps = episode_params["max_steps"]
        self._curriculum_termination_mode = episode_params["termination_mode"]
        self._curriculum_wheels_required = episode_params["wheels_required_inside"]
        
    def _create_curriculum_track(self, width_multiplier: float):
        """Create a modified track with adjusted width for curriculum learning."""
        if width_multiplier == 1.0:
            self.track = self.base_track
            return
            
        # Create a new track with modified width
        new_track = Track(
            name=f"{self.base_track.name}_curriculum_{width_multiplier:.1f}x",
            centerline=self.base_track.centerline.copy(),
            width=self.base_track.width * width_multiplier,
            interpolation_resolution=self.base_track.interpolation_resolution
        )
        self.track = new_track
        
    def _get_curriculum_checkpoint_reward(self, base_reward: float) -> float:
        """Apply curriculum multiplier to checkpoint rewards."""
        if not self.curriculum:
            return base_reward
        reward_params = self.curriculum.get_reward_parameters()
        return base_reward * reward_params["checkpoint_multiplier"]
        
    def _get_curriculum_progress_bonus(self) -> float:
        """Get curriculum-specific progress bonus."""
        if not self.curriculum:
            return 0.0
        reward_params = self.curriculum.get_reward_parameters()
        return reward_params["progress_bonus"]
        
    def _get_checkpoint_magnetism_reward(self, current_pos: np.ndarray) -> float:
        """Calculate magnetic attraction reward toward next checkpoint."""
        # Find next checkpoint to hit
        next_checkpoint_idx = len(self.checkpoints_hit)
        if next_checkpoint_idx >= self.num_checkpoints:
            return 0.0  # All checkpoints hit
            
        # Get position of next checkpoint
        checkpoint_progress = next_checkpoint_idx / self.num_checkpoints
        checkpoint_idx = int(checkpoint_progress * len(self.track.interpolated_centerline))
        checkpoint_pos = self.track.interpolated_centerline[checkpoint_idx]
        
        # Calculate distance to next checkpoint
        distance_to_checkpoint = np.linalg.norm(current_pos - checkpoint_pos)
        
        # Inverse distance reward - closer is better
        # Scale to provide meaningful but not overwhelming reward
        base_magnetism = 2.0 / (1.0 + distance_to_checkpoint * 0.1)  # Max ~2 points
        
        # Direction alignment bonus - are we moving toward the checkpoint?
        if hasattr(self, '_last_pos'):
            movement_vector = current_pos - self._last_pos
            to_checkpoint_vector = checkpoint_pos - current_pos
            
            # Normalize vectors
            movement_norm = np.linalg.norm(movement_vector)
            checkpoint_norm = np.linalg.norm(to_checkpoint_vector)
            
            if movement_norm > 0.1 and checkpoint_norm > 0.1:  # Avoid division by zero
                movement_unit = movement_vector / movement_norm
                checkpoint_unit = to_checkpoint_vector / checkpoint_norm
                
                # Dot product gives alignment (-1 to 1)
                alignment = np.dot(movement_unit, checkpoint_unit)
                
                # Bonus for moving toward checkpoint
                if alignment > 0:
                    direction_bonus = alignment * 1.0  # Up to +1 for perfect alignment
                    base_magnetism += direction_bonus
        
        # Store position for next step
        self._last_pos = current_pos.copy()
        
        # Scale based on curriculum stage - stronger magnetism in early stages
        stage_name = self.curriculum.get_stage_name() if self.curriculum else "racing_pro"
        magnetism_multipliers = {
            "driving_school": 2.0,      # Strong guidance in early learning
            "learners_permit": 1.5,     # Moderate guidance
            "provisional_license": 1.0, # Normal guidance
            "full_license": 0.7,        # Reduced guidance
            "racing_pro": 0.3          # Minimal guidance for advanced racing
        }
        
        multiplier = magnetism_multipliers.get(stage_name, 1.0)
        
        return base_magnetism * multiplier
        
    def _should_terminate_curriculum(self, wheels_inside: int) -> bool:
        """Check termination based on curriculum rules."""
        if not self.curriculum:
            return wheels_inside < 2  # Default rule
            
        mode = self._curriculum_termination_mode
        required = self._curriculum_wheels_required
        
        if mode == "never":
            return False  # Never terminate for wheel violations
        elif mode == "soft":
            # Only terminate if WAY off track for multiple steps
            return wheels_inside == 0 and hasattr(self, '_off_track_count') and self._off_track_count > 20
        elif mode == "normal":
            return wheels_inside < required
        elif mode == "strict":
            return wheels_inside < required
        
        return False

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        
        # Log episode completion stats every 100 episodes
        self.termination_stats["total_episodes"] += 1
        
        # Update training phase based on episode count
        self._update_training_phase()
        
        if self.termination_stats["total_episodes"] % 100 == 0:
            self._print_termination_stats()
            # Save telemetry data every 100 episodes
            if self.telemetry:
                self.telemetry.save_summaries()
        
        self.step_count = 0
        self.track_progress = 0.0
        self.last_track_progress = 0.0
        self.lap_completed = False
        self.max_track_idx_reached = 0
        self.checkpoints_hit = set()
        self.current_checkpoint = 0
        self.last_steer_command = 0.0  # Initialize steering tracking for smooth steering reward
        x0, y0 = self.track.centerline[0]
        self.state = VehicleState(x=x0, y=y0, yaw=0.0, vx=3.0, vy=0.0, yaw_rate=0.0, gear=2, rpm=1500.0)
        
        # Initialize off-track counter for soft termination
        self._off_track_count = 0
        
        # Initialize position tracking for checkpoint magnetism
        self._last_pos = np.array([self.state.x, self.state.y])
        
        # Check for curriculum progression
        if self.curriculum and hasattr(self, '_last_checkpoints_hit'):
            # Record previous episode result for curriculum
            graduated = self.curriculum.record_episode_result(
                checkpoints_hit=getattr(self, '_last_checkpoints_hit', 0),
                lap_completed=getattr(self, '_last_lap_completed', False),
                episode_num=self.termination_stats["total_episodes"]
            )
            
            if graduated:
                # Update settings for new curriculum stage
                self._update_curriculum_settings()
                self.curriculum.print_curriculum_status()
        
        # Start telemetry for new episode
        if self.telemetry:
            self.telemetry.start_episode(self.termination_stats["total_episodes"])
        
        return self._get_obs(), {}

    def _print_termination_stats(self):
        """Print statistics about why episodes are terminating."""
        total = self.termination_stats["total_episodes"]
        if total == 0:
            return
            
        print(f"\n=== Episode Termination Stats (last 100 episodes) ===")
        print(f"Training Phase: {self.training_phase.upper()}")
        if self.training_phase == "learning":
            print(f"  Rule: Only terminate if ALL wheels leave track (very lenient)")
        elif self.training_phase == "intermediate":
            print(f"  Rule: Terminate if 0 wheels inside track (moderate)")
        else:
            print(f"  Rule: Terminate if <2 wheels inside track (STRICT)")
        print(f"Wheel violations: {self.termination_stats['wheel_violations']} ({self.termination_stats['wheel_violations']/min(100,total)*100:.1f}%)")
        print(f"Off track: {self.termination_stats['off_track']} ({self.termination_stats['off_track']/min(100,total)*100:.1f}%)")
        print(f"Max steps reached: {self.termination_stats['max_steps']} ({self.termination_stats['max_steps']/min(100,total)*100:.1f}%)")
        print(f"Lap completed: {self.termination_stats['lap_completed']} ({self.termination_stats['lap_completed']/min(100,total)*100:.1f}%)")
        print(f"Total episodes: {total}")
        print("=" * 50)
        
        # Additional useful metrics
        avg_checkpoints = sum(1 for i in range(max(1, total-99), total+1) if i <= total) # Simple placeholder
        print(f"Recent checkpoint progress: Avg {len(self.checkpoints_hit)}/{self.num_checkpoints} per episode")
        
        # Reset counters for next 100 episodes
        if total % 100 == 0:
            self.termination_stats = {key: 0 if key != "total_episodes" else total for key in self.termination_stats}

    def _update_training_phase(self):
        """Update training phase based on episode count for progressive difficulty."""
        episodes = self.termination_stats["total_episodes"]
        
        if episodes < self.phase_episode_threshold:
            new_phase = "learning"
        elif episodes < self.phase_episode_threshold * 2:
            new_phase = "intermediate" 
        else:
            new_phase = "strict"
            
        if new_phase != self.training_phase:
            self.training_phase = new_phase
            print(f"\nTRAINING PHASE CHANGE: Now in '{new_phase}' phase (episode {episodes})")
            if new_phase == "learning":
                print("   - Very lenient: Only terminate if ALL wheels leave track")
            elif new_phase == "intermediate":
                print("   - Moderate: Terminate if only 1 wheel inside track") 
            else:
                print("   - STRICT: Terminate if fewer than 2 wheels inside track")
            print()

    def _is_point_inside_track(self, point: np.ndarray) -> bool:
        """Check if a point is inside the interpolated track boundaries."""
        return self.track.is_point_inside_track(point)
    
    def _count_wheels_inside_track(self) -> int:
        """Count how many wheels are inside the track boundaries."""
        wheel_positions = get_wheel_positions(self.spec, self.state)
        wheels_inside = 0
        
        for wheel_pos in wheel_positions:
            if self._is_point_inside_track(wheel_pos):
                wheels_inside += 1
                
        return wheels_inside

    def _check_checkpoint(self, current_progress: float) -> bool:
        """Check if we've reached the next checkpoint in sequence based on track progress."""
        checkpoint_spacing = 1.0 / self.num_checkpoints
        
        # Calculate which checkpoint this progress represents
        checkpoint_id = int(current_progress / checkpoint_spacing)
        checkpoint_id = min(checkpoint_id, self.num_checkpoints - 1)  # Clamp to valid range
        
        # Only allow hitting checkpoints in order
        if checkpoint_id == self.current_checkpoint:
            if checkpoint_id not in self.checkpoints_hit:
                self.checkpoints_hit.add(checkpoint_id)
                self.current_checkpoint = (checkpoint_id + 1) % self.num_checkpoints
                return True
        return False

    def _get_obs(self):
        s = self.state
        
        # Get track center for normalization
        track_center = np.mean(self.track.centerline, axis=0)
        
        # Normalize position relative to track center
        pos_x = np.clip(s.x - track_center[0], -1000.0, 1000.0)
        pos_y = np.clip(s.y - track_center[1], -1000.0, 1000.0)
        
        # Normalize angle to [-π, π]
        yaw = np.arctan2(np.sin(s.yaw), np.cos(s.yaw))
        
        # Clip velocities to reasonable bounds
        vx = np.clip(s.vx, -100.0, 100.0)  # m/s
        vy = np.clip(s.vy, -100.0, 100.0)  # m/s
        yaw_rate = np.clip(s.yaw_rate, -10.0, 10.0)  # rad/s
        
        # Get next points and normalize relative to current position
        try:
            # Use interpolated track for smoother lookahead points
            closest_idx = np.argmin(cdist([np.array([s.x, s.y])], self.track.interpolated_centerline)[0])
            
            # Get k points ahead on the interpolated centerline
            lookahead_indices = [(closest_idx + i*20) % len(self.track.interpolated_centerline) for i in range(1, self.k+1)]
            nxt = self.track.interpolated_centerline[lookahead_indices]
            
            # Normalize next points relative to current position
            nxt_rel = nxt - np.array([s.x, s.y])
            nxt_rel = np.clip(nxt_rel, -100.0, 100.0)  # Clip to reasonable bounds
        except:
            # Fallback if sampling fails
            nxt_rel = np.zeros((self.k, 2))
        
        # Construct observation with proper bounds
        obs_values = [pos_x, pos_y, yaw, vx, vy, yaw_rate] + nxt_rel.flatten().tolist()
        
        # Final safety check for NaN/inf values
        obs_values = [np.clip(float(val), -1e6, 1e6) if np.isfinite(val) else 0.0 for val in obs_values]
        
        obs = np.array(obs_values, dtype=np.float32)
        return obs
        
        
    def step(self, action: np.ndarray):
        throttle, brake, steer_command = float(action[0]), float(action[1]), float(action[2])    
        
        # Map agent action [-1, 1] to actual steering angle in radians
        # This ensures agent commands make physical sense
        steer_angle = steer_command * self.spec.max_steering_angle  # [-0.6, +0.6] rad max
        
        # Debug steering disabled - using telemetry system instead
        # (Old debug code removed to reduce console noise)
        
        # Store control inputs for telemetry
        control_inputs = (throttle, brake, steer_command, steer_angle)
        
        # Physics engine will further limit based on speed
        self.state = step_dynamics(self.spec, self.state, self.cfg.dt, throttle, brake, steer_angle)
        self.step_count += 1
        
        # Initialize termination flag
        terminated = False
        
        # Get current position and interpolated track progress
        s = self.state
        pos = np.array([s.x, s.y])
        
        # Use interpolated track for smooth progress tracking
        current_progress = self.track.get_track_progress(pos)
        center_dist = np.min(cdist([pos], self.track.interpolated_centerline)[0])
        checkpoint_hit = False
        
        # Only allow checkpoint progress if close to track
        if center_dist <= (self.track.width/2 + 2.0):  # Within 2m of track edge
            checkpoint_hit = self._check_checkpoint(current_progress)
            
        # FIXED REWARD STRUCTURE - Proper per-step scaling
        reward = 0.0
        
        # 1. Core objective: Stay on track (SMALL per-step rewards)
        if center_dist <= self.track.width/2:
            reward += 0.01  # Tiny reward for being on track per step
        elif center_dist <= self.track.width/2 + 1.0:
            reward += 0.005   # Even smaller reward for being close
        else:
            reward -= 0.02  # Small penalty for being off track
            
        # 2. Progress reward: Use checkpoint system (LARGE milestone bonuses)
        if checkpoint_hit:
            base_checkpoint_reward = 50.0
            curriculum_checkpoint_reward = self._get_curriculum_checkpoint_reward(base_checkpoint_reward)
            reward += curriculum_checkpoint_reward
            print(f"CHECKPOINT {len(self.checkpoints_hit)} HIT! Progress: {len(self.checkpoints_hit)}/{self.num_checkpoints} (+{curriculum_checkpoint_reward:.1f} reward)")
            
        # Calculate progress based on checkpoints hit
        self.track_progress = len(self.checkpoints_hit) / self.num_checkpoints
        
        # Check for lap completion - need ALL checkpoints
        if len(self.checkpoints_hit) >= self.num_checkpoints and not self.lap_completed:
            self.lap_completed = True
            
        # 3. Speed optimization: Encourage optimal racing speeds
        speed = np.hypot(s.vx, s.vy)
        if center_dist <= self.track.width/2:
            # Optimal speed range: 15-40 m/s (54-144 km/h)
            target_speed = 25.0  # Target racing speed
            speed_diff = abs(speed - target_speed)
            
            if speed < 5.0:
                # Penalty for being too slow
                reward -= (5.0 - speed) * 0.01
            elif speed_diff < 5.0:
                # Reward for being in optimal range
                speed_bonus = (5.0 - speed_diff) * 0.02
                reward += speed_bonus
            else:
                # Small speed bonus for any forward motion
                reward += np.clip(speed * 0.001, 0, 0.05)
            
        # 4. Lap completion (BIG milestone bonus)
        if self.lap_completed:
            reward += self.cfg.lap_completion_bonus
            
        # 5. Racing line reward (SMALL per-step)
        if center_dist <= self.track.width/2:  # Only when on track
            # Distance to interpolated centerline (racing line)
            racing_line_dist = np.linalg.norm(pos - self.track.interpolated_centerline[
                np.argmin(np.linalg.norm(self.track.interpolated_centerline - pos, axis=1))
            ])
            racing_line_reward = max(0, 0.03 - racing_line_dist * 0.01)  # Max +0.03 per step
            reward += racing_line_reward
        
        # 6. Steering extremes penalty (SMALL per-step penalty)
        steering_extremes_penalty = abs(steer_command) ** 2 * 0.05  # Much smaller penalty
        reward -= steering_extremes_penalty
        
        # 7. Smooth steering reward (SMALL penalty for jerky movements)
        if hasattr(self, 'last_steer_command'):
            steering_change = abs(steer_command - self.last_steer_command)
            if steering_change > 0.3:  # Penalize jerky steering
                reward -= steering_change * 0.1  # Much smaller penalty
        self.last_steer_command = steer_command
        
        # 8. Curriculum progress bonus (encourages forward movement)
        curriculum_bonus = self._get_curriculum_progress_bonus()
        if curriculum_bonus > 0:
            reward += curriculum_bonus
            
        # 8.5. Checkpoint magnetism - attract agent toward next checkpoint
        checkpoint_magnetism_reward = self._get_checkpoint_magnetism_reward(pos)
        reward += checkpoint_magnetism_reward
        
        # 9. Time penalty (TINY per-step)
        reward += self.cfg.time_penalty * 0.1  # Scale down time penalty
            
        # 10. Stagnation penalty: discourage staying in same area
        if self.step_count > 1000 and len(self.checkpoints_hit) == 0:
            reward -= 1.0  # Smaller stagnation penalty
            
        # Curriculum-aware track boundary enforcement
        wheels_inside = self._count_wheels_inside_track()
        
        # Update off-track counter for soft termination mode
        if wheels_inside == 0:
            self._off_track_count += 1
        else:
            self._off_track_count = 0
        
        # Check curriculum-aware termination conditions
        if self._should_terminate_curriculum(wheels_inside):
            reward += self.cfg.illegal_position_penalty
            terminated = True
            self.termination_stats["wheel_violations"] += 1
            
            # Get current curriculum stage for logging
            stage_name = self.curriculum.get_stage_name() if self.curriculum else "standard"
            required = self._curriculum_wheels_required if hasattr(self, '_curriculum_wheels_required') else 2
            
            if self.step_count % 500 == 0:  # Occasional detailed logging
                print(f"WHEEL VIOLATION [{stage_name.upper()}] at step {self.step_count}: {wheels_inside}/4 wheels inside, need ≥{required}")
        elif wheels_inside < 4:
            # Some wheels outside - graduated penalty (softer during early curriculum stages)
            wheels_outside = 4 - wheels_inside
            stage_name = self.curriculum.get_stage_name() if self.curriculum else "strict"
            penalty_multiplier = {"driving_school": 0.05, "learners_permit": 0.1, "provisional_license": 0.3, "full_license": 0.7, "racing_pro": 1.0}.get(stage_name, 1.0)
            reward += self.cfg.track_violation_penalty * wheels_outside * penalty_multiplier
            
        # Also terminate if extremely far from track (safety check)
        if center_dist > (self.track.width/2 + 15.0):  # Very generous 15m margin
            reward += self.cfg.offtrack_penalty * 3
            terminated = True
            self.termination_stats["off_track"] += 1
            if self.step_count % 500 == 0:  # Occasional detailed logging
                print(f"OFF TRACK at step {self.step_count}: distance {center_dist:.1f}m from centerline, checkpoints: {len(self.checkpoints_hit)}/{self.num_checkpoints}")
            
        # Episode ends on lap completion or max steps
        if self.lap_completed:
            truncated = True
            self.termination_stats["lap_completed"] += 1
            print(f"LAP COMPLETED! Steps: {self.step_count}, All {self.num_checkpoints} checkpoints hit")
        elif self.step_count >= self.cfg.max_steps:
            truncated = True
            self.termination_stats["max_steps"] += 1
            if len(self.checkpoints_hit) > 0:  # Only log if some progress was made
                print(f"MAX STEPS reached: {self.step_count}, progress: {len(self.checkpoints_hit)}/{self.num_checkpoints} checkpoints")
        else:
            truncated = False

        # Collect telemetry data
        if self.telemetry:
            # Calculate additional metrics for telemetry
            current_speed = np.hypot(self.state.vx, self.state.vy)
            throttle, brake, steer_command, steer_angle = control_inputs
            
            # Get max allowed steering at current speed
            max_steer_allowed = get_max_steering_angle(self.spec, current_speed)
            
            # Calculate racing line distance
            racing_line_dist = 0.0
            if center_dist <= self.track.width/2:
                closest_centerline_point = self.track.interpolated_centerline[
                    np.argmin(np.linalg.norm(self.track.interpolated_centerline - pos, axis=1))
                ]
                racing_line_dist = np.linalg.norm(pos - closest_centerline_point)
            
            # Break down reward components for telemetry (UPDATED to match new reward structure)
            track_reward = 0.01 if center_dist <= self.track.width/2 else (-0.02 if center_dist > self.track.width/2 + 1.0 else 0.005)
            checkpoint_reward = self._get_curriculum_checkpoint_reward(50.0) if checkpoint_hit else 0.0
            
            # Calculate speed reward breakdown
            speed_reward = 0.0
            if center_dist <= self.track.width/2:
                target_speed = 25.0
                speed_diff = abs(current_speed - target_speed)
                if current_speed < 5.0:
                    speed_reward = -(5.0 - current_speed) * 0.01
                elif speed_diff < 5.0:
                    speed_reward = (5.0 - speed_diff) * 0.02
                else:
                    speed_reward = np.clip(current_speed * 0.001, 0, 0.05)
            
            racing_line_reward = max(0, 0.03 - racing_line_dist * 0.01) if center_dist <= self.track.width/2 else 0.0
            magnetism_reward = self._get_checkpoint_magnetism_reward(pos)
            steering_penalty = abs(steer_command) ** 2 * 0.05
            smooth_steering_penalty = 0.0
            if hasattr(self, 'last_steer_command'):
                steering_change = abs(steer_command - self.last_steer_command)
                if steering_change > 0.3:
                    smooth_steering_penalty = steering_change * 0.1
                    
            frame = TelemetryFrame(
                timestamp=time.time(),
                step=self.step_count,
                # Vehicle state
                x=self.state.x,
                y=self.state.y,
                yaw=self.state.yaw,
                speed=current_speed,
                vx=self.state.vx,
                vy=self.state.vy,
                yaw_rate=self.state.yaw_rate,
                # Control inputs
                throttle=throttle,
                brake=brake,
                steer_command=steer_command,
                steer_angle=steer_angle,
                max_steer_allowed=max_steer_allowed,
                # Track information
                track_progress=current_progress,
                center_distance=center_dist,
                racing_line_distance=racing_line_dist,
                wheels_inside=wheels_inside,
                current_checkpoint=len(self.checkpoints_hit),
                # Rewards
                total_reward=reward,
                track_reward=track_reward,
                speed_reward=speed_reward,
                checkpoint_reward=checkpoint_reward,
                racing_line_reward=racing_line_reward,
                magnetism_reward=magnetism_reward,
                steering_penalty=steering_penalty,
                smooth_steering_penalty=smooth_steering_penalty
            )
            
            self.telemetry.log_frame(frame)
            
            # End episode telemetry if terminated
            if terminated or truncated:
                # Determine termination reason
                if self.lap_completed:
                    termination_reason = "lap_completed"
                elif truncated:
                    termination_reason = "max_steps"
                elif wheels_inside < 2 and self.training_phase == "strict":
                    termination_reason = "wheel_violation"
                elif wheels_inside == 0 and self.training_phase == "intermediate":
                    termination_reason = "wheel_violation"
                else:
                    termination_reason = "wheel_violation"
                    
                # Store episode results for curriculum tracking
                self._last_checkpoints_hit = len(self.checkpoints_hit)
                self._last_lap_completed = self.lap_completed
                
                # Include curriculum stage in telemetry
                current_stage = self.curriculum.get_stage_name() if self.curriculum else self.training_phase
                self.telemetry.end_episode(self.lap_completed, termination_reason, current_stage)
                
                # Print training progress every 100 episodes
                if self.termination_stats["total_episodes"] % 100 == 0:
                    self.telemetry.print_training_progress(self.termination_stats["total_episodes"])
                    # Also print curriculum status
                    if self.curriculum:
                        self.curriculum.print_curriculum_status()
            else:
                # Store episode results for curriculum tracking (non-telemetry case)
                self._last_checkpoints_hit = len(self.checkpoints_hit)
                self._last_lap_completed = self.lap_completed
        
        return self._get_obs(), reward, terminated, truncated, {}


    def render(self):
        pass