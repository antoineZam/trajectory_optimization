from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from utils.track import Track
from utils.map_processing import sample_next_centerline_points, speed_along_tangent
from physics.physics_engine import VehicleSpec, VehicleState, step_dynamics, get_wheel_positions




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
    
    def __init__(self, track: Track, veh_spec: VehicleSpec, cfg: RLConfig | None = None):
        super().__init__()
        self.track = track
        self.spec = veh_spec
        self.cfg = cfg or RLConfig()

        # Observation: [norm_x, norm_y, yaw, vx, vy, yaw_rate] + next 5 relative points (x,y)
        self.k = 5
        self.obs_dim = 6 + 2*self.k
        
        # Define reasonable observation bounds matching our clipping
        obs_low = np.array([-1000.0, -1000.0, -np.pi, -100.0, -100.0, -10.0] + [-100.0]*10)  # 6 + 2*5
        obs_high = np.array([1000.0, 1000.0, np.pi, 100.0, 100.0, 10.0] + [100.0]*10)  # 6 + 2*5
        
        self.action_space = spaces.Box(low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32) # throttle, brake, steer
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.state = None
        self.step_count = 0
        self.track_progress = 0.0  # Progress around track [0-1]
        self.last_track_idx = 0   # Last closest track point index
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


    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        
        # Log episode completion stats every 100 episodes
        self.termination_stats["total_episodes"] += 1
        
        # Update training phase based on episode count
        self._update_training_phase()
        
        if self.termination_stats["total_episodes"] % 100 == 0:
            self._print_termination_stats()
        
        self.step_count = 0
        self.track_progress = 0.0
        self.last_track_idx = 0
        self.lap_completed = False
        self.max_track_idx_reached = 0
        self.checkpoints_hit = set()
        self.current_checkpoint = 0
        x0, y0 = self.track.centerline[0]
        self.state = VehicleState(x=x0, y=y0, yaw=0.0, vx=3.0, vy=0.0, yaw_rate=0.0, gear=2, rpm=1500.0)
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
        """Check if a point is inside the track boundaries."""
        # Find closest track point
        distances = np.linalg.norm(self.track.centerline - point, axis=1)
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        
        # Check if within track width (with small safety margin)
        return closest_distance <= (self.track.width / 2.0 - 0.1)  # 0.1m safety margin
    
    def _count_wheels_inside_track(self) -> int:
        """Count how many wheels are inside the track boundaries."""
        wheel_positions = get_wheel_positions(self.spec, self.state)
        wheels_inside = 0
        
        for wheel_pos in wheel_positions:
            if self._is_point_inside_track(wheel_pos):
                wheels_inside += 1
                
        return wheels_inside

    def _check_checkpoint(self, track_idx: int) -> bool:
        """Check if we've reached the next checkpoint in sequence."""
        track_len = len(self.track.centerline)
        checkpoint_spacing = track_len // self.num_checkpoints
        
        # Calculate which checkpoint this track index represents
        checkpoint_id = track_idx // checkpoint_spacing
        
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
            nxt = sample_next_centerline_points(self.track.centerline, np.array([s.x, s.y]), k=self.k, lookahead=5.0)
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
        throttle, brake, steer = float(action[0]), float(action[1]), float(action[2])    
        self.state = step_dynamics(self.spec, self.state, self.cfg.dt, throttle, brake, steer)
        self.step_count += 1
        
        # Initialize termination flag
        terminated = False
        
        # Find closest track point
        s = self.state
        pos = np.array([s.x, s.y])
        idx = int(np.argmin(((self.track.centerline - pos)**2).sum(axis=1)))
        
        # Checkpoint-based progress tracking
        track_len = len(self.track.centerline)
        center_dist = np.linalg.norm(self.track.centerline[idx] - pos)
        checkpoint_hit = False
        
        # Only allow checkpoint progress if close to track
        if center_dist <= (self.track.width/2 + 2.0):  # Within 2m of track edge
            checkpoint_hit = self._check_checkpoint(idx)
            
        # Calculate progress based on checkpoints hit
        self.track_progress = len(self.checkpoints_hit) / self.num_checkpoints
        
        # Check for lap completion - need ALL checkpoints
        if len(self.checkpoints_hit) >= self.num_checkpoints and not self.lap_completed:
            self.lap_completed = True
        
        # Rewards
        reward = 0.0
        
        # Distance-based reward: stay on track
        if center_dist <= self.track.width/2:
            # On track - good base reward
            reward += 5.0
        elif center_dist <= self.track.width/2 + 1.0:
            # Close to track - small reward
            reward += 1.0
        else:
            # Off track - penalty
            reward -= 5.0
        
        # MASSIVE checkpoint reward: heavily incentivize reaching new checkpoints
        if checkpoint_hit:
            reward += 200.0  # HUGE reward for hitting a new checkpoint in sequence
            print(f"CHECKPOINT {len(self.checkpoints_hit)-1} HIT! Progress: {len(self.checkpoints_hit)}/{self.num_checkpoints}, step: {self.step_count}")
            
        # Progress-based rewards for being at correct part of track
        expected_checkpoint = len(self.checkpoints_hit)
        if expected_checkpoint < self.num_checkpoints:
            # Calculate how close we are to the next checkpoint
            target_checkpoint_idx = (expected_checkpoint * track_len) // self.num_checkpoints
            distance_to_checkpoint = abs(idx - target_checkpoint_idx) / track_len
            
            # Reward being close to the next checkpoint
            if distance_to_checkpoint < 0.1:  # Within 10% of track length
                reward += 10.0 * (0.1 - distance_to_checkpoint) * 100  # Higher reward when closer
        
        # Speed reward along track direction - only when on track
        if center_dist <= self.track.width/2 + 1.0:
            nxt = self.track.centerline[(idx+1) % track_len]
            tangent = nxt - self.track.centerline[idx]
            vprog = speed_along_tangent(s.vx, s.vy, tangent)
            reward += np.clip(vprog * 0.1, 0, 2)  # Small speed bonus, capped
        
        # Time penalty to encourage fast completion
        reward += self.cfg.time_penalty
        
        # Lap completion bonus
        if self.lap_completed:
            reward += self.cfg.lap_completion_bonus
            
        # Stagnation penalty: discourage staying in same area
        if self.step_count > 1000 and len(self.checkpoints_hit) == 0:
            reward -= 10.0  # Penalty for not hitting any checkpoints after many steps
            
        # Progressive track boundary enforcement based on training phase
        wheels_inside = self._count_wheels_inside_track()
        
        # Determine termination threshold based on training phase
        if self.training_phase == "learning":
            min_wheels_required = 0  # Only terminate if ALL wheels leave track
        elif self.training_phase == "intermediate":
            min_wheels_required = 1  # Terminate if only 0 wheels inside
        else:  # strict phase
            min_wheels_required = 2  # Racing rule: at least 2 wheels inside
            
        # Check for wheel violation termination
        if wheels_inside < min_wheels_required:
            reward += self.cfg.illegal_position_penalty
            terminated = True
            self.termination_stats["wheel_violations"] += 1
            if self.step_count % 500 == 0:  # Occasional detailed logging
                print(f"WHEEL VIOLATION ({self.training_phase} phase) at step {self.step_count}: {wheels_inside}/4 wheels inside, need ≥{min_wheels_required}")
        elif wheels_inside < 4:
            # Some wheels outside - graduated penalty based on phase
            wheels_outside = 4 - wheels_inside
            penalty_multiplier = {"learning": 0.1, "intermediate": 0.5, "strict": 1.0}[self.training_phase]
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

        return self._get_obs(), reward, terminated, truncated, {}


    def render(self):
        pass