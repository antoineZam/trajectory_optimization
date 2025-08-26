"""
Racing telemetry system for detailed performance analysis and monitoring.
"""
from __future__ import annotations
import numpy as np
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class TelemetryFrame:
    """Single frame of telemetry data."""
    timestamp: float
    step: int
    # Vehicle state
    x: float
    y: float
    yaw: float
    speed: float  # Total speed (m/s)
    vx: float
    vy: float
    yaw_rate: float
    # Control inputs
    throttle: float
    brake: float
    steer_command: float  # Agent command [-1, 1]
    steer_angle: float   # Actual steering angle (rad)
    max_steer_allowed: float  # Physics limit (rad)
    # Track information
    track_progress: float  # [0-1]
    center_distance: float  # Distance to track center (m)
    racing_line_distance: float  # Distance to racing line (m)
    wheels_inside: int  # Number of wheels inside track
    current_checkpoint: int
    # Rewards
    total_reward: float
    track_reward: float
    speed_reward: float
    checkpoint_reward: float
    racing_line_reward: float
    magnetism_reward: float  # New: checkpoint magnetism reward
    steering_penalty: float
    smooth_steering_penalty: float


@dataclass
class EpisodeSummary:
    """Summary statistics for a completed episode."""
    episode: int
    # Completion metrics
    lap_completed: bool
    lap_time: float  # seconds
    checkpoints_hit: int
    max_checkpoints: int
    # Performance metrics
    average_speed: float  # m/s
    max_speed: float  # m/s
    total_distance: float  # m
    # Racing quality
    average_center_distance: float  # m
    average_racing_line_distance: float  # m
    time_on_track: float  # percentage
    # Control quality
    average_steering_magnitude: float
    max_steering_used: float  # degrees
    steering_smoothness: float  # lower is smoother
    # Rewards
    total_reward: float
    average_reward_per_step: float
    # Termination reason
    termination_reason: str
    # Training phase
    training_phase: str


class RacingTelemetry:
    """Comprehensive telemetry system for racing analysis."""
    
    def __init__(self, save_dir: str = "telemetry"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Current episode data
        self.current_episode_frames: List[TelemetryFrame] = []
        self.episode_start_time: float = 0.0
        self.episode_summaries: List[EpisodeSummary] = []
        
        # Real-time monitoring
        self.last_print_time: float = 0.0
        self.print_interval: float = 2.0  # seconds
        
        # Performance tracking
        self.best_lap_time: float = float('inf')
        self.best_episode: int = -1
        
        # Terminal formatting
        self.BOLD = '\033[1m'
        self.RESET = '\033[0m'
        self.GREEN = '\033[92m'
        self.RED = '\033[91m'
        self.YELLOW = '\033[93m'
        self.BLUE = '\033[94m'
        
    def start_episode(self, episode_num: int):
        """Start a new episode."""
        self.current_episode = episode_num
        self.current_episode_frames = []
        self.episode_start_time = time.time()
        self.last_checkpoint_time = self.episode_start_time
        
    def log_frame(self, frame: TelemetryFrame):
        """Log a single telemetry frame."""
        self.current_episode_frames.append(frame)
        
        # Real-time monitoring
        current_time = time.time()
        if current_time - self.last_print_time > self.print_interval:
            self._print_realtime_telemetry(frame)
            self.last_print_time = current_time
    
    def end_episode(self, lap_completed: bool, termination_reason: str, training_phase: str) -> EpisodeSummary:
        """End current episode and generate summary."""
        if not self.current_episode_frames:
            return None
            
        frames = self.current_episode_frames
        lap_time = time.time() - self.episode_start_time
        
        # Calculate summary statistics
        speeds = [f.speed for f in frames]
        center_distances = [f.center_distance for f in frames]
        racing_line_distances = [f.racing_line_distance for f in frames]
        steering_commands = [abs(f.steer_command) for f in frames]
        steering_angles = [abs(f.steer_angle) for f in frames]
        rewards = [f.total_reward for f in frames]
        
        # Calculate distance traveled
        positions = [(f.x, f.y) for f in frames]
        total_distance = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        # Calculate steering smoothness (variance in steering changes)
        steering_changes = []
        for i in range(1, len(frames)):
            change = abs(frames[i].steer_command - frames[i-1].steer_command)
            steering_changes.append(change)
        steering_smoothness = np.var(steering_changes) if steering_changes else 0.0
        
        # Time on track percentage
        wheels_inside_counts = [f.wheels_inside for f in frames]
        time_on_track = sum(1 for w in wheels_inside_counts if w >= 2) / len(wheels_inside_counts) * 100
        
        # Create summary
        summary = EpisodeSummary(
            episode=self.current_episode,
            lap_completed=lap_completed,
            lap_time=lap_time,
            checkpoints_hit=max(f.current_checkpoint for f in frames) if frames else 0,
            max_checkpoints=4,  # TODO: Make this configurable
            average_speed=np.mean(speeds) if speeds else 0.0,
            max_speed=max(speeds) if speeds else 0.0,
            total_distance=total_distance,
            average_center_distance=np.mean(center_distances) if center_distances else 0.0,
            average_racing_line_distance=np.mean(racing_line_distances) if racing_line_distances else 0.0,
            time_on_track=time_on_track,
            average_steering_magnitude=np.mean(steering_commands) if steering_commands else 0.0,
            max_steering_used=max(np.degrees(steering_angles)) if steering_angles else 0.0,
            steering_smoothness=steering_smoothness,
            total_reward=sum(rewards) if rewards else 0.0,
            average_reward_per_step=np.mean(rewards) if rewards else 0.0,
            termination_reason=termination_reason,
            training_phase=training_phase
        )
        
        self.episode_summaries.append(summary)
        
        # Track best performance
        if lap_completed and lap_time < self.best_lap_time:
            self.best_lap_time = lap_time
            self.best_episode = self.current_episode
            
        # Print episode summary
        self._print_episode_summary(summary)
        
        return summary
    
    def _print_realtime_telemetry(self, frame: TelemetryFrame):
        """Print real-time telemetry data."""
        speed_kmh = frame.speed * 3.6
        steer_deg = np.degrees(frame.steer_angle)
        max_steer_deg = np.degrees(frame.max_steer_allowed)
        
        print(f"\rLIVE: Step {frame.step:4d} | "
              f"Speed: {speed_kmh:5.1f} km/h | "
              f"Steer: {steer_deg:+5.1f}°/{max_steer_deg:4.1f}° | "
              f"Progress: {frame.track_progress*100:5.1f}% | "
              f"Center: {frame.center_distance:4.1f}m | "
              f"CP: {frame.current_checkpoint}/4 | "
              f"Reward: {frame.total_reward:+6.1f}", 
              end="", flush=True)
    
    def _print_episode_summary(self, summary: EpisodeSummary):
        """Print detailed episode summary."""
        print(f"\n")
        print(f"{'='*80}")
        print(f"{self.BOLD}EPISODE {summary.episode} SUMMARY [{summary.training_phase.upper()}]{self.RESET}")
        print(f"{'='*80}")
        
        # Completion status
        if summary.lap_completed:
            status = f"{self.GREEN}✅ LAP COMPLETED{self.RESET}"
        else:
            status = f"{self.RED}❌ {summary.termination_reason.upper()}{self.RESET}"
        print(f"Status: {status}")
        print(f"Duration: {summary.lap_time:.1f}s")
        print(f"Progress: {summary.checkpoints_hit}/{summary.max_checkpoints} checkpoints")
        
        # Performance metrics
        print(f"\n{self.BOLD}PERFORMANCE:{self.RESET}")
        print(f"  Average Speed: {summary.average_speed*3.6:6.1f} km/h")
        print(f"  Max Speed:     {summary.max_speed*3.6:6.1f} km/h") 
        print(f"  Distance:      {summary.total_distance:6.1f} m")
        print(f"  Time on Track: {summary.time_on_track:6.1f}%")
        
        # Racing quality
        print(f"\n{self.BOLD}RACING QUALITY:{self.RESET}")
        print(f"  Avg Center Dist:     {summary.average_center_distance:5.2f} m")
        print(f"  Avg Racing Line Dist: {summary.average_racing_line_distance:5.2f} m")
        print(f"  Steering Smoothness:  {summary.steering_smoothness:5.3f}")
        print(f"  Max Steering Used:    {summary.max_steering_used:5.1f}°")
        
        # Rewards
        print(f"\n{self.BOLD}REWARDS:{self.RESET}")
        print(f"  Total:      {summary.total_reward:8.1f}")
        print(f"  Per Step:   {summary.average_reward_per_step:8.3f}")
        
        # Best performance tracking
        if summary.lap_completed and hasattr(self, 'best_episode') and self.current_episode == self.best_episode:
            print(f"\n{self.YELLOW}🏆 NEW BEST LAP TIME! {summary.lap_time:.1f}s{self.RESET}")
            
        print(f"{'='*80}\n")
    
    def print_training_progress(self, episodes_completed: int):
        """Print overall training progress."""
        if len(self.episode_summaries) < 10:
            return
            
        recent_episodes = self.episode_summaries[-10:]
        
        # Calculate trends
        completion_rate = sum(1 for ep in recent_episodes if ep.lap_completed) / len(recent_episodes) * 100
        avg_speed = np.mean([ep.average_speed * 3.6 for ep in recent_episodes])
        avg_time_on_track = np.mean([ep.time_on_track for ep in recent_episodes])
        avg_reward = np.mean([ep.total_reward for ep in recent_episodes])
        
        print(f"\n{self.BOLD}TRAINING PROGRESS (Last 10 episodes):{self.RESET}")
        print(f"  Episodes Completed: {episodes_completed}")
        print(f"  Lap Completion Rate: {completion_rate:5.1f}%")
        print(f"  Average Speed: {avg_speed:6.1f} km/h")
        print(f"  Time on Track: {avg_time_on_track:6.1f}%")
        print(f"  Average Reward: {avg_reward:8.1f}")
        
        if hasattr(self, 'best_lap_time') and self.best_lap_time < float('inf'):
            print(f"  {self.GREEN}Best Lap Time: {self.best_lap_time:.1f}s (Episode {self.best_episode}){self.RESET}")
    
    def save_episode_data(self, episode_num: int):
        """Save episode telemetry data to file."""
        if not self.current_episode_frames:
            return
            
        # Save frames data
        frames_file = self.save_dir / f"episode_{episode_num:06d}_frames.json"
        frames_data = [asdict(frame) for frame in self.current_episode_frames]
        
        with open(frames_file, 'w') as f:
            json.dump(frames_data, f, indent=2)
    
    def save_summaries(self):
        """Save all episode summaries."""
        summaries_file = self.save_dir / "episode_summaries.json"
        summaries_data = [asdict(summary) for summary in self.episode_summaries]
        
        with open(summaries_file, 'w') as f:
            json.dump(summaries_data, f, indent=2)
    
    def export_csv(self):
        """Export summaries to CSV for external analysis."""
        import pandas as pd
        
        if not self.episode_summaries:
            return
            
        df = pd.DataFrame([asdict(summary) for summary in self.episode_summaries])
        csv_file = self.save_dir / "training_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"Telemetry data exported to: {csv_file}")
