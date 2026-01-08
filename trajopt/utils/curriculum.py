"""
Curriculum Learning System for Racing Agent Training

Implements a progressive training approach that starts with easy scenarios
and gradually increases difficulty as the agent learns basic competencies.

Design Principles:
- Realistic thresholds for RL (10-30% success rates are normal during learning)
- Faster stage progression to maintain training momentum
- Smooth difficulty curves with gradual parameter changes
- Early graduation when agent significantly exceeds expectations
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class CurriculumStage:
    """Defines a single stage of the curriculum."""
    
    name: str
    description: str
    
    # Track parameters
    track_width_multiplier: float  # Multiply base track width
    
    # Episode parameters
    max_episode_steps: int
    checkpoint_threshold: int  # Minimum checkpoints to consider "success"
    
    # Termination conditions
    termination_mode: str  # "never", "soft", "normal", "strict"
    wheels_required_inside: int  # For termination check
    
    # Graduation criteria (realistic for RL)
    min_episodes: int  # Minimum episodes before graduation possible
    target_episodes: int  # Expected episodes to graduate
    success_rate_threshold: float  # Minimum success rate to graduate
    early_graduation_rate: float  # If success rate exceeds this, graduate early
    
    # Reward modifications
    checkpoint_reward_multiplier: float
    progress_bonus: float  # Additional per-step bonus


class CurriculumLearning:
    """
    Manages progressive training curriculum for racing agent.
    
    The curriculum is designed with realistic RL expectations:
    - Success rates of 15-30% are considered good during training
    - Early stages focus on basic competencies (forward motion, not crashing)
    - Later stages refine racing line and speed optimization
    """
    
    def __init__(self):
        self.current_stage = 0
        self.stage_start_episode = 0
        self.stage_episodes_completed = 0
        self.stage_successes = 0
        
        # Rolling window for recent success rate (more responsive than all-time)
        self.recent_results: List[bool] = []
        self.recent_window_size = 50
        
        # Define curriculum stages with realistic RL thresholds
        self.stages = [
            # Stage 0: "Driving School" - Just learn to move forward
            CurriculumStage(
                name="driving_school",
                description="Learn to drive forward and hit first checkpoint",
                track_width_multiplier=5.0,  # Very wide - almost impossible to fail
                max_episode_steps=2000,
                checkpoint_threshold=1,  # Just need to hit first checkpoint (25% of lap)
                termination_mode="never",  # Never terminate for going off track
                wheels_required_inside=0,
                min_episodes=50,  # At least 50 episodes
                target_episodes=150,  # Expect to graduate around 150
                success_rate_threshold=0.15,  # 15% hitting checkpoint 1
                early_graduation_rate=0.40,  # Graduate early if 40%+ succeed
                checkpoint_reward_multiplier=2.0,  # Double checkpoint rewards
                progress_bonus=0.1,  # Extra encouragement
            ),
            
            # Stage 1: "Learner's Permit" - Learn to stay on track
            CurriculumStage(
                name="learners_permit",
                description="Learn to stay on track and reach checkpoint 2",
                track_width_multiplier=3.5,  # Still quite wide
                max_episode_steps=3000,
                checkpoint_threshold=2,  # Need 2 checkpoints (50% of lap)
                termination_mode="soft",  # Only terminate if completely off for 20+ steps
                wheels_required_inside=0,
                min_episodes=75,
                target_episodes=200,
                success_rate_threshold=0.12,  # 12% hitting checkpoint 2
                early_graduation_rate=0.35,
                checkpoint_reward_multiplier=1.75,
                progress_bonus=0.05,
            ),
            
            # Stage 2: "Provisional License" - Learn boundaries
            CurriculumStage(
                name="provisional_license",
                description="Learn track boundaries and reach checkpoint 3",
                track_width_multiplier=2.5,  # Moderately wide
                max_episode_steps=4000,
                checkpoint_threshold=3,  # Need 3 checkpoints (75% of lap)
                termination_mode="soft",
                wheels_required_inside=1,  # At least 1 wheel inside
                min_episodes=100,
                target_episodes=300,
                success_rate_threshold=0.10,  # 10% hitting checkpoint 3
                early_graduation_rate=0.30,
                checkpoint_reward_multiplier=1.5,
                progress_bonus=0.03,
            ),
            
            # Stage 3: "Full License" - Complete laps with some tolerance
            CurriculumStage(
                name="full_license",
                description="Complete full laps with reasonable boundaries",
                track_width_multiplier=1.8,  # Slightly wider than normal
                max_episode_steps=6000,
                checkpoint_threshold=4,  # Full lap required
                termination_mode="normal",
                wheels_required_inside=2,  # Standard 2-wheel rule
                min_episodes=150,
                target_episodes=400,
                success_rate_threshold=0.08,  # 8% completing laps
                early_graduation_rate=0.25,
                checkpoint_reward_multiplier=1.25,
                progress_bonus=0.01,
            ),
            
            # Stage 4: "Racing Pro" - Standard racing conditions
            CurriculumStage(
                name="racing_pro",
                description="Race on standard track with normal rules",
                track_width_multiplier=1.2,  # Near-normal width
                max_episode_steps=8000,
                checkpoint_threshold=4,
                termination_mode="normal",
                wheels_required_inside=2,
                min_episodes=200,
                target_episodes=500,
                success_rate_threshold=0.05,  # 5% completing laps
                early_graduation_rate=0.20,
                checkpoint_reward_multiplier=1.0,
                progress_bonus=0.0,
            ),
            
            # Stage 5: "Champion" - Final stage, optimal performance
            CurriculumStage(
                name="champion",
                description="Master the track at full difficulty",
                track_width_multiplier=1.0,  # Normal track width
                max_episode_steps=10000,
                checkpoint_threshold=4,
                termination_mode="strict",
                wheels_required_inside=2,
                min_episodes=500,  # Extended training at final level
                target_episodes=2000,
                success_rate_threshold=0.03,  # 3% - focus on lap time optimization
                early_graduation_rate=1.0,  # Never early graduate (final stage)
                checkpoint_reward_multiplier=1.0,
                progress_bonus=0.0,
            ),
        ]
        
        # Statistics tracking
        self.graduation_history: List[Dict[str, Any]] = []
        self.total_episodes = 0
    
    def get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        return self.stages[min(self.current_stage, len(self.stages) - 1)]
    
    def get_stage_name(self) -> str:
        """Get current stage name for logging."""
        return self.get_current_stage().name
    
    def record_episode_result(
        self, checkpoints_hit: int, lap_completed: bool, episode_num: int
    ) -> bool:
        """
        Record episode result and check for stage progression.
        
        Args:
            checkpoints_hit: Number of checkpoints reached this episode.
            lap_completed: Whether a full lap was completed.
            episode_num: Current episode number.
            
        Returns:
            True if stage advanced, False otherwise.
        """
        self.stage_episodes_completed += 1
        self.total_episodes = episode_num
        
        current_stage = self.get_current_stage()
        
        # Check if episode was successful for this stage
        success = checkpoints_hit >= current_stage.checkpoint_threshold
        if success:
            self.stage_successes += 1
        
        # Update rolling window
        self.recent_results.append(success)
        if len(self.recent_results) > self.recent_window_size:
            self.recent_results.pop(0)
        
        # Check for stage advancement
        if self._should_advance_stage():
            return self._advance_stage(episode_num)
        
        return False
    
    def _get_recent_success_rate(self) -> float:
        """Get success rate from recent episodes (more responsive)."""
        if not self.recent_results:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)
    
    def _get_overall_success_rate(self) -> float:
        """Get overall success rate for current stage."""
        if self.stage_episodes_completed == 0:
            return 0.0
        return self.stage_successes / self.stage_episodes_completed
    
    def _should_advance_stage(self) -> bool:
        """Check if criteria are met to advance to next stage."""
        if self.current_stage >= len(self.stages) - 1:
            return False  # Already at final stage
        
        current_stage = self.get_current_stage()
        
        # Must have completed minimum episodes
        if self.stage_episodes_completed < current_stage.min_episodes:
            return False
        
        # Check for early graduation (exceptional performance)
        recent_rate = self._get_recent_success_rate()
        if recent_rate >= current_stage.early_graduation_rate:
            return True
        
        # Check for normal graduation
        if self.stage_episodes_completed >= current_stage.target_episodes:
            overall_rate = self._get_overall_success_rate()
            if overall_rate >= current_stage.success_rate_threshold:
                return True
        
        # Also allow graduation if doing well for extended period
        if self.stage_episodes_completed >= current_stage.target_episodes * 1.5:
            # Lower the bar if taking too long (agent might be stuck)
            overall_rate = self._get_overall_success_rate()
            if overall_rate >= current_stage.success_rate_threshold * 0.5:
                return True
        
        return False
    
    def _advance_stage(self, episode_num: int) -> bool:
        """Advance to next curriculum stage."""
        if self.current_stage >= len(self.stages) - 1:
            return False
        
        old_stage = self.get_current_stage()
        overall_rate = self._get_overall_success_rate()
        recent_rate = self._get_recent_success_rate()
        
        # Determine graduation type
        if recent_rate >= old_stage.early_graduation_rate:
            graduation_type = "EARLY (exceptional performance!)"
        elif self.stage_episodes_completed >= old_stage.target_episodes * 1.5:
            graduation_type = "EXTENDED (minimum criteria met)"
        else:
            graduation_type = "NORMAL"
        
        # Record graduation
        graduation_info = {
            "stage": old_stage.name,
            "episode": episode_num,
            "episodes_at_stage": self.stage_episodes_completed,
            "overall_success_rate": overall_rate,
            "recent_success_rate": recent_rate,
            "graduation_type": graduation_type,
            "timestamp": time.time(),
        }
        self.graduation_history.append(graduation_info)
        
        # Advance stage
        self.current_stage += 1
        self.stage_start_episode = episode_num
        self.stage_episodes_completed = 0
        self.stage_successes = 0
        self.recent_results.clear()
        
        # Print graduation message
        new_stage = self.get_current_stage()
        print("\n" + "=" * 60)
        print(f"🎓 CURRICULUM GRADUATION - {graduation_type}")
        print("=" * 60)
        print(f"   Completed: {old_stage.name.upper()} → {new_stage.name.upper()}")
        print(f"   Episodes at stage: {graduation_info['episodes_at_stage']}")
        print(f"   Overall success rate: {overall_rate:.1%}")
        print(f"   Recent success rate: {recent_rate:.1%}")
        print(f"")
        print(f"   NEW CHALLENGE: {new_stage.description}")
        print(f"   Track width: {new_stage.track_width_multiplier:.1f}x")
        print(f"   Checkpoint goal: {new_stage.checkpoint_threshold}/4")
        print(f"   Termination mode: {new_stage.termination_mode}")
        print("=" * 60 + "\n")
        
        return True
    
    def get_track_parameters(self) -> Dict[str, float]:
        """Get current track modification parameters."""
        stage = self.get_current_stage()
        return {
            "width_multiplier": stage.track_width_multiplier,
        }
    
    def get_episode_parameters(self) -> Dict[str, Any]:
        """Get current episode parameters."""
        stage = self.get_current_stage()
        return {
            "max_steps": stage.max_episode_steps,
            "termination_mode": stage.termination_mode,
            "wheels_required_inside": stage.wheels_required_inside,
        }
    
    def get_reward_parameters(self) -> Dict[str, float]:
        """Get current reward modifications."""
        stage = self.get_current_stage()
        return {
            "checkpoint_multiplier": stage.checkpoint_reward_multiplier,
            "progress_bonus": stage.progress_bonus,
        }
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current curriculum progress information."""
        stage = self.get_current_stage()
        overall_rate = self._get_overall_success_rate()
        recent_rate = self._get_recent_success_rate()
        
        # Calculate progress toward graduation
        episode_progress = min(1.0, self.stage_episodes_completed / stage.target_episodes)
        success_progress = min(1.0, overall_rate / stage.success_rate_threshold) if stage.success_rate_threshold > 0 else 1.0
        
        return {
            "current_stage": self.current_stage,
            "total_stages": len(self.stages),
            "stage_name": stage.name,
            "stage_description": stage.description,
            "episodes_completed": self.stage_episodes_completed,
            "min_episodes": stage.min_episodes,
            "target_episodes": stage.target_episodes,
            "episode_progress": episode_progress,
            "overall_success_rate": overall_rate,
            "recent_success_rate": recent_rate,
            "target_success_rate": stage.success_rate_threshold,
            "early_graduation_rate": stage.early_graduation_rate,
            "success_progress": success_progress,
            "ready_to_advance": self._should_advance_stage(),
        }
    
    def print_curriculum_status(self) -> None:
        """Print detailed curriculum status."""
        info = self.get_progress_info()
        stage = self.get_current_stage()
        
        # Determine status emoji
        if info["ready_to_advance"]:
            status = "✅ READY TO GRADUATE"
        elif info["recent_success_rate"] >= stage.success_rate_threshold:
            status = "📈 On track"
        elif info["episodes_completed"] < stage.min_episodes:
            status = "🔄 Warming up"
        else:
            status = "📊 Training"
        
        print(f"\n{'─' * 50}")
        print(f"CURRICULUM: Stage {info['current_stage'] + 1}/{info['total_stages']} - {info['stage_name'].upper()}")
        print(f"{'─' * 50}")
        print(f"Status: {status}")
        print(f"Goal: {info['stage_description']}")
        print(f"")
        print(f"Episodes: {info['episodes_completed']}/{info['target_episodes']} (min: {info['min_episodes']})")
        print(f"Success rate: {info['overall_success_rate']:.1%} overall, {info['recent_success_rate']:.1%} recent")
        print(f"Need: {info['target_success_rate']:.1%} to graduate, {info['early_graduation_rate']:.1%} for early")
        print(f"")
        print(f"Track: {stage.track_width_multiplier:.1f}x width | Max steps: {stage.max_episode_steps}")
        print(f"Checkpoint goal: {stage.checkpoint_threshold}/4 | Termination: {stage.termination_mode}")
        print(f"{'─' * 50}")
    
    def get_summary(self) -> str:
        """Get a one-line summary of curriculum status."""
        info = self.get_progress_info()
        return (
            f"Stage {info['current_stage'] + 1}/{info['total_stages']} "
            f"({info['stage_name']}) - "
            f"{info['episodes_completed']} eps, "
            f"{info['recent_success_rate']:.0%} recent success"
        )
