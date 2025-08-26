"""
Curriculum Learning System for Racing Agent Training

Implements a progressive training approach that starts with easy scenarios
and gradually increases difficulty as the agent learns basic competencies.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List
import time


@dataclass
class CurriculumStage:
    """Defines a single stage of the curriculum."""
    name: str
    description: str
    
    # Track parameters
    track_width_multiplier: float  # Multiply base track width
    track_complexity: float  # 0.0-1.0, affects curvature/difficulty
    
    # Episode parameters
    max_episode_steps: int
    checkpoint_threshold: int  # Minimum checkpoints to advance
    
    # Termination conditions
    termination_mode: str  # "never", "soft", "normal", "strict"
    wheels_required_inside: int  # For termination
    
    # Success criteria for stage progression
    episodes_required: int  # Min episodes at this stage
    success_rate_threshold: float  # % of episodes that must succeed
    min_checkpoints_avg: float  # Average checkpoints per episode
    
    # Reward modifications
    checkpoint_reward_multiplier: float
    progress_reward_bonus: float


class CurriculumLearning:
    """Manages progressive training curriculum for racing agent."""
    
    def __init__(self):
        self.current_stage = 0
        self.stage_start_episode = 0
        self.stage_episodes_completed = 0
        self.stage_successes = 0
        
        # Define curriculum stages
        self.stages = [
            # Stage 0: "Driving School" - Learn basic steering and movement
            CurriculumStage(
                name="driving_school",
                description="Learn basic steering and forward movement",
                track_width_multiplier=4.0,  # 4x wider track - almost impossible to fail
                track_complexity=0.1,  # Very gentle curves
                max_episode_steps=1000,
                checkpoint_threshold=1,  # Just need to hit first checkpoint
                termination_mode="never",  # Never terminate for track violations
                wheels_required_inside=0,  # No wheel requirements
                episodes_required=200,
                success_rate_threshold=0.8,  # 80% must reach first checkpoint
                min_checkpoints_avg=1.2,
                checkpoint_reward_multiplier=2.0,  # Double rewards for encouragement
                progress_reward_bonus=0.05
            ),
            
            # Stage 1: "Learner's Permit" - Basic track following
            CurriculumStage(
                name="learners_permit", 
                description="Learn to follow track boundaries with some tolerance",
                track_width_multiplier=2.5,  # 2.5x wider track
                track_complexity=0.3,
                max_episode_steps=2000,
                checkpoint_threshold=2,  # Need to hit 2 checkpoints
                termination_mode="soft",  # Gentle termination after many violations
                wheels_required_inside=1,  # At least 1 wheel inside
                episodes_required=300,
                success_rate_threshold=0.6,  # 60% must hit 2+ checkpoints
                min_checkpoints_avg=2.0,
                checkpoint_reward_multiplier=1.5,
                progress_reward_bonus=0.03
            ),
            
            # Stage 2: "Provisional License" - More realistic constraints
            CurriculumStage(
                name="provisional_license",
                description="Learn proper racing with moderate constraints", 
                track_width_multiplier=1.8,  # 1.8x wider track
                track_complexity=0.6,
                max_episode_steps=4000,
                checkpoint_threshold=3,  # Need to hit 3 checkpoints
                termination_mode="normal",  # Normal termination rules
                wheels_required_inside=2,  # At least 2 wheels inside (realistic)
                episodes_required=400,
                success_rate_threshold=0.4,  # 40% must hit 3+ checkpoints
                min_checkpoints_avg=2.5,
                checkpoint_reward_multiplier=1.2,
                progress_reward_bonus=0.02
            ),
            
            # Stage 3: "Full License" - Standard racing conditions
            CurriculumStage(
                name="full_license",
                description="Standard racing with full constraints",
                track_width_multiplier=1.2,  # Slightly wider for final learning
                track_complexity=0.8,
                max_episode_steps=6000,
                checkpoint_threshold=4,  # Need to complete full lap
                termination_mode="normal",
                wheels_required_inside=2,
                episodes_required=500,
                success_rate_threshold=0.3,  # 30% must complete laps
                min_checkpoints_avg=3.0,
                checkpoint_reward_multiplier=1.0,
                progress_reward_bonus=0.01
            ),
            
            # Stage 4: "Racing Pro" - Competition conditions
            CurriculumStage(
                name="racing_pro",
                description="Professional racing with strict enforcement",
                track_width_multiplier=1.0,  # Normal track width
                track_complexity=1.0,  # Full complexity
                max_episode_steps=8000,
                checkpoint_threshold=4,
                termination_mode="strict",  # Strict enforcement
                wheels_required_inside=2,
                episodes_required=1000,  # Extended training at final level
                success_rate_threshold=0.2,  # 20% lap completion
                min_checkpoints_avg=3.5,
                checkpoint_reward_multiplier=1.0,
                progress_reward_bonus=0.0
            )
        ]
        
        # Statistics tracking
        self.stage_stats = []
        self.graduation_times = []
        
    def get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        return self.stages[min(self.current_stage, len(self.stages) - 1)]
    
    def get_stage_name(self) -> str:
        """Get current stage name for logging."""
        return self.get_current_stage().name
    
    def record_episode_result(self, checkpoints_hit: int, lap_completed: bool, episode_num: int) -> bool:
        """
        Record episode result and check for stage progression.
        Returns True if stage advanced.
        """
        self.stage_episodes_completed += 1
        
        current_stage = self.get_current_stage()
        
        # Check if episode was successful for this stage
        if checkpoints_hit >= current_stage.checkpoint_threshold:
            self.stage_successes += 1
            
        # Check if we should advance to next stage
        if self._should_advance_stage():
            return self._advance_stage(episode_num)
        
        return False
    
    def _should_advance_stage(self) -> bool:
        """Check if criteria are met to advance to next stage."""
        if self.current_stage >= len(self.stages) - 1:
            return False  # Already at final stage
            
        current_stage = self.get_current_stage()
        
        # Need minimum episodes at this stage
        if self.stage_episodes_completed < current_stage.episodes_required:
            return False
            
        # Calculate success rate
        success_rate = self.stage_successes / max(1, self.stage_episodes_completed)
        
        # Must meet success threshold
        return success_rate >= current_stage.success_rate_threshold
    
    def _advance_stage(self, episode_num: int) -> bool:
        """Advance to next curriculum stage."""
        if self.current_stage >= len(self.stages) - 1:
            return False
            
        # Record graduation statistics
        old_stage = self.get_current_stage()
        success_rate = self.stage_successes / max(1, self.stage_episodes_completed)
        
        graduation_info = {
            "stage": old_stage.name,
            "episode": episode_num,
            "episodes_completed": self.stage_episodes_completed,
            "success_rate": success_rate,
            "graduation_time": time.time()
        }
        self.graduation_times.append(graduation_info)
        
        # Advance stage
        self.current_stage += 1
        self.stage_start_episode = episode_num
        self.stage_episodes_completed = 0
        self.stage_successes = 0
        
        # Print graduation message
        new_stage = self.get_current_stage()
        print(f"\nCURRICULUM GRADUATION!")
        print(f"   Completed: {old_stage.name} → {new_stage.name}")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Episodes: {graduation_info['episodes_completed']}")
        print(f"   New Challenge: {new_stage.description}")
        print(f"   Track Width: {new_stage.track_width_multiplier:.1f}x")
        print(f"   Max Steps: {new_stage.max_episode_steps}")
        print("=" * 60)
        
        return True
    
    def get_track_parameters(self) -> Dict[str, float]:
        """Get current track modification parameters."""
        stage = self.get_current_stage()
        return {
            "width_multiplier": stage.track_width_multiplier,
            "complexity": stage.track_complexity
        }
    
    def get_episode_parameters(self) -> Dict[str, Any]:
        """Get current episode parameters."""
        stage = self.get_current_stage()
        return {
            "max_steps": stage.max_episode_steps,
            "termination_mode": stage.termination_mode,
            "wheels_required_inside": stage.wheels_required_inside
        }
    
    def get_reward_parameters(self) -> Dict[str, float]:
        """Get current reward modifications."""
        stage = self.get_current_stage()
        return {
            "checkpoint_multiplier": stage.checkpoint_reward_multiplier,
            "progress_bonus": stage.progress_reward_bonus
        }
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current curriculum progress information."""
        stage = self.get_current_stage()
        progress = min(1.0, self.stage_episodes_completed / stage.episodes_required)
        success_rate = self.stage_successes / max(1, self.stage_episodes_completed)
        
        return {
            "current_stage": self.current_stage,
            "stage_name": stage.name,
            "stage_description": stage.description,
            "stage_progress": progress,
            "episodes_completed": self.stage_episodes_completed,
            "episodes_required": stage.episodes_required,
            "success_rate": success_rate,
            "target_success_rate": stage.success_rate_threshold,
            "ready_to_advance": self._should_advance_stage()
        }
    
    def print_curriculum_status(self):
        """Print detailed curriculum status."""
        info = self.get_progress_info()
        stage = self.get_current_stage()
        
        print(f"\nCURRICULUM STATUS:")
        print(f"   Stage: {info['current_stage'] + 1}/{len(self.stages)} - {info['stage_name'].upper()}")
        print(f"   Goal: {info['stage_description']}")
        print(f"   Progress: {info['episodes_completed']}/{info['episodes_required']} episodes ({info['stage_progress']:.1%})")
        print(f"   Success Rate: {info['success_rate']:.1%} (need {info['target_success_rate']:.1%})")
        print(f"   Track Width: {stage.track_width_multiplier:.1f}x normal")
        print(f"   Max Episode Steps: {stage.max_episode_steps:,}")
        
        if info['ready_to_advance']:
            print(f"READY TO GRADUATE to next stage!")
        else:
            needed = stage.episodes_required - info['episodes_completed']
            print(f"Need {needed} more episodes or higher success rate")
