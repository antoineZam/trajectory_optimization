"""
Optimal racing line finder using Reinforcement Learning.

Uses vectorized environments with SubprocVecEnv for parallel training
across multiple CPU cores. All hyperparameters are configurable through
Hydra configuration.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs.rl_environment import RacingEnv
from physics.physics_engine import VehicleSpec, get_gear_speed_info, get_steering_info
from utils.curriculum import CurriculumLearning
from utils.track import load_track_json


# Default number of parallel environments
DEFAULT_N_ENVS = min(mp.cpu_count(), 8)


class CurriculumCallback(BaseCallback):
    """Centralized curriculum that pushes stage params to all vectorized envs."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.curriculum = CurriculumLearning()
        self.total_episodes = 0
    
    def _on_training_start(self) -> None:
        self._push_curriculum_params()
    
    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        
        any_graduated = False
        for done, info in zip(dones, infos):
            if done and "checkpoints_hit" in info:
                self.total_episodes += 1
                graduated = self.curriculum.record_episode_result(
                    checkpoints_hit=info["checkpoints_hit"],
                    lap_completed=info["lap_completed"],
                    episode_num=self.total_episodes,
                )
                if graduated:
                    any_graduated = True
        
        if any_graduated:
            self._push_curriculum_params()
        
        return True
    
    def _push_curriculum_params(self) -> None:
        stage = self.curriculum.get_current_stage()
        params = {
            "width_multiplier": stage.track_width_multiplier,
            "max_steps": stage.max_episode_steps,
            "termination_mode": stage.termination_mode,
            "wheels_required": stage.wheels_required_inside,
            "checkpoint_multiplier": stage.checkpoint_reward_multiplier,
            "progress_bonus": stage.progress_bonus,
            "stage_name": stage.name,
        }
        self.training_env.env_method("update_curriculum_params", params)


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults.
    
    These defaults can be overridden by Hydra configuration.
    """
    # Total training timesteps
    timesteps: int = 500_000
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Network architecture
    pi_layers: tuple = (256, 256, 128)
    vf_layers: tuple = (256, 256, 128)
    activation: str = "tanh"
    
    # Environment settings
    n_envs: int = DEFAULT_N_ENVS
    normalize_obs: bool = True
    normalize_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    
    # Logging
    verbose: int = 1
    tensorboard_log: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TrainingConfig":
        """Create TrainingConfig from a dictionary (e.g., from Hydra)."""
        # Extract values with defaults
        kwargs = {}
        
        # Direct mappings
        direct_fields = [
            "timesteps", "learning_rate", "n_steps", "batch_size", "n_epochs",
            "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef", 
            "max_grad_norm", "verbose", "tensorboard_log"
        ]
        for field in direct_fields:
            if field in config:
                kwargs[field] = config[field]
        
        # Nested env config
        if "env" in config:
            env_cfg = config["env"]
            if "n_envs" in env_cfg:
                kwargs["n_envs"] = env_cfg["n_envs"]
            if "normalize_obs" in env_cfg:
                kwargs["normalize_obs"] = env_cfg["normalize_obs"]
            if "normalize_reward" in env_cfg:
                kwargs["normalize_reward"] = env_cfg["normalize_reward"]
            if "clip_obs" in env_cfg:
                kwargs["clip_obs"] = env_cfg["clip_obs"]
            if "clip_reward" in env_cfg:
                kwargs["clip_reward"] = env_cfg["clip_reward"]
        
        # Network architecture
        if "network" in config:
            net_cfg = config["network"]
            if "pi_layers" in net_cfg:
                kwargs["pi_layers"] = tuple(net_cfg["pi_layers"])
            if "vf_layers" in net_cfg:
                kwargs["vf_layers"] = tuple(net_cfg["vf_layers"])
            if "activation" in net_cfg:
                kwargs["activation"] = net_cfg["activation"]
        
        return cls(**kwargs)


def make_env_factory(
    track_path: str,
    vehicle_cfg: dict,
    interpolation_resolution: int = 2000,
    enable_telemetry: bool = False,
    enable_curriculum: bool = True,
) -> Callable[[], RacingEnv]:
    """Create a factory function for environment instantiation.
    
    This factory is picklable and can be used with SubprocVecEnv.
    
    Args:
        track_path: Path to track JSON file.
        vehicle_cfg: Vehicle configuration dictionary.
        interpolation_resolution: Track interpolation resolution.
        enable_telemetry: Whether to enable telemetry (only for rank 0).
        enable_curriculum: Whether to enable curriculum learning.
        
    Returns:
        A callable that creates a RacingEnv instance.
    """
    def _init() -> RacingEnv:
        track = load_track_json(track_path, interpolation_resolution=interpolation_resolution)
        spec = VehicleSpec.from_config(vehicle_cfg)
        return RacingEnv(
            track=track,
            veh_spec=spec,
            enable_telemetry=enable_telemetry,
            enable_curriculum=enable_curriculum,
        )
    return _init


def create_vec_env(
    track_path: str,
    vehicle_cfg: dict,
    training_cfg: TrainingConfig,
    interpolation_resolution: int = 2000,
    use_subproc: bool = True,
) -> tuple:
    """Create a vectorized environment for parallel training.
    
    Args:
        track_path: Path to track JSON file.
        vehicle_cfg: Vehicle configuration dictionary.
        training_cfg: Training configuration with env settings.
        interpolation_resolution: Track interpolation resolution.
        use_subproc: If True, use SubprocVecEnv for true parallelism.
                     If False, use DummyVecEnv (sequential, for debugging).
        
    Returns:
        Vectorized training environment (optionally wrapped with VecNormalize).
    """
    n_envs = training_cfg.n_envs
    
    # Create environment factories
    # Only enable telemetry on the first environment to avoid conflicts
    env_fns = [
        make_env_factory(
            track_path=track_path,
            vehicle_cfg=vehicle_cfg,
            interpolation_resolution=interpolation_resolution,
            enable_telemetry=(i == 0),  # Only first env has telemetry
            enable_curriculum=True,
        )
        for i in range(n_envs)
    ]
    
    # Create vectorized environment
    if use_subproc and n_envs > 1:
        # Use SubprocVecEnv for true parallel execution
        # start_method="spawn" is required on Windows
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
        print(f"  Using SubprocVecEnv with {n_envs} parallel environments")
    else:
        # Use DummyVecEnv for sequential execution (debugging)
        vec_env = DummyVecEnv(env_fns)
        print(f"  Using DummyVecEnv with {n_envs} sequential environments")
    
    # Optionally wrap with VecNormalize for better training stability
    if training_cfg.normalize_obs or training_cfg.normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=training_cfg.normalize_obs,
            norm_reward=training_cfg.normalize_reward,
            clip_obs=training_cfg.clip_obs,
            clip_reward=training_cfg.clip_reward,
        )
        print(f"  Normalization: obs={training_cfg.normalize_obs}, reward={training_cfg.normalize_reward}")
    
    return vec_env


def _get_activation_fn(name: str):
    """Get PyTorch activation function by name."""
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }
    return activations.get(name.lower(), nn.Tanh)


def train_and_export(
    track_path: str,
    vehicle_cfg: Union[str, dict],
    out_path: str,
    training_cfg: Optional[Union[Dict[str, Any], TrainingConfig]] = None,
    interpolation_resolution: int = 2000,
    use_subproc: bool = True,
    # Legacy parameters for backwards compatibility
    timesteps: Optional[int] = None,
    n_envs: Optional[int] = None,
):
    """Train RL agent and export optimal trajectory.
    
    Args:
        track_path: Path to track JSON file.
        vehicle_cfg: Either a path to vehicle config JSON or a config dict.
        out_path: Path to save the optimal trajectory.
        training_cfg: Training configuration (dict from Hydra or TrainingConfig).
        interpolation_resolution: Track interpolation resolution.
        use_subproc: Whether to use SubprocVecEnv (True) or DummyVecEnv (False).
        timesteps: (Legacy) Override total timesteps.
        n_envs: (Legacy) Override number of environments.
    
    Returns:
        Optimal trajectory as numpy array.
    """
    # Build training config
    if training_cfg is None:
        cfg = TrainingConfig()
    elif isinstance(training_cfg, dict):
        cfg = TrainingConfig.from_dict(training_cfg)
    else:
        cfg = training_cfg
    
    # Apply legacy overrides if provided
    if timesteps is not None:
        cfg.timesteps = timesteps
    if n_envs is not None:
        cfg.n_envs = n_envs
    
    # Load track for info display
    track = load_track_json(track_path, interpolation_resolution=interpolation_resolution)
    
    # Accept either a path string or a config dict
    if isinstance(vehicle_cfg, str):
        with open(vehicle_cfg, "r", encoding="utf-8") as f:
            veh_cfg = json.load(f)
    else:
        veh_cfg = vehicle_cfg
    
    spec = VehicleSpec.from_config(veh_cfg)
    
    # Print training configuration
    _print_training_info(cfg, track, spec, use_subproc)
    
    # Create vectorized training environment
    print("\nInitializing vectorized environments...")
    env = create_vec_env(
        track_path=track_path,
        vehicle_cfg=veh_cfg,
        training_cfg=cfg,
        interpolation_resolution=interpolation_resolution,
        use_subproc=use_subproc,
    )
    
    # Build policy kwargs from config
    policy_kwargs = dict(
        activation_fn=_get_activation_fn(cfg.activation),
        net_arch=dict(
            pi=list(cfg.pi_layers),
            vf=list(cfg.vf_layers),
        )
    )
    
    # Handle tensorboard_log - set to None if empty or not a valid path
    tb_log = cfg.tensorboard_log
    if tb_log is not None and (tb_log == "" or tb_log == "null"):
        tb_log = None
    
    # Create PPO model with config hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        verbose=cfg.verbose,
        tensorboard_log=tb_log,
        device="auto",
    )
    
    print(f"\nStarting training for {cfg.timesteps:,} timesteps...")
    print(f"Expected updates: {cfg.timesteps // (cfg.n_envs * cfg.n_steps)}")
    
    curriculum_cb = CurriculumCallback(verbose=cfg.verbose)
    model.learn(total_timesteps=cfg.timesteps, callback=curriculum_cb)
    
    print("\nTRAINING COMPLETED")
    
    # Build eval environment with the same normalization stats as training
    print("\nGenerating optimal trajectory (looking for completed lap)...")
    raw_eval_env = DummyVecEnv([
        make_env_factory(
            track_path=track_path,
            vehicle_cfg=veh_cfg,
            interpolation_resolution=interpolation_resolution,
            enable_telemetry=True,
            enable_curriculum=False,
        )
    ])
    
    # Wrap eval env with VecNormalize using training stats (no reward normalization for eval)
    if isinstance(env, VecNormalize):
        eval_env = VecNormalize(
            raw_eval_env,
            norm_obs=env.norm_obs,
            norm_reward=False,
            clip_obs=env.clip_obs,
        )
        eval_env.obs_rms = env.obs_rms
        eval_env.training = False
    else:
        eval_env = raw_eval_env
    
    # Close training environment (after copying stats)
    env.close()
    
    # Access the underlying RacingEnv (inside DummyVecEnv, inside VecNormalize)
    def _get_racing_env():
        if isinstance(eval_env, VecNormalize):
            return eval_env.venv.envs[0]
        return eval_env.envs[0]
    
    best_trajectory = None
    best_checkpoints = 0
    max_attempts = 10
    
    for attempt in range(max_attempts):
        obs = eval_env.reset()
        done = False
        xs, ys, vs = [], [], []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            s = _get_racing_env().state
            xs.append(s.x)
            ys.append(s.y)
            vs.append(np.hypot(s.vx, s.vy))
            done = bool(done[0])
        
        racing_env = _get_racing_env()
        checkpoints_hit = len(racing_env.checkpoints_hit) if hasattr(racing_env, 'checkpoints_hit') else 0
        lap_completed = racing_env.lap_completed if hasattr(racing_env, 'lap_completed') else False
        
        print(f"  Attempt {attempt + 1}/{max_attempts}: {checkpoints_hit}/4 checkpoints, "
              f"{'LAP COMPLETED!' if lap_completed else 'incomplete'}, {len(xs)} steps")
        
        if checkpoints_hit > best_checkpoints:
            best_checkpoints = checkpoints_hit
            best_trajectory = (xs, ys, vs, lap_completed, checkpoints_hit)
        
        if lap_completed:
            print(f"\nSuccessfully captured completed lap trajectory!")
            break
    else:
        if best_trajectory is not None:
            xs, ys, vs, lap_completed, checkpoints_hit = best_trajectory
            print(f"\nNo completed lap in {max_attempts} attempts. "
                  f"Using best trajectory with {checkpoints_hit}/4 checkpoints.")
        else:
            print(f"\nFailed to generate any valid trajectory.")
    
    # Export telemetry from eval environment
    racing_env = _get_racing_env()
    if hasattr(racing_env, 'telemetry') and racing_env.telemetry:
        print("Exporting telemetry data...")
        racing_env.telemetry.save_summaries()
        try:
            racing_env.telemetry.export_csv()
        except ImportError:
            print("   Note: Install pandas for CSV export")
        print("   Telemetry data saved to ./telemetry/ directory")
    
    eval_env.close()
    
    traj = np.stack([np.array(xs), np.array(ys), np.array(vs)], axis=1)
    np.save(out_path, traj)
    print(f"Optimal trajectory saved to {out_path} ({len(xs)} points)")
    
    return traj


def _print_training_info(cfg: TrainingConfig, track, spec, use_subproc: bool) -> None:
    """Print detailed training configuration."""
    print("=" * 70)
    print("TRAJECTORY OPTIMIZATION - REINFORCEMENT LEARNING")
    print("=" * 70)
    
    # Training parameters
    print("\nTRAINING CONFIGURATION:")
    print(f"  Timesteps: {cfg.timesteps:,}")
    print(f"  Parallel environments: {cfg.n_envs}")
    print(f"  Steps per env per update: {cfg.n_steps}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Effective samples per update: {cfg.n_envs * cfg.n_steps:,}")
    
    # PPO hyperparameters
    print("\nPPO HYPERPARAMETERS:")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Epochs per update: {cfg.n_epochs}")
    print(f"  Gamma (discount): {cfg.gamma}")
    print(f"  GAE lambda: {cfg.gae_lambda}")
    print(f"  Clip range: {cfg.clip_range}")
    print(f"  Entropy coef: {cfg.ent_coef}")
    print(f"  Value function coef: {cfg.vf_coef}")
    print(f"  Max gradient norm: {cfg.max_grad_norm}")
    
    # Network architecture
    print("\nNETWORK ARCHITECTURE:")
    print(f"  Policy layers: {cfg.pi_layers}")
    print(f"  Value layers: {cfg.vf_layers}")
    print(f"  Activation: {cfg.activation}")
    
    # Track info
    print("\nTRACK:")
    print(f"  Points: {len(track.centerline)}")
    print(f"  Width: {track.width}m")
    print(f"  Interpolation: {track.interpolation_resolution} points")
    
    # Vehicle info
    print("\nVEHICLE:")
    print(f"  Wheelbase: {spec.wheelbase}m")
    print(f"  Track width: {spec.track_width}m")
    print(f"  Max steering: {np.degrees(spec.max_steering_angle):.1f}°")
    
    # Observation space
    print("\nOBSERVATION SPACE (21D - Track-Relative):")
    print("  Vehicle Dynamics (5D): speed, lateral vel, yaw rate, slip angle, steering")
    print("  Track Position (5D): lateral offset, heading error, progress, boundaries")
    print("  Lookahead (8D): 4 points ahead (distance, angle)")
    print("  Control Context (3D): prev throttle, brake, steer")
    
    # Curriculum
    print("\nCURRICULUM LEARNING:")
    print("  Stage 1: Driving School (5x track, never terminate)")
    print("  Stage 2: Learner's Permit (3.5x track, soft termination)")
    print("  Stage 3: Provisional License (2.5x track)")
    print("  Stage 4: Full License (1.8x track)")
    print("  Stage 5: Racing Pro (1.2x track)")
    print("  Stage 6: Champion (1x track, strict)")
    
    # Vectorization
    print("\nVECTORIZED TRAINING:")
    print(f"  Method: {'SubprocVecEnv' if use_subproc else 'DummyVecEnv'}")
    print(f"  Obs normalization: {cfg.normalize_obs}")
    print(f"  Reward normalization: {cfg.normalize_reward}")
    
    # Gear info
    gear_info = get_gear_speed_info(spec)
    print(f"\nSPEED LIMITS:")
    print(f"  Top speed: {gear_info['top_speed_kmh']:.1f} km/h")
    print(f"  RPM limiter: {gear_info['rpm_limiter']:,.0f}")
    
    print("=" * 70)
