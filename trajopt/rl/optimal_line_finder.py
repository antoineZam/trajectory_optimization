"""
Optimal racing line finder using Reinforcement Learning.

Uses vectorized environments with SubprocVecEnv for parallel training
across multiple CPU cores.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from typing import Callable, Union

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs.rl_environment import RacingEnv
from physics.physics_engine import VehicleSpec, get_gear_speed_info, get_steering_info
from utils.track import load_track_json

# Default number of parallel environments
DEFAULT_N_ENVS = min(mp.cpu_count(), 8)


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
    n_envs: int = DEFAULT_N_ENVS,
    interpolation_resolution: int = 2000,
    use_subproc: bool = True,
    normalize: bool = True,
) -> tuple:
    """Create a vectorized environment for parallel training.
    
    Args:
        track_path: Path to track JSON file.
        vehicle_cfg: Vehicle configuration dictionary.
        n_envs: Number of parallel environments.
        interpolation_resolution: Track interpolation resolution.
        use_subproc: If True, use SubprocVecEnv for true parallelism.
                     If False, use DummyVecEnv (sequential, for debugging).
        normalize: If True, wrap with VecNormalize for observation/reward normalization.
        
    Returns:
        Tuple of (vectorized_env, eval_env) where eval_env is a single DummyVecEnv
        for deterministic evaluation.
    """
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
    if normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        print("  Observation and reward normalization enabled")
    
    # Create a separate eval environment (single, deterministic)
    eval_env = DummyVecEnv([
        make_env_factory(
            track_path=track_path,
            vehicle_cfg=vehicle_cfg,
            interpolation_resolution=interpolation_resolution,
            enable_telemetry=True,
            enable_curriculum=False,  # No curriculum for eval
        )
    ])
    
    return vec_env, eval_env


def train_and_export(
    track_path: str,
    vehicle_cfg: Union[str, dict],
    out_path: str,
    timesteps: int = 500_000,
    interpolation_resolution: int = 2000,
    n_envs: int = DEFAULT_N_ENVS,
    use_subproc: bool = True,
):
    """Train RL agent and export optimal trajectory.
    
    Args:
        track_path: Path to track JSON file.
        vehicle_cfg: Either a path to vehicle config JSON or a config dict.
        out_path: Path to save the optimal trajectory.
        timesteps: Total training timesteps.
        interpolation_resolution: Track interpolation resolution.
        n_envs: Number of parallel environments for training.
        use_subproc: Whether to use SubprocVecEnv (True) or DummyVecEnv (False).
    
    Returns:
        Optimal trajectory as numpy array.
    """
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
    print("=" * 60)
    print("TRAJECTORY OPTIMIZATION - VECTORIZED PARALLEL TRAINING")
    print("=" * 60)
    print(f"Training timesteps: {timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Effective samples per update: {n_envs * 2048:,}")
    print(f"Track points: {len(track.centerline)}")
    print(f"Interpolated track points: {track.interpolation_resolution}")
    print(f"Track width: {track.width}m")
    print(f"Checkpoints: 4 (every {len(track.centerline)//4} track points)")
    print(f"Vehicle wheelbase: {spec.wheelbase}m, track width: {spec.track_width}m")
    print("FIXED ACTION SPACE:")
    print(f"  - Agent steer_command [-1, 1] → steering angle [-{np.degrees(spec.max_steering_angle):.1f}°, +{np.degrees(spec.max_steering_angle):.1f}°]")
    print("  - No more unrealistic 57° steering commands!")
    print("REALISTIC VEHICLE DYNAMICS:")
    print("  - Speed-dependent steering limitations (agent learns to work within limits)")
    print(f"  - Max steering angle: {spec.max_steering_angle:.2f} rad ({np.degrees(spec.max_steering_angle):.1f}°)")
    print(f"  - Minimum turn radius: {spec.min_turn_radius}m")
    print("  - Physics engine applies additional speed-based clipping")
    print("OBSERVATION SPACE (21 Dimensions - Track-Relative):")
    print("  Vehicle Dynamics (5D): speed, lateral velocity, yaw rate, slip angle, steering")
    print("  Track Position (5D): lateral offset (Frenet), heading error, progress,")
    print("     left/right boundary distances - all normalized to [-1, 1] or [0, 1]")
    print("  Lookahead (8D): 4 points ahead in vehicle-relative polar coordinates")
    print("     (distance, angle) - tells agent about upcoming track geometry")
    print("  Control Context (3D): previous throttle, brake, steering for smooth control")
    print("CURRICULUM LEARNING SYSTEM:")
    print("  Stage 1: Driving School (4x wide track, never terminate)")
    print("  Stage 2: Learner's Permit (2.5x wide, soft termination)")
    print("  Stage 3: Provisional License (1.8x wide, normal rules)")
    print("  Stage 4: Full License (1.2x wide, realistic racing)")
    print("  Stage 5: Racing Pro (1x wide, professional level)")
    print("VECTORIZED TRAINING:")
    print(f"  - {n_envs} parallel simulations running simultaneously")
    print(f"  - {'SubprocVecEnv (multi-process)' if use_subproc else 'DummyVecEnv (single-process)'}")
    print("  - VecNormalize for observation/reward scaling")
    print("  - ~{:.1f}x faster training vs single environment".format(n_envs * 0.7))
    
    gear_info = get_gear_speed_info(spec)
    print(f"\nREALISTIC GEAR-BASED SPEED LIMITS (RPM limited):")
    print(f"  RPM Limiter: {gear_info['rpm_limiter']:,.0f} RPM")
    print(f"  Final Drive: {gear_info['final_drive']:.2f}:1")
    print(f"  Absolute Top Speed: {gear_info['absolute_top_speed_kmh']:.1f} km/h ({gear_info['absolute_top_speed_ms']:.1f} m/s)")
    print(f"  Effective Top Speed: {gear_info['top_speed_kmh']:.1f} km/h ({gear_info['top_speed_ms']:.1f} m/s)")
    print("  Gear Speed Limits:")
    for gear in range(1, len(spec.gear_ratios) + 1):
        gear_data = gear_info['gear_speeds'][f'gear_{gear}']
        print(f"    Gear {gear}: {gear_data['max_speed_kmh']:5.1f} km/h ({gear_data['max_speed_ms']:4.1f} m/s) @ {gear_data['rpm_at_max_speed']:,.0f} RPM")
    
    print("\nSTEERING LIMITATIONS (Speed-dependent):")
    for speed_kmh in [0, 30, 60, 120, 180]:
        if speed_kmh <= gear_info['top_speed_kmh']:
            speed_ms = speed_kmh / 3.6
            info = get_steering_info(spec, speed_ms)
            reduction = info['max_steering_angle_deg'] / np.degrees(spec.max_steering_angle)
            print(f"  {speed_kmh:3d} km/h: max {info['max_steering_angle_deg']:4.1f}° ({reduction:4.1%} of base) → {info['turn_radius_m']:4.1f}m radius")
    print("=" * 60)
    
    # Create vectorized environments
    print("\nInitializing vectorized environments...")
    env, eval_env = create_vec_env(
        track_path=track_path,
        vehicle_cfg=veh_cfg,
        n_envs=n_envs,
        interpolation_resolution=interpolation_resolution,
        use_subproc=use_subproc,
        normalize=True,
    )
    
    # Custom hyperparameters optimized for vectorized training
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])  # Larger network for parallel data
    )

    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,        # Standard PPO learning rate
        n_steps=2048,              # Steps per environment before update
        batch_size=256,            # Larger batch for vectorized training
        n_epochs=10,               # Epochs per update
        gamma=0.99,                # Discount factor
        gae_lambda=0.95,           # GAE lambda
        clip_range=0.2,            # PPO clip range
        ent_coef=0.01,             # Entropy coefficient for exploration
        vf_coef=0.5,               # Value function coefficient
        max_grad_norm=0.5,         # Gradient clipping
        verbose=1, 
        tensorboard_log=None,
        device="auto",             # Auto-select CPU/GPU
    )

    print(f"\nStarting training for {timesteps:,} timesteps...")
    print(f"Expected updates: {timesteps // (n_envs * 2048)}")
    
    model.learn(total_timesteps=timesteps)
    
    print("\nTRAINING COMPLETED")
    
    # Close training environment
    env.close()

    # Rollout best trajectory using eval environment (deterministic)
    print("\nGenerating optimal trajectory...")
    obs = eval_env.reset()
    done = False
    xs, ys, vs = [], [], []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        s = eval_env.envs[0].state
        xs.append(s.x)
        ys.append(s.y)
        vs.append(np.hypot(s.vx, s.vy))
        done = bool(done[0])
    
    # Export telemetry from eval environment
    if hasattr(eval_env.envs[0], 'telemetry') and eval_env.envs[0].telemetry:
        print("Exporting telemetry data...")
        eval_env.envs[0].telemetry.save_summaries()
        try:
            eval_env.envs[0].telemetry.export_csv()
        except ImportError:
            print("   Note: Install pandas for CSV export")
        print("   Telemetry data saved to ./telemetry/ directory")
    
    eval_env.close()
    
    traj = np.stack([np.array(xs), np.array(ys), np.array(vs)], axis=1)
    np.save(out_path, traj)
    print(f"Optimal trajectory saved to {out_path}")
    
    return traj