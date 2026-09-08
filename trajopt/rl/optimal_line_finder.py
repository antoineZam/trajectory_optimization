"""
Optimal racing line finder using Reinforcement Learning.

Uses vectorized environments with SubprocVecEnv for parallel training
across multiple CPU cores. All hyperparameters are configurable through
Hydra configuration.

Training runs on a *set* of tracks: every worker holds all of them and draws
one per episode, so the policy is fitted to a distribution of circuits rather
than to one memorised lap. A racing line is then exported per track, since the
optimal line is a property of the circuit, not of the policy.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs.rl_environment import RacingEnv, RLConfig
from physics.physics_engine import VehicleSpec, get_gear_speed_info
from utils.track import Track, load_track_json

# Default number of parallel environments
DEFAULT_N_ENVS = min(mp.cpu_count(), 8)

# Metres between interpolated centerline samples. A point count cannot be
# shared by a 1.9 km and a 7 km circuit; a spatial step can.
DEFAULT_ARC_STEP = 1.0

# Evaluation laps attempted per track before exporting the best one.
#
# One is enough as things stand: the export forces `randomize_reset` off and
# asks the policy for deterministic actions, so every attempt replays the same
# lap -- ten attempts on twelve circuits printed twelve groups of ten identical
# lines and cost ten times the evaluation. Raise it only alongside a source of
# variation, such as a randomised start.
EXPORT_ATTEMPTS = 1


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
    gamma: float = 0.995  # 10 s horizon at dt=0.05; a lap is ~27 s
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0  # 0.01 pinned the action std at 1.0; see conf/training
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

    # Reproducibility: seeds PPO, the vec env and every env reset.
    seed: int = 0

    # Logging
    verbose: int = 1
    tensorboard_log: str | None = None
    log_interval: int = 1        # ROLLOUTS between SB3 log dumps (not steps)
    save_freq: int = 50_000      # Timesteps between intermediate checkpoints (0 = off)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> TrainingConfig:
        """Create TrainingConfig from a dictionary (e.g., from Hydra)."""
        # Extract values with defaults
        kwargs = {}

        # Direct mappings
        direct_fields = [
            "timesteps", "learning_rate", "n_steps", "batch_size", "n_epochs",
            "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef",
            "max_grad_norm", "verbose", "tensorboard_log", "log_interval",
            "save_freq", "seed",
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


def load_tracks(
    track_paths: str | Sequence[str], arc_step: float = DEFAULT_ARC_STEP
) -> list[Track]:
    """Load one or several track JSON files.

    Args:
        track_paths: A path, or paths, to track JSON files.
        arc_step: Metres between interpolated centerline samples.

    Returns:
        The loaded tracks, in the order given.
    """
    paths = [track_paths] if isinstance(track_paths, str) else list(track_paths)
    if not paths:
        raise ValueError("no track paths given")
    return [
        load_track_json(path, interpolation_resolution=None, arc_step=arc_step)
        for path in paths
    ]


def make_env_factory(
    track_paths: str | Sequence[str],
    vehicle_cfg: dict,
    arc_step: float = DEFAULT_ARC_STEP,
    enable_telemetry: bool = False,
    env_cfg: dict | None = None,
    seed: int | None = None,
) -> Callable[[], Monitor]:
    """Create a factory function for environment instantiation.

    This factory is picklable and can be used with SubprocVecEnv.

    Args:
        track_paths: Path, or paths, to track JSON files. Given several, the
            environment draws one per episode.
        vehicle_cfg: Vehicle configuration dictionary.
        arc_step: Metres between interpolated centerline samples.
        enable_telemetry: Whether to enable telemetry (only for rank 0).
        env_cfg: RLConfig overrides (the Hydra `env` group). None uses defaults.
        seed: Per-env seed. Seeds the first reset and the action space so runs
            are reproducible; None leaves seeding to the caller.

    Returns:
        A callable that creates a Monitor-wrapped RacingEnv instance.
    """
    def _init() -> Monitor:
        tracks = load_tracks(track_paths, arc_step)
        spec = VehicleSpec.from_config(vehicle_cfg)
        env = RacingEnv(
            track=tracks,
            veh_spec=spec,
            cfg=RLConfig.from_dict(env_cfg),
            enable_telemetry=enable_telemetry,
        )
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)

        # Monitor is what populates info["episode"], which is the only source
        # SB3 has for rollout/ep_rew_mean and rollout/ep_len_mean -- the two
        # curves you actually read to tell whether the agent is learning.
        # Without it the TensorBoard run has train/* scalars only.
        return Monitor(env)
    return _init


def create_vec_env(
    track_paths: str | Sequence[str],
    vehicle_cfg: dict,
    training_cfg: TrainingConfig,
    arc_step: float = DEFAULT_ARC_STEP,
    use_subproc: bool = True,
    env_cfg: dict | None = None,
) -> tuple:
    """Create a vectorized environment for parallel training.

    Every worker gets the whole track set rather than one track each: with
    four workers and twelve circuits, a track per worker would train on a
    third of the library and give each track a quarter of the batch. The cost
    is that each worker holds every track's splines, boundaries and polygon --
    a few MB per circuit at the default arc step, built lazily on the first
    episode that draws it. Raise `arc_step` to trade accuracy for that memory.

    Args:
        track_paths: Path, or paths, to track JSON files.
        vehicle_cfg: Vehicle configuration dictionary.
        training_cfg: Training configuration with env settings.
        arc_step: Metres between interpolated centerline samples.
        use_subproc: If True, use SubprocVecEnv for true parallelism.
                     If False, use DummyVecEnv (sequential, for debugging).
        env_cfg: RLConfig overrides (the Hydra `env` group).

    Returns:
        Vectorized training environment (optionally wrapped with VecNormalize).
    """
    n_envs = training_cfg.n_envs

    # Create environment factories
    # Only enable telemetry on the first environment to avoid conflicts.
    # Each worker gets a distinct but deterministic seed so rollouts are
    # reproducible without all workers exploring the identical trajectory.
    env_fns = [
        make_env_factory(
            track_paths=track_paths,
            vehicle_cfg=vehicle_cfg,
            arc_step=arc_step,
            enable_telemetry=(i == 0),  # Only first env has telemetry
            env_cfg=env_cfg,
            seed=training_cfg.seed + i,
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
        print(
            f"  Normalization: obs={training_cfg.normalize_obs},"
            f" reward={training_cfg.normalize_reward}"
        )

    # Seed the vec env itself (action space sampling, per-env reset streams)
    vec_env.seed(training_cfg.seed)

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
    track_paths: str | Sequence[str],
    vehicle_cfg: str | dict,
    out_path: str,
    training_cfg: dict[str, Any] | TrainingConfig | None = None,
    arc_step: float = DEFAULT_ARC_STEP,
    use_subproc: bool = True,
    env_cfg: dict | None = None,
    # Legacy parameters for backwards compatibility
    timesteps: int | None = None,
    n_envs: int | None = None,
) -> dict[str, np.ndarray]:
    """Train one policy over a set of tracks and export a lap on each.

    Args:
        track_paths: Path, or paths, to track JSON files. Training draws one
            track per episode; the export then drives each of them in turn.
        vehicle_cfg: Either a path to vehicle config JSON or a config dict.
        out_path: Where to save the exported trajectory. A single track writes
            this path; several write `<stem>_<track><suffix>` beside it.
        training_cfg: Training configuration (dict from Hydra or TrainingConfig).
        arc_step: Metres between interpolated centerline samples.
        use_subproc: Whether to use SubprocVecEnv (True) or DummyVecEnv (False).
        env_cfg: Environment/reward configuration (the Hydra `env` group).
        timesteps: (Legacy) Override total timesteps.
        n_envs: (Legacy) Override number of environments.

    Returns:
        Exported trajectory per track name, each an (N, 3) array of x, y, speed.
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

    # Load every track: for the info display now, and for the export later.
    paths = [track_paths] if isinstance(track_paths, str) else list(track_paths)
    tracks = load_tracks(paths, arc_step)

    # Accept either a path string or a config dict
    if isinstance(vehicle_cfg, str):
        with open(vehicle_cfg, encoding="utf-8") as f:
            veh_cfg = json.load(f)
    else:
        veh_cfg = vehicle_cfg

    spec = VehicleSpec.from_config(veh_cfg)

    # Print training configuration
    _print_training_info(cfg, tracks, spec, use_subproc)

    # Seed every global RNG before anything stochastic happens, so that a run is
    # reproducible from (config, seed) alone.
    set_random_seed(cfg.seed)

    # Create vectorized training environment
    print("\nInitializing vectorized environments...")
    env = create_vec_env(
        track_paths=paths,
        vehicle_cfg=veh_cfg,
        training_cfg=cfg,
        arc_step=arc_step,
        use_subproc=use_subproc,
        env_cfg=env_cfg,
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
        seed=cfg.seed,
        device="auto",
    )

    print(f"\nStarting training for {cfg.timesteps:,} timesteps...")
    print(f"Expected updates: {cfg.timesteps // (cfg.n_envs * cfg.n_steps)}")
    if tb_log:
        print(f"TensorBoard logs: {tb_log}  (view with: tensorboard --logdir {tb_log})")
    else:
        print("WARNING: tensorboard_log is unset - this run will produce no learning curves.")

    import os
    model_dir = os.path.join(os.path.dirname(out_path), "saved_model")
    os.makedirs(model_dir, exist_ok=True)

    callbacks: list[BaseCallback] = []
    if cfg.save_freq > 0:
        # save_freq counts calls per env, so divide to get global timesteps
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, cfg.save_freq // cfg.n_envs),
                save_path=os.path.join(model_dir, "checkpoints"),
                name_prefix="ppo_racing",
                save_vecnormalize=True,
                verbose=cfg.verbose,
            )
        )

    model.learn(
        total_timesteps=cfg.timesteps,
        callback=CallbackList(callbacks) if callbacks else None,
        log_interval=cfg.log_interval,
    )

    print("\nTRAINING COMPLETED")

    # Save model and normalization stats
    model.save(os.path.join(model_dir, "ppo_racing"))
    if isinstance(env, VecNormalize):
        env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    print(f"Model saved to {model_dir}/")

    # Copy the observation statistics before closing the training env: the
    # policy was trained on normalised observations and the export has to feed
    # it the same scaling.
    obs_rms = env.obs_rms if isinstance(env, VecNormalize) else None
    clip_obs = float(env.clip_obs) if isinstance(env, VecNormalize) else 10.0
    env.close()

    print(f"\nExporting a racing line on each of the {len(tracks)} track(s)...")
    trajectories: dict[str, np.ndarray] = {}
    for track_path, track in zip(paths, tracks, strict=True):
        traj, lap_completed, checkpoints_hit = _export_best_lap(
            model=model,
            track_path=track_path,
            track_name=track.name,
            vehicle_cfg=veh_cfg,
            env_cfg=env_cfg,
            seed=cfg.seed,
            arc_step=arc_step,
            obs_rms=obs_rms,
            clip_obs=clip_obs,
        )

        destination = _trajectory_path(out_path, track.name, single=len(tracks) == 1)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, traj)
        trajectories[track.name] = traj

        status = "lap completed" if lap_completed else f"{checkpoints_hit} checkpoints only"
        print(f"  {track.name:22s} {len(traj):5d} points, {status} -> {destination}")

    return trajectories


def _trajectory_path(out_path: str, track_name: str, single: bool) -> Path:
    """Where one track's exported lap is written.

    A single-track run writes `out_path` itself, which is what `mode=simulate`
    reads. With several tracks one file cannot hold them all, so each gets
    `<stem>_<track><suffix>` next to it.

    Args:
        out_path: The configured output path.
        track_name: Name of the track the lap was driven on.
        single: Whether this is the only track in the run.

    Returns:
        The path to save to.
    """
    base = Path(out_path)
    if single:
        return base
    return base.with_name(f"{base.stem}_{track_name}{base.suffix}")


def _export_best_lap(
    model: PPO,
    track_path: str,
    track_name: str,
    vehicle_cfg: dict,
    env_cfg: dict | None,
    seed: int,
    arc_step: float,
    obs_rms=None,
    clip_obs: float = 10.0,
    attempts: int = EXPORT_ATTEMPTS,
) -> tuple[np.ndarray, bool, int]:
    """Drive the trained policy on one track and keep its best lap.

    Args:
        model: The trained policy.
        track_path: Track JSON to drive.
        track_name: Track name, for the progress lines.
        vehicle_cfg: Vehicle configuration dictionary.
        env_cfg: RLConfig overrides; the start is forced onto the line.
        seed: Seed for the evaluation environment.
        arc_step: Metres between interpolated centerline samples.
        obs_rms: Observation statistics from training, or None if the training
            environment was not normalised.
        clip_obs: Observation clipping threshold used during training.
        attempts: Laps to try before settling for the best incomplete run.

    Returns:
        The trajectory as an (N, 3) array of x, y, speed; whether a lap was
        completed; and how many checkpoints the exported run hit.
    """
    raw_eval_env = DummyVecEnv([
        make_env_factory(
            track_paths=track_path,
            vehicle_cfg=vehicle_cfg,
            arc_step=arc_step,
            enable_telemetry=True,
            # Deterministic start on the line: the exported trajectory has to
            # be a comparable lap, not one begun from a random point.
            env_cfg={**(env_cfg or {}), "randomize_reset": False},
            seed=seed,
        )
    ])

    # Normalise observations with the training statistics; rewards are not
    # normalised for evaluation.
    if obs_rms is not None:
        eval_env = VecNormalize(
            raw_eval_env, norm_obs=True, norm_reward=False, clip_obs=clip_obs
        )
        eval_env.obs_rms = obs_rms
        eval_env.training = False
    else:
        eval_env = raw_eval_env

    # envs[0] is a Monitor wrapper; .unwrapped reaches the RacingEnv.
    venv = eval_env.venv if isinstance(eval_env, VecNormalize) else eval_env
    racing_env: RacingEnv = venv.envs[0].unwrapped

    best_trajectory = None
    best_score = None

    for attempt in range(attempts):
        obs = eval_env.reset()
        xs, ys, vs = [], [], []

        def _record() -> None:
            s = racing_env.state
            xs.append(s.x)
            ys.append(s.y)
            vs.append(float(np.hypot(s.vx, s.vy)))

        _record()  # starting state

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)

            if dones[0]:
                # DummyVecEnv auto-resets on done, so by the time control
                # returns here racing_env.state, .checkpoints_hit and
                # .lap_completed already describe the NEXT episode. Reading
                # them after the loop is why every attempt reported 0/4
                # checkpoints and lap_completed False, which made
                # `checkpoints_hit > best_checkpoints` permanently false,
                # left best_trajectory as None, and still exported the raw
                # xs/ys/vs of the last attempt -- whose final point was the
                # reset state, visible as [120.0, 0.0, 5.0] in the shipped
                # optimal_traj.npy. The episode results survive only here.
                checkpoints_hit = int(infos[0].get("checkpoints_hit", 0))
                lap_completed = bool(infos[0].get("lap_completed", False))
                break

            _record()

        print(f"  {track_name}: lap {attempt + 1}/{attempts}: "
              f"{checkpoints_hit}/{racing_env.num_checkpoints} checkpoints, "
              f"{'LAP COMPLETED!' if lap_completed else 'incomplete'}, {len(xs)} points")

        # Prefer a completed lap, then more checkpoints, then a faster run.
        score = (lap_completed, checkpoints_hit, -len(xs))
        if best_score is None or score > best_score:
            best_score = score
            best_trajectory = (xs, ys, vs, lap_completed, checkpoints_hit)

        if lap_completed:
            break

    if best_trajectory is None:
        raise RuntimeError(
            f"Evaluation produced no trajectory at all on {track_name}. The "
            "trained model is saved; re-run the export separately."
        )

    xs, ys, vs, lap_completed, checkpoints_hit = best_trajectory
    if not lap_completed:
        print(
            f"  WARNING: no completed lap on {track_name} in {attempts} attempt(s). "
            f"Exporting the best run ({checkpoints_hit}/"
            f"{racing_env.num_checkpoints} checkpoints, {len(xs)} points). "
            "This is NOT an optimal racing line -- treat it as a diagnostic trace."
        )

    if racing_env.telemetry:
        racing_env.telemetry.save_summaries()
        try:
            racing_env.telemetry.export_csv()
        except ImportError:
            print("   Note: Install pandas for CSV export")

    eval_env.close()

    traj = np.stack([np.array(xs), np.array(ys), np.array(vs)], axis=1)
    return traj, lap_completed, checkpoints_hit


def _print_training_info(
    cfg: TrainingConfig, tracks: list[Track], spec, use_subproc: bool
) -> None:
    """Print detailed training configuration.

    Args:
        cfg: Training configuration.
        tracks: Every track the run will draw episodes from.
        spec: Vehicle specification.
        use_subproc: Whether workers run in subprocesses.
    """
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
    print(f"\nTRACKS ({len(tracks)}, one drawn per episode):")
    for track in tracks:
        print(
            f"  {track.name:22s} {track.lap_length:7.0f} m  "
            f"width {track.width:4.1f} m  "
            f"{track.interpolation_resolution:6d} samples "
            f"({track.arc_step:.2f} m apart)"
        )

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

    # Vectorization
    print("\nVECTORIZED TRAINING:")
    print(f"  Method: {'SubprocVecEnv' if use_subproc else 'DummyVecEnv'}")
    print(f"  Obs normalization: {cfg.normalize_obs}")
    print(f"  Reward normalization: {cfg.normalize_reward}")

    # Gear info
    gear_info = get_gear_speed_info(spec)
    print("\nSPEED LIMITS:")
    print(f"  Top speed: {gear_info['top_speed_kmh']:.1f} km/h")
    print(f"  RPM limiter: {gear_info['rpm_limiter']:,.0f}")

    print("=" * 70)
