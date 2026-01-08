"""
TrajOpt - Trajectory Optimization with Reinforcement Learning

Usage:
    poetry run python main.py                          # Default optimize mode
    poetry run python main.py mode=simulate            # Simulate mode
    poetry run python main.py training=fast            # Fast training preset
    poetry run python main.py training.timesteps=100000  # Override specific param
"""
from __future__ import annotations

import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from utils.track import load_track_json, make_oval_track, save_track_json
from utils.visualization import plot_track, plot_trajectories


def ensure_sample_track(path: str) -> None:
    """Create a sample oval track if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tr = make_oval_track(a=120.0, b=70.0, n=800, width=12.0)
        save_track_json(tr, path)
        print(f"Sample track saved to {path}")


def mode_optimize(cfg: DictConfig) -> None:
    """Run trajectory optimization using RL."""
    from rl.optimal_line_finder import train_and_export

    track_path = cfg.paths.track
    output_path = cfg.paths.optimal_trajectory
    timesteps = cfg.training.timesteps
    n_envs = cfg.training.env.n_envs

    ensure_sample_track(track_path)

    # Convert vehicle config to dict for compatibility
    vehicle_spec = OmegaConf.to_container(cfg.vehicle, resolve=True)

    traj = train_and_export(
        track_path,
        vehicle_spec,
        output_path,
        timesteps=timesteps,
        n_envs=n_envs,
        use_subproc=True,
    )

    tr = load_track_json(track_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_track(ax, tr.centerline, tr.left_boundary, tr.right_boundary)
    plot_trajectories(ax, optimal=traj[:, :2])
    ax.set_title("Optimized racing line")
    plt.show()


def mode_simulate(cfg: DictConfig) -> None:
    """Visualize a pre-trained optimal trajectory with toy prediction."""
    track_path = cfg.paths.track
    traj_path = cfg.paths.optimal_trajectory

    if not os.path.exists(traj_path):
        raise FileNotFoundError(
            f"Optimal trajectory not found at {traj_path}. "
            "Run 'python main.py mode=optimize' first."
        )

    tr = load_track_json(track_path)
    traj = np.load(traj_path)
    past = traj[:200, :2]

    # Toy prediction: extrapolation + noise
    pred = past[-1] + (past[-1] - past[-20]) * np.linspace(0, 1, 30)[:, None]
    pred += np.random.normal(scale=0.5, size=pred.shape)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_track(ax, tr.centerline, tr.left_boundary, tr.right_boundary)
    plot_trajectories(ax, optimal=traj[:, :2], past=past, pred=pred)
    ax.set_title("Simulation: past vs predicted (demo)")
    plt.show()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    # Print resolved config for debugging
    print(OmegaConf.to_yaml(cfg))

    # Change to original working directory (Hydra changes cwd by default)
    os.chdir(hydra.utils.get_original_cwd())

    if cfg.mode == "optimize":
        mode_optimize(cfg)
    elif cfg.mode == "simulate":
        mode_simulate(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Use 'optimize' or 'simulate'.")


if __name__ == "__main__":
    main()
