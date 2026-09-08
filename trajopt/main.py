"""
TrajOpt - Trajectory Optimization with Reinforcement Learning

Usage:
    poetry run python main.py                          # Default optimize mode
    poetry run python main.py mode=simulate            # Simulate mode
    poetry run python main.py training=fast            # Fast training preset
    poetry run python main.py training.timesteps=100000  # Override specific param
    poetry run python main.py training.learning_rate=0.001  # Custom learning rate
    poetry run python main.py paths.tracks=easy        # Only the easy circuits
    poetry run python main.py paths.tracks=null        # Just paths.track
    poetry run python main.py paths.tracks=[monza,spa_francorchamps]
"""
from __future__ import annotations

import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from utils.run_logging import capture_console
from utils.track import load_track_json, make_oval_track, save_track_json
from utils.track_library import LAYOUTS, layouts_with_tags, write_library
from utils.visualization import plot_track, plot_trajectories


def ensure_sample_track(path: str) -> None:
    """Create a sample oval track if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tr = make_oval_track(a=120.0, b=70.0, n=800, width=12.0)
        save_track_json(tr, path)
        print(f"Sample track saved to {path}")


def ensure_track_library(track_dir: str) -> None:
    """Generate the library track files if they are not there yet.

    The generated JSON is derived from `utils.track_library`, so it is not
    committed -- the layouts are the source, the files are a build product.

    Args:
        track_dir: Directory the library is generated into.
    """
    missing = [name for name in LAYOUTS if not (Path(track_dir) / f"{name}.json").exists()]
    if not missing:
        return
    print(f"Generating {len(missing)} missing library track(s) in {track_dir}/")
    write_library(track_dir)


def resolve_training_tracks(cfg: DictConfig) -> list[str]:
    """Work out which tracks a run trains on.

    `paths.tracks` accepts, in order of what it is checked against:

    * `null` -- train on `paths.track` alone, as before the library existed;
    * `"library"` -- every circuit in `utils.track_library`;
    * a tag from the library, e.g. `"easy"` or `"hairpin"`, for curriculum runs;
    * a list of layout names, or of paths to track JSON files.

    Args:
        cfg: The resolved Hydra config.

    Returns:
        Paths to the track JSON files to train on.

    Raises:
        ValueError: If the spec matches no track.
    """
    spec = cfg.paths.get("tracks")
    track_dir = cfg.paths.get("track_dir", "data/tracks")

    if not spec:
        ensure_sample_track(cfg.paths.track)
        return [cfg.paths.track]

    def library_paths(names: list[str]) -> list[str]:
        ensure_track_library(track_dir)
        return [str(Path(track_dir) / f"{name}.json") for name in names]

    if isinstance(spec, str):
        if spec == "library":
            return library_paths(list(LAYOUTS))
        tagged = [layout.name for layout in layouts_with_tags(spec)]
        if not tagged:
            raise ValueError(
                f"paths.tracks={spec!r} is neither 'library', a library tag, nor a "
                f"list; known layouts: {', '.join(LAYOUTS)}"
            )
        return library_paths(tagged)

    names = [str(entry) for entry in spec]
    unknown = [name for name in names if name not in LAYOUTS and not name.endswith(".json")]
    if unknown:
        raise ValueError(
            f"unknown track(s) in paths.tracks: {', '.join(unknown)}; "
            f"known layouts: {', '.join(LAYOUTS)}"
        )
    ensure_track_library(track_dir)
    paths = [
        name if name.endswith(".json") else str(Path(track_dir) / f"{name}.json")
        for name in names
    ]

    # Fail on a mistyped path here rather than after the training banner.
    absent = [path for path in paths if not Path(path).exists()]
    if absent:
        raise FileNotFoundError(f"track file(s) not found: {', '.join(absent)}")
    return paths


def mode_optimize(cfg: DictConfig) -> None:
    """Run trajectory optimization using RL."""
    from rl.optimal_line_finder import train_and_export

    track_paths = resolve_training_tracks(cfg)
    output_path = cfg.paths.optimal_trajectory

    # Convert configs to dict for compatibility
    vehicle_spec = OmegaConf.to_container(cfg.vehicle, resolve=True)
    training_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)

    # Train one policy over the whole set, then export a lap on each track
    trajectories = train_and_export(
        track_paths=track_paths,
        vehicle_cfg=vehicle_spec,
        out_path=output_path,
        training_cfg=training_cfg,
        env_cfg=env_cfg,
        use_subproc=True,
    )

    plot_exported_laps(track_paths, trajectories)


def plot_exported_laps(track_paths: list[str], trajectories: dict) -> None:
    """Show each exported lap on the track it was driven on.

    Args:
        track_paths: The track files that were trained on, in order.
        trajectories: Exported trajectory per track name.
    """
    columns = min(3, len(track_paths))
    rows = (len(track_paths) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False)

    for ax, track_path in zip(axes.flat, track_paths, strict=False):
        tr = load_track_json(track_path)
        plot_track(ax, tr.centerline, tr.left_boundary, tr.right_boundary)
        traj = trajectories.get(tr.name)
        if traj is not None:
            plot_trajectories(ax, optimal=traj[:, :2])
        ax.set_title(f"{tr.name} ({tr.lap_length:.0f} m)")

    for ax in list(axes.flat)[len(track_paths):]:
        ax.axis("off")

    fig.suptitle("Optimized racing lines")
    plt.tight_layout()
    plt.show()


def mode_simulate(cfg: DictConfig) -> None:
    """Visualize a pre-trained optimal trajectory with toy prediction."""
    track_path = cfg.paths.track
    traj_path = cfg.paths.optimal_trajectory

    if not os.path.exists(traj_path):
        # A multi-track run writes one file per circuit, never the bare path.
        stem = Path(traj_path)
        exported = sorted(stem.parent.glob(f"{stem.stem}_*{stem.suffix}"))
        hint = (
            "\nExports from the last multi-track run:\n  "
            + "\n  ".join(str(path) for path in exported)
            + "\nPass one of them, with its track, e.g.:\n"
            f"  python main.py mode=simulate paths.optimal_trajectory={exported[0]} "
            f"paths.track=data/tracks/{exported[0].stem.split('_', 1)[1]}.json"
            if exported
            else ""
        )
        raise FileNotFoundError(
            f"Optimal trajectory not found at {traj_path}. "
            f"Run 'python main.py mode=optimize' first.{hint}"
        )

    tr = load_track_json(track_path)
    traj = np.load(traj_path)
    past = traj[:200, :2]

    # Toy prediction: extrapolation + noise
    pred = past[-1] + (past[-1] - past[-20]) * np.linspace(0, 1, 30)[:, None]
    pred += np.random.normal(scale=0.5, size=pred.shape)

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_track(ax, tr.centerline, tr.left_boundary, tr.right_boundary)
    plot_trajectories(ax, optimal=traj[:, :2], past=past, pred=pred)
    ax.set_title("Simulation: Past vs Predicted (Demo)")
    plt.tight_layout()
    plt.show()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    # Resolve the Hydra run directory before chdir'ing away from it, so the
    # console log lands next to the run's config snapshot.
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    # Change to original working directory (Hydra changes cwd by default, but
    # every path in conf/paths is relative to the project root).
    os.chdir(hydra.utils.get_original_cwd())

    with capture_console(run_dir / "console.log"):
        # Record the fully resolved config: without it a run is not reproducible.
        resolved = OmegaConf.to_yaml(cfg)
        (run_dir / "resolved_config.yaml").write_text(resolved, encoding="utf-8")
        if cfg.training.get("verbose", 0):
            print(resolved)
        print(f"Hydra run directory: {run_dir}")

        if cfg.mode == "optimize":
            mode_optimize(cfg)
        elif cfg.mode == "simulate":
            mode_simulate(cfg)
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}. Use 'optimize' or 'simulate'.")


if __name__ == "__main__":
    main()
