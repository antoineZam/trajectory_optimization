"""Selection of the tracks a run trains on, and where each lap is exported.

`paths.tracks` is the handle on the whole library -- a tag for a curriculum
run, a list for a targeted one -- so a typo in it must fail loudly rather than
silently training on one circuit.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from conftest import TRACK_PATH, VEHICLE_CONF
from main import resolve_training_tracks
from rl.optimal_line_finder import _trajectory_path, load_tracks, make_env_factory
from utils.track_library import LAYOUTS, layouts_with_tags


def _cfg(tmp_path, **paths):
    """A config with only the `paths` group the resolver reads."""
    return OmegaConf.create(
        {
            "paths": {
                "track": str(TRACK_PATH),
                "track_dir": str(tmp_path),
                "optimal_trajectory": str(tmp_path / "optimal_traj.npy"),
                **paths,
            }
        }
    )


def test_no_track_set_means_the_reference_track_alone(tmp_path):
    """The single-track flow that predates the library must still work."""
    assert resolve_training_tracks(_cfg(tmp_path, tracks=None)) == [str(TRACK_PATH)]


def test_the_library_keyword_selects_every_circuit(tmp_path):
    paths = resolve_training_tracks(_cfg(tmp_path, tracks="library"))

    assert len(paths) == len(LAYOUTS)
    # The JSON files are a build product: the resolver generates what is missing.
    assert all((tmp_path / f"{name}.json").exists() for name in LAYOUTS)
    assert [Path(p).name for p in paths] == [f"{name}.json" for name in LAYOUTS]


def test_a_tag_selects_a_curriculum_subset(tmp_path):
    paths = resolve_training_tracks(_cfg(tmp_path, tracks="easy"))

    expected = [layout.name for layout in layouts_with_tags("easy")]
    assert expected, "the library must keep an 'easy' end for curriculum runs"
    assert len(paths) == len(expected)
    assert len(paths) < len(LAYOUTS)


def test_an_explicit_list_of_layouts_is_honoured(tmp_path):
    paths = resolve_training_tracks(
        _cfg(tmp_path, tracks=["monza", "spa_francorchamps"])
    )

    assert len(paths) == 2
    assert paths[0].endswith("monza.json")
    assert paths[1].endswith("spa_francorchamps.json")


def test_an_explicit_json_path_passes_through(tmp_path):
    paths = resolve_training_tracks(_cfg(tmp_path, tracks=[str(TRACK_PATH)]))
    assert paths == [str(TRACK_PATH)]


def test_an_unknown_tag_is_rejected_with_the_alternatives(tmp_path):
    with pytest.raises(ValueError, match="neither"):
        resolve_training_tracks(_cfg(tmp_path, tracks="fastest"))


def test_an_unknown_layout_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown track"):
        resolve_training_tracks(_cfg(tmp_path, tracks=["monza", "nurburgring"]))


def test_one_track_writes_the_configured_path():
    """`mode=simulate` reads the configured path, so it must stay exact."""
    destination = _trajectory_path("data/optimal_traj.npy", "monza", single=True)
    assert destination == Path("data/optimal_traj.npy")


def test_several_tracks_write_one_file_each():
    """One file cannot hold laps of different circuits."""
    monza = _trajectory_path("data/optimal_traj.npy", "monza", single=False)
    spa = _trajectory_path("data/optimal_traj.npy", "spa_francorchamps", single=False)

    assert monza.name == "optimal_traj_monza.npy"
    assert spa.name == "optimal_traj_spa_francorchamps.npy"
    assert monza.parent == spa.parent


def test_a_worker_holds_every_configured_track(tmp_path):
    """Each worker samples the whole set, rather than owning one track.

    With four workers and twelve circuits, a track per worker would train on a
    third of the library and give each track a quarter of the batch.
    """
    resolve_training_tracks(_cfg(tmp_path, tracks="library"))
    track_paths = [str(tmp_path / f"{name}.json") for name in ("monza", "hungaroring")]

    with VEHICLE_CONF.open(encoding="utf-8") as handle:
        vehicle_cfg = yaml.safe_load(handle)

    env = make_env_factory(
        track_paths=track_paths,
        vehicle_cfg=vehicle_cfg,
        arc_step=4.0,
        enable_telemetry=False,
        seed=0,
    )().unwrapped

    assert {track.name for track in env.tracks} == {"monza", "hungaroring"}


def test_the_centerline_resolution_follows_the_track(tmp_path):
    """A point count fixed for one track is a different spatial step on another."""
    resolve_training_tracks(_cfg(tmp_path, tracks=["monza", "brands_hatch_indy"]))
    monza, brands = load_tracks(
        [str(tmp_path / "monza.json"), str(tmp_path / "brands_hatch_indy.json")],
        arc_step=1.0,
    )

    assert monza.arc_step == pytest.approx(1.0, abs=0.05)
    assert brands.arc_step == pytest.approx(1.0, abs=0.05)
    # Which means the sample counts differ, by the ratio of the lap lengths.
    assert monza.interpolation_resolution > 2 * brands.interpolation_resolution


def test_an_empty_track_list_is_rejected():
    with pytest.raises(ValueError, match="no track paths"):
        load_tracks([])
