"""Shared pytest fixtures.

The package imports are top-level (`from utils.track import ...`), so the
`trajopt/` directory must be importable. `pythonpath = ["."]` in
pyproject.toml handles that when pytest runs from `trajopt/`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from envs.rl_environment import RacingEnv, RLConfig
from physics.physics_engine import VehicleSpec
from utils.track import Track, load_track_json

PROJECT_ROOT = Path(__file__).parent
TRACK_PATH = PROJECT_ROOT / "data" / "tracks" / "sample_track.json"
VEHICLE_CONF = PROJECT_ROOT / "conf" / "vehicle" / "default.yaml"


@pytest.fixture(scope="session")
def vehicle_spec() -> VehicleSpec:
    """The default vehicle, loaded through the Pydantic-validated path."""
    with VEHICLE_CONF.open(encoding="utf-8") as handle:
        return VehicleSpec.from_config(yaml.safe_load(handle))


@pytest.fixture(scope="session")
def track() -> Track:
    """The sample oval at its real 12 m width."""
    return load_track_json(str(TRACK_PATH))


@pytest.fixture
def env(track: Track, vehicle_spec: VehicleSpec) -> RacingEnv:
    """A curriculum-free environment on the real track.

    Curriculum off and telemetry off: tests must exercise the task the export
    step actually evaluates on, not a widened training variant.

    max_steps is 3000 rather than the 1500 default so a slow lap still fits,
    but small enough that a controller which fails to lap fails fast.
    Randomization is off so reference lap times are comparable; the random
    starts are exercised by their own tests.
    """
    return RacingEnv(
        track=track,
        veh_spec=vehicle_spec,
        cfg=RLConfig(max_steps=3_000, randomize_reset=False),
        enable_telemetry=False,
        enable_curriculum=False,
    )
