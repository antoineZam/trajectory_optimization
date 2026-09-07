"""Configuration wiring tests.

The saved model in `data/saved_model/` matches no config file in the repo
(n_steps 4096 vs 2048, gamma 0.995 vs 0.99, 2.0M timesteps vs 500k) and no
seed was set anywhere, so no run was reproducible. These tests guard the
plumbing that makes a run recoverable from (config, seed).
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir

from envs.rl_environment import RLConfig
from rl.optimal_line_finder import TrainingConfig

CONF_DIR = Path(__file__).parent.parent / "conf"
TRAINING_PRESETS = sorted(p.name for p in (CONF_DIR / "training").glob("*.yaml"))


@pytest.fixture(scope="module")
def composed_config():
    """The fully composed default Hydra config."""
    with initialize_config_dir(config_dir=str(CONF_DIR.resolve()), version_base=None):
        return compose(config_name="config")


def test_hydra_config_composes_with_an_env_group(composed_config):
    """`env` must be part of the default config, not just a file on disk."""
    for group in ("paths", "vehicle", "training", "env"):
        assert group in composed_config, f"missing config group: {group}"


def test_env_group_covers_every_rlconfig_field(composed_config):
    """conf/env/default.yaml and RLConfig must not drift apart.

    Any field missing from the YAML is a parameter that can only be changed by
    editing source, which is what forced reward experiments to be commits.
    """
    dataclass_fields = {f.name for f in fields(RLConfig)}
    yaml_keys = set(composed_config.env.keys())

    assert not dataclass_fields - yaml_keys, (
        f"RLConfig fields absent from conf/env/default.yaml: "
        f"{sorted(dataclass_fields - yaml_keys)}"
    )
    assert not yaml_keys - dataclass_fields, (
        f"conf/env/default.yaml keys that RLConfig ignores: "
        f"{sorted(yaml_keys - dataclass_fields)}"
    )


def test_env_group_values_survive_the_round_trip(composed_config):
    """Values in the YAML must actually reach the dataclass."""
    cfg = RLConfig.from_dict(dict(composed_config.env))
    for field in fields(RLConfig):
        assert getattr(cfg, field.name) == composed_config.env[field.name]


@pytest.mark.parametrize("preset", TRAINING_PRESETS)
def test_every_training_preset_is_observable_and_seeded(preset: str):
    """No preset may ship with logging off or an unset seed.

    All 42 historical runs produced 0-byte logs and no TensorBoard output, so
    none of the 40 "fix" commits could ever be evaluated.
    """
    with (CONF_DIR / "training" / preset).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    assert data.get("tensorboard_log"), f"{preset}: tensorboard_log is unset"
    assert data.get("seed") is not None, f"{preset}: no seed"
    assert "save_freq" in data and "log_interval" in data


@pytest.mark.parametrize("preset", TRAINING_PRESETS)
def test_every_preset_produces_a_usable_learning_curve(preset: str):
    """A run must yield enough log points to read a trend from.

    `log_interval` counts ROLLOUTS, not timesteps. At the old value of 10, a
    500k-step run with n_steps=2048 x n_envs=4 dumped 6 points total -- an
    event file that is technically enabled but useless, and in short runs
    completely empty.
    """
    with (CONF_DIR / "training" / preset).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    cfg = TrainingConfig.from_dict(data)
    steps_per_rollout = cfg.n_steps * cfg.n_envs
    log_points = cfg.timesteps / steps_per_rollout / cfg.log_interval

    # 10 is the floor for reading a trend at all; the `fast` smoke preset sits
    # just above it, `default` and `long` have far more headroom.
    assert log_points >= 10, (
        f"{preset}: only {log_points:.0f} log points for {cfg.timesteps:,} steps "
        f"(n_steps={cfg.n_steps}, n_envs={cfg.n_envs}, log_interval={cfg.log_interval})"
    )


@pytest.mark.parametrize("preset", TRAINING_PRESETS)
def test_every_preset_writes_intermediate_checkpoints(preset: str):
    """`save_freq` must produce several checkpoints, not one at the very end."""
    with (CONF_DIR / "training" / preset).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    cfg = TrainingConfig.from_dict(data)
    assert cfg.save_freq > 0, f"{preset}: checkpointing disabled"
    assert cfg.timesteps / cfg.save_freq >= 4, (
        f"{preset}: only {cfg.timesteps / cfg.save_freq:.1f} checkpoints over the run"
    )


@pytest.mark.parametrize("preset", TRAINING_PRESETS)
def test_training_config_reads_the_observability_fields(preset: str):
    """`TrainingConfig.from_dict` must not silently drop keys.

    `save_freq` and `log_interval` were present in the YAML but never parsed,
    so no checkpoints were written and `log_interval` never reached
    `model.learn()`.
    """
    with (CONF_DIR / "training" / preset).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    cfg = TrainingConfig.from_dict(data)
    assert cfg.tensorboard_log == data["tensorboard_log"]
    assert cfg.seed == data["seed"]
    assert cfg.save_freq == data["save_freq"]
    assert cfg.log_interval == data["log_interval"]
