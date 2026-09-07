"""Trajectory export contract.

The export loop read its results from the environment *after* the episode
ended, but DummyVecEnv auto-resets on done, so it was reading the next
episode's freshly-zeroed state. Every attempt reported 0/4 checkpoints and
lap_completed False, `best_trajectory` stayed None, and the file shipped
anyway with the reset state as its last point.
"""
from __future__ import annotations

import numpy as np
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv

from conftest import TRACK_PATH, VEHICLE_CONF
from controllers.pure_pursuit import PurePursuitController


def _vec_env():
    from rl.optimal_line_finder import make_env_factory

    with VEHICLE_CONF.open(encoding="utf-8") as handle:
        vehicle_cfg = yaml.safe_load(handle)

    return DummyVecEnv([
        make_env_factory(
            track_path=str(TRACK_PATH),
            vehicle_cfg=vehicle_cfg,
            enable_telemetry=False,
            seed=0,
        )
    ])


def test_episode_results_survive_the_auto_reset():
    """On the terminal step, results must be read from info, not from the env.

    Drives a full lap with the reference controller through a DummyVecEnv and
    checks that the terminal info reports the lap, while the env object itself
    has already been reset -- which is exactly why reading it was wrong.
    """
    vec_env = _vec_env()
    racing_env = vec_env.envs[0].unwrapped
    controller = PurePursuitController(racing_env.track, racing_env.spec)

    vec_env.reset()
    terminal_info = None
    for _ in range(racing_env.cfg.max_steps):
        action = controller.act(racing_env.state)
        _, _, dones, infos = vec_env.step(np.array([action]))
        if dones[0]:
            terminal_info = infos[0]
            break

    assert terminal_info is not None, "the reference lap never terminated"

    # The lap did happen, and info is the only place that still says so.
    assert terminal_info["lap_completed"] is True
    assert terminal_info["checkpoints_hit"] == racing_env.cfg.num_checkpoints

    # The env has been auto-reset: reading it here is what produced 0/4.
    assert racing_env.checkpoints_hit == set()
    assert racing_env.lap_completed is False

    vec_env.close()


def test_recorded_trajectory_excludes_the_reset_state():
    """The last recorded point must be on the driven path, not the reset pose.

    `data/optimal_traj.npy` used to end with [120.0, 0.0, 5.0] -- the reset
    position and reset speed -- because the state was sampled after the
    auto-reset had already happened.
    """
    vec_env = _vec_env()
    racing_env = vec_env.envs[0].unwrapped
    controller = PurePursuitController(racing_env.track, racing_env.spec)

    reset_position = np.array(racing_env.track.centerline[0])

    vec_env.reset()
    xs, ys, vs = [], [], []
    for _ in range(racing_env.cfg.max_steps):
        action = controller.act(racing_env.state)
        _, _, dones, _ = vec_env.step(np.array([action]))
        if dones[0]:
            break
        s = racing_env.state
        xs.append(s.x)
        ys.append(s.y)
        vs.append(float(np.hypot(s.vx, s.vy)))

    assert len(xs) > 100, f"only {len(xs)} points recorded"

    # A completed lap returns near the start line, so proximity alone is not
    # the tell -- the reset SPEED is. reset() always starts at 5.0 m/s, and a
    # lapping car crosses the line far faster than that.
    assert vs[-1] > 10.0, (
        f"final recorded speed {vs[-1]:.2f} m/s looks like the reset state"
    )
    assert not (
        np.allclose([xs[-1], ys[-1]], reset_position, atol=1e-6)
        and abs(vs[-1] - 5.0) < 1e-6
    ), "final point is the reset state"

    vec_env.close()
