"""Environment contract tests.

Four tests here are `xfail(strict=True)` and encode the Phase 1 blockers: the
lookahead horizon, checkpoint reachability, progress-reward conservativeness and
lap termination semantics. Removing a marker is part of landing each fix.
"""
from __future__ import annotations

import numpy as np
import pytest

from controllers.pure_pursuit import PurePursuitController
from envs.rl_environment import RacingEnv, RLConfig


def _synthetic_track_state(env: RacingEnv, progress: float) -> dict:
    """A track state at a given progress, with the vehicle dead centre.

    Only the fields `_compute_reward` reads are populated, so the progress term
    can be isolated from the speed and centring terms.
    """
    centerline = env.track.interpolated_centerline
    idx = int(progress * len(centerline)) % len(centerline)
    return {
        "closest_idx": idx,
        "closest_point": centerline[idx],
        "center_dist": 0.0,
        "track_progress": progress,
        "tangent": np.array([1.0, 0.0]),
        "normal": np.array([0.0, 1.0]),
        "track_heading": 0.0,
        "signed_offset": 0.0,
    }


# =============================================================================
# Observation and configuration plumbing
# =============================================================================


def test_observation_matches_declared_space(env: RacingEnv):
    """Observations must stay inside the declared Box at reset and while driving."""
    obs, _ = env.reset()
    assert obs.shape == (RacingEnv.OBS_DIM,)
    assert env.observation_space.contains(obs), "reset observation is out of bounds"

    controller = PurePursuitController(env.track, env.spec)
    for _ in range(200):
        obs, _, terminated, truncated, _ = env.step(controller.act(env.state))
        assert np.all(np.isfinite(obs)), "observation contains NaN/Inf"
        assert env.observation_space.contains(obs), f"observation out of bounds: {obs}"
        if terminated or truncated:
            break


def test_rlconfig_is_reachable_from_a_config_dict():
    """Reward weights must be settable without editing source.

    Every parameter that matters -- dt, max_steps, all reward weights -- lived
    only in the `RLConfig` dataclass, so experiments meant code commits. The
    Hydra `env` group now maps onto it.
    """
    cfg = RLConfig.from_dict(
        {"max_steps": 800, "progress_reward_scale": 0.25, "unknown_key": 1}
    )
    assert cfg.max_steps == 800
    assert cfg.progress_reward_scale == 0.25
    assert cfg.dt == RLConfig().dt  # untouched fields keep their defaults


def test_same_seed_gives_identical_rollouts(track, vehicle_spec):
    """Reproducibility: identical seeds must produce identical trajectories."""
    def rollout(seed: int) -> np.ndarray:
        e = RacingEnv(
            track=track, veh_spec=vehicle_spec, cfg=RLConfig(max_steps=500),
            enable_telemetry=False,
        )
        e.reset(seed=seed)
        e.action_space.seed(seed)
        positions = []
        for _ in range(60):
            e.step(e.action_space.sample())
            positions.append((e.state.x, e.state.y))
        return np.array(positions)

    np.testing.assert_allclose(rollout(7), rollout(7))


# =============================================================================
# Phase 1 blockers
# =============================================================================


def test_lookahead_points_are_at_the_requested_distances(env: RacingEnv):
    """The four lookahead points must sit 20/50/80/100 m ahead.

    With a zero horizon the agent has no information about the geometry ahead,
    which makes learning a racing line impossible in principle -- and 8 of the
    21 observation dimensions become duplicated noise.
    """
    env.reset()
    track_state = env._compute_track_state(np.array([env.state.x, env.state.y]))
    lookahead = env._compute_lookahead_points(track_state)

    distances = lookahead[:, 0] * env.cfg.max_lookahead_distance
    expected = np.array(RacingEnv.LOOKAHEAD_SPACING_M)

    assert len(set(np.round(distances, 3))) == 4, (
        f"lookahead points are not distinct: {distances}"
    )
    # Straight-line distance to a point on a curve is shorter than the arc,
    # so allow a generous chord tolerance.
    np.testing.assert_allclose(distances, expected, rtol=0.25)


def test_a_full_lap_awards_all_checkpoints_and_completes(env: RacingEnv):
    """Driving one full lap must award 4/4 checkpoints and set lap_completed.

    The reference controller demonstrably drives three clean laps, yet the
    environment reports checkpoints {1, 2, 3} and lap_completed False.
    """
    controller = PurePursuitController(env.track, env.spec)
    env.reset()

    previous = env.track.get_track_progress(np.array([env.state.x, env.state.y]))
    cumulative = 0.0
    for _ in range(3000):
        env.step(controller.act(env.state))
        current = env.track.get_track_progress(np.array([env.state.x, env.state.y]))
        delta = current - previous
        if delta > 0.5:
            delta -= 1.0
        elif delta < -0.5:
            delta += 1.0
        cumulative += delta
        previous = current
        if cumulative >= 1.02:  # comfortably past the line
            break

    assert cumulative >= 1.0, "the reference controller failed to complete a lap"
    assert len(env.checkpoints_hit) == env.cfg.num_checkpoints, (
        f"checkpoints hit: {sorted(env.checkpoints_hit)}"
    )
    assert env.lap_completed


def test_progress_reward_is_conservative(env: RacingEnv):
    """Moving forward then back must net zero progress reward.

    Clipping the negative half of a difference signal means a round trip is
    paid for the outbound leg and not charged for the return.
    """
    env.reset()
    start, end = 0.30, 0.32

    # Baseline: same progress twice, so the progress delta is zero and only the
    # speed/centring terms contribute.
    baseline = env._compute_reward(_synthetic_track_state(env, start), False, 0.0)
    forward = env._compute_reward(_synthetic_track_state(env, end), False, end - start)
    backward = env._compute_reward(_synthetic_track_state(env, start), False, start - end)

    net = (forward - baseline) + (backward - baseline)
    assert net == pytest.approx(0.0, abs=1e-6), (
        f"zero net displacement earned {net:.1f} reward"
    )


def test_lap_completion_is_terminal_not_truncated(env: RacingEnv):
    """`lap_completed` must map to `terminated`, not `truncated`."""
    env.reset()
    env.lap_completed = True
    terminated, truncated = env._check_termination(
        env._compute_track_state(np.array([env.state.x, env.state.y]))
    )
    assert terminated, "a completed lap must terminate the episode"
    assert not truncated, "a completed lap is not a time-limit truncation"


def test_reset_randomizes_the_initial_state(track, vehicle_spec):
    """Resets must spread over the whole track, not repeat one pose.

    reset() previously always placed the vehicle at centerline[0] with vx=5.0
    and no noise, so four parallel workers produced identical rollouts: an
    8192-step batch had the diversity of 2048 samples and the value function
    was only ever trained inside one narrow tube.
    """
    env = RacingEnv(
        track=track, veh_spec=vehicle_spec, cfg=RLConfig(max_steps=100),
        enable_telemetry=False,
    )

    starts = []
    for seed in range(40):
        env.reset(seed=seed)
        starts.append((env.state.x, env.state.y, env.state.yaw, env.state.vx))

    assert len({tuple(np.round(s, 6)) for s in starts}) == len(starts), (
        "resets are repeating the same pose"
    )

    # Start points must cover the lap, not cluster near the line.
    progresses = np.array([
        track.get_track_progress(np.array(s[:2])) for s in starts
    ])
    assert progresses.min() < 0.25 and progresses.max() > 0.75, (
        f"start points span only {progresses.min():.2f}..{progresses.max():.2f} of the lap"
    )

    # Speeds must span the configured range, not sit at min_speed.
    speeds = np.array([s[3] for s in starts])
    assert speeds.max() - speeds.min() > 10.0, f"speed spread is only {speeds.ptp():.1f} m/s"

    # Every start must be a legal state: on track, and inside the speed bounds.
    assert speeds.min() >= env.cfg.min_speed - 1e-9
    assert speeds.max() <= env.cfg.reset_max_speed + 1e-9


def test_randomization_can_be_switched_off(track, vehicle_spec):
    """The export needs a deterministic lap from the start line."""
    env = RacingEnv(
        track=track, veh_spec=vehicle_spec,
        cfg=RLConfig(randomize_reset=False),
        enable_telemetry=False,
    )

    def start_state(seed: int) -> tuple:
        env.reset(seed=seed)
        return (env.state.x, env.state.y, env.state.yaw, env.state.vx)

    assert start_state(1) == start_state(2) == start_state(99)
    env.reset(seed=0)
    assert env.state.vx == env.cfg.min_speed
    np.testing.assert_allclose(
        [env.state.x, env.state.y], track.interpolated_centerline[0], atol=1e-9
    )


def test_training_and_evaluation_see_the_same_track(track, vehicle_spec):
    """No environment may present a different track from the real one.

    This was the fifth blocking defect: training spent its entire budget on a
    curriculum stage that multiplied the track width by 5 (12 m -> 60 m) and
    never terminated, while the export evaluated on the real 12 m track with
    strict termination. The trained policy had never seen the task it was
    scored on, and the exported "racing line" was an 18-step off-track
    excursion. Removing the curriculum is what closes it.
    """
    env = RacingEnv(track=track, veh_spec=vehicle_spec, enable_telemetry=False)

    assert env.track is track, "the env substituted a different track object"
    assert env.track.width == track.width

    # Termination must be the real rule, not a relaxed training variant.
    assert env.cfg.wheels_required_inside == 2
    env.reset(seed=0)
    terminated, _ = env._check_termination(
        env._compute_track_state(np.array([1e5, 1e5]))
    )
    assert terminated, "a vehicle far off track must terminate the episode"
