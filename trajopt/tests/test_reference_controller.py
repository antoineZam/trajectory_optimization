"""Integration test: an analytic controller must be able to lap the track.

This is the project's decisive test. If a pure-pursuit controller with a
curvature-based speed profile cannot complete a lap on the real 12 m track,
the environment or the physics is broken and no RL agent will do better --
so a failure here should be read as "fix the simulator", not "tune the agent".

It also produces the reference lap time that learned policies are compared to.
"""
from __future__ import annotations

import numpy as np

from controllers.pure_pursuit import (
    PurePursuitController,
    compute_speed_profile,
    drive,
)
from envs.rl_environment import RacingEnv, RLConfig


def test_speed_profile_respects_cornering_and_braking_limits(track, vehicle_spec):
    """The offline profile must be finite, bounded and braking-feasible."""
    profile = compute_speed_profile(track, vehicle_spec)

    assert len(profile) == len(track.interpolated_centerline)
    assert np.all(np.isfinite(profile))
    assert np.all(profile > 0.0)

    # Slowest point must be the tightest corner (max curvature)
    assert profile.min() < profile.max(), "profile is flat: curvature is being ignored"

    # Feasibility: v[i]^2 <= v[i+1]^2 + 2*a*ds around the closed loop
    nxt = np.roll(profile, -1)
    limit = np.sqrt(nxt ** 2 + 2.0 * 6.0 * track.arc_step)
    assert np.all(profile <= limit + 1e-6), "profile demands more braking than allowed"


def test_reference_controller_completes_a_lap(env: RacingEnv):
    """A full lap on the real 12 m track, without leaving it."""
    controller = PurePursuitController(env.track, env.spec)
    result = drive(env, controller, laps=1.0)

    assert result["laps_completed"] >= 1.0, (
        f"only completed {result['laps_completed']:.3f} laps "
        f"({result['distance_m']:.0f} m) in {result['steps']} steps; "
        f"left_track={result['left_track']}"
    )
    assert not result["left_track"], "the reference controller left the track"
    assert result["min_wheels_inside"] == 4, (
        f"only {result['min_wheels_inside']} wheels stayed inside the track"
    )
    # The environment must agree that the lap happened, not just the odometer.
    assert result["lap_completed"], "env.lap_completed did not register the lap"
    assert result["terminated"], "a finished lap must terminate the episode"


def test_reference_lap_time_is_physically_plausible(env: RacingEnv):
    """The lap time gives learned policies something to be measured against.

    The bounds are deliberately loose: they catch a simulator that has become
    absurdly fast or slow, not small controller regressions.
    """
    controller = PurePursuitController(env.track, env.spec)
    result = drive(env, controller, laps=1.0)

    lap_time = result["lap_time_s"] / result["laps_completed"]
    lap_length = env.track.lap_length

    # 607 m at a mean 15-40 m/s
    assert lap_length / 40.0 < lap_time < lap_length / 15.0, (
        f"implausible reference lap time: {lap_time:.1f} s for {lap_length:.0f} m"
    )
    assert result["mean_speed"] > 15.0, f"mean speed only {result['mean_speed']:.1f} m/s"


def test_reference_lap_return_matches_the_documented_budget(env: RacingEnv):
    """A completed lap must return an O(100) reward, milestones ~1/3 of it.

    The reward scale drifted from its docstring by 5x historically (documented
    ~0.65 per step, actually ~3.1), and nothing caught it. This pins the
    budget so a future reward change has to be deliberate.

    Bounds are loose on purpose: they catch an order-of-magnitude slip, not
    a small retune.
    """
    controller = PurePursuitController(env.track, env.spec)
    env.reset()

    total = 0.0
    steps = 0
    for _ in range(env.cfg.max_steps):
        _, reward, terminated, truncated, _ = env.step(controller.act(env.state))
        total += reward
        steps += 1
        if terminated or truncated:
            break

    assert env.lap_completed, "the reference lap did not complete"

    milestones = (
        env.cfg.num_checkpoints * env.cfg.checkpoint_bonus + env.cfg.lap_completion_bonus
    )
    per_step = (total - milestones) / steps

    assert 50.0 < total < 200.0, f"lap return {total:.1f} is outside the O(100) budget"
    assert 0.2 < milestones / total < 0.5, (
        f"milestones are {milestones / total:.0%} of the return, expected ~1/3"
    )
    # Dense reward must stay O(0.1) per step: this is what keeps the return
    # standard deviation low enough that the value loss does not swamp the
    # policy gradient.
    assert 0.05 < per_step < 0.5, f"dense reward is {per_step:.3f} per step"


def test_repeated_lap_episodes_stay_stable(env: RacingEnv):
    """Three consecutive lap episodes, to catch state that leaks across resets.

    A finished lap terminates the episode, so covering several laps means
    several episodes. This exercises the progress bookkeeping being cleared on
    reset, and the start/finish seam, which both the boundary offsets and the
    track polygon used to get wrong -- flagging a centred car as off-track on
    every crossing.
    """
    controller = PurePursuitController(env.track, env.spec)

    lap_times = []
    for lap in range(3):
        result = drive(env, controller, laps=1.0)
        assert result["lap_completed"], f"lap {lap + 1} did not complete"
        assert not result["left_track"], f"left the track on lap {lap + 1}"
        assert result["min_wheels_inside"] == 4
        lap_times.append(result["lap_time_s"])

    # Deterministic env and controller: the laps must be identical, which is
    # what proves nothing leaked from the previous episode.
    assert len(set(lap_times)) == 1, f"lap times drifted across episodes: {lap_times}"


def test_reference_controller_recovers_from_random_starts(track, vehicle_spec):
    """A lap must be completable from anywhere on the track, at any speed.

    This is what validates that initial-state randomization produces legal,
    recoverable states rather than episodes that are lost at step 0 -- a start
    at 30 m/s in the tightest corner would demand 2.25 g against the 1.6 g
    available. Measured: 60/60 random starts complete a lap.
    """
    env = RacingEnv(
        track=track,
        veh_spec=vehicle_spec,
        cfg=RLConfig(max_steps=3_000),
        enable_telemetry=False,
    )
    controller = PurePursuitController(track, vehicle_spec)

    failures = []
    attempts = 25
    for seed in range(attempts):
        env.reset(seed=seed)
        start_speed = env.state.vx
        for _ in range(env.cfg.max_steps):
            _, _, terminated, truncated, _ = env.step(controller.act(env.state))
            if terminated or truncated:
                break
        if not env.lap_completed:
            failures.append((seed, round(start_speed, 1)))

    assert not failures, f"{len(failures)}/{attempts} random starts failed: {failures}"
