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
from envs.rl_environment import RacingEnv


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
    """A full lap on the real 12 m track, without leaving it.

    Asserts on cumulative unwrapped arc length rather than `env.lap_completed`,
    because the environment's own lap detection is unreachable until B2 is
    fixed -- see `test_environment.py`.
    """
    controller = PurePursuitController(env.track, env.spec)
    result = drive(env, controller, laps=1.0)

    assert result["laps_completed"] >= 1.0, (
        f"only completed {result['laps_completed']:.3f} laps "
        f"({result['distance_m']:.0f} m) in {result['steps']} steps; "
        f"terminated={result['terminated']}"
    )
    assert not result["terminated"], "the reference controller left the track"
    assert result["min_wheels_inside"] == 4, (
        f"only {result['min_wheels_inside']} wheels stayed inside the track"
    )


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


def test_multiple_laps_stay_stable(env: RacingEnv):
    """Three consecutive laps, to catch drift that a single lap would hide.

    In particular this covers the start/finish seam: the boundary offsets and
    the track polygon both used to be wrong there, flagging a centred car as
    off-track on every crossing.
    """
    controller = PurePursuitController(env.track, env.spec)
    result = drive(env, controller, laps=3.0)

    assert result["laps_completed"] >= 3.0, (
        f"only {result['laps_completed']:.2f} laps completed"
    )
    assert result["min_wheels_inside"] == 4
    assert not result["terminated"]
