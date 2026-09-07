"""Vehicle dynamics invariants.

The yaw-stability tests are regressions for a sign error that made the vehicle
uncontrollable by any policy. The remaining tests encode physical properties the
model does not have yet; they are marked `xfail(strict=True)` so that fixing
them fails the suite until the marker is removed (Phase 3).
"""
from __future__ import annotations

import numpy as np
import pytest

from physics.physics_engine import (
    VehicleSpec,
    VehicleState,
    aero_forces,
    get_max_steering_angle,
    step_dynamics,
    tire_mu,
)

GRAVITY = 9.81
DT = 0.05


def _state(vx: float = 25.0, **kwargs) -> VehicleState:
    defaults = dict(x=0.0, y=0.0, yaw=0.0, vx=vx, vy=0.0, yaw_rate=0.0, gear=4, rpm=4000.0)
    defaults.update(kwargs)
    return VehicleState(**defaults)


# =============================================================================
# Yaw stability -- regressions for the inverted slip-angle sign
# =============================================================================


@pytest.mark.parametrize("speed", [10.0, 25.0, 40.0])
def test_yaw_mode_is_damped_not_divergent(vehicle_spec: VehicleSpec, speed: float):
    """With no steering input, a yaw perturbation must decay.

    The slip angles were previously defined with the opposite sign on both
    axles, which flips the sign of the yaw damping term while leaving the
    kinematic coupling intact. The yaw mode became exponentially divergent: a
    0.01 rad/s perturbation doubled every ~73 ms and saturated the +/-20 rad/s
    safety clamp within a few steps.
    """
    state = _state(vx=speed, yaw_rate=0.05)
    initial = abs(state.yaw_rate)

    for _ in range(20):
        state = step_dynamics(vehicle_spec, state, DT, 0.0, 0.0, 0.0)

    assert abs(state.yaw_rate) < 0.1 * initial, (
        f"yaw rate grew from {initial} to {state.yaw_rate} at {speed} m/s"
    )


@pytest.mark.parametrize("steer_deg", [1.0, 2.0, 4.0])
def test_steady_state_yaw_rate_matches_kinematic_model(
    vehicle_spec: VehicleSpec, steer_deg: float
):
    """At modest lateral load, yaw rate must approach v/L * tan(delta).

    Checks both the sign (steering left must yaw left) and the magnitude. A
    real vehicle understeers slightly, so the ratio is allowed below 1 but not
    above it.
    """
    steer = np.radians(steer_deg)
    state = _state(vx=25.0)
    for _ in range(60):
        state = step_dynamics(vehicle_spec, state, DT, 0.25, 0.0, steer)

    speed = float(np.hypot(state.vx, state.vy))
    kinematic = speed / vehicle_spec.wheelbase * np.tan(steer)

    assert state.yaw_rate > 0.0, "positive steering produced a negative yaw rate"
    assert 0.8 <= state.yaw_rate / kinematic <= 1.05, (
        f"yaw rate {state.yaw_rate:.4f} vs kinematic reference {kinematic:.4f}"
    )


def test_lateral_acceleration_stays_within_tire_grip(vehicle_spec: VehicleSpec):
    """Full steering lock must not exceed the tire's lateral capability.

    The quantity the tires actually have to supply is the turn rate of the
    velocity vector in the global frame (path lateral acceleration), not
    `speed * yaw_rate` -- the latter also counts the body's own rotation while
    the car is sliding, and reads high for legitimate slides.

    Previously the vehicle spun up into the +/-20 rad/s yaw clamp; here the
    steady-state value should sit around mu0.
    """
    state = _state(vx=25.0)
    peak = 0.0
    previous_course = None

    for step in range(60):
        state = step_dynamics(
            vehicle_spec, state, DT, 1.0, 0.0, vehicle_spec.max_steering_angle
        )
        # Velocity in the global frame
        global_vx = state.vx * np.cos(state.yaw) - state.vy * np.sin(state.yaw)
        global_vy = state.vx * np.sin(state.yaw) + state.vy * np.cos(state.yaw)
        course = np.arctan2(global_vy, global_vx)
        speed = float(np.hypot(global_vx, global_vy))

        if previous_course is not None and step > 4:  # skip the step-input transient
            delta = np.arctan2(
                np.sin(course - previous_course), np.cos(course - previous_course)
            )
            peak = max(peak, abs(speed * delta / DT) / GRAVITY)
        previous_course = course

    limit = vehicle_spec.mu0 * 1.3  # headroom for aero downforce and Euler error
    assert peak <= limit, f"peak path lateral acceleration {peak:.2f} g exceeds {limit:.2f} g"


def test_safety_clamps_are_not_load_bearing(vehicle_spec: VehicleSpec):
    """Tire physics, not the +/-20 rad/s clamp, must bound the yaw rate."""
    state = _state(vx=25.0)
    for _ in range(60):
        state = step_dynamics(
            vehicle_spec, state, DT, 1.0, 0.0, vehicle_spec.max_steering_angle
        )
    assert abs(state.yaw_rate) < 19.0, "yaw rate is being held by the safety clamp"
    assert abs(state.vy) < 49.0, "lateral velocity is being held by the safety clamp"


# =============================================================================
# Known gaps -- Phase 3
# =============================================================================


def test_braking_stays_within_the_available_grip(vehicle_spec: VehicleSpec):
    """Longitudinal deceleration cannot exceed mu * Fz / m.

    Fz includes aerodynamic downforce, which is why braking legitimately
    exceeds mu0 at speed: at 60 m/s the downforce is 93% of the car's weight
    and 3.09 g is genuinely available. The test therefore brakes from a low
    speed, where aero is a few percent, and compares against the real budget.
    """
    initial_speed = 15.0
    state = _state(vx=initial_speed)

    peak_decel = 0.0
    previous_speed = initial_speed
    for _ in range(60):
        state = step_dynamics(vehicle_spec, state, DT, 0.0, 1.0, 0.0)
        speed = float(np.hypot(state.vx, state.vy))
        peak_decel = max(peak_decel, (previous_speed - speed) / DT)
        previous_speed = speed

    _, downforce = aero_forces(vehicle_spec, initial_speed)
    normal_load = vehicle_spec.mass * GRAVITY + downforce
    available_g = tire_mu(vehicle_spec, normal_load / 2.0) * normal_load / (
        vehicle_spec.mass * GRAVITY
    )

    assert peak_decel / GRAVITY <= available_g * 1.05, (
        f"peak {peak_decel / GRAVITY:.2f} g exceeds the {available_g:.2f} g available"
    )


def test_steering_eats_into_the_braking_budget(vehicle_spec: VehicleSpec):
    """The friction circle: Fx and Fy share one grip budget per axle.

    This is the property every feature of a racing line rests on -- braking
    in a straight line, the apex, progressive reacceleration. Previously Fx
    and Fy were computed independently, so the car could brake at its limit
    and corner at its limit simultaneously, and the optimal policy for this
    simulator was "full throttle everywhere while steering".
    """
    def one_step_decel(steer_deg: float) -> float:
        state = _state(vx=20.0)
        after = step_dynamics(
            vehicle_spec, state, DT, 0.0, 1.0, np.radians(steer_deg)
        )
        speed = float(np.hypot(after.vx, after.vy))
        return (20.0 - speed) / DT

    decels = [one_step_decel(d) for d in (0.0, 2.0, 5.0, 10.0)]

    # More steering must leave less grip for braking, monotonically.
    assert all(a > b for a, b in zip(decels, decels[1:])), (
        f"braking did not decrease with steering angle: {decels}"
    )
    # And the effect must be material, not numerical noise.
    assert decels[0] - decels[-1] > 0.2, (
        f"grip sharing is negligible: {decels[0]:.3f} -> {decels[-1]:.3f} m/s^2"
    )


def test_braking_never_reverses_the_vehicle(vehicle_spec: VehicleSpec):
    """Braking must decelerate towards zero, not through it."""
    state = _state(vx=8.0)
    for _ in range(200):
        state = step_dynamics(vehicle_spec, state, DT, 0.0, 1.0, 0.0)
        assert state.vx >= -0.05, f"braking drove vx to {state.vx:.3f} m/s"


@pytest.mark.xfail(
    strict=True,
    reason="Phase 3: the `if speed > 5.0` branch makes the limiter jump 31.25 "
    "-> 19.46 deg, and reset() starts the vehicle at exactly 5.0 m/s.",
)
def test_steering_limiter_is_continuous_and_monotonic(vehicle_spec: VehicleSpec):
    """Max steering must decrease smoothly with speed.

    A discontinuity at 5.0 m/s is especially harmful because `reset()`
    initialises the vehicle at exactly that speed: the same command yields
    dynamics differing by 38% on either side, at every episode start.
    """
    speeds = np.linspace(0.0, 60.0, 601)
    angles = np.array([get_max_steering_angle(vehicle_spec, v) for v in speeds])

    steps = np.diff(angles)
    assert np.all(steps <= 1e-9), "steering limiter is not monotonically decreasing"
    # No jump larger than 1 degree between adjacent 0.1 m/s samples
    assert np.max(np.abs(steps)) < np.radians(1.0), "steering limiter is discontinuous"
