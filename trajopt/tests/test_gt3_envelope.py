"""The default car against the performance of current GT3 cars.

The default vehicle is calibrated on FIA GT3 data. These tests drive it
through the manoeuvres for which public figures exist and check that it lands
inside them, so a physics change that makes the car unrealistically fast or
slow fails here rather than in a lap time weeks later.

- Top speed: LMGT3 speed traps at Le Mans 2024-2026, 284-304 km/h.
- Braking: Brembo, IMSA GTD 2025, 2.0 g (Daytona T1) to 2.5 g (Road Atlanta).
- Acceleration: the road-going 992 GT3 does 0-200 km/h in 10.2 s; the race
  car is no heavier and runs on slicks.
- Cornering: about 1.6 g in slow corners and 2 g or more in fast ones. These
  are estimates, as no GT3 telemetry is public.
"""
from __future__ import annotations

import numpy as np
import pytest

from physics.physics_engine import VehicleSpec, VehicleState, step_dynamics

GRAVITY = 9.81
DT = 0.05
KMH = 1.0 / 3.6


def _state(speed: float, gear: int = 1) -> VehicleState:
    return VehicleState(x=0.0, y=0.0, yaw=0.0, vx=speed, vy=0.0, yaw_rate=0.0,
                        gear=gear, rpm=1000.0)


def test_top_speed_matches_the_le_mans_speed_traps(vehicle_spec: VehicleSpec):
    """Flat out on an endless straight, the car settles at 280-300 km/h."""
    state = _state(150.0 * KMH, gear=4)
    for _ in range(int(90.0 / DT)):
        state = step_dynamics(vehicle_spec, state, DT, 1.0, 0.0, 0.0)

    assert 280.0 <= state.vx / KMH <= 300.0, f"top speed {state.vx / KMH:.1f} km/h"


def test_standing_start_beats_the_road_car(vehicle_spec: VehicleSpec):
    """0-200 km/h under the road 992 GT3's 10.2 s, but not absurdly so."""
    state, elapsed = _state(0.5), 0.0
    while state.vx < 200.0 * KMH and elapsed < 30.0:
        state = step_dynamics(vehicle_spec, state, DT, 1.0, 0.0, 0.0)
        elapsed += DT

    assert 8.0 <= elapsed <= 10.2, f"0-200 km/h in {elapsed:.2f} s"


def test_braking_at_speed_matches_the_brembo_data(vehicle_spec: VehicleSpec):
    """Full brakes at 290 km/h peak at 2.0-2.5 g, downforce and drag included."""
    state, peak = _state(290.0 * KMH, gear=6), 0.0
    for _ in range(int(1.0 / DT)):
        after = step_dynamics(vehicle_spec, state, DT, 0.0, 1.0, 0.0)
        peak = max(peak, (state.vx - after.vx) / DT)
        state = after

    assert 2.0 <= peak / GRAVITY <= 2.5, f"peak deceleration {peak / GRAVITY:.2f} g"


def _sustained_lateral_g(spec: VehicleSpec, speed: float) -> float:
    """Peak lateral acceleration while the steering ramps to full lock in 12 s.

    Speed is held with the pedals, and the path's turn rate is averaged over
    half a second, so this is what the tyres sustain rather than a spike.
    """
    state, course, lateral = _state(speed, gear=3), None, []
    for k in range(int(12.0 / DT)):
        pedal = float(np.clip(0.5 * (speed - np.hypot(state.vx, state.vy)), -1.0, 1.0))
        steer = spec.max_steering_angle * k * DT / 12.0
        state = step_dynamics(spec, state, DT, max(pedal, 0.0), max(-pedal, 0.0), steer)

        heading = state.yaw + np.arctan2(state.vy, state.vx)
        if course is not None:
            turn = np.arctan2(np.sin(heading - course), np.cos(heading - course))
            lateral.append(np.hypot(state.vx, state.vy) * turn / DT)
        course = heading

    window = int(0.5 / DT)
    return float(np.convolve(lateral, np.ones(window) / window, mode="valid").max()) / GRAVITY


@pytest.mark.parametrize(
    ("speed_kmh", "low_g", "high_g"),
    [(90.0, 1.5, 1.8), (230.0, 1.9, 2.5)],
    ids=["slow-corner", "fast-corner"],
)
def test_cornering_grip_matches_gt3_estimates(
    vehicle_spec: VehicleSpec, speed_kmh: float, low_g: float, high_g: float
):
    """About 1.6 g in a slow corner, and 2 g or more once downforce builds."""
    lateral = _sustained_lateral_g(vehicle_spec, speed_kmh * KMH)

    assert low_g <= lateral <= high_g, f"{lateral:.2f} g at {speed_kmh:.0f} km/h"
