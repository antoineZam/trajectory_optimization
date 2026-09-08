"""Invariants of the straight/corner layout builder.

The library describes circuits as segment sequences and trusts the builder to
turn them into a closed centerline without quietly reshaping them. These tests
pin down both halves of that promise: the geometry it produces, and the
corrections it refuses to make.
"""
from __future__ import annotations

import numpy as np
import pytest

from utils.track_builder import Corner, Layout, Straight, build_centerline


def _rounded_rectangle(radius: float = 50.0) -> Layout:
    """A layout that already closes exactly: 4 right-angle corners."""
    return Layout(
        name="rectangle",
        width=12.0,
        segments=(
            Straight(200.0),
            Corner(radius, -90.0),
            Straight(100.0),
            Corner(radius, -90.0),
            Straight(200.0),
            Corner(radius, -90.0),
            Straight(100.0),
            Corner(radius, -90.0),
        ),
    )


def test_a_closed_sequence_needs_no_correction():
    """Nothing is scaled or tapered when the description already closes."""
    layout = _rounded_rectangle()
    centerline, report = build_centerline(layout)

    assert report.turn_scale == pytest.approx(1.0)
    assert report.gap == pytest.approx(0.0, abs=1e-6)
    assert report.lap_length == pytest.approx(layout.length)
    # The closing point is dropped: the centerline must not repeat its start.
    assert not np.allclose(centerline[0], centerline[-1])


def test_corner_radius_is_exact_not_chorded():
    """A corner sampled by chords would come out tighter than requested.

    A single 360 deg corner is a circle, so its lap length is 2*pi*r and its
    bounding box is exactly one diameter across.
    """
    radius = 80.0
    layout = Layout(name="circle", width=10.0, segments=(Corner(radius, 360.0),))
    centerline, report = build_centerline(layout)

    assert report.lap_length == pytest.approx(2 * np.pi * radius, rel=1e-6)
    extent = centerline.max(axis=0) - centerline.min(axis=0)
    assert extent == pytest.approx([2 * radius, 2 * radius], rel=1e-3)


def test_sample_spacing_bounds_the_control_point_gap():
    """Control points must be dense enough to pin the shape down."""
    centerline, _ = build_centerline(_rounded_rectangle(), sample_spacing=4.0)
    steps = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    assert steps.max() <= 4.0 + 1e-9


def test_heading_error_is_spread_over_every_corner():
    """A sequence that turns 350 deg is closed by opening the corners equally."""
    layout = Layout(
        name="short-of-a-lap",
        width=12.0,
        segments=(
            Straight(200.0),
            Corner(50.0, -85.0),
            Straight(100.0),
            Corner(50.0, -90.0),
            Straight(200.0),
            Corner(50.0, -85.0),
            Straight(100.0),
            Corner(50.0, -90.0),
        ),
    )
    centerline, report = build_centerline(layout)

    assert layout.total_turn == pytest.approx(-350.0)
    assert report.turn_scale == pytest.approx(360.0 / 350.0)
    # Closed to numerical precision, whatever the raw sequence did.
    assert np.linalg.norm(centerline[0] - centerline[-1]) < 2.0 * 4.0


def test_a_sequence_that_is_not_a_lap_is_rejected():
    """Rescaling a 180 deg sequence into a lap would invent a circuit."""
    layout = Layout(
        name="half-a-lap",
        width=12.0,
        segments=(Straight(200.0), Corner(50.0, -90.0), Straight(200.0), Corner(50.0, -90.0)),
    )
    with pytest.raises(ValueError, match="needs a"):
        build_centerline(layout)


def test_a_sequence_that_misses_its_own_start_is_rejected():
    """The taper is a correction, not a way to close an arbitrary shape."""
    layout = Layout(
        name="wrong-straights",
        width=12.0,
        segments=(
            Straight(600.0),  # 400 m longer than the straight it faces
            Corner(50.0, -90.0),
            Straight(100.0),
            Corner(50.0, -90.0),
            Straight(200.0),
            Corner(50.0, -90.0),
            Straight(100.0),
            Corner(50.0, -90.0),
        ),
    )
    with pytest.raises(ValueError, match="misses its start"):
        build_centerline(layout)


@pytest.mark.parametrize(
    ("segment", "message"),
    [
        (lambda: Straight(0.0), "positive"),
        (lambda: Straight(-10.0), "positive"),
        (lambda: Corner(0.0, 90.0), "positive"),
        (lambda: Corner(50.0, 0.0), "non-zero"),
    ],
)
def test_degenerate_segments_are_rejected_at_construction(segment, message):
    with pytest.raises(ValueError, match=message):
        segment()


def test_layout_reports_its_own_geometry():
    layout = _rounded_rectangle(radius=40.0)
    assert layout.num_corners == 4
    assert layout.min_radius == pytest.approx(40.0)
    assert layout.total_turn == pytest.approx(-360.0)
    assert layout.length == pytest.approx(600.0 + 2 * np.pi * 40.0)
