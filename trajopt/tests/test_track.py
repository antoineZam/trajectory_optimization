"""Track geometry invariants.

These are the assumptions the environment, the reward and the lookahead all
rest on. Two of them were silently violated before Phase 0.
"""
from __future__ import annotations

import numpy as np
import pytest

from utils.track import Track

# Width multipliers to check the geometry against. 1.0 is the real track; the
# wider ones are a robustness sweep -- the boundary offset and the polygon both
# used to break above ~3.5x, and any future variable-width track will land
# somewhere in this range.
WIDTH_MULTIPLIERS = [1.0, 1.2, 1.8, 2.5, 3.5, 5.0]


def test_arc_step_is_meters_not_point_count(track: Track):
    """`arc_step` must be a spatial step, and tile the lap exactly."""
    n = len(track.interpolated_centerline)
    assert track.arc_step * n == pytest.approx(track.lap_length)
    # Sanity: the sample oval is ~607 m over 2000 samples -> ~0.3 m per index.
    assert 0.1 < track.arc_step < 1.0
    assert track.arc_step != track.interpolation_resolution


def test_index_offset_for_distance_matches_arc_length(track: Track):
    """A requested distance in meters must map to that distance on the track."""
    centerline = track.interpolated_centerline
    start = 250  # away from the seam, so this tests the mapping, not wrapping

    for requested in (5.0, 20.0, 50.0, 100.0, 200.0):
        offset = track.index_offset_for_distance(requested)
        assert offset > 0, f"{requested} m collapsed to a zero index offset"

        segment = centerline[start:start + offset + 1]
        travelled = float(np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1)))
        # Within one arc step of the request
        assert travelled == pytest.approx(requested, abs=2.0 * track.arc_step)


@pytest.mark.parametrize("multiplier", WIDTH_MULTIPLIERS)
def test_track_polygon_is_valid_at_every_width(track: Track, multiplier: float):
    """Shapely predicates are undefined on invalid geometry.

    The polygon used to be built as a single self-intersecting ring, and the
    boundary offsets used a non-periodic tangent that broke at the seam.
    """
    widened = Track(
        name=f"x{multiplier}",
        centerline=track.centerline.copy(),
        width=track.width * multiplier,
        interpolation_resolution=track.interpolation_resolution,
    )
    widened.is_point_inside_track(widened.interpolated_centerline[0])  # force build

    polygon = widened._track_polygon
    assert polygon.is_valid, f"invalid track polygon at {multiplier}x"
    # A track is an annulus: an outer ring with the inner boundary as a hole.
    assert len(polygon.interiors) == 1


@pytest.mark.parametrize("multiplier", WIDTH_MULTIPLIERS)
def test_centerline_is_entirely_inside_the_track(track: Track, multiplier: float):
    """Regression: the two samples at the start/finish seam read as off-track.

    Every interpolated centerline point is by construction the middle of the
    track, so any of them testing as outside means the car is flagged off-track
    while driving dead centre -- which happened on every lap crossing.
    """
    widened = Track(
        name=f"x{multiplier}",
        centerline=track.centerline.copy(),
        width=track.width * multiplier,
        interpolation_resolution=track.interpolation_resolution,
    )
    centerline = widened.interpolated_centerline
    outside = [i for i, p in enumerate(centerline) if not widened.is_point_inside_track(p)]
    assert not outside, f"centerline indices reported off-track at {multiplier}x: {outside[:10]}"


def test_inside_test_agrees_with_lateral_offset(track: Track):
    """`is_point_inside_track` must agree with |e_lat| <= half_width.

    This is the invariant that lets the wheel check be replaced by a vectorised
    lateral-offset comparison later (Phase 5 performance work).
    """
    centerline = track.interpolated_centerline
    half_width = track.width / 2.0

    # Periodic tangents -> outward normals, sampled around the whole lap
    tangents = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    for idx in range(0, len(centerline), 97):
        for fraction, expected_inside in ((0.0, True), (0.8, True), (1.3, False)):
            for sign in (+1.0, -1.0):
                point = centerline[idx] + sign * fraction * half_width * normals[idx]
                assert track.is_point_inside_track(point) is expected_inside, (
                    f"index {idx}, offset {sign * fraction:.1f} x half_width"
                )
