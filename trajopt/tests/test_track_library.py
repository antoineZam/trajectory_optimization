"""Every library circuit must be a valid, drivable, recognisable track.

A layout is a hand-written reconstruction, so it can be wrong in ways an oval
never is: it can miss its own start line, fold back onto itself, or end up half
the length of the circuit it claims to reproduce. Each of those is a test here,
because each of them was hit while the library was being written.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from utils.track import Track, load_track_json
from utils.track_builder import Corner, Layout, Straight, build_centerline
from utils.track_library import (
    LAYOUTS,
    build_track,
    get_layout,
    layouts_with_tags,
    write_library,
)

# Coarse enough to keep the geometry checks quick, fine enough that a fold in
# the layout still shows up.
TEST_ARC_STEP = 2.0

LAYOUT_NAMES = list(LAYOUTS)


@pytest.fixture(scope="module")
def built_tracks() -> dict[str, Track]:
    """Every library track, built once for the whole module."""
    tracks = {}
    for name in LAYOUT_NAMES:
        track = build_track(name, arc_step=TEST_ARC_STEP)
        track.is_point_inside_track(track.interpolated_centerline[0])  # force build
        tracks[name] = track
    return tracks


@pytest.mark.parametrize("name", LAYOUT_NAMES)
def test_layouts_are_already_closed_loops(name: str):
    """The committed numbers must close on their own.

    The builder can absorb a small error, but a layout that leans on it is one
    whose corners have all been silently rescaled away from the real circuit.
    """
    layout = get_layout(name)
    _, report = build_centerline(layout)

    assert report.turn_scale == pytest.approx(1.0, abs=0.01)
    assert report.gap_fraction < 0.005, (
        f"{name} misses its own start by {report.gap:.1f} m"
    )


@pytest.mark.parametrize("name", LAYOUT_NAMES)
def test_lap_length_matches_the_real_circuit(name: str, built_tracks: dict[str, Track]):
    """A reconstruction that is 20% short is no longer the same circuit."""
    layout = get_layout(name)
    assert layout.reference_length > 0, f"{name} declares no reference length"

    lap = built_tracks[name].lap_length
    assert lap == pytest.approx(layout.reference_length, rel=0.03)


@pytest.mark.parametrize("name", LAYOUT_NAMES)
def test_the_drivable_surface_is_a_valid_annulus(name: str, built_tracks: dict[str, Track]):
    """`Track` builds an outer ring with the inner boundary as a hole.

    Shapely predicates -- every off-track check in the environment -- are
    undefined if that polygon is invalid.
    """
    track = built_tracks[name]
    polygon = track._track_polygon

    assert polygon.is_valid, f"{name} produces an invalid track polygon"
    assert len(polygon.interiors) == 1


@pytest.mark.parametrize("name", LAYOUT_NAMES)
def test_the_centerline_is_inside_its_own_track(name: str, built_tracks: dict[str, Track]):
    """Every centerline point is the middle of the track by construction.

    One reading as outside means the car is flagged off-track while driving
    dead centre, which is what a self-overlapping layout does.
    """
    track = built_tracks[name]
    centerline = track.interpolated_centerline
    outside = [i for i, p in enumerate(centerline[::5]) if not track.is_point_inside_track(p)]
    assert not outside, f"{name}: {len(outside)} centerline samples read as off-track"


@pytest.mark.parametrize("name", LAYOUT_NAMES)
def test_the_track_never_runs_into_itself(name: str, built_tracks: dict[str, Track]):
    """Two parts of the lap must not be closer than the track is wide.

    The Monaco reconstruction originally brought the swimming pool section
    within 7 m of the start straight on a 9 m wide circuit, which merges the
    two into one drivable blob: the boundaries cross, and a lap can be cut.
    """
    track = built_tracks[name]
    centerline = track.interpolated_centerline
    lap = track.lap_length
    arc = np.arange(len(centerline)) * track.arc_step

    pairs = cKDTree(centerline).query_pairs(r=track.width, output_type="ndarray")
    if len(pairs) == 0:
        return
    first, second = pairs[:, 0], pairs[:, 1]
    along = np.abs(arc[first] - arc[second])
    separation = np.minimum(along, lap - along)

    # Neighbours a few metres apart along the lap are trivially close together.
    remote = separation > 4.0 * track.width
    assert not np.any(remote), (
        f"{name}: the lap passes within {track.width} m of itself at "
        f"{int(np.count_nonzero(remote))} sample pairs"
    )


@pytest.mark.parametrize("name", LAYOUT_NAMES)
def test_corners_are_wider_than_the_track(name: str):
    """A corner radius below the half-width turns the inner boundary inside out."""
    layout = get_layout(name)
    assert layout.min_radius > layout.width, (
        f"{name}: tightest corner is {layout.min_radius:.1f} m "
        f"on a {layout.width:.1f} m wide track"
    )


def test_the_library_spans_a_range_of_difficulty():
    """Curriculum work needs both ends: an easy circuit and a hard one."""
    assert layouts_with_tags("easy")
    assert layouts_with_tags("hard")

    laps = [layout.reference_length for layout in LAYOUTS.values()]
    assert min(laps) < 2500.0  # a short circuit, for quick episodes
    assert max(laps) > 6500.0  # and a long one

    radii = [layout.min_radius for layout in LAYOUTS.values()]
    assert min(radii) < 15.0  # a hairpin
    assert max(radii) > 100.0  # and a circuit with no slow corner at all


def test_tag_filter_requires_every_tag():
    high_speed = {layout.name for layout in layouts_with_tags("high-speed")}
    ovals = {layout.name for layout in layouts_with_tags("high-speed", "oval")}

    assert ovals < high_speed
    assert "indianapolis_oval" in ovals
    assert len(layouts_with_tags()) == len(LAYOUTS)


def test_unknown_layout_names_are_rejected_with_the_alternatives():
    with pytest.raises(KeyError, match="monza"):
        get_layout("mnoza")


def test_generated_json_reloads_through_the_validated_loader(tmp_path):
    """The library is consumed as JSON files, so that path has to work."""
    paths = write_library(tmp_path)
    assert len(paths) == len(LAYOUTS)

    for path in paths:
        track = load_track_json(str(path))
        layout = get_layout(path.stem)
        assert track.name == layout.name
        assert track.width == layout.width
        assert len(track.centerline) > 100


def test_arc_step_sets_the_centerline_resolution():
    """A fixed sample count is meaningless across a 1.9 km to 7 km library."""
    fine = build_track("monza", arc_step=1.0)
    coarse = build_track("monza", arc_step=4.0)

    assert fine.arc_step == pytest.approx(1.0, abs=0.05)
    assert coarse.arc_step == pytest.approx(4.0, abs=0.2)


def test_a_layout_that_does_not_close_is_reported_not_silently_bent():
    """The guard rails belong to the builder; this is the library's use of them."""
    broken = Layout(
        name="broken",
        width=12.0,
        segments=(Straight(500.0), Corner(40.0, -90.0), Straight(500.0), Corner(40.0, -90.0)),
    )
    with pytest.raises(ValueError):
        build_centerline(broken)
