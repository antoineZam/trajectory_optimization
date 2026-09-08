"""Build a track centerline from a straight/corner description of a layout.

Real circuits are documented as a *sequence*: a 1.1 km straight, a 12 m radius
hairpin, a 340 m run to a 90 m radius double-right. Describing them that way
here keeps two properties that a hand-drawn coordinate list loses:

* corner radii are real, so the grip limit of the physics engine decides the
  apex speed the way it does on the real circuit;
* straight lengths are real, so the braking zones are where they actually are.

Integrating such a sequence almost never closes exactly -- a few tenths of a
degree of accumulated heading error over 5 km of track leaves a gap of tens of
metres. `build_centerline` closes it, but only within limits, so a layout whose
description is genuinely wrong fails loudly instead of being silently bent into
a loop that no longer resembles the circuit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# A closed circuit turns through exactly one full revolution.
FULL_TURN_DEG = 360.0


@dataclass(frozen=True)
class Straight:
    """A straight of `length` metres."""

    length: float
    name: str = ""

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError(f"straight length must be positive, got {self.length}")


@dataclass(frozen=True)
class Corner:
    """A constant-radius corner.

    Args:
        radius: Corner radius in metres, measured on the centerline.
        angle: Arc swept, in degrees. Positive turns left, negative right.
        name: Corner name, for readability of the layouts.
    """

    radius: float
    angle: float
    name: str = ""

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"corner radius must be positive, got {self.radius}")
        if self.angle == 0:
            raise ValueError("corner angle must be non-zero; use Straight instead")

    @property
    def length(self) -> float:
        """Arc length of the corner in metres."""
        return float(self.radius * abs(np.radians(self.angle)))


Segment = Straight | Corner


@dataclass(frozen=True)
class Layout:
    """A circuit description: what it is drawn from, how wide, and its sequence.

    Args:
        name: Identifier, also the stem of the generated track file.
        width: Uniform track width in metres.
        segments: The lap, as straights and corners in driving order.
        inspiration: The real circuit this reconstructs.
        tags: Free-form labels used to select subsets for training.
    """

    name: str
    width: float
    segments: tuple[Segment, ...]
    inspiration: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def length(self) -> float:
        """Nominal lap length in metres, before the closure correction."""
        return float(sum(s.length for s in self.segments))

    @property
    def total_turn(self) -> float:
        """Net heading change over the lap, in degrees."""
        return float(sum(s.angle for s in self.segments if isinstance(s, Corner)))

    @property
    def min_radius(self) -> float:
        """Tightest corner radius in metres."""
        radii = [s.radius for s in self.segments if isinstance(s, Corner)]
        return float(min(radii)) if radii else float("inf")

    @property
    def num_corners(self) -> int:
        """Number of corner segments (a double-apex counts as its two arcs)."""
        return sum(1 for s in self.segments if isinstance(s, Corner))


@dataclass(frozen=True)
class ClosureReport:
    """How much correction it took to turn a segment sequence into a loop.

    A layout stays faithful to its inspiration only as long as both numbers
    stay small: `turn_scale` away from 1.0 means every corner was tightened or
    opened, and `gap` is how far the raw sequence missed its own start line.
    """

    turn_scale: float
    gap: float
    lap_length: float

    @property
    def gap_fraction(self) -> float:
        """Closure gap as a fraction of the lap length."""
        return self.gap / self.lap_length


def build_centerline(
    layout: Layout,
    sample_spacing: float = 4.0,
    max_turn_scale: float = 1.25,
    max_gap_fraction: float = 0.06,
) -> tuple[np.ndarray, ClosureReport]:
    """Integrate a layout into a closed centerline.

    The sequence is first made to turn through exactly one revolution by
    scaling every corner angle equally, then the remaining position gap is
    removed by a taper spread over the whole lap -- proportional to arc length,
    so no single corner absorbs the error.

    Args:
        layout: The circuit description.
        sample_spacing: Target distance in metres between control points. The
            `Track` spline re-samples these, so this only has to be fine enough
            to pin the shape down.
        max_turn_scale: Reject the layout if closing the heading needs a
            correction beyond this factor (or its inverse).
        max_gap_fraction: Reject the layout if the raw sequence misses its own
            start by more than this fraction of the lap length.

    Returns:
        The closed centerline of shape (N, 2), and the closure report.

    Raises:
        ValueError: If the sequence is too far from a loop to be closed without
            distorting the layout beyond recognition.
    """
    turn = layout.total_turn
    if turn == 0:
        raise ValueError(f"{layout.name}: layout has no net rotation, it cannot close")

    turn_scale = FULL_TURN_DEG * np.sign(turn) / turn
    if not 1.0 / max_turn_scale <= turn_scale <= max_turn_scale:
        raise ValueError(
            f"{layout.name}: turns through {turn:.1f} deg, which needs a "
            f"{turn_scale:.2f}x correction to close (max {max_turn_scale:.2f}x). "
            "Fix the corner angles rather than letting them all be rescaled."
        )

    points, arc = _integrate(layout.segments, turn_scale, sample_spacing)

    # The sequence ends where it ends; the loop has to end where it started.
    gap = points[0] - points[-1]
    gap_norm = float(np.linalg.norm(gap))
    lap_length = float(arc[-1])
    if gap_norm > max_gap_fraction * lap_length:
        raise ValueError(
            f"{layout.name}: the sequence misses its start by {gap_norm:.1f} m "
            f"over a {lap_length:.0f} m lap "
            f"({gap_norm / lap_length:.1%} > {max_gap_fraction:.1%}). "
            "Adjust the straight lengths or the corner angles."
        )

    # Taper the gap out along the lap, then drop the duplicated closing point.
    points = points + gap * (arc / lap_length)[:, None]
    centerline = points[:-1]

    return centerline, ClosureReport(
        turn_scale=float(turn_scale), gap=gap_norm, lap_length=lap_length
    )


def _integrate(
    segments: tuple[Segment, ...], turn_scale: float, sample_spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Walk the segment sequence, sampling positions and cumulative arc length.

    Starts at the origin heading along +x. The returned arrays include both the
    first point and the final point of the last segment, so the caller can read
    the closure gap off their difference.

    Args:
        segments: Straights and corners in driving order.
        turn_scale: Factor applied to every corner angle to close the heading.
        sample_spacing: Target distance in metres between samples.

    Returns:
        Positions of shape (N, 2) and cumulative arc length of shape (N,).
    """
    position = np.zeros(2)
    heading = 0.0
    points = [position.copy()]
    arc = [0.0]

    for segment in segments:
        steps = max(2, int(np.ceil(segment.length / sample_spacing)))
        step_length = segment.length / steps

        if isinstance(segment, Straight):
            direction = np.array([np.cos(heading), np.sin(heading)])
            for _ in range(steps):
                position = position + direction * step_length
                points.append(position.copy())
                arc.append(arc[-1] + step_length)
            continue

        angle = np.radians(segment.angle * turn_scale)
        # Rotate the entry point about the corner's centre, so the radius is
        # exact instead of a chord approximation that shortens every corner.
        centre = position + np.sign(angle) * segment.radius * np.array(
            [-np.sin(heading), np.cos(heading)]
        )
        entry = position.copy()
        for i in range(steps):
            sweep = angle * (i + 1) / steps
            rotation = np.array(
                [[np.cos(sweep), -np.sin(sweep)], [np.sin(sweep), np.cos(sweep)]]
            )
            position = centre + rotation @ (entry - centre)
            points.append(position.copy())
            arc.append(arc[-1] + step_length)
        heading += angle

    return np.array(points), np.array(arc)
