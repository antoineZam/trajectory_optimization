from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point, Polygon

from schemas import validate_track_data


@dataclass
class Track:
    name: str
    centerline: np.ndarray # shape (N, 2)
    width: float # uniform width for simplicity
    interpolation_resolution: int = 2000  # Default resolution
    # Interpolated track data for smooth boundaries
    _interpolated_centerline: np.ndarray = None
    _interpolated_left: np.ndarray = None
    _interpolated_right: np.ndarray = None
    # K-D Tree for fast spatial queries
    _centerline_kdtree: cKDTree = None
    _left_boundary_kdtree: cKDTree = None
    _right_boundary_kdtree: cKDTree = None
    # Polygon for accurate inside/outside checks
    _track_polygon: Polygon = None
    # Arc-length metadata (meters), derived from the interpolated centerline
    _lap_length: float = None
    _arc_step: float = None

    def __post_init__(self):
        # Allow lazy computation of interpolated track
        pass

    @property
    def left_boundary(self) -> np.ndarray:
        if self._interpolated_left is None:
            self._compute_interpolated_track()
        return self._interpolated_left

    @property
    def right_boundary(self) -> np.ndarray:
        if self._interpolated_right is None:
            self._compute_interpolated_track()
        return self._interpolated_right

    @property
    def interpolated_centerline(self) -> np.ndarray:
        """Get high-resolution interpolated centerline for smooth path following."""
        if self._interpolated_centerline is None:
            self._compute_interpolated_track()
        return self._interpolated_centerline

    @property
    def lap_length(self) -> float:
        """Total arc length of one lap along the interpolated centerline, in meters."""
        if self._lap_length is None:
            self._compute_interpolated_track()
        return self._lap_length

    @property
    def arc_step(self) -> float:
        """Arc length between two consecutive interpolated centerline points, in meters.

        This is the conversion factor between a *distance* in meters and an *index
        offset* into `interpolated_centerline`. Note that `interpolation_resolution`
        is a point count, not a spatial resolution -- dividing a distance by it is a
        bug (see `arc_step` usages).
        """
        if self._arc_step is None:
            self._compute_interpolated_track()
        return self._arc_step

    def index_offset_for_distance(self, distance: float) -> int:
        """Convert a distance ahead in meters to an index offset on the centerline.

        Args:
            distance: Distance along the track in meters.

        Returns:
            Number of centerline indices corresponding to that distance.
        """
        return int(round(distance / self.arc_step))

    def _compute_interpolated_track(self):
        """Compute smooth interpolated centerline and boundaries using splines."""
        try:
            # Ensure track is closed (add first point at end if needed)
            centerline = self.centerline
            if not np.allclose(centerline[0], centerline[-1], atol=1e-6):
                centerline = np.vstack([centerline, centerline[0]])

            # Parameterize the centerline with spline interpolation
            # Use periodic spline for closed tracks
            tck, u = splprep([centerline[:, 0], centerline[:, 1]],
                           s=0, k=3, per=True)  # s=0 for exact interpolation, k=3 for cubic

            # Generate high-resolution interpolated points
            u_new = np.linspace(0, 1, self.interpolation_resolution, endpoint=False)
            interp_x, interp_y = splev(u_new, tck)
            self._interpolated_centerline = np.column_stack([interp_x, interp_y])

            # Compute smooth boundaries by offsetting the interpolated centerline
            self._interpolated_left = self._offset_interpolated_line(
                self._interpolated_centerline, self.width/2)
            self._interpolated_right = self._offset_interpolated_line(
                self._interpolated_centerline, -self.width/2)

        except Exception as e:
            print(f"Warning: Spline interpolation failed ({e}), using original boundaries")
            # Fallback to original method
            self._interpolated_centerline = self.centerline
            self._interpolated_left = offset_polyline(self.centerline, +self.width/2)
            self._interpolated_right = offset_polyline(self.centerline, -self.width/2)

        # Arc length of the closed centerline (includes the wrap-around segment)
        segments = np.diff(
            np.vstack([self._interpolated_centerline, self._interpolated_centerline[0]]),
            axis=0,
        )
        self._lap_length = float(np.sum(np.linalg.norm(segments, axis=1)))
        self._arc_step = self._lap_length / len(self._interpolated_centerline)

        # Build K-D trees for fast lookups
        self._centerline_kdtree = cKDTree(self._interpolated_centerline)
        self._left_boundary_kdtree = cKDTree(self._interpolated_left)
        self._right_boundary_kdtree = cKDTree(self._interpolated_right)

        # Create a Shapely Polygon for the track for accurate inside/outside checks.
        #
        # The track is an ANNULUS: one boundary is the outer ring, the other the
        # inner hole. Concatenating them into a single ring instead closes the
        # shape with a chord across the start/finish line, so points near the
        # seam -- including the centerline itself -- read as outside the track.
        # Which offset ends up outside depends on the centerline's traversal
        # direction, so pick by enclosed area rather than assuming.
        ring_a = Polygon(self._interpolated_left)
        ring_b = Polygon(self._interpolated_right)
        if ring_a.area >= ring_b.area:
            outer, inner = self._interpolated_left, self._interpolated_right
        else:
            outer, inner = self._interpolated_right, self._interpolated_left
        self._track_polygon = Polygon(shell=outer, holes=[inner])

    def _offset_interpolated_line(self, line: np.ndarray, offset: float) -> np.ndarray:
        """Create offset boundary from interpolated centerline.

        The centerline is a closed curve, so tangents are computed with
        *periodic* central differences. `np.gradient` falls back to one-sided
        differences at the array ends, which puts a wrong normal on the two
        samples at the start/finish seam; the resulting misplaced offset points
        made the boundary self-intersect there (visible as an invalid track
        polygon for any track wider than ~3.5x this one).
        """
        # Periodic central differences: tangent[i] = (p[i+1] - p[i-1]) / 2
        tangents = (np.roll(line, -1, axis=0) - np.roll(line, 1, axis=0)) / 2.0
        # Normalize tangents
        tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangent_norms = np.where(tangent_norms > 1e-8, tangent_norms, 1.0)
        unit_tangents = tangents / tangent_norms

        # Compute normal vectors (perpendicular to tangents)
        normals = np.stack([-unit_tangents[:, 1], unit_tangents[:, 0]], axis=1)

        # Offset points by normal * offset distance
        boundary = line + normals * offset
        return boundary

    def get_distance_to_boundaries(self, point: np.ndarray) -> tuple[float, float]:
        """
        Get distance from point to left and right boundaries using interpolated track.
        Returns (distance_to_left, distance_to_right).
        """
        if self._interpolated_left is None:
            self._compute_interpolated_track()

        # Query K-D trees for closest distances
        dist_left, _ = self._left_boundary_kdtree.query(point, k=1)
        dist_right, _ = self._right_boundary_kdtree.query(point, k=1)

        return dist_left, dist_right

    def is_point_inside_track(self, point: np.ndarray) -> bool:
        """Check if point is inside interpolated track boundaries using a robust polygon check."""
        if self._track_polygon is None:
            self._compute_interpolated_track()
        return self._track_polygon.contains(Point(point))

    def closest_index(self, point: np.ndarray) -> int:
        """Index of the nearest interpolated centerline sample to `point`.

        Uses the K-D tree, which is ~2x faster than a full argmin over the
        centerline and is built once at interpolation time anyway.

        Args:
            point: Query position [x, y].

        Returns:
            Index into `interpolated_centerline`.
        """
        if self._centerline_kdtree is None:
            self._compute_interpolated_track()
        _, closest_idx = self._centerline_kdtree.query(point, k=1)
        return int(closest_idx)

    def get_track_progress(self, point: np.ndarray) -> float:
        """Get progress around track (0-1) based on closest point on interpolated centerline."""
        return self.closest_index(point) / len(self.interpolated_centerline)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "centerline": self.centerline.tolist(),
        }


    @staticmethod
    def from_json(d: dict, interpolation_resolution: int = 2000) -> Track:
        return Track(
            name=d["name"],
            width=float(d["width"]),
            centerline=np.array(d["centerline"], dtype=float),
            interpolation_resolution=interpolation_resolution,
    )




def offset_polyline(poly: np.ndarray, offset: float) -> np.ndarray:
    """Offset approximatif via buffer shapely sur LineString puis extraction d'edge."""
    ls = LineString(poly)
    buff = ls.buffer(offset, cap_style=2, join_style=2)
    # Pour une polyline, le buffer crée un polygone; on récupère l’edge pertinent
    if not isinstance(buff, Polygon):
        buff = max(buff.geoms, key=lambda g: g.length)
    # Approx: échantillonne l'edge plus long
    coords = np.array(buff.exterior.coords)
    return coords




def make_oval_track(a: float = 120.0, b: float = 70.0, n: int = 600, width: float = 12.0,
    rotate_deg: float = 0.0, dx: float = 0.0, dy: float = 0.0,
    interpolation_resolution: int = 2000) -> Track:
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)
    center = np.stack([x, y], axis=1)
    ls = LineString(center)
    ls = rotate(ls, rotate_deg, origin=(0, 0), use_radians=False)
    ls = translate(ls, dx, dy)
    return Track(name="oval", centerline=np.array(ls.coords), width=width,
                 interpolation_resolution=interpolation_resolution)




def save_track_json(track: Track, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(track.to_json(), f, ensure_ascii=False, indent=2)




def load_track_json(path: str, interpolation_resolution: int = 2000) -> Track:
    """Load and validate track from JSON file.

    Args:
        path: Path to track JSON file.
        interpolation_resolution: Resolution for track interpolation.

    Returns:
        Validated Track instance.

    Raises:
        pydantic.ValidationError: If track data fails validation.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Validate data with Pydantic schema
    validated = validate_track_data(data)

    return Track(
        name=validated.name,
        width=validated.width,
        centerline=np.array(validated.centerline, dtype=float),
        interpolation_resolution=interpolation_resolution,
    )
