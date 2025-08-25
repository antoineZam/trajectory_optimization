from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass
from shapely.geometry import LineString, Polygon
from shapely.affinity import rotate, translate
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


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
        
        # Build K-D trees for fast lookups
        self._centerline_kdtree = cKDTree(self._interpolated_centerline)
        self._left_boundary_kdtree = cKDTree(self._interpolated_left)
        self._right_boundary_kdtree = cKDTree(self._interpolated_right)

    def _offset_interpolated_line(self, line: np.ndarray, offset: float) -> np.ndarray:
        """Create offset boundary from interpolated centerline."""
        # Compute tangent vectors
        tangents = np.gradient(line, axis=0)
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
        """Check if point is inside interpolated track boundaries."""
        dist_left, dist_right = self.get_distance_to_boundaries(point)
        
        # Simple approximation: if closer to centerline than to either boundary
        dist_center, _ = self._centerline_kdtree.query(point, k=1)
        
        # Point is inside if it's reasonable close to centerline and boundaries
        max_reasonable_dist = self.width/2 + 1.0  # Small safety margin
        return dist_center <= max_reasonable_dist

    def get_track_progress(self, point: np.ndarray) -> float:
        """Get progress around track (0-1) based on closest point on interpolated centerline."""
        if self._interpolated_centerline is None:
            self._compute_interpolated_track()
            
        # Find closest point on interpolated centerline using K-D Tree
        _, closest_idx = self._centerline_kdtree.query(point, k=1)
        
        # Progress is the index divided by total points
        return closest_idx / len(self._interpolated_centerline)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "centerline": self.centerline.tolist(),
        }


    @staticmethod
    def from_json(d: dict, interpolation_resolution: int = 2000) -> "Track":
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
    with open(path, "r", encoding="utf-8") as f:
        return Track.from_json(json.load(f), interpolation_resolution=interpolation_resolution)