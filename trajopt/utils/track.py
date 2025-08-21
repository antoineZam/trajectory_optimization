from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass
from shapely.geometry import LineString, Polygon
from shapely.affinity import rotate, translate


@dataclass
class Track:
    name: str
    centerline: np.ndarray # shape (N, 2)
    width: float # uniform width for simplicity


    @property
    def left_boundary(self) -> np.ndarray:
        return offset_polyline(self.centerline, +self.width/2)


    @property
    def right_boundary(self) -> np.ndarray:
        return offset_polyline(self.centerline, -self.width/2)


    def to_json(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "centerline": self.centerline.tolist(),
        }


    @staticmethod
    def from_json(d: dict) -> "Track":
        return Track(
            name=d["name"],
            width=float(d["width"]),
            centerline=np.array(d["centerline"], dtype=float),
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
    rotate_deg: float = 0.0, dx: float = 0.0, dy: float = 0.0) -> Track:
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)
    center = np.stack([x, y], axis=1)
    ls = LineString(center)
    ls = rotate(ls, rotate_deg, origin=(0, 0), use_radians=False)
    ls = translate(ls, dx, dy)
    return Track(name="oval", centerline=np.array(ls.coords), width=width)




def save_track_json(track: Track, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(track.to_json(), f, ensure_ascii=False, indent=2)




def load_track_json(path: str) -> Track:
    with open(path, "r", encoding="utf-8") as f:
        return Track.from_json(json.load(f))