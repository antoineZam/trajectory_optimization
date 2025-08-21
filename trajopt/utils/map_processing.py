from __future__ import annotations
import numpy as np
from typing import Tuple
from .track import Track


def sample_next_centerline_points(centerline: np.ndarray, pos_xy: np.ndarray, k: int = 5, lookahead: float = 5.0) -> np.ndarray:
    """Renvoie k points de la ligne centrale à venir, espacés par lookahead (approx)."""
    # Trouve le point le plus proche
    d2 = ((centerline - pos_xy) ** 2).sum(axis=1)
    i0 = int(np.argmin(d2))
    # Échantillonnage circulaire
    idxs = (i0 + np.arange(1, k + 1)) % len(centerline)
    return centerline[idxs]


def speed_along_tangent(vx: float, vy: float, tangent: np.ndarray) -> float:
    t = tangent / (np.linalg.norm(tangent) + 1e-6)
    return vx * t[0] + vy * t[1]


def get_track_width(track: Track) -> float:
    return track.width