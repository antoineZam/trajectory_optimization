from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt




def plot_track(ax, center: np.ndarray, left: np.ndarray, right: np.ndarray):
    ax.plot(center[:,0], center[:,1], linewidth=1.5, label="Center")
    ax.plot(left[:,0], left[:,1], linestyle=":", linewidth=1.0, label="Left")
    ax.plot(right[:,0], right[:,1], linestyle=":", linewidth=1.0, label="Right")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)




def plot_trajectories(ax, optimal: np.ndarray | None = None, past: np.ndarray | None = None, pred: np.ndarray | None = None):
    if optimal is not None:
        ax.plot(optimal[:,0], optimal[:,1], label="Racing line (opt)", linewidth=2.0)
    if past is not None:
        ax.plot(past[:,0], past[:,1], label="Past", linewidth=1.5)
    if pred is not None:
        ax.plot(pred[:,0], pred[:,1], label="Predicted", linewidth=1.5)
    ax.legend(loc="best")




def plot_trajectory(ax, trajectory: np.ndarray, label: str = "Trajectory", linewidth: float = 1.5):
    ax.plot(trajectory[:,0], trajectory[:,1], label=label, linewidth=linewidth)
    ax.legend(loc="best")