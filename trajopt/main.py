from __future__ import annotations
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from utils.track import make_oval_track, save_track_json, load_track_json
from utils.visualization import plot_track, plot_trajectories
from rl.optimal_line_finder import train_and_export


def ensure_sample_track(path: str):
    if not os.path.exists(path):
        tr = make_oval_track(a=120.0, b=70.0, n=800, width=12.0)
        save_track_json(tr, path)
        print(f"Sample track saved to {path}")


def mode_optimize(args):
    ensure_sample_track(args.track)
    traj = train_and_export(args.track, args.vehicle, args.output, timesteps=args.timesteps)
    tr = load_track_json(args.track)

    fig, ax = plt.subplots(figsize=(10,6))
    plot_track(ax, tr.centerline, tr.left_boundary, tr.right_boundary)
    plot_trajectories(ax, optimal=traj[:, :2])
    ax.set_title("Optimized racing line")
    plt.show()


def mode_simulate(args):
    # Affiche une trajectoire optimale pré-entrainée + une prédiction jouet (bruitée)
    tr = load_track_json(args.track)
    traj = np.load(args.opt_traj)
    past = traj[:200, :2]
    # prédiction jouet: extrapolation + bruit
    pred = past[-1] + (past[-1] - past[-20]) * np.linspace(0, 1, 30)[:,None]
    pred += np.random.normal(scale=0.5, size=pred.shape)


    fig, ax = plt.subplots(figsize=(10,6))
    plot_track(ax, tr.centerline, tr.left_boundary, tr.right_boundary)
    plot_trajectories(ax, optimal=traj[:, :2], past=past, pred=pred)
    ax.set_title("Simulation: past vs predicted (demo)")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)


    p_opt = sub.add_parser("optimize")
    p_opt.add_argument("--track", default="data/tracks/sample_track.json")
    p_opt.add_argument("--vehicle", default="configs/vehicle_spec.example.json")
    p_opt.add_argument("--output", default="data/optimal_traj.npy")
    p_opt.add_argument("--timesteps", type=int, default=50_000)


    p_sim = sub.add_parser("simulate")
    p_sim.add_argument("--track", default="data/tracks/sample_track.json")
    p_sim.add_argument("--opt_traj", default="data/optimal_traj.npy")


    args = p.parse_args()
    if args.mode == "optimize":
        mode_optimize(args)
    elif args.mode == "simulate":
        mode_simulate(args)




if __name__ == "__main__":
    main()
