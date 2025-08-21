from __future__ import annotations
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.track import load_track_json
from physics.physics_engine import VehicleSpec
from envs.rl_environment import RacingEnv


def train_and_export(track_path: str, vehicle_cfg_path: str, out_path: str, timesteps: int = 200_000):
    track = load_track_json(track_path)
    with open(vehicle_cfg_path, "r", encoding="utf-8") as f:
        veh_cfg = json.load(f)
    spec = VehicleSpec.from_config(veh_cfg)
    
    def make_env():
        return RacingEnv(track, spec)
    
    # Train model
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=None)
    model.learn(total_timesteps=timesteps)


    # Rollout best trajectory (greedy eval)
    obs, _ = env.reset()
    done = False
    xs, ys, vs = [], [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        s = env.envs[0].state
        xs.append(s.x); ys.append(s.y); vs.append(np.hypot(s.vx, s.vy))
        done = bool(term or trunc)
    
    traj = np.stack([np.array(xs), np.array(ys), np.array(vs)], axis=1)
    np.save(out_path, traj)
    return traj