from __future__ import annotations
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.track import load_track_json
from physics.physics_engine import VehicleSpec
from envs.rl_environment import RacingEnv


def train_and_export(track_path: str, vehicle_cfg_path: str, out_path: str, timesteps: int = 500_000):
    track = load_track_json(track_path)
    with open(vehicle_cfg_path, "r", encoding="utf-8") as f:
        veh_cfg = json.load(f)
    spec = VehicleSpec.from_config(veh_cfg)
    
    def make_env():
        return RacingEnv(track, spec)
    
    # Extended training with checkpoint system and detailed logging
    print("=" * 60)
    print("TRAJECTORY OPTIMIZATION - INTERPOLATED TRACK TRAINING")
    print("=" * 60)
    print(f"Training timesteps: {timesteps:,}")
    print(f"Original track points: {len(track.centerline)}")
    print(f"Interpolated track points: {track._interpolation_resolution}")
    print(f"Track width: {track.width}m")
    print(f"Checkpoints: 4 (based on smooth progress tracking)")
    print(f"Vehicle wheelbase: {spec.wheelbase}m, track width: {spec.track_width}m")
    print("REALISTIC VEHICLE DYNAMICS:")
    print("  - Speed-dependent steering limitations")
    print(f"  - Max steering angle: {spec.max_steering_angle:.2f} rad ({np.degrees(spec.max_steering_angle):.1f}°)")
    print(f"  - Minimum turn radius: {spec.min_turn_radius}m")
    print("  - Steering reduces with speed (realistic physics)")
    print("INTERPOLATED FEATURES:")
    print("  - Smooth spline-based track boundaries")
    print("  - High-resolution progress tracking") 
    print("  - Accurate wheel position validation")
    print("PROGRESSIVE TRAINING:")
    print("  Episodes 0-500: LEARNING phase (only terminate if ALL wheels leave track)")
    print("  Episodes 500-1000: INTERMEDIATE phase (terminate if 0 wheels inside)")  
    print("  Episodes 1000+: STRICT phase (≥2 wheels must stay inside track)")
    print("Rewards: +200 per checkpoint, +5 on-track, +500 lap completion")
    print("Logging: Stats every 100 episodes, phase transitions, detailed events")
    print("\nSTEERING LIMITATIONS PREVIEW:")
    from physics.physics_engine import get_steering_info
    for speed_kmh in [0, 30, 60, 100, 150]:
        speed_ms = speed_kmh / 3.6
        info = get_steering_info(spec, speed_ms)
        print(f"  {speed_kmh:3d} km/h: max {info['max_steering_angle_deg']:4.1f}° (turn radius: {info['turn_radius_m']:4.1f}m)")
    print("=" * 60)
    
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=None)
    model.learn(total_timesteps=timesteps)


    # Rollout best trajectory (greedy eval)
    obs = env.reset()
    done = False
    xs, ys, vs = [], [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        s = env.envs[0].state
        xs.append(s.x); ys.append(s.y); vs.append(np.hypot(s.vx, s.vy))
        done = bool(done[0])  # Extract boolean from array
    
    traj = np.stack([np.array(xs), np.array(ys), np.array(vs)], axis=1)
    np.save(out_path, traj)
    return traj