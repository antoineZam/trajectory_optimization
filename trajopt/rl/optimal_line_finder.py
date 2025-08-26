from __future__ import annotations
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.track import load_track_json
from physics.physics_engine import VehicleSpec
from envs.rl_environment import RacingEnv


def train_and_export(track_path: str, vehicle_cfg_path: str, out_path: str, 
                     timesteps: int = 500_000, interpolation_resolution: int = 2000):
    track = load_track_json(track_path, interpolation_resolution=interpolation_resolution)
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
    print(f"Track points: {len(track.centerline)}")
    print(f"Interpolated track points: {track.interpolation_resolution}")
    print(f"Track width: {track.width}m")
    print(f"Checkpoints: 4 (every {len(track.centerline)//4} track points)")
    print(f"Vehicle wheelbase: {spec.wheelbase}m, track width: {spec.track_width}m")
    print("FIXED ACTION SPACE:")
    print(f"  - Agent steer_command [-1, 1] → steering angle [-{np.degrees(spec.max_steering_angle):.1f}°, +{np.degrees(spec.max_steering_angle):.1f}°]")
    print("  - No more unrealistic 57° steering commands!")
    print("REALISTIC VEHICLE DYNAMICS:")
    print("  - Speed-dependent steering limitations (agent learns to work within limits)")
    print(f"  - Max steering angle: {spec.max_steering_angle:.2f} rad ({np.degrees(spec.max_steering_angle):.1f}°)")
    print(f"  - Minimum turn radius: {spec.min_turn_radius}m")
    print("  - Physics engine applies additional speed-based clipping")
    print("ENHANCED REWARD SYSTEM:")
    print("  - Stay on track: +10 points")
    print("  - Checkpoint progress: +200 points") 
    print("  - Speed bonus: +0-5 points (when on track)")
    print("  - Racing line following: +0-3 points (distance to centerline)")
    print("  - Steering extremes penalty: -0-2 points (quadratic)")
    print("  - Smooth steering: penalty for jerky movements")
    print("  - Lap completion: +500 points")
    print("COMPREHENSIVE TELEMETRY:")
    print("  - Real-time metrics: speed, steering, position, rewards")
    print("  - Episode summaries: lap times, racing quality, control analysis")
    print("  - Training progress: completion rates, performance trends")
    print("  - Data export: JSON/CSV for external analysis")
    print("INTERPOLATED FEATURES:")
    print("  - Smooth spline-based track boundaries")
    print("  - High-resolution progress tracking") 
    print("  - Accurate wheel position validation")
    print("PROGRESSIVE TRAINING:")
    print("  Episodes 0-500: LEARNING phase (only terminate if ALL wheels leave track)")
    print("  Episodes 500-1000: INTERMEDIATE phase (terminate if 0 wheels inside)")  
    print("  Episodes 1000+: STRICT phase (≥2 wheels must stay inside track)")
    print("Logging: Stats every 100 episodes, phase transitions, checkpoint progress")
    print("\nIMPROVED STEERING LIMITATIONS (More learning-friendly):")
    from physics.physics_engine import get_steering_info
    for speed_kmh in [0, 15, 30, 60, 100, 150]:
        speed_ms = speed_kmh / 3.6
        info = get_steering_info(spec, speed_ms)
        reduction = info['max_steering_angle_deg'] / np.degrees(spec.max_steering_angle)
        print(f"  {speed_kmh:3d} km/h: max {info['max_steering_angle_deg']:4.1f}° ({reduction:4.1%} of base) → {info['turn_radius_m']:4.1f}m radius")
    print("  Low speeds: Near-full steering for learning")
    print("  High speeds: Realistic physics constraints")
    print("  Agent learns proper speed-cornering relationship!")
    print("=" * 60)
    
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=None)
    model.learn(total_timesteps=timesteps)

    # Final telemetry export
    if hasattr(env.envs[0], 'telemetry') and env.envs[0].telemetry:
        print("\nTRAINING COMPLETED - Exporting telemetry data...")
        env.envs[0].telemetry.save_summaries()
        try:
            env.envs[0].telemetry.export_csv()
        except ImportError:
            print("   Note: Install pandas for CSV export: pip install pandas")
        print("   Telemetry data saved to ./telemetry/ directory")

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