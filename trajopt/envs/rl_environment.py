from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from utils.track import Track
from utils.map_processing import sample_next_centerline_points, speed_along_tangent
from physics.physics_engine import VehicleSpec, VehicleState, step_dynamics




@dataclass
class RLConfig:
    dt: float = 0.05
    max_steps: int = 4000
    offtrack_penalty: float = -10.0
    progress_reward_scale: float = 1.0
    time_penalty: float = -0.001
    max_steer_rate: float = 0.25


class RacingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}
    
    def __init__(self, track: Track, veh_spec: VehicleSpec, cfg: RLConfig | None = None):
        super().__init__()
        self.track = track
        self.spec = veh_spec
        self.cfg = cfg or RLConfig()

        high = np.array([np.inf]*2 + [np.pi] + [100.0, 50.0, 10.0])
        low = -high
        # Observation: [x, y, yaw, vx, vy, yaw_rate] + next 5 points (x,y)
        self.k = 5
        self.obs_dim = 6 + 2*self.k
        self.action_space = spaces.Box(low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32) # throttle, brake, steer
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.state = None
        self.step_count = 0


    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        x0, y0 = self.track.centerline[0]
        self.state = VehicleState(x=x0, y=y0, yaw=0.0, vx=3.0, vy=0.0, yaw_rate=0.0, gear=2, rpm=1500.0)
        return self._get_obs()

    def _get_obs(self):
        s = self.state
        nxt = sample_next_centerline_points(self.track.centerline, np.array([s.x, s.y]), k=self.k, lookahead=5.0)
        obs = np.array([s.x, s.y, s.yaw, s.vx, s.vy, s.yaw_rate] + nxt.flatten().tolist(), dtype=np.float32)
        return obs
        
        
    def step(self, action: np.ndarray):
        throttle, brake, steer = float(action[0]), float(action[1]), float(action[2])    
        self.state = step_dynamics(self.spec, self.state, self.cfg.dt, throttle, brake, steer)
        self.step_count += 1
        
        # Rewards
        # Progress along tangent: approx via next centerline vector
        s = self.state
        idx = (np.argmin(((self.track.centerline - np.array([s.x, s.y]))**2).sum(axis=1)))
        nxt = self.track.centerline[(idx+1) % len(self.track.centerline)]
        tangent = nxt - self.track.centerline[idx]
        vprog = speed_along_tangent(s.vx, s.vy, tangent)
        
        reward = self.cfg.progress_reward_scale * vprog + self.cfg.time_penalty

        # Offtrack check (distance to center > width/2)
        dist = np.linalg.norm(self.track.centerline[idx] - np.array([s.x, s.y]))
        terminated = dist > (self.track.width/2 + 2.0)
        if terminated:
            reward += self.cfg.offtrack_penalty

        truncated = self.step_count >= self.cfg.max_steps
        return self._get_obs(), reward, terminated, truncated, {}


    def render(self):
        pass