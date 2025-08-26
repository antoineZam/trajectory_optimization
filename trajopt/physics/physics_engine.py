from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np


# -------------------------
# Données véhicule & utilitaires
# -------------------------


@dataclass
class VehicleSpec:
    mass: float
    cg: np.ndarray # x,y,z (m)
    moi: np.ndarray # I_x, I_y, I_z (kg*m^2)
    Cx: float
    Cz_front: float
    Cz_rear: float
    mass_split_front: float # 0..1
    # Vehicle dimensions for track boundary checking
    wheelbase: float # distance between front and rear axles (m)
    track_width: float # distance between left and right wheels (m)
    # Steering system limitations
    max_steering_angle: float # maximum physical steering angle (rad)
    steering_speed_factor: float # how much speed reduces steering (s/m)
    min_turn_radius: float # minimum turning radius at low speed (m)
    # Powertrain
    torque_curve: np.ndarray # shape (M, 2): RPM, Torque(Nm)    
    rpm_limiter: float
    gear_ratios: np.ndarray
    final_drive: float
    driveline_eff: float # 0..1
    # Tires & suspension (simplified)
    k_spring_front: float
    k_spring_rear: float
    camber: float
    toe: float
    mu0: float # base friction coefficient
    alpha_muFz: float # mu = mu0 * (1 + alpha*(Fz-Ref)/Ref)
    # Brakes
    brake_torque_max: float
    brake_split_front: float

    @staticmethod
    def from_config(cfg: Dict) -> "VehicleSpec":
        ch, pw, st, br = cfg["chassis"], cfg["powertrain"], cfg["suspension_tires"], cfg["brakes"]
        return VehicleSpec(
            mass=float(ch["masse_totale"]),
            cg=np.array(ch["centre_de_gravite"], dtype=float),
            moi=np.array(ch["moment_inertie"], dtype=float),
            Cx=float(ch["coefficient_trainee"]),
            Cz_front=float(ch["coefficient_portance"]["front"]),
            Cz_rear=float(ch["coefficient_portance"]["rear"]),
            mass_split_front=float(ch["repartition_masses"]["front"]),
            # Vehicle dimensions with defaults for racing car
            wheelbase=float(ch.get("empattement", 2.65)),  # Default wheelbase ~2.65m
            track_width=float(ch.get("voie", 1.55)),       # Default track width ~1.55m
            # Steering limitations with realistic defaults
            max_steering_angle=float(ch.get("angle_braquage_max", 0.6)),      # ~34 degrees max
            steering_speed_factor=float(ch.get("facteur_vitesse_braquage", 0.02)),  # Reduces steering at speed
            min_turn_radius=float(ch.get("rayon_braquage_min", 6.0)),         # 6m minimum turn radius
            torque_curve=np.array(pw["courbe_couple_moteur"], dtype=float),
            rpm_limiter=float(pw["limiteur_rpm"]),
            gear_ratios=np.array(pw["rapports_boite_de_vitesse"], dtype=float),
            final_drive=float(pw["rapport_pont_final"]),
            driveline_eff=float(pw["efficacite_transmission"]),
            k_spring_front=float(st["raideur_suspension"]["front"]),
            k_spring_rear=float(st["raideur_suspension"]["rear"]),
            camber=float(st["geometrie_pneus"]["carrossage"]),
            toe=float(st["geometrie_pneus"]["pincement"]),
            mu0=float(st["modele_pneu_adherence"]["mu0"]),
            alpha_muFz=float(st["modele_pneu_adherence"]["alpha"]),
            brake_torque_max=float(br["couple_freinage_max"]),
            brake_split_front=float(br["repartition_freinage"]["front"]),
        )


# -------------------------
# Modèle dynamique (bicycle 2D + aéro + transmission simplifiée)
# -------------------------


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    yaw_rate: float
    gear: int
    rpm: float


def interp_torque(torque_curve: np.ndarray, rpm: float) -> float:
    rpm = np.clip(rpm, torque_curve[0,0], torque_curve[-1,0])
    return float(np.interp(rpm, torque_curve[:,0], torque_curve[:,1]))


def aero_forces(spec: VehicleSpec, v: float, rho_air: float = 1.225, area: float = 2.0) -> Tuple[float, float]:
    """
    Drag ~ 0.5*rho*Cx*A*v^2 ; Downforce (front+rear) ~ 0.5*rho*Cz*A*v^2
    """
    drag = 0.5 * rho_air * spec.Cx * area * v**2
    downforce = 0.5 * rho_air * (abs(spec.Cz_front) + abs(spec.Cz_rear)) * area * v**2
    return drag, downforce


def tire_mu(spec: VehicleSpec, Fz: float, Fz_ref: float = 4000.0) -> float:
    return spec.mu0 * (1.0 + spec.alpha_muFz * (Fz - Fz_ref) / max(Fz_ref, 1.0))


def get_max_steering_angle(spec: VehicleSpec, speed: float) -> float:
    """
    Calculate maximum allowed steering angle based on vehicle speed.
    
    At low speeds: Full steering angle available
    At high speeds: Reduced steering to prevent unrealistic sharp turns
    
    Args:
        spec: Vehicle specification with steering limits
        speed: Current vehicle speed (m/s)
    
    Returns:
        Maximum allowed steering angle (rad)
    """
    # Base maximum steering angle (physical limit)
    max_angle = spec.max_steering_angle
    
    # Speed-dependent reduction factor
    # At 0 m/s: full steering, at higher speeds: progressively less
    speed_reduction = 1.0 / (1.0 + spec.steering_speed_factor * speed)
    
    # Apply minimum turn radius constraint ONLY at higher speeds
    # Allow more aggressive steering at low speeds for learning
    if speed > 5.0:  # Only apply radius constraints above 18 km/h
        # Using bicycle model: tan(δ) = wheelbase / turn_radius
        # Speed-adjusted minimum radius (much less aggressive)
        speed_adjusted_radius = spec.min_turn_radius * (1.0 + (speed - 5.0) * 0.05)  # Gentler increase
        speed_radius_angle = np.arctan(spec.wheelbase / speed_adjusted_radius)
        
        # Take the most restrictive limit
        return min(max_angle * speed_reduction, speed_radius_angle)
    else:
        # At low speeds, allow near-full steering for learning
        low_speed_factor = max(0.8, speed / 5.0)  # Minimum 80% of max steering
        return max_angle * speed_reduction * low_speed_factor


def step_dynamics(spec: VehicleSpec, s: VehicleState, dt: float,
                throttle: float, brake: float, steer: float,
                wheel_radius: float = 0.33, CdA_area: float = 2.0) -> VehicleState:
    # Clamp inputs
    throttle = float(np.clip(throttle, 0.0, 1.0))
    brake = float(np.clip(brake, 0.0, 1.0))
    
    # Calculate current speed for steering limitations
    v = np.hypot(s.vx, s.vy)
    
    # Apply realistic speed-dependent steering limitations
    max_steer_angle = get_max_steering_angle(spec, v)
    steer_before = steer
    steer = float(np.clip(steer, -max_steer_angle, max_steer_angle))
    
    # Physics debug disabled - using telemetry system for comprehensive monitoring
    # (Debug code removed to reduce console noise)

    # Aéro
    drag, downforce = aero_forces(spec, v, area=CdA_area)

    # Répartition verticale (statique + aéro) — simplifiée
    Fz_front = (spec.mass * 9.81 * spec.mass_split_front) + downforce * 0.5
    Fz_rear = (spec.mass * 9.81 * (1.0 - spec.mass_split_front)) + downforce * 0.5

    # Capacité de friction
    mu_f = tire_mu(spec, Fz_front)
    mu_r = tire_mu(spec, Fz_rear)
    Fy_max_front = mu_f * Fz_front
    Fy_max_rear = mu_r * Fz_rear
    
    # Propulsion : estimate wheel speed from gear
    gear = int(np.clip(s.gear, 1, len(spec.gear_ratios)))
    ratio = spec.gear_ratios[gear-1] * spec.final_drive
    wheel_omega = (s.vx / max(wheel_radius,1e-3)) if v>0.1 else 0.0
    est_rpm = wheel_omega * ratio * 9.5493 # rad/s -> RPM
    rpm = np.clip(est_rpm, 800.0, spec.rpm_limiter)
    eng_torque = interp_torque(spec.torque_curve, rpm) * throttle
    wheel_torque = eng_torque * ratio * spec.driveline_eff
    Fx_driven = wheel_torque / max(wheel_radius,1e-3)

    # Freinage
    brake_torque = brake * spec.brake_torque_max
    Fx_brake = brake_torque / max(wheel_radius,1e-3)
    Fx_long = Fx_driven - Fx_brake - np.sign(s.vx) * drag

    # Direction (bicycle) — saturée par Fy_max
    # Simplification : Fy_front ~ k*steer*vx , saturée ; Fy_rear ~ -C*beta
    beta = np.arctan2(s.vy, max(s.vx, 1e-3))
    Cf = Fy_max_front / max(abs(steer)*1.0 + 1e-3, 1.0) # heuristique
    Cr = Fy_max_rear / max(abs(beta)*5.0 + 1e-3, 1.0)
    Fy_front = np.clip(steer * Cf, -Fy_max_front, Fy_max_front)
    Fy_rear = np.clip(-beta * Cr, -Fy_max_rear, Fy_max_rear)

    # Équations de mouvement 2D (plan)
    ax = (Fx_long - Fy_front * np.sin(steer)) / spec.mass + s.vy * s.yaw_rate
    ay = (Fy_front * np.cos(steer) + Fy_rear) / spec.mass - s.vx * s.yaw_rate
    yaw_acc = (Fy_front * 1.2 - Fy_rear * 1.4) / max(spec.moi[2],1e-3) # bras de levier heuristiques

    # Intégration Euler
    vx = s.vx + dt * ax
    vy = s.vy + dt * ay
    yaw_rate = s.yaw_rate + dt * yaw_acc
    yaw = s.yaw + dt * yaw_rate
    x = s.x + dt * (s.vx * np.cos(s.yaw) - s.vy * np.sin(s.yaw))
    y = s.y + dt * (s.vx * np.sin(s.yaw) + s.vy * np.cos(s.yaw))

    # Mise à jour RPM/gear (boîte auto simple)
    upshift_rpm = 0.92 * spec.rpm_limiter
    downshift_rpm = 2000.0
    new_gear = gear
    if rpm > upshift_rpm and gear < len(spec.gear_ratios):
        new_gear += 1
    elif rpm < downshift_rpm and gear > 1:
        new_gear -= 1

    # Safety checks to prevent extreme values
    x = np.clip(x, -1e6, 1e6)
    y = np.clip(y, -1e6, 1e6)
    vx = np.clip(vx, -200.0, 200.0)  # Limit speed to 200 m/s (720 km/h)
    vy = np.clip(vy, -200.0, 200.0)
    yaw_rate = np.clip(yaw_rate, -20.0, 20.0)  # Limit angular velocity
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))  # Normalize angle
    rpm = np.clip(rpm, 500.0, spec.rpm_limiter * 1.1)
    
    # Check for NaN/inf values
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(vx) and np.isfinite(vy) and 
            np.isfinite(yaw) and np.isfinite(yaw_rate) and np.isfinite(rpm)):
        # Reset to safe values if any NaN/inf detected
        print("Warning: NaN/inf detected in vehicle state, resetting to safe values")
        return VehicleState(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2, 1500.0)
    
    return VehicleState(x, y, yaw, vx, vy, yaw_rate, new_gear, rpm)


def get_steering_info(spec: VehicleSpec, speed: float) -> dict:
    """Get detailed steering information for a given speed."""
    max_angle = get_max_steering_angle(spec, speed)
    max_angle_deg = np.degrees(max_angle)
    reduction_factor = max_angle / spec.max_steering_angle
    
    # Calculate turn radius at this speed and max steering
    if abs(max_angle) > 1e-6:
        turn_radius = spec.wheelbase / np.tan(abs(max_angle))
    else:
        turn_radius = float('inf')
    
    return {
        'speed_ms': speed,
        'speed_kmh': speed * 3.6,
        'max_steering_angle_rad': max_angle,
        'max_steering_angle_deg': max_angle_deg,
        'reduction_factor': reduction_factor,
        'turn_radius_m': turn_radius
    }


def get_wheel_positions(spec: VehicleSpec, state: VehicleState) -> np.ndarray:
    """
    Calculate the positions of all 4 wheels based on vehicle state.
    Returns array of shape (4, 2) with [x, y] positions for [FL, FR, RL, RR] wheels.
    """
    # Vehicle center position
    cx, cy = state.x, state.y
    yaw = state.yaw
    
    # Half dimensions
    half_wheelbase = spec.wheelbase / 2.0
    half_track = spec.track_width / 2.0
    
    # Calculate wheel positions in vehicle frame, then transform to global frame
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    
    # Front left wheel
    fl_x = cx + cos_yaw * half_wheelbase - sin_yaw * half_track
    fl_y = cy + sin_yaw * half_wheelbase + cos_yaw * half_track
    
    # Front right wheel  
    fr_x = cx + cos_yaw * half_wheelbase + sin_yaw * half_track
    fr_y = cy + sin_yaw * half_wheelbase - cos_yaw * half_track
    
    # Rear left wheel
    rl_x = cx - cos_yaw * half_wheelbase - sin_yaw * half_track
    rl_y = cy - sin_yaw * half_wheelbase + cos_yaw * half_track
    
    # Rear right wheel
    rr_x = cx - cos_yaw * half_wheelbase + sin_yaw * half_track
    rr_y = cy - sin_yaw * half_wheelbase - cos_yaw * half_track
    
    return np.array([[fl_x, fl_y], [fr_x, fr_y], [rl_x, rl_y], [rr_x, rr_y]])