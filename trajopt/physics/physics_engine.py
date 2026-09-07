from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from schemas import validate_vehicle_data

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
    def from_config(cfg: dict) -> VehicleSpec:
        """Create VehicleSpec from configuration dictionary.

        Args:
            cfg: Vehicle configuration dictionary (from JSON, YAML, or Hydra).

        Returns:
            Validated VehicleSpec instance.

        Raises:
            pydantic.ValidationError: If configuration fails validation.
        """
        # Validate configuration with Pydantic schema
        validated = validate_vehicle_data(cfg)

        ch = validated.chassis
        pw = validated.powertrain
        st = validated.suspension_tires
        br = validated.brakes

        return VehicleSpec(
            mass=ch.masse_totale,
            cg=np.array(ch.centre_de_gravite, dtype=float),
            moi=np.array(ch.moment_inertie, dtype=float),
            Cx=ch.coefficient_trainee,
            Cz_front=ch.coefficient_portance.front,
            Cz_rear=ch.coefficient_portance.rear,
            mass_split_front=ch.repartition_masses.front,
            wheelbase=ch.empattement,
            track_width=ch.voie,
            max_steering_angle=ch.angle_braquage_max,
            steering_speed_factor=ch.facteur_vitesse_braquage,
            min_turn_radius=ch.rayon_braquage_min,
            torque_curve=np.array(pw.courbe_couple_moteur, dtype=float),
            rpm_limiter=pw.limiteur_rpm,
            gear_ratios=np.array(pw.rapports_boite_de_vitesse, dtype=float),
            final_drive=pw.rapport_pont_final,
            driveline_eff=pw.efficacite_transmission,
            k_spring_front=st.raideur_suspension.front,
            k_spring_rear=st.raideur_suspension.rear,
            camber=st.geometrie_pneus.carrossage,
            toe=st.geometrie_pneus.pincement,
            mu0=st.modele_pneu_adherence.mu0,
            alpha_muFz=st.modele_pneu_adherence.alpha,
            brake_torque_max=br.couple_freinage_max,
            brake_split_front=br.repartition_freinage.front,
        )


# Below this speed the wheels are treated as stopped, so the brake holds the
# car rather than reversing it.
BRAKE_DEADBAND_MPS = 0.1

# How fast the usable turn radius grows with speed, per m/s.
STEERING_RADIUS_GAIN = 0.05


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


def aero_forces(
    spec: VehicleSpec,
    v: float,
    rho_air: float = 1.225,
    area: float = 2.0,
) -> tuple[float, float]:
    """
    Drag ~ 0.5*rho*Cx*A*v^2 ; Downforce (front+rear) ~ 0.5*rho*Cz*A*v^2
    """
    drag = 0.5 * rho_air * spec.Cx * area * v**2
    downforce = 0.5 * rho_air * (abs(spec.Cz_front) + abs(spec.Cz_rear)) * area * v**2
    return drag, downforce


def tire_mu(spec: VehicleSpec, Fz: float, Fz_ref: float = 4000.0) -> float:
    return spec.mu0 * (1.0 + spec.alpha_muFz * (Fz - Fz_ref) / max(Fz_ref, 1.0))


def _apply_friction_circle(
    Fx: float, Fy: float, grip_budget: float
) -> tuple[float, float]:
    """Scale an axle's force vector back onto its friction circle.

    Args:
        Fx: Longitudinal force at the axle (N).
        Fy: Lateral force at the axle (N).
        grip_budget: Maximum force magnitude the axle can transmit, mu * Fz (N).

    Returns:
        (Fx, Fy) scaled so that sqrt(Fx^2 + Fy^2) <= grip_budget, preserving
        the direction of the demand.
    """
    magnitude = float(np.hypot(Fx, Fy))
    if magnitude <= grip_budget or magnitude < 1e-9:
        return Fx, Fy
    scale = grip_budget / magnitude
    return Fx * scale, Fy * scale


def get_max_speed_for_gear(
    spec: VehicleSpec,
    gear: int,
    wheel_radius: float = 0.33,
    absolute_top_speed: float = 88.0,
) -> float:
    """
    Calculate maximum realistic speed for a given gear based on RPM limiter.

    Args:
        spec: Vehicle specification with gear ratios and RPM limiter
        gear: Current gear (1-based)
        wheel_radius: Wheel radius in meters
        absolute_top_speed: Absolute maximum speed limit in m/s (default 88 m/s = 317 km/h)

    Returns:
        Maximum speed in m/s for this gear, capped at absolute_top_speed
    """
    if gear < 1 or gear > len(spec.gear_ratios):
        return min(50.0, absolute_top_speed)  # Fallback speed limit

    # Get gear ratio and final drive
    gear_ratio = spec.gear_ratios[gear-1]
    total_ratio = gear_ratio * spec.final_drive

    # Calculate max wheel speed from RPM limiter
    # RPM → rad/s → wheel speed → vehicle speed
    max_wheel_omega = spec.rpm_limiter / 9.5493  # RPM to rad/s
    max_wheel_speed = max_wheel_omega / total_ratio  # Account for gear reduction
    max_vehicle_speed = max_wheel_speed * wheel_radius  # Linear speed

    # ENFORCE ABSOLUTE TOP SPEED LIMIT: No gear can exceed 88 m/s
    return min(max_vehicle_speed, absolute_top_speed)


def get_gear_speed_limits(
    spec: VehicleSpec,
    wheel_radius: float = 0.33,
    absolute_top_speed: float = 88.0,
) -> np.ndarray:
    """
    Calculate speed limits for all gears.

    Args:
        spec: Vehicle specification
        wheel_radius: Wheel radius in meters
        absolute_top_speed: Absolute maximum speed limit in m/s

    Returns:
        Array of max speeds for each gear [gear1_max, gear2_max, ...]
    """
    return np.array([get_max_speed_for_gear(spec, gear+1, wheel_radius, absolute_top_speed)
                     for gear in range(len(spec.gear_ratios))])


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
    # Two continuous, monotonically decreasing limits; their minimum is
    # therefore also continuous and monotonically decreasing.
    #
    # This used to branch on `speed > 5.0`, applying the turn-radius limit
    # only above that point and an extra low_speed_factor below it. The result
    # was non-monotonic and discontinuous: 27.50 deg at 0 m/s, rising to a
    # local maximum of 31.25 deg at 5 m/s, then dropping 38% to 19.46 deg at
    # 10 m/s. reset() starts the vehicle at exactly 5.0 m/s, so the same
    # steering command produced dynamics differing by 38% either side of the
    # jump, at the start of every single episode.
    speed = max(speed, 0.0)

    # Steering-system limit: full lock at rest, progressively less with speed
    system_limit = spec.max_steering_angle / (1.0 + spec.steering_speed_factor * speed)

    # Turn-radius limit, bicycle model: tan(delta) = wheelbase / radius.
    # The usable radius grows with speed, so the angle shrinks.
    radius_limit = np.arctan(
        spec.wheelbase / (spec.min_turn_radius * (1.0 + STEERING_RADIUS_GAIN * speed))
    )

    return float(min(system_limit, radius_limit))


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

    # Freinage. Brakes dissipate energy, so the force always OPPOSES motion --
    # it was previously subtracted unconditionally, which made braking at low
    # speed push the car backwards (measured: braking from 60 m/s drove vx to
    # 0.34 m/s and then through zero into reverse).
    brake_torque = brake * spec.brake_torque_max
    Fx_brake_capacity = brake_torque / max(wheel_radius, 1e-3)

    # Cap the brake impulse at what brings the car exactly to rest over this
    # step. Without it a deadband alone cannot help: full brakes give 18.8
    # m/s^2, so a 50 ms step jumps straight from +0.5 to -0.44 m/s and steps
    # over any deadband narrow enough to be meaningful.
    Fx_brake_capacity = min(Fx_brake_capacity, abs(s.vx) * spec.mass / dt)

    if abs(s.vx) > BRAKE_DEADBAND_MPS:
        Fx_brake = -np.sign(s.vx) * Fx_brake_capacity
    else:
        # At a standstill the brake can hold the car against the driveline,
        # but not accelerate it in either direction.
        Fx_brake = -np.clip(Fx_driven, -Fx_brake_capacity, Fx_brake_capacity)


    # CG-to-axle distances from wheelbase and mass split
    lf = spec.wheelbase * (1.0 - spec.mass_split_front)  # CG to front axle
    lr = spec.wheelbase * spec.mass_split_front           # CG to rear axle

    # Tire slip angles (bicycle model), standard SAE convention:
    #   alpha = atan(v_lat_at_axle / v_long) - steer
    # so that a positive slip angle produces a NEGATIVE lateral force
    # (Fy = -Ca * alpha) and the yaw mode is damped.
    #
    # These were previously written with the opposite sign on both axles, which
    # negates Fy_front and Fy_rear while leaving the kinematic coupling terms
    # (-vx*yaw_rate in ay) untouched. That flips the sign of the yaw damping
    # coefficient and makes the yaw mode exponentially DIVERGENT: with zero
    # steering, a 0.01 rad/s perturbation doubled every ~73 ms and ran into the
    # +/-20 rad/s safety clamp. The vehicle was not controllable by any policy.
    vx_safe = max(abs(s.vx), 1.0)
    alpha_f = np.arctan2(s.vy + lf * s.yaw_rate, vx_safe) - steer
    alpha_r = np.arctan2(s.vy - lr * s.yaw_rate, vx_safe)

    # Linear cornering stiffness with saturation at Fy_max.
    # Peak grip reached at ~7 deg slip angle (typical racing tire).
    ALPHA_PEAK = 0.12  # rad
    Ca_f = Fy_max_front / ALPHA_PEAK
    Ca_r = Fy_max_rear / ALPHA_PEAK
    Fy_front = np.clip(-Ca_f * alpha_f, -Fy_max_front, Fy_max_front)
    Fy_rear = np.clip(-Ca_r * alpha_r, -Fy_max_rear, Fy_max_rear)

    # --- Friction circle -----------------------------------------------------
    # A tire has ONE grip budget shared between longitudinal and lateral force:
    # sqrt(Fx^2 + Fy^2) <= mu * Fz. Fx and Fy were computed independently, so
    # the model could brake at 2.02 g on a mu of 1.6 while simultaneously
    # generating full cornering force. Every property of a racing line --
    # braking in a straight line, the apex, progressive reacceleration --
    # follows from this constraint, so without it the optimal policy for this
    # simulator was "full throttle everywhere while steering", which has
    # nothing to do with driving.
    #
    # Longitudinal force is split per axle: brakes by their designed bias,
    # drive torque by mass distribution. The latter is an approximation --
    # the spec carries no drive-layout parameter -- but it keeps the budget
    # honest at both ends. Aerodynamic drag acts on the body, not through the
    # contact patch, so it does not consume tire grip.
    Fx_front = Fx_driven * spec.mass_split_front + Fx_brake * spec.brake_split_front
    Fx_rear = (
        Fx_driven * (1.0 - spec.mass_split_front)
        + Fx_brake * (1.0 - spec.brake_split_front)
    )

    Fx_front, Fy_front = _apply_friction_circle(Fx_front, Fy_front, mu_f * Fz_front)
    Fx_rear, Fy_rear = _apply_friction_circle(Fx_rear, Fy_rear, mu_r * Fz_rear)

    Fx_long = Fx_front + Fx_rear - np.sign(s.vx) * drag

    # Equations of motion (planar bicycle model)
    ax = (Fx_long - Fy_front * np.sin(steer)) / spec.mass + s.vy * s.yaw_rate
    ay = (Fy_front * np.cos(steer) + Fy_rear) / spec.mass - s.vx * s.yaw_rate
    yaw_acc = (lf * Fy_front * np.cos(steer) - lr * Fy_rear) / max(spec.moi[2], 1e-3)

    # Semi-implicit Euler: update velocities first, then positions with new velocities
    vx = s.vx + dt * ax
    vy = s.vy + dt * ay
    yaw_rate = s.yaw_rate + dt * yaw_acc
    yaw = s.yaw + dt * yaw_rate
    x = s.x + dt * (vx * np.cos(yaw) - vy * np.sin(yaw))
    y = s.y + dt * (vx * np.sin(yaw) + vy * np.cos(yaw))

    # Auto gearbox
    upshift_rpm = 0.92 * spec.rpm_limiter
    downshift_rpm = 2000.0
    new_gear = gear
    if rpm > upshift_rpm and gear < len(spec.gear_ratios):
        new_gear += 1
    elif rpm < downshift_rpm and gear > 1:
        new_gear -= 1

    # Gear-based speed limiting (88 m/s absolute cap)
    current_gear = int(np.clip(new_gear, 1, len(spec.gear_ratios)))
    max_speed_current_gear = get_max_speed_for_gear(
        spec, current_gear, wheel_radius, absolute_top_speed=88.0,
    )
    speed_limit = max_speed_current_gear * 1.05
    current_speed = np.hypot(vx, vy)

    if current_speed > speed_limit:
        scale_factor = speed_limit / current_speed
        vx *= scale_factor
        vy *= scale_factor

        wheel_omega = (vx / max(wheel_radius, 1e-3)) if current_speed > 0.1 else 0.0
        ratio = spec.gear_ratios[current_gear-1] * spec.final_drive
        rpm = np.clip(wheel_omega * ratio * 9.5493, 800.0, spec.rpm_limiter)

    # Safety clamps
    x = np.clip(x, -1e6, 1e6)
    y = np.clip(y, -1e6, 1e6)
    vy = np.clip(vy, -50.0, 50.0)
    yaw_rate = np.clip(yaw_rate, -20.0, 20.0)
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    rpm = np.clip(rpm, 500.0, spec.rpm_limiter * 1.1)

    # NaN/inf guard: return previous state unchanged instead of teleporting
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(vx) and np.isfinite(vy) and
            np.isfinite(yaw) and np.isfinite(yaw_rate) and np.isfinite(rpm)):
        return s

    return VehicleState(x, y, yaw, vx, vy, yaw_rate, new_gear, rpm)


def get_gear_speed_info(
    spec: VehicleSpec,
    wheel_radius: float = 0.33,
    absolute_top_speed: float = 88.0,
) -> dict:
    """
    Get detailed gear and speed information for vehicle analysis.

    Args:
        spec: Vehicle specification
        wheel_radius: Wheel radius in meters
        absolute_top_speed: Absolute maximum speed limit in m/s

    Returns:
        Dictionary with gear speeds, RPM limits, and other drivetrain info
    """
    speed_limits = get_gear_speed_limits(spec, wheel_radius, absolute_top_speed)

    gear_info = {}
    for gear in range(1, len(spec.gear_ratios) + 1):
        max_speed_ms = speed_limits[gear-1]
        max_speed_kmh = max_speed_ms * 3.6
        gear_ratio = spec.gear_ratios[gear-1]
        total_ratio = gear_ratio * spec.final_drive

        gear_info[f'gear_{gear}'] = {
            'max_speed_ms': max_speed_ms,
            'max_speed_kmh': max_speed_kmh,
            'gear_ratio': gear_ratio,
            'total_ratio': total_ratio,
            'rpm_at_max_speed': spec.rpm_limiter
        }

    return {
        'gear_speeds': gear_info,
        'rpm_limiter': spec.rpm_limiter,
        'final_drive': spec.final_drive,
        'wheel_radius': wheel_radius,
        'absolute_top_speed_ms': absolute_top_speed,
        'absolute_top_speed_kmh': absolute_top_speed * 3.6,
        'top_speed_ms': min(max(speed_limits), absolute_top_speed),
        'top_speed_kmh': min(max(speed_limits), absolute_top_speed) * 3.6
    }


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
