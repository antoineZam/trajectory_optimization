"""
Pydantic schemas for data validation.

This module defines strict schemas for Track and Vehicle configuration data,
ensuring type safety and validation before execution.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# Track Schema
# =============================================================================


class TrackSchema(BaseModel):
    """Schema for track configuration data."""

    name: str = Field(..., min_length=1, description="Track name identifier")
    width: Annotated[float, Field(gt=0, description="Track width in meters")]
    centerline: list[tuple[float, float]] = Field(
        ..., min_length=3, description="List of (x, y) coordinates defining the track centerline"
    )

    @field_validator("centerline", mode="before")
    @classmethod
    def validate_centerline(cls, v):
        """Ensure centerline points are valid coordinate pairs."""
        if not isinstance(v, list):
            raise ValueError("centerline must be a list")

        validated = []
        for i, point in enumerate(v):
            if not isinstance(point, list | tuple) or len(point) != 2:
                raise ValueError(f"centerline[{i}] must be a pair of coordinates, got {point}")
            try:
                validated.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError) as e:
                raise ValueError(f"centerline[{i}] contains non-numeric values: {point}") from e
        return validated

    model_config = {"extra": "forbid"}


# =============================================================================
# Vehicle Schema - Nested Components
# =============================================================================


class LiftCoefficients(BaseModel):
    """Front and rear lift/downforce coefficients."""

    front: float = Field(..., description="Front downforce coefficient (negative = downforce)")
    rear: float = Field(..., description="Rear downforce coefficient (negative = downforce)")

    model_config = {"extra": "forbid"}


class MassDistribution(BaseModel):
    """Front/rear mass distribution."""

    front: Annotated[float, Field(ge=0, le=1, description="Front mass ratio (0-1)")]
    rear: Annotated[float, Field(ge=0, le=1, description="Rear mass ratio (0-1)")]

    @model_validator(mode="after")
    def validate_sum(self):
        """Ensure front + rear ≈ 1.0."""
        total = self.front + self.rear
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Mass distribution must sum to 1.0, got {total}")
        return self

    model_config = {"extra": "forbid"}


class ChassisSchema(BaseModel):
    """Vehicle chassis configuration."""

    masse_totale: Annotated[float, Field(gt=0, description="Total vehicle mass in kg")]
    centre_de_gravite: tuple[float, float, float] = Field(
        ..., description="Center of gravity (x, y, z) in meters"
    )
    moment_inertie: tuple[float, float, float] = Field(
        ..., description="Moments of inertia (Ix, Iy, Iz) in kg·m²"
    )
    coefficient_trainee: Annotated[float, Field(ge=0, description="Drag coefficient (Cx)")]
    coefficient_portance: LiftCoefficients
    repartition_masses: MassDistribution
    empattement: Annotated[float, Field(gt=0, description="Wheelbase in meters")]
    voie: Annotated[float, Field(gt=0, description="Track width in meters")]
    angle_braquage_max: Annotated[
        float, Field(gt=0, le=1.5, description="Max steering angle in radians")
    ]
    facteur_vitesse_braquage: Annotated[
        float, Field(ge=0, description="Speed-dependent steering reduction factor")
    ]
    rayon_braquage_min: Annotated[float, Field(gt=0, description="Minimum turn radius in meters")]

    @field_validator("centre_de_gravite", "moment_inertie", mode="before")
    @classmethod
    def validate_tuple3(cls, v):
        """Convert list to tuple and validate."""
        if isinstance(v, list):
            if len(v) != 3:
                raise ValueError(f"Expected 3 values, got {len(v)}")
            return tuple(float(x) for x in v)
        return v

    model_config = {"extra": "forbid"}


class PowertrainSchema(BaseModel):
    """Vehicle powertrain configuration."""

    courbe_couple_moteur: list[tuple[float, float]] = Field(
        ..., min_length=2, description="Torque curve as [(RPM, Torque_Nm), ...]"
    )
    limiteur_rpm: Annotated[float, Field(gt=0, description="RPM limiter")]
    rapports_boite_de_vitesse: list[float] = Field(
        ..., min_length=1, description="Gear ratios (1st, 2nd, ...)"
    )
    rapport_pont_final: Annotated[float, Field(gt=0, description="Final drive ratio")]
    efficacite_transmission: Annotated[
        float, Field(gt=0, le=1, description="Driveline efficiency (0-1)")
    ]

    @field_validator("courbe_couple_moteur", mode="before")
    @classmethod
    def validate_torque_curve(cls, v):
        """Validate torque curve format."""
        if not isinstance(v, list):
            raise ValueError("courbe_couple_moteur must be a list")

        validated = []
        for i, point in enumerate(v):
            if not isinstance(point, list | tuple) or len(point) != 2:
                raise ValueError(f"courbe_couple_moteur[{i}] must be (RPM, Torque) pair")
            rpm, torque = float(point[0]), float(point[1])
            if rpm < 0:
                raise ValueError(f"RPM must be positive, got {rpm}")
            if torque < 0:
                raise ValueError(f"Torque must be positive, got {torque}")
            validated.append((rpm, torque))
        return validated

    @field_validator("rapports_boite_de_vitesse", mode="before")
    @classmethod
    def validate_gear_ratios(cls, v):
        """Ensure gear ratios are positive and in descending order."""
        if not isinstance(v, list):
            raise ValueError("rapports_boite_de_vitesse must be a list")
        ratios = [float(r) for r in v]
        if any(r <= 0 for r in ratios):
            raise ValueError("All gear ratios must be positive")
        return ratios

    model_config = {"extra": "forbid"}


class SuspensionStiffness(BaseModel):
    """Front/rear suspension stiffness."""

    front: Annotated[float, Field(gt=0, description="Front spring rate in N/m")]
    rear: Annotated[float, Field(gt=0, description="Rear spring rate in N/m")]

    model_config = {"extra": "forbid"}


class TireGeometry(BaseModel):
    """Tire alignment geometry."""

    carrossage: float = Field(..., description="Camber angle in degrees")
    pincement: float = Field(..., description="Toe angle in degrees")

    model_config = {"extra": "forbid"}


class TireGripModel(BaseModel):
    """Tire grip model parameters."""

    mu0: Annotated[float, Field(gt=0, description="Base friction coefficient")]
    alpha: float = Field(..., description="Load sensitivity coefficient")

    model_config = {"extra": "forbid"}


class SuspensionTiresSchema(BaseModel):
    """Suspension and tire configuration."""

    type_suspension: str = Field(..., description="Suspension type description")
    raideur_suspension: SuspensionStiffness
    geometrie_pneus: TireGeometry
    modele_pneu_adherence: TireGripModel

    model_config = {"extra": "forbid"}


class BrakeDistribution(BaseModel):
    """Front/rear brake force distribution."""

    front: Annotated[float, Field(ge=0, le=1, description="Front brake ratio (0-1)")]
    rear: Annotated[float, Field(ge=0, le=1, description="Rear brake ratio (0-1)")]

    @model_validator(mode="after")
    def validate_sum(self):
        """Ensure front + rear ≈ 1.0."""
        total = self.front + self.rear
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Brake distribution must sum to 1.0, got {total}")
        return self

    model_config = {"extra": "forbid"}


class BrakesSchema(BaseModel):
    """Brake system configuration."""

    couple_freinage_max: Annotated[float, Field(gt=0, description="Maximum brake torque in N·m")]
    repartition_freinage: BrakeDistribution

    model_config = {"extra": "forbid"}


# =============================================================================
# Complete Vehicle Schema
# =============================================================================


class VehicleSchema(BaseModel):
    """Complete vehicle configuration schema."""

    chassis: ChassisSchema
    powertrain: PowertrainSchema
    suspension_tires: SuspensionTiresSchema
    brakes: BrakesSchema

    model_config = {"extra": "forbid"}


# =============================================================================
# Utility Functions
# =============================================================================


def validate_track_data(data: dict) -> TrackSchema:
    """Validate track JSON data and return a TrackSchema instance.

    Args:
        data: Raw dictionary from JSON file.

    Returns:
        Validated TrackSchema instance.

    Raises:
        pydantic.ValidationError: If validation fails.
    """
    return TrackSchema.model_validate(data)


def validate_vehicle_data(data: dict) -> VehicleSchema:
    """Validate vehicle JSON/YAML data and return a VehicleSchema instance.

    Args:
        data: Raw dictionary from JSON/YAML file or Hydra config.

    Returns:
        Validated VehicleSchema instance.

    Raises:
        pydantic.ValidationError: If validation fails.
    """
    return VehicleSchema.model_validate(data)

