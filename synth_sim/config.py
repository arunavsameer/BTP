"""Hyperparameters, seed pools, and domain-randomization ranges.

Coordinate convention (right-handed world frame):
    +X right (lateral), +Y up (elevation), +Z forward (along-street).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class CameraIntrinsicsConfig:
    width: int = 1280
    height: int = 720
    fov_x_deg: float = 90.0
    near: float = 0.1
    far: float = 100.0

    @property
    def aspect(self) -> float:
        return float(self.width) / float(self.height)

    @property
    def fx(self) -> float:
        return self.width / (2.0 * math.tan(math.radians(self.fov_x_deg) * 0.5))

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.width * 0.5

    @property
    def cy(self) -> float:
        return self.height * 0.5

    @property
    def fov_y_rad(self) -> float:
        """Vertical FOV matching the pinhole model with square pixels."""
        return 2.0 * math.atan(math.tan(math.radians(self.fov_x_deg) * 0.5) / self.aspect)


@dataclass(frozen=True)
class GaitConfig:
    """Inverted-pendulum gait and vestibular / cervical perturbation amplitudes."""

    v0_min: float = 1.1
    v0_max: float = 1.5
    alpha_v: float = 0.08
    f_step: float = 1.8
    h_nominal_mean: float = 1.60
    h_nominal_std: float = 0.10
    a_heave: float = 0.045
    a_sway: float = 0.035
    a_pitch_deg: float = 2.5
    a_yaw_deg: float = 4.0
    a_roll_deg: float = 1.5
    pitch_explore_deg: float = 6.0
    yaw_explore_deg: float = 18.0
    roll_explore_deg: float = 4.0
    beta_pitch: float = 3.0
    beta_yaw: float = 8.0
    beta_roll: float = 1.5
    noise_heave: float = 0.006
    noise_sway: float = 0.005

    @property
    def f_stride(self) -> float:
        return 0.5 * self.f_step


@dataclass(frozen=True)
class StreetConfig:
    road_width_min: float = 7.0
    road_width_max: float = 12.0
    curb_width: float = 0.25
    curb_height: float = 0.15
    sidewalk_width_min: float = 2.0
    sidewalk_width_max: float = 4.0
    street_length: float = 160.0
    spline_resolution_m: float = 0.45
    n_control_min: int = 8
    n_control_max: int = 12
    lateral_wander: float = 3.5


@dataclass(frozen=True)
class PoissonRadii:
    dustbin: float = 1.2
    bollard: float = 1.2
    scooter: float = 1.8
    bicycle: float = 1.8
    pothole: float = 2.0
    tree: float = 3.5
    building_gap: float = 2.5


@dataclass(frozen=True)
class ThreatThresholds:
    static_speed: float = 0.15
    safe_static_cpa: float = 1.2
    safe_dynamic_ttc: float = 4.0
    safe_dynamic_cpa: float = 1.8
    near_miss_ttc_lo: float = 1.8
    near_miss_ttc_hi: float = 3.5
    near_miss_cpa_lo: float = 0.4
    near_miss_cpa_hi: float = 1.2
    critical_ttc: float = 2.0
    critical_cpa: float = 0.45
    critical_offset: Tuple[float, float] = (0.0, 0.35)
    near_miss_offset: Tuple[float, float] = (0.6, 1.2)


@dataclass(frozen=True)
class PopulationConfig:
    """How many moving agents share the street with the camera wearer."""

    ped_walk_min: int = 9
    ped_walk_max: int = 16
    ped_stand_min: int = 2
    ped_stand_max: int = 5
    ped_pair_min: int = 1
    ped_pair_max: int = 3
    ped_cross_min: int = 3
    ped_cross_max: int = 6
    veh_moving_min: int = 6
    veh_moving_max: int = 10
    veh_parked_min: int = 4
    veh_parked_max: int = 7
    cyclist_min: int = 1
    cyclist_max: int = 3
    scooter_moving: int = 2


@dataclass(frozen=True)
class KinematicEnvelopes:
    pedestrian: Tuple[float, float] = (0.8, 2.8)
    vehicle: Tuple[float, float] = (5.0, 18.0)
    cyclist: Tuple[float, float] = (3.0, 8.0)
    scooter: Tuple[float, float] = (3.0, 8.0)


@dataclass(frozen=True)
class DomainRandomization:
    sun_elevation_deg: Tuple[float, float] = (5.0, 85.0)
    sun_azimuth_deg: Tuple[float, float] = (0.0, 360.0)
    sun_kelvin: Tuple[float, float] = (2500.0, 7500.0)
    ambient_sky: Tuple[float, float] = (0.1, 0.6)
    wetness: Tuple[float, float] = (0.05, 0.85)
    fog_density: Tuple[float, float] = (0.0012, 0.0045)


@dataclass
class SimulationConfig:
    """Top-level deterministic generator configuration."""

    camera: CameraIntrinsicsConfig = field(default_factory=CameraIntrinsicsConfig)
    gait: GaitConfig = field(default_factory=GaitConfig)
    street: StreetConfig = field(default_factory=StreetConfig)
    poisson: PoissonRadii = field(default_factory=PoissonRadii)
    threat: ThreatThresholds = field(default_factory=ThreatThresholds)
    envelopes: KinematicEnvelopes = field(default_factory=KinematicEnvelopes)
    domain: DomainRandomization = field(default_factory=DomainRandomization)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    fps: int = 30
    frames_per_episode: int = 600
    min_episode_seconds: float = 15.0
    default_episode_seconds: float = 20.0
    seed: int = 20260910
    class_ratios: Dict[str, float] = field(
        default_factory=lambda: {
            "SAFE": 0.35,
            "NEAR_MISS": 0.30,
            "CRITICAL_THREAT": 0.35,
        }
    )
    occlusion_grid: int = 7
    occlusion_flag_threshold: float = 0.25
    writer_workers: int = 4
    output_mode: str = "both"
    shadow_resolution: int = 2048
    generator_name: str = "EgocentricAccessibilitySim-Scratch"
    generator_version: str = "2.3.0"

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)

    @property
    def episode_duration(self) -> float:
        return self.frames_per_episode * self.dt


CLASS_NAME_IDS: Dict[str, int] = {
    "background": 0,
    "road": 1,
    "sidewalk": 2,
    "curb": 3,
    "building": 4,
    "vehicle": 5,
    "pedestrian": 6,
    "cyclist": 7,
    "scooter": 8,
    "dustbin": 9,
    "bollard": 10,
    "pothole": 11,
    "head_obstacle": 12,
    "sign": 13,
    "awning": 14,
    "tree": 15,
}

DYNAMIC_CLASSES = frozenset({"vehicle", "pedestrian", "cyclist", "scooter"})
HAZARD_CLASSES = frozenset(
    {"dustbin", "bollard", "pothole", "head_obstacle", "sign", "awning", "tree", "scooter"}
)

PED_SPEED = (0.8, 2.8)
VEH_SPEED = (5.0, 18.0)
CYC_SPEED = (3.0, 8.0)
