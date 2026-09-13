"""Hyperparameters for the egocentric pedestrian synthetic-data pipeline.

All SI units unless noted. Blender is Z-up internally; exported JSON uses the
spec's Y-up convention (X right, Y height, Z forward) via threat_math converters.
"""

from __future__ import annotations

import copy
import math
from typing import Any


CONFIG: dict[str, Any] = {
    # -------------------------------------------------------------------------
    # Render / sensor
    # -------------------------------------------------------------------------
    "render": {
        # Blender 4.2–4.5: BLENDER_EEVEE_NEXT. Blender 5.x: BLENDER_EEVEE.
        # configure_eevee() picks whichever the running binary actually exposes.
        "engine": "BLENDER_EEVEE",
        "resolution_x": 1920,
        "resolution_y": 1080,
        "fps": 30,
        "frames_per_episode": 150,  # 5.0 s at 30 fps
        "filepath_format": "PNG",
        "color_depth": "8",
        "film_transparent": False,
        # EEVEE knobs. TAA 16 + reprojection matches 32 samples on this
        # low-poly lighting; raytracing stays on so car paint / windows match.
        "taa_render_samples": 16,
        "use_raytracing": True,
        "use_shadows": True,
        "use_volumetric_shadows": False,
        "fast_gi_ray_count": 4,
        "fast_gi_step_count": 6,
        "fast_gi_quality": 0.30,
        "volumetric_samples": 16,
        "volumetric_tile_size": "8",
        "volumetric_start": 0.1,
        "volumetric_end": 80.0,
        # PNG zlib 0–100. 1 is nearly as fast as uncompressed and much smaller.
        "png_compression": 1,
    },
    # Head-mounted RGB camera (full-frame equivalent).
    # Horizontal FOV = 2 atan((sensor_width/2) / lens_mm). Default 24 mm / 36 mm ≈ 73.7°.
    # Per episode, lens is drawn from hfov_deg_range unless CLI locks it.
    "camera": {
        "name": "EgoHeadCam",
        "lens_mm": 24.0,
        "sensor_width_mm": 36.0,
        "sensor_fit": "HORIZONTAL",
        "clip_start": 0.05,
        "clip_end": 120.0,
        "eye_height_m": 1.6,
        # Domain-randomized standing eye height. Seated uses ego.seated_*.
        # short ≈ child/teen / stooped adult; tall ≈ 95th-percentile adult.
        "stature_weights": {
            "short": 0.22,
            "typical": 0.56,
            "tall": 0.22,
        },
        "eye_height_m_by_stature": {
            "short": (1.35, 1.50),
            "typical": (1.55, 1.72),
            "tall": (1.75, 1.90),
        },
        "randomize_hfov": True,
        # ~35 mm (50°) through ~18 mm (90°). Inclusive.
        "hfov_deg_range": (50.0, 90.0),
    },
    # -------------------------------------------------------------------------
    # Biomechanical locomotion (Part 2)
    # -------------------------------------------------------------------------
    "gait": {
        "walk_speed_min": 0.70,  # m/s — slow stroll floor (seated is exactly 0)
        "walk_speed_max": 2.05,  # m/s — hurry / late-for-the-bus
        "amplitude_m": 0.04,  # 4 cm vertical bounce
        "frequency_hz": 1.8,  # step frequency
        # Optional small pitch bob locked to the gait cycle (radians).
        "pitch_bob_amp_rad": 0.015,
        # Per-episode pace band. Uniform draw inside the chosen band.
        "pace_weights": {
            "stroll": 0.22,
            "walk": 0.50,
            "hurry": 0.28,
        },
        "pace_speed": {
            "stroll": (0.70, 1.00),
            "walk": (1.00, 1.50),
            "hurry": (1.50, 2.05),
        },
    },
    # -------------------------------------------------------------------------
    # Ego trajectory modes (Part 2). The walker is not always a metronome on
    # a straight sidewalk: they cut across intersections, hesitate, sidestep,
    # and sometimes sit still. `seated` sets walk_speed = 0, which every
    # downstream solver is required to handle without a divide-by-zero.
    # -------------------------------------------------------------------------
    "ego": {
        "mode_weights": {
            "walk": 0.44,
            "diagonal_cross": 0.08,
            "crosswalk": 0.12,
            "erratic": 0.22,
            "seated": 0.14,
        },
        # crosswalk: full kerb-to-kerb turn; heading follows Frenet motion.
        "crosswalk_target_frac": (0.88, 1.00),
        "crosswalk_span": (0.22, 0.78),
        # Bench / kerb height rather than standing eye height.
        "seated_eye_height_m": (0.95, 1.28),
        # Erratic: fBm lateral sidestep + speed modulation, plus a chance of
        # a full stop partway through the episode.
        "sidestep_amp_m": (0.22, 0.80),
        "sidestep_rate_hz": (0.10, 0.38),
        "speed_wobble": (0.15, 0.55),
        "hesitate_prob": 0.55,
        "hesitate_window_s": (0.9, 2.6),
        "hesitate_duration_s": (0.5, 1.6),
        # Diagonal: fraction of the way to the opposite kerb (1.0 = all the
        # way across), and when the crossing starts / ends as a fraction of
        # the episode.
        "diagonal_target_frac": (0.45, 1.00),
        "diagonal_span": (0.12, 0.85),
    },
    # 1-D fractal Perlin (fBm) applied to local camera Euler angles.
    "jitter": {
        "yaw": {
            "amplitude_deg": 15.0,
            "frequency_hz": 0.22,  # slow environmental scanning
            "octaves": 4,
            "persistence": 0.5,
            "lacunarity": 2.0,
        },
        "pitch": {
            "amplitude_deg": 5.0,
            "frequency_hz": 0.55,  # medium-frequency nod
            "octaves": 3,
            "persistence": 0.45,
            "lacunarity": 2.15,
        },
        "roll": {
            "amplitude_deg": 2.0,
            "frequency_hz": 1.9,  # heel-strike / pavement impact
            "octaves": 2,
            "persistence": 0.35,
            "lacunarity": 2.4,
        },
    },
    # -------------------------------------------------------------------------
    # Procedural street
    # -------------------------------------------------------------------------
    "world": {
        "path_length_min": 48.0,
        "path_length_max": 78.0,
        "path_types": ("straight", "gentle_curve", "s_curve", "corner_90"),
        "road_width": 7.0,
        "lane_offset": 1.75,
        "sidewalk_width": 2.4,
        "curb_height": 0.12,
        "sample_ds": 0.40,
        "building_depth": (4.0, 10.0),
        "building_width": (4.2, 14.0),
        "building_height": (6.0, 18.0),
        "building_gap": (0.4, 2.2),
        # Metres of planting strip between sidewalk outer edge and facade.
        # Trees sit in this strip only. Canopies must not reach the wall.
        "building_setback": 3.6,
        # Planted centre strip (metres). 0 = no median, no in-road trees.
        # Avenue sets this so median trees sit in grass, not on asphalt.
        "median_width": 0.0,
        # Chance the carriageway uses cobble tiles instead of asphalt.
        "cobble_prob": 0.10,
        "n_streetlamps": 6,
        "streetlamp_height": 5.6,
        "streetlamp_energy_night": 900.0,
        # Poisson-disk radii (minimum spacing) for sidewalk clutter.
        "poisson": {
            "static_radius": 3.2,
            "head_hazard_radius": 7.5,
            "k_candidates": 20,
            "n_ground_static": (3, 7),
            # Floating head-height boxes (branch / AC / sign) are off by
            # default — they read as junk on the gait line. Re-enable via
            # config if a pack specifically wants that clutter.
            "n_head_hazards": (0, 0),
        },
        "n_background_pedestrians": (3, 6),
        "n_background_vehicles": (2, 5),
        # Urban street, not a highway — cars stay in frame for several seconds.
        "vehicle_speed": (5.0, 9.0),
        "pedestrian_speed": (0.90, 1.45),
        "bicycle_speed": (3.2, 5.5),
        "cube_speed": (1.15, 2.20),
        "shape_speed": (1.15, 2.20),
        "cross_car_speed": (3.2, 4.8),
        "head_hazard_height": (1.2, 1.8),
        # Pavement outer edge minus this stays inside the facade setback.
        "corridor_margin": 0.22,
        # ---------------------------------------------------------------
        # Biomes. The road spline is the universal substrate — every biome
        # is the same Frenet corridor with different widths, ground cover,
        # and props, so injectors and annotation are unchanged.
        # ---------------------------------------------------------------
        "biome_weights": {
            "street": 0.26,
            "avenue": 0.12,
            "park": 0.16,
            "plaza": 0.12,
            "alley": 0.12,
            "residential": 0.14,
            "market": 0.08,
        },
        "biomes": {
            "street": {
                "n_trees": (6, 12),
                "n_street_trees": (0, 0),
                "n_median_trees": (0, 0),
                "median_width": 0.0,
                "cobble_prob": 0.08,
            },
            "avenue": {
                "road_width": 13.0,
                "lane_offset": 3.40,
                "sidewalk_width": 3.4,
                "building_setback": 4.2,
                "building_height": (10.0, 30.0),
                "building_width": (6.0, 16.0),
                "n_background_vehicles": (4, 9),
                "n_streetlamps": 9,
                "n_trees": (10, 18),
                "n_street_trees": (0, 0),
                "n_median_trees": (2, 4),
                "median_width": 2.6,
                "cobble_prob": 0.04,
            },
            "park": {
                # A 3 m gravel path through open grass: no kerb, no traffic.
                "road_width": 3.0,
                "lane_offset": 0.85,
                "sidewalk_width": 2.6,
                "curb_height": 0.0,
                "buildings": False,
                "lane_paint": False,
                "verge_width": 22.0,
                "n_background_vehicles": (0, 0),
                "n_background_pedestrians": (4, 9),
                "n_streetlamps": 3,
                "n_trees": (10, 26),
                "n_path_trees": (1, 4),
                "n_grass_clumps": (60, 160),
                "ground": "grass",
                "path_types": ("gentle_curve", "s_curve", "straight"),
            },
            "plaza": {
                # Wide paved open space, buildings on one side only.
                "road_width": 11.0,
                "lane_offset": 2.4,
                "sidewalk_width": 4.2,
                "curb_height": 0.02,
                "lane_paint": False,
                "building_sides": (1.0,),
                "building_setback": 3.4,
                "n_background_vehicles": (0, 1),
                "n_background_pedestrians": (5, 11),
                "n_streetlamps": 5,
                "n_trees": (4, 10),
                "n_street_trees": (0, 0),
                "n_median_trees": (0, 0),
                "median_width": 0.0,
                "n_grass_clumps": (0, 30),
                "ground": "paved",
                "cobble_prob": 0.55,
            },
            "alley": {
                # Tight service lane: narrow carriageway, close facades.
                "road_width": 4.2,
                "lane_offset": 1.05,
                "sidewalk_width": 1.35,
                "building_setback": 0.85,
                "building_depth": (3.0, 6.5),
                "building_width": (3.6, 8.5),
                "building_height": (4.5, 14.0),
                "building_gap": (0.15, 1.10),
                "n_background_vehicles": (0, 2),
                "n_background_pedestrians": (2, 6),
                "n_streetlamps": 4,
                "n_trees": (0, 3),
                "n_street_trees": (0, 0),
                "n_median_trees": (0, 0),
                "median_width": 0.0,
                "n_grass_clumps": (0, 6),
                "path_types": ("straight", "gentle_curve", "corner_90"),
                "cobble_prob": 0.70,
            },
            "residential": {
                # Quieter street, deeper front gardens, lower houses.
                "road_width": 6.4,
                "lane_offset": 1.60,
                "sidewalk_width": 2.0,
                "building_setback": 5.2,
                "building_depth": (5.0, 12.0),
                "building_width": (5.0, 12.0),
                "building_height": (4.5, 10.5),
                "building_gap": (1.2, 4.5),
                "n_background_vehicles": (1, 3),
                "n_background_pedestrians": (1, 4),
                "n_streetlamps": 5,
                "n_trees": (8, 16),
                "n_street_trees": (0, 0),
                "n_median_trees": (0, 0),
                "median_width": 0.0,
                "n_grass_clumps": (16, 40),
                "path_types": ("straight", "gentle_curve", "s_curve"),
                "cobble_prob": 0.22,
            },
            "market": {
                # Wide sidewalks, shop-front clutter, more people than cars.
                "road_width": 8.0,
                "lane_offset": 2.05,
                "sidewalk_width": 4.8,
                "building_setback": 2.2,
                "building_depth": (4.0, 9.0),
                "building_width": (4.0, 11.0),
                "building_height": (5.0, 14.0),
                "building_gap": (0.25, 1.40),
                "n_background_vehicles": (1, 3),
                "n_background_pedestrians": (8, 16),
                "n_streetlamps": 6,
                "n_trees": (2, 6),
                "n_street_trees": (0, 0),
                "n_median_trees": (0, 0),
                "median_width": 0.0,
                "poisson": {
                    "static_radius": 2.4,
                    "head_hazard_radius": 7.5,
                    "k_candidates": 20,
                    "n_ground_static": (6, 14),
                    "n_head_hazards": (0, 0),
                },
                "path_types": ("straight", "gentle_curve"),
                "cobble_prob": 0.60,
            },
        },
        # Procedural nature. Counts are per episode; geometry is instanced
        # from a handful of shared datablocks (see MeshLibrary).
        # n_trees = planting-strip / field trees. Street-tree / median /
        # path counts are extra. Curb/median roles never sit on asphalt
        # unless a planted median_width strip exists (avenue).
        "n_trees": (6, 12),
        "n_street_trees": (0, 0),
        "n_median_trees": (0, 0),
        "n_path_trees": (0, 0),
        "n_grass_clumps": (8, 28),
        "tree": {
            "trunk_height": (2.0, 4.8),
            "trunk_radius": (0.08, 0.28),
            "canopy_radius": (1.0, 3.0),
            "shapes": ("round", "conical", "columnar", "spreading", "bare"),
            "lean_deg": (0.0, 6.0),
        },
        "grass": {
            "blades": 26,
            "clump_radius": (0.30, 0.85),
            "blade_height": (0.10, 0.42),
        },
    },
    # -------------------------------------------------------------------------
    # Threat taxonomy thresholds (Part 4)
    # -------------------------------------------------------------------------
    "threat": {
        "safe_static_distance": 5.0,
        "safe_dynamic_cpa": 1.5,
        "near_miss_ttc": 4.0,
        "near_miss_cpa_min": 0.5,
        "near_miss_cpa_max": 1.5,
        "critical_ttc": 2.5,
        "critical_cpa": 0.5,
        "static_speed_eps": 0.05,  # m/s: treat as static
        "rel_speed_eps": 1e-4,
        "converging_eps": 0.0,
        # User hitbox used when injecting intercepts (not the visual mesh).
        "user_hitbox_radius": 0.30,
    },
    # Target mix across episodes when --scenario auto (Part 4.3).
    "scenarios": {
        "ratios": {
            "safe": 0.40,
            "near_miss": 0.30,
            "critical": 0.30,
        },
        # Named injectors. Motion is Frenet (s, lateral) on the road ribbon
        # so actors cannot chord through buildings on a curve.
        "safe_pool": (
            "safe_walk",
            "empty_street",
            "oncoming_pedestrian",
            "parallel_pedestrian",
            "cyclist_same_way",
            "car_pass_far",
            "car_approaching",
            "distant_jaywalk",
            "pothole_offset",
            "parked_car_opposite",
            "crossing_street",
            "cyclist_overtake",
        ),
        "near_miss_pool": (
            "near_miss_pass",
            "jaywalker_offset",
            "jaywalker_from_left",
            "jaywalker_from_right",
            "jaywalker_turn_away",
            "cyclist_near_miss",
            "car_near_miss_lane",
            "car_cross_front",
            "cube_near_miss",
            "shape_near_miss",
            "pothole_near",
            "cyclist_weaving",
            "group_crossing",
            "scooter_from_sidewalk",
            "parked_car_door",
        ),
        "critical_pool": (
            "jaywalker",
            "jaywalker_turn_toward",
            "sudden_stop",
            "swerve_vehicle",
            "pothole_on_path",
            "cube_head_on",
            "cube_from_left",
            "cube_from_right",
            "cube_on_path",
            "shape_head_on",
            "shape_from_left",
            "shape_from_right",
            "shapes_on_path",
            "car_cut_in",
            "cyclist_head_on",
            "head_level_projectile",
            "car_cross_critical",
            "car_erratic_swerve",
            "car_runs_off_road",
            "child_darting",
            "crossing_car_side",
            "crossing_head_on",
            "backing_vehicle",
        ),
        # Empty-center / side-threat pack (``gen_dataset.py --theme peripheral``).
        # Not drawn by ``--scenario auto`` so mixed packs stay unchanged.
        "peripheral_safe": (
            "periph_empty",
            "periph_car_side",
            "periph_parked",
            "periph_ped_side",
        ),
        "peripheral_near": (),
        "peripheral_critical": (
            "periph_car_turn",
            "periph_car_runoff",
            "periph_ped_cut",
            "periph_child_cut",
        ),
        "swerve_trigger_s": 1.8,
        "cut_in_trigger_s": 1.2,
        "sudden_stop_lead_m": 3.0,
        "sudden_stop_trigger_s": 1.6,
        # On-gait hole. 4.2 m sat below a typical walking VFOV (eye 1.6 m,
        # half-VFOV ~14–22°). 7.8 m is in the lower third at 50–90° HFOV.
        "pothole_on_path_lead_m": 7.8,
        "pothole_near_lead_m": 7.2,
        "pothole_offset_lead_m": 8.0,
        # Time until closest approach for a walking-speed crosser (not a sprint).
        "jaywalker_ttc": 3.2,
        "cross_person_speed": (1.00, 1.35),
        "projectile_ttc": 1.8,
        "projectile_speed": 3.6,
        "near_miss_cpa_target": 1.0,
        "critical_cpa_target": 0.12,
        "oncoming_tau": 2.8,
        "cross_car_tau": 2.6,
        # Compound injectors (``--scenario jaywalker,car,pothole``). Occupancy
        # is Frenet (s, lateral); this runs once per episode, not per frame.
        "compose": {
            "cross_stride_m": 4.0,
            "along_stride_m": 3.2,
            "static_stride_m": 2.4,
            "nudge_s_m": 2.6,
            "max_nudges": 14,
            "horizon_s": 5.0,
            "dt_sample": 0.12,
            "clearance_s": 0.20,
            "clearance_lat": 0.18,
            "look_ahead_m": 32.0,
        },
    },
    # -------------------------------------------------------------------------
    # Domain randomization bounds (Part 3.3)
    # -------------------------------------------------------------------------
    "domain_randomization": {
        # harsh_glare: sun almost on the horizon at many times noon energy,
        #              i.e. the sunset-blindness case a white cane cannot see.
        # overcast:    high turbidity, flat, near-shadowless.
        "sun_energy": {
            "dawn": (1.5, 4.0),
            "noon": (8.0, 14.0),
            "dusk": (1.2, 3.5),
            "night": (0.02, 0.15),
            "harsh_glare": (18.0, 42.0),
            "overcast": (0.8, 2.2),
        },
        "sun_elevation_deg": {
            "dawn": (4.0, 18.0),
            "noon": (55.0, 85.0),
            "dusk": (3.0, 16.0),
            "night": (-12.0, -2.0),
            "harsh_glare": (1.5, 7.5),
            "overcast": (25.0, 70.0),
        },
        "sun_azimuth_deg": (0.0, 360.0),
        "lighting_weights": {
            "dawn": 0.16,
            "noon": 0.24,
            "dusk": 0.16,
            "night": 0.16,
            "harsh_glare": 0.14,
            "overcast": 0.14,
        },
        # Probability that an overhead canopy gobo dapples the whole street.
        "dappled_prob": 0.26,
        # Per-episode appearance-chaos dial handed to materials.py.
        "material_chaos": (0.20, 1.00),
        "weather_weights": {
            "clear": 0.50,
            "light_fog": 0.30,
            "heavy_smog": 0.20,
        },
        # Per-episode wind. Leaves rustle; the trunk stays still.
        "wind_weights": {
            "calm": 0.32,
            "breeze": 0.48,
            "windy": 0.20,
        },
        "wind_strength": {
            "calm": (0.02, 0.08),
            "breeze": (0.28, 0.55),
            "windy": (0.70, 1.00),
        },
        "volume_density": {
            "clear": (0.0000, 0.0025),
            "light_fog": (0.012, 0.035),
            "heavy_smog": (0.055, 0.12),
        },
        "volume_anisotropy": (-0.15, 0.35),
        "horizon_color": {
            "dawn": ((0.72, 0.38, 0.22), (0.95, 0.62, 0.38)),
            "noon": ((0.45, 0.62, 0.92), (0.62, 0.78, 1.00)),
            "dusk": ((0.55, 0.22, 0.28), (0.85, 0.40, 0.25)),
            "night": ((0.01, 0.02, 0.05), (0.04, 0.05, 0.09)),
            "harsh_glare": ((0.98, 0.72, 0.42), (1.00, 0.88, 0.62)),
            "overcast": ((0.60, 0.62, 0.66), (0.74, 0.76, 0.80)),
        },
        "road_color": ((0.05, 0.05, 0.06), (0.18, 0.18, 0.20)),
        "cobble_color": ((0.28, 0.22, 0.18), (0.52, 0.42, 0.32)),
        "sidewalk_color": ((0.28, 0.28, 0.26), (0.55, 0.54, 0.50)),
        # Per-episode ribbon jitter after biome fold-in (fraction of the value).
        "layout_jitter": {
            "road_width": 0.12,
            "sidewalk_width": 0.14,
            "building_setback": 0.16,
            "lane_offset": 0.08,
            "median_width": 0.10,
        },
        "roughness": (0.35, 0.95),
        "vehicle_palette": (
            (0.12, 0.12, 0.13),
            (0.55, 0.08, 0.08),
            (0.08, 0.18, 0.48),
            (0.82, 0.82, 0.80),
            (0.08, 0.35, 0.18),
            (0.75, 0.55, 0.08),
        ),
        "pedestrian_palette": (
            (0.18, 0.18, 0.22),
            (0.42, 0.22, 0.14),
            (0.10, 0.28, 0.40),
            (0.55, 0.45, 0.30),
            (0.25, 0.25, 0.25),
        ),
    },
    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------
    "output": {
        "root": "output",
        "rgb_dirname": "rgb",
        "ann_dirname": "annotations",
        "manifest_name": "episode.json",
        # Visual product written next to the per-frame JSON.
        # "frames" = PNG only, "video" = H.264 only, "both" = keep PNGs and .mp4.
        "media": "both",
        "video_name": "preview.mp4",
        "video_crf": 18,
        "ffmpeg_bin": "ffmpeg",
        # auto → h264_nvenc when the NVIDIA encoder loads, else libx264.
        "video_encoder": "auto",
        "x264_preset": "veryfast",
        "nvenc_preset": "p4",
    },
    "annotation": {
        "max_distance": 40.0,  # skip objects beyond this (metres)
        "occlusion_epsilon": 0.08,
    },
}


def lens_mm_from_hfov_deg(hfov_deg: float, sensor_width_mm: float = 36.0) -> float:
    """Full-frame pinhole: lens = (sensor/2) / tan(HFOV/2)."""
    half = math.radians(float(hfov_deg) * 0.5)
    tan_h = math.tan(half)
    if tan_h <= 1e-8:
        return 24.0
    return (float(sensor_width_mm) * 0.5) / tan_h


def hfov_deg_from_lens_mm(lens_mm: float, sensor_width_mm: float = 36.0) -> float:
    """Inverse of `lens_mm_from_hfov_deg`."""
    lens = max(1e-3, float(lens_mm))
    return math.degrees(2.0 * math.atan((float(sensor_width_mm) * 0.5) / lens))


def apply_camera_fov(
    cfg: dict[str, Any],
    *,
    hfov_deg: float | None = None,
    lens_mm: float | None = None,
    rng: Any | None = None,
    lock: bool = False,
) -> dict[str, Any]:
    """Write `lens_mm` and `hfov_deg` onto `cfg['camera']` for this episode.

    Priority: explicit HFOV, then explicit lens, then (if unlocked) a uniform
    draw from `hfov_deg_range`, else the config default lens.
    """
    cam = cfg["camera"]
    sensor = float(cam["sensor_width_mm"])
    if hfov_deg is not None:
        fov = float(hfov_deg)
        cam["lens_mm"] = lens_mm_from_hfov_deg(fov, sensor)
        cam["hfov_deg"] = fov
        return cfg
    if lens_mm is not None:
        cam["lens_mm"] = float(lens_mm)
        cam["hfov_deg"] = hfov_deg_from_lens_mm(float(lens_mm), sensor)
        return cfg
    randomize = (not lock) and bool(cam.get("randomize_hfov", True))
    if randomize and rng is not None:
        lo, hi = cam["hfov_deg_range"]
        fov = float(rng.uniform(float(lo), float(hi)))
        cam["lens_mm"] = lens_mm_from_hfov_deg(fov, sensor)
        cam["hfov_deg"] = fov
        return cfg
    cam["hfov_deg"] = hfov_deg_from_lens_mm(float(cam["lens_mm"]), sensor)
    return cfg


def get_config() -> dict[str, Any]:
    """Deep-copy so an episode can mutate local overrides without leaking."""
    return copy.deepcopy(CONFIG)
