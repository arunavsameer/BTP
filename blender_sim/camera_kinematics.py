"""Egocentric head-mounted camera rig: gait bounce + Perlin micro-saccades.

The walker shares the *road* spline with every actor: arc-length `s` along
the centreline plus a constant sidewalk `lateral`. Each frame the local
camera transform is *not* the Frenet frame of that path — it is the Frenet
frame composed with:

  1. Vertical gait   Y_spec(t) = 1.6 + A sin(2 π f t)
                     (applied on Blender +Z, above the curb)
  2. Head jitter     yaw / pitch / roll from independent 1-D fractal Perlin
                     streams, matching the amplitudes in the spec.

Blender cameras look down local −Z, with +Y as the camera up axis. The
base orientation is therefore the matrix whose columns are

    [ right | world_up_corrected | −tangent ]

so that local −Z equals the path tangent.

Ego modes
---------
A dataset of nothing but constant-velocity straight-line walks teaches the
network that optical flow is always a pure forward translation. Four modes
break that assumption:

``walk``            the original constant-speed sidewalk traverse.
``diagonal_cross``  the walker cuts from one kerb to the other, so the
                    lateral coordinate ramps across the episode and the
                    flow field acquires a sustained sideways component.
``crosswalk``       first-class street crossing: full kerb-to-kerb Frenet
                    lateral change; gaze follows the motion vector so the
                    walker visibly turns onto the crossing.
``erratic``         fBm sidesteps plus speed modulation, with a chance of a
                    complete stop partway through. The halt is a raised-cosine
                    ramp, not a hard 0, and gaze stays the road tangent so a
                    sidestep cannot yaw the world 90°.
``hasty``           high-frequency look jitter (large yaw/pitch/roll). Body
                    path is still the gait tangent so TTC does not flicker.
``seated``          on a bench: ``walk_speed = 0`` and a lower eye height.

Arc length under a varying speed
--------------------------------
``walk`` has the closed form :math:`s(t) = s_0 + v t`. ``erratic`` does not:
its speed is an fBm signal with ramped hesitation windows, so ``s(t)`` is
:math:`s_0 + \\int_0^t v(\\tau)\\,d\\tau` with no analytic antiderivative.
The rig therefore integrates once at construction onto a fixed-step table
and interpolates it. That matters for correctness, not just speed: scenario
injectors query ``s`` at arbitrary future times *before* the simulation loop
runs, and they must agree with the positions the loop later produces to the
last metre, or a "critical" intercept lands behind the walker.

Zero ego speed
--------------
``seated`` (and every hesitation window) makes ``walk_speed`` exactly 0.
Nothing here divides by it: :meth:`max_time` reads the arc table instead of
dividing remaining distance by speed, and the gait bounce is scaled by
speed so a stationary head does not bob.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

try:
    from mathutils import Euler, Matrix, Vector
except ImportError:  # system Python: halt-gain self-test still runs
    Euler = Matrix = Vector = object  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# Seeded 1-D improved Perlin (Ken Perlin 2002 fade), with fractal Brownian
# motion. Independent of Blender's `mathutils.noise` so a given seed is
# bit-identical across Blender versions.
# ---------------------------------------------------------------------------

def _fade(t: float) -> float:
    """Perlin quintic: 6t^5 - 15t^4 + 10t^3. C2-continuous at lattice nodes."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


class Perlin1D:
    """Classic improved Perlin evaluated along a single axis."""

    def __init__(self, seed: int, table_size: int = 256) -> None:
        rng = random.Random(int(seed) & 0xFFFFFFFF)
        perm = list(range(table_size))
        rng.shuffle(perm)
        # Duplicate so we can index with (i + 1) & mask without wrap logic.
        self._perm = perm + perm
        self._grads = [rng.uniform(-1.0, 1.0) for _ in range(table_size)]
        self._grads.extend(self._grads)
        self._mask = table_size - 1
        if table_size & self._mask:
            raise ValueError("Perlin1D table_size must be a power of two")

    def noise(self, x: float) -> float:
        """Return a smooth value in approximately [-1, 1] at real `x`."""
        xi = math.floor(x)
        xf = x - xi
        i0 = xi & self._mask
        i1 = (xi + 1) & self._mask
        g0 = self._grads[i0]
        g1 = self._grads[i1]
        # Dot the lattice gradient with the offset from that lattice point.
        n0 = g0 * xf
        n1 = g1 * (xf - 1.0)
        return _fade(xf) * (n1 - n0) + n0

    def fbm(
        self,
        x: float,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ) -> float:
        """Fractal Brownian motion: sum of octaves of `noise`.

        Amplitude is re-normalized by the geometric series so the result
        stays in roughly [-1, 1] regardless of octave count.
        """
        total = 0.0
        amp = 1.0
        freq = 1.0
        amp_sum = 0.0
        for _ in range(max(1, int(octaves))):
            total += self.noise(x * freq) * amp
            amp_sum += amp
            amp *= persistence
            freq *= lacunarity
        if amp_sum <= 0.0:
            return 0.0
        return total / amp_sum


# ---------------------------------------------------------------------------
# Halt window (raised cosine). Pure math so it can be tested without bpy.
# ---------------------------------------------------------------------------

HALT_RAMP_S = 0.40  # default ease in/out inside [t0, t1]


def halt_speed_gain(
    t: float,
    t0: Optional[float],
    t1: Optional[float],
    ramp_s: float = HALT_RAMP_S,
) -> float:
    """Cruise multiplier: 1 outside a halt, 0 in the middle, C1 at the joints.

    Ramps live *inside* ``[t0, t1]`` so injectors that read the window still
    see a real stop, not a dip that never reaches zero. A hard box
    ``t0 ≤ t < t1 → 0`` freezes optical flow in one frame and, when gaze
    used the sidestep rate, yawed the world ~90°.
    """
    if t0 is None or t1 is None:
        return 1.0
    t0f = float(t0)
    t1f = float(t1)
    if t1f <= t0f:
        return 1.0
    span = t1f - t0f
    ramp = min(max(0.0, float(ramp_s)), 0.45 * span)
    tt = float(t)
    if ramp < 1e-4:
        return 0.0 if t0f <= tt < t1f else 1.0
    if tt < t0f:
        return 1.0
    if tt < t0f + ramp:
        u = (tt - t0f) / ramp
        return 0.5 * (1.0 + math.cos(math.pi * u))  # 1 → 0, derivative 0 at ends
    if tt < t1f - ramp:
        return 0.0
    if tt < t1f:
        u = (tt - (t1f - ramp)) / ramp
        return 0.5 * (1.0 - math.cos(math.pi * u))  # 0 → 1
    return 1.0


def _look_blends_lateral(mode: str) -> bool:
    """Only a street crossing should yaw the body toward the lateral rate."""
    return mode in ("diagonal_cross", "crosswalk")


# ---------------------------------------------------------------------------
# Camera state
# ---------------------------------------------------------------------------

@dataclass
class CameraState:
    """Snapshot consumed by the annotator and written into camera_data."""

    t: float
    position: Vector  # Blender world (Z-up)
    velocity: Vector  # Blender world m/s
    tangent: Vector
    right: Vector
    up: Vector
    pitch: float  # local jitter, radians
    yaw: float
    roll: float
    matrix_world: Matrix
    walk_speed: float  # instantaneous ground speed, 0 when seated / halted
    arc_length: float
    lateral: float
    mode: str = "walk"

    def pitch_yaw_roll(self) -> tuple[float, float, float]:
        return (self.pitch, self.yaw, self.roll)


EGO_MODES: tuple[str, ...] = (
    "walk", "diagonal_cross", "crosswalk", "erratic", "hasty", "seated",
)


@dataclass
class EgoProfile:
    """Per-episode ego trajectory parameters, drawn once before the sim.

    Separated from :class:`CameraRig` so it can be built, logged, and
    unit-tested without a Blender camera object.
    """

    mode: str = "walk"
    eye_height: float = 1.6
    stature: str = "typical"
    # diagonal_cross: fraction of the way to the *opposite* kerb, traversed
    # over [t0, t1]. Stored as a fraction rather than metres because the
    # profile is drawn before the rig knows how wide this biome's corridor
    # is — a 1.3 m shift crosses a park path but only steps off a kerb on a
    # four-lane avenue.
    diag_frac: float = 0.0
    diag_t0: float = 0.0
    diag_t1: float = 1.0
    # erratic: fBm sidestep and speed modulation.
    sidestep_amp: float = 0.0
    sidestep_rate: float = 0.20
    speed_wobble: float = 0.0
    halt_t0: Optional[float] = None
    halt_t1: Optional[float] = None
    halt_ramp: float = HALT_RAMP_S
    # hasty: high-frequency look; body path still uses the gait tangent.
    hasty_yaw_deg: float = 32.0
    hasty_pitch_deg: float = 16.0
    hasty_roll_deg: float = 10.0
    hasty_hz: float = 4.0
    hasty_bob_m: float = 0.035

    @property
    def stationary(self) -> bool:
        return self.mode == "seated"


def choose_ego_profile(cfg: dict, rng: random.Random, episode_seconds: float = 10.0) -> EgoProfile:
    """Draw an ego mode and its parameters from `cfg['ego']`.

    `episode_seconds` scales the timing windows so a hesitation actually
    lands inside a short episode instead of after the last frame. Config
    windows are written for a 10 s clip.
    """
    ecfg = dict(cfg.get("ego") or {})
    weights = dict(ecfg.get("mode_weights") or {"walk": 1.0})
    keys = [k for k in weights if k in EGO_MODES]
    if not keys:
        keys = ["walk"]
    total = sum(max(0.0, float(weights.get(k, 0.0))) for k in keys)
    if total <= 0.0:
        mode = "walk"
    else:
        x = rng.uniform(0.0, total)
        acc = 0.0
        mode = keys[-1]
        for k in keys:
            acc += max(0.0, float(weights.get(k, 0.0)))
            if x <= acc:
                mode = k
                break

    eye = float(cfg["camera"]["eye_height_m"])
    stature = "typical"
    locked_h = ecfg.get("eye_height_lock_m")
    locked_band = str(ecfg.get("stature_lock") or "").strip().lower()
    cam = dict(cfg.get("camera") or {})
    bands = dict(cam.get("eye_height_m_by_stature") or {})
    if locked_h is not None:
        eye = max(0.85, min(2.15, float(locked_h)))
        if eye < 1.52:
            stature = "short"
        elif eye > 1.74:
            stature = "tall"
    elif locked_band in bands:
        stature = locked_band
        lo, hi = bands[stature]
        eye = float(rng.uniform(float(lo), float(hi)))
    elif mode != "seated" and bands:
        sw = dict(cam.get("stature_weights") or {"typical": 1.0})
        keys = [k for k in sw if k in bands]
        if keys:
            total_s = sum(max(0.0, float(sw.get(k, 0.0))) for k in keys)
            x_s = rng.uniform(0.0, total_s if total_s > 0.0 else 1.0)
            acc_s = 0.0
            stature = keys[-1]
            for k in keys:
                acc_s += max(0.0, float(sw.get(k, 0.0)))
                if x_s <= acc_s:
                    stature = k
                    break
            lo, hi = bands[stature]
            eye = float(rng.uniform(float(lo), float(hi)))
    prof = EgoProfile(mode=mode, eye_height=eye, stature=stature)
    span = max(0.5, float(episode_seconds))

    if mode == "seated":
        lo, hi = ecfg.get("seated_eye_height_m", (0.95, 1.28))
        prof.eye_height = float(rng.uniform(float(lo), float(hi)))
        prof.stature = "seated"
        return prof

    if mode in ("diagonal_cross", "crosswalk"):
        if mode == "crosswalk":
            lo, hi = ecfg.get("crosswalk_target_frac", (0.88, 1.00))
            f0, f1 = ecfg.get("crosswalk_span", (0.22, 0.78))
            min_start, min_dur = (0.80, 4.50) if span >= 8.0 else (0.45, 3.00)
        else:
            lo, hi = ecfg.get("diagonal_target_frac", (0.45, 1.00))
            f0, f1 = ecfg.get("diagonal_span", (0.12, 0.85))
            min_start, min_dur = (0.35, 4.00) if span >= 8.0 else (0.20, 3.00)
        prof.diag_frac = float(rng.uniform(float(lo), float(hi)))
        # Floor in seconds so a 6-frame QA clip does not compress the turn
        # into a 90° snap at t=0 (injectors still see the same schedule).
        prof.diag_t0 = max(min_start, span * float(f0))
        prof.diag_t1 = max(prof.diag_t0 + min_dur, span * float(f1))
        return prof

    if mode == "erratic":
        a_lo, a_hi = ecfg.get("sidestep_amp_m", (0.22, 0.80))
        r_lo, r_hi = ecfg.get("sidestep_rate_hz", (0.10, 0.38))
        w_lo, w_hi = ecfg.get("speed_wobble", (0.15, 0.55))
        prof.sidestep_amp = float(rng.uniform(float(a_lo), float(a_hi)))
        prof.sidestep_rate = float(rng.uniform(float(r_lo), float(r_hi)))
        prof.speed_wobble = float(rng.uniform(float(w_lo), float(w_hi)))
        if rng.random() < float(ecfg.get("hesitate_prob", 0.55)):
            h_lo, h_hi = ecfg.get("hesitate_window_s", (1.8, 5.2))
            d_lo, d_hi = ecfg.get("hesitate_duration_s", (0.7, 2.0))
            # Config windows are for a 10 s clip.
            scale = span / 10.0
            t0 = float(rng.uniform(float(h_lo), float(h_hi))) * scale
            dur = float(rng.uniform(float(d_lo), float(d_hi)))
            if scale < 1.0:
                dur = max(0.45, dur * max(0.55, scale))
            prof.halt_t0 = min(t0, max(0.2, span - 0.8))
            prof.halt_t1 = min(span - 0.15, prof.halt_t0 + dur)
            prof.halt_ramp = float(ecfg.get("halt_ramp_s", HALT_RAMP_S))
        return prof

    if mode == "hasty":
        hj = dict(ecfg.get("hasty") or {})
        y_lo, y_hi = hj.get("yaw_amp_deg", (25.0, 40.0))
        p_lo, p_hi = hj.get("pitch_amp_deg", (12.0, 20.0))
        r_lo, r_hi = hj.get("roll_amp_deg", (8.0, 12.0))
        f_lo, f_hi = hj.get("freq_hz", (2.5, 6.0))
        b_lo, b_hi = hj.get("bob_m", (0.025, 0.050))
        prof.hasty_yaw_deg = float(rng.uniform(float(y_lo), float(y_hi)))
        prof.hasty_pitch_deg = float(rng.uniform(float(p_lo), float(p_hi)))
        prof.hasty_roll_deg = float(rng.uniform(float(r_lo), float(r_hi)))
        prof.hasty_hz = float(rng.uniform(float(f_lo), float(f_hi)))
        prof.hasty_bob_m = float(rng.uniform(float(b_lo), float(b_hi)))
        return prof

    return prof


def apply_ego_height(cfg: dict, spec: str | None) -> None:
    """Lock standing eye height from CLI: short|typical|tall|auto|metres."""
    raw = str(spec or "auto").strip().lower()
    ego = cfg.setdefault("ego", {})
    ego.pop("stature_lock", None)
    ego.pop("eye_height_lock_m", None)
    if raw in ("", "auto", "random"):
        return
    if raw in ("short", "typical", "tall"):
        ego["stature_lock"] = raw
        return
    try:
        height = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"unknown --ego-height {spec!r}. Use short|typical|tall|auto or metres."
        ) from exc
    ego["eye_height_lock_m"] = max(0.85, min(2.15, height))


@dataclass
class CameraRig:
    """Owns the Blender camera object and writes its 4×4 transform each frame."""

    cam_obj: Any
    spline: Any  # PathSpline (duck-typed: evaluate / tangent / length)
    cfg: dict
    rng: random.Random
    walk_speed: float
    sidewalk_s0: float = 3.0
    lateral: float = 0.0
    curb_height: float = 0.12
    profile: EgoProfile = field(default_factory=EgoProfile)
    lateral_limit: float = 1e9
    _prev_position: Optional[Vector] = field(default=None, init=False, repr=False)
    _yaw_noise: Perlin1D = field(init=False, repr=False)
    _pitch_noise: Perlin1D = field(init=False, repr=False)
    _roll_noise: Perlin1D = field(init=False, repr=False)
    _step_noise: Perlin1D = field(init=False, repr=False)
    _speed_noise: Perlin1D = field(init=False, repr=False)
    _yaw_phase: float = field(init=False, repr=False)
    _pitch_phase: float = field(init=False, repr=False)
    _roll_phase: float = field(init=False, repr=False)
    _arc_dt: float = field(default=1.0 / 120.0, init=False, repr=False)
    _arc: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        # Six independent seeds so no two channels are correlated.
        base = self.rng.randrange(1, 2**31)
        self._yaw_noise = Perlin1D(base + 17)
        self._pitch_noise = Perlin1D(base + 101)
        self._roll_noise = Perlin1D(base + 233)
        self._step_noise = Perlin1D(base + 331)
        self._speed_noise = Perlin1D(base + 457)
        # Phase offsets so two episodes with the same walk speed still differ.
        self._yaw_phase = self.rng.uniform(0.0, 64.0)
        self._pitch_phase = self.rng.uniform(0.0, 64.0)
        self._roll_phase = self.rng.uniform(0.0, 64.0)

        cam = self.cam_obj.data
        ccfg = self.cfg["camera"]
        cam.lens = float(ccfg["lens_mm"])
        cam.sensor_width = float(ccfg["sensor_width_mm"])
        cam.sensor_fit = str(ccfg["sensor_fit"])
        cam.clip_start = float(ccfg["clip_start"])
        cam.clip_end = float(ccfg["clip_end"])
        wcfg = self.cfg.get("world") or {}
        self.curb_height = float(wcfg.get("curb_height", self.curb_height))
        if self.profile.mode == "seated":
            # A bench sits back from the kerb, toward the building line.
            self.lateral = self._clamp_lateral(
                self.lateral + math.copysign(0.35, self.lateral or 1.0)
            )
        self._build_arc_table()

    # -- trajectory --------------------------------------------------------

    def _clamp_lateral(self, lat: float) -> float:
        lim = abs(float(self.lateral_limit))
        return max(-lim, min(lim, float(lat)))

    def speed_at(self, t: float) -> float:
        """Instantaneous ground speed (m/s). Never negative.

        This is the single source of truth for "is the ego moving"; the arc
        table, the gait amplitude, and the exported ``walk_speed`` all read
        it, so a hesitation cannot desynchronise them.
        """
        prof = self.profile
        if prof.stationary:
            return 0.0
        v = float(self.walk_speed)
        if prof.speed_wobble > 1e-6:
            n = self._speed_noise.fbm(t * 0.45, octaves=3, persistence=0.5, lacunarity=2.0)
            v *= 1.0 + prof.speed_wobble * n
        v = max(0.0, v)
        gain = halt_speed_gain(t, prof.halt_t0, prof.halt_t1, prof.halt_ramp)
        return v * gain

    def _build_arc_table(self, horizon_s: float = 90.0) -> None:
        """Cumulative-trapezoid table of s(t) over [0, horizon].

        Built once. ``walk`` and ``seated`` have exact closed forms and skip
        the table entirely; only the modulated modes pay for it, and even
        then it is ~10k floats.
        """
        self._arc = []
        prof = self.profile
        if prof.stationary or (prof.speed_wobble <= 1e-6 and prof.halt_t0 is None):
            return  # constant speed: s(t) is analytic
        dt = self._arc_dt
        n = int(horizon_s / dt) + 2
        s = 0.0
        prev_v = self.speed_at(0.0)
        table = [0.0] * n
        for i in range(1, n):
            t = i * dt
            v = self.speed_at(t)
            s += 0.5 * (prev_v + v) * dt
            table[i] = s
            prev_v = v
        self._arc = table

    def travelled(self, t: float) -> float:
        """Distance walked since t = 0 (metres), for any mode."""
        tt = max(0.0, float(t))
        if not self._arc:
            if self.profile.stationary:
                return 0.0
            return float(self.walk_speed) * tt
        dt = self._arc_dt
        x = tt / dt
        i = int(x)
        if i >= len(self._arc) - 1:
            return self._arc[-1]
        u = x - i
        a = self._arc[i]
        return a + (self._arc[i + 1] - a) * u

    def arc_length_at(self, t: float) -> float:
        """Road arc-length of the ego at time `t`, clamped to the spline."""
        s = self.sidewalk_s0 + self.travelled(t)
        return min(max(0.05, s), max(0.10, self.spline.length - 0.05))

    def lateral_at(self, t: float) -> float:
        """Signed offset from the road centreline at time `t`.

        ``walk`` and ``seated`` are constant. ``diagonal_cross`` ramps with a
        smoothstep so the heading turns continuously rather than snapping.
        ``erratic`` adds a zero-mean fBm sidestep about the base lateral.
        """
        prof = self.profile
        lat = float(self.lateral)
        if prof.mode in ("diagonal_cross", "crosswalk") and abs(prof.diag_frac) > 1e-6:
            t0, t1 = prof.diag_t0, prof.diag_t1
            u = 0.0 if t <= t0 else (1.0 if t >= t1 else (t - t0) / max(t1 - t0, 1e-3))
            u = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
            # Cross toward the far kerb: opposite sign to the start lateral.
            # Full span is base → −base, i.e. 2|base|, with a floor so a
            # centreline path (base ≈ 0) still produces a real crossing.
            span = max(2.0 * abs(lat), 2.5)
            lat = lat - math.copysign(prof.diag_frac * span, lat if lat else 1.0) * u
        elif prof.mode == "erratic" and prof.sidestep_amp > 1e-6:
            # Drive the sidestep by distance walked, not wall-clock. A halt
            # then freezes the pose instead of sliding the body sideways
            # while "stopped" (and yawing the camera into that slide).
            dist = self.travelled(t)
            spatial = float(prof.sidestep_rate) / max(float(self.walk_speed), 0.35)
            n = self._step_noise.fbm(
                dist * spatial, octaves=3, persistence=0.5, lacunarity=2.05
            )
            lat = lat + prof.sidestep_amp * n
        return self._clamp_lateral(lat)

    # -- public API --------------------------------------------------------

    def max_time(self) -> float:
        """Largest t that still keeps the walker on the road spline.

        Read off the arc table rather than dividing remaining distance by
        speed, so a hesitating or seated walker (speed 0) does not divide by
        zero and does not get an episode truncated to nothing.
        """
        remaining = max(0.5, self.spline.length - self.sidewalk_s0 - 2.0)
        if self.profile.stationary:
            return 1e6  # never leaves the spline; frame count rules instead
        if not self._arc:
            return remaining / max(self.walk_speed, 1e-3)
        dt = self._arc_dt
        for i, s in enumerate(self._arc):
            if s >= remaining:
                return i * dt
        return len(self._arc) * dt

    def sample_angles(self, t: float) -> tuple[float, float, float]:
        """Return (pitch, yaw, roll) in radians from the Perlin streams."""
        if self.profile.mode == "hasty":
            hz = max(0.5, float(self.profile.hasty_hz))

            def hasty_axis(amp_deg: float, noise: Perlin1D, phase: float) -> float:
                n = noise.fbm(
                    t * hz + phase,
                    octaves=3,
                    persistence=0.48,
                    lacunarity=2.15,
                )
                return math.radians(float(amp_deg)) * n

            pitch = hasty_axis(self.profile.hasty_pitch_deg, self._pitch_noise, self._pitch_phase)
            yaw = hasty_axis(self.profile.hasty_yaw_deg, self._yaw_noise, self._yaw_phase)
            roll = hasty_axis(self.profile.hasty_roll_deg, self._roll_noise, self._roll_phase)
            return pitch, yaw, roll

        j = self.cfg["jitter"]

        def axis(name: str, noise: Perlin1D, phase: float) -> float:
            spec = j[name]
            # Time is scaled by the axis frequency so "slow yaw" really is slow.
            x = t * float(spec["frequency_hz"]) + phase
            n = noise.fbm(
                x,
                octaves=int(spec["octaves"]),
                persistence=float(spec["persistence"]),
                lacunarity=float(spec["lacunarity"]),
            )
            return math.radians(float(spec["amplitude_deg"])) * n

        pitch = axis("pitch", self._pitch_noise, self._pitch_phase)
        yaw = axis("yaw", self._yaw_noise, self._yaw_phase)
        roll = axis("roll", self._roll_noise, self._roll_phase)
        return pitch, yaw, roll

    def _gait_gain(self, t: float) -> float:
        """Bounce amplitude as a fraction of the nominal walking bounce.

        A head does not bob while its owner is sitting on a bench or has
        stopped dead, and a bobbing camera with zero ground velocity is a
        strong, wrong cue: it looks like motion that the labels deny.
        """
        if self.profile.stationary:
            return 0.0
        nominal = max(float(self.walk_speed), 1e-6)
        return max(0.0, min(1.0, self.speed_at(t) / nominal))

    def gait_height(self, t: float) -> float:
        """Spec equation: eye + A sin(2 π f t), with A scaled by ground speed."""
        g = self.cfg["gait"]
        a = float(g["amplitude_m"]) * self._gait_gain(t)
        f = float(g["frequency_hz"])
        z = self.profile.eye_height + a * math.sin(2.0 * math.pi * f * t)
        if self.profile.mode == "hasty":
            z += float(self.profile.hasty_bob_m) * math.sin(
                2.0 * math.pi * max(0.5, float(self.profile.hasty_hz)) * t
            )
        return z

    def gait_pitch_bob(self, t: float) -> float:
        """Tiny nod locked to the bounce (optional, not in the spec minimum)."""
        g = self.cfg["gait"]
        amp = float(g.get("pitch_bob_amp_rad", 0.0)) * self._gait_gain(t)
        f = float(g["frequency_hz"])
        # Phase-quadrature with the vertical sine so the head dips at mid-stance.
        return amp * math.cos(2.0 * math.pi * f * t)

    def predict_position(self, t: float) -> Vector:
        """World position of the *eyes* at time t (no jitter, gait included).

        Used by scenario injectors to place intercepts on the future gait
        line, so it must follow the same arc table and lateral schedule the
        simulation loop will follow.
        """
        p, _, _ = self._frenet(self.arc_length_at(t), self.lateral_at(t))
        p = p.copy()
        p.z = p.z + self.gait_height(t)
        return p

    def predict_velocity(self, t: float, dt: float = 1.0 / 30.0) -> Vector:
        """Central-difference velocity of the eye point (includes gait dZ/dt)."""
        h = max(float(dt), 1e-4)
        return (self.predict_position(t + h) - self.predict_position(max(0.0, t - h))) / (2.0 * h)

    def update(self, t: float, dt: float) -> CameraState:
        """Write `cam_obj.matrix_world` and return the kinematics snapshot."""
        s = self.arc_length_at(t)
        lat = self.lateral_at(t)

        origin, path_tan, _path_right = self._frenet(s, lat)
        look = self._look_direction(t, path_tan)
        world_up = Vector((0.0, 0.0, 1.0))
        right = look.cross(world_up)
        if right.length < 1e-6:
            right = Vector((1.0, 0.0, 0.0))
        else:
            right.normalize()
        up = right.cross(look)
        if up.length < 1e-6:
            up = world_up.copy()
        else:
            up.normalize()

        height = self.gait_height(t)
        position = Vector((origin.x, origin.y, origin.z + height))

        pitch, yaw, roll = self.sample_angles(t)
        pitch = pitch + self.gait_pitch_bob(t)

        matrix = self._compose_matrix(position, look, up, right, pitch, yaw, roll)
        self._apply_camera_matrix(self.cam_obj, matrix)

        if self._prev_position is None or dt <= 1e-8:
            # Frame 0 has no previous sample. Central-difference the analytic
            # trajectory instead of assuming tangent × walk_speed, which is
            # wrong for a diagonal crossing and for a halted walker.
            h = max(dt, 1.0 / 240.0)
            velocity = (
                self.predict_position(t + h) - self.predict_position(max(0.0, t - h))
            ) / (2.0 * h)
        else:
            velocity = (position - self._prev_position) / dt
        self._prev_position = position.copy()

        return CameraState(
            t=t,
            position=position,
            velocity=velocity,
            tangent=look,
            right=right,
            up=up,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            matrix_world=matrix.copy(),
            walk_speed=self.speed_at(t),
            arc_length=s,
            lateral=lat,
            mode=self.profile.mode,
        )

    # -- internals ---------------------------------------------------------

    def _frenet(self, s: float, lateral: Optional[float] = None) -> tuple[Vector, Vector, Vector]:
        """(origin_on_ground, unit_tangent, unit_right) in Blender world.

        `origin` is the gait point: centreline plus `lateral` along road-right,
        so the camera shares the same Frenet `(s, lateral)` as every actor.
        Gaze stays the *road* tangent (forward along the street).
        """
        lat = float(self.lateral if lateral is None else lateral)
        p = Vector(self.spline.evaluate(s))
        tangent = Vector(self.spline.tangent(s))
        if tangent.length < 1e-8:
            tangent = Vector((0.0, 1.0, 0.0))
        else:
            tangent.normalize()
        world_up = Vector((0.0, 0.0, 1.0))
        # right = tangent × up  (see module docstring)
        right = tangent.cross(world_up)
        if right.length < 1e-6:
            right = Vector((1.0, 0.0, 0.0))
        else:
            right.normalize()
        origin = p + right * lat
        origin.z = float(self.curb_height)
        return origin, tangent, right

    def _look_direction(self, t: float, path_tan: Vector) -> Vector:
        """Gaze along the gait, or along Frenet motion on a street crossing.

        Walk / seated / hasty / erratic / halt keep the *road* tangent so a
        sidestep is a sway, not a 90° world cut. Crosswalk and diagonal
        blend in the lateral rate so heading turns onto the crossing.
        """
        tan = Vector(path_tan)
        if tan.length > 1e-8:
            tan.normalize()
        else:
            tan = Vector((0.0, 1.0, 0.0))
        if self.profile.stationary or not _look_blends_lateral(self.profile.mode):
            return tan
        ds = float(self.speed_at(t))
        h = 0.06
        dlat = (self.lateral_at(t + h) - self.lateral_at(max(0.0, t - h))) / (2.0 * h)
        if abs(dlat) < 1e-4:
            return tan
        _o, _t, right = self._frenet(self.arc_length_at(t), self.lateral_at(t))
        look = tan * max(ds, 0.45) + right * float(dlat)
        if look.length < 1e-5:
            return tan
        look.normalize()
        # A pedestrian turns onto a crossing; they do not spin to face a wall.
        max_off = math.radians(50.0)
        min_fwd = math.cos(max_off)
        fwd = look.dot(tan)
        if fwd < min_fwd:
            side = look - tan * fwd
            if side.length < 1e-6:
                return tan
            side.normalize()
            look = tan * min_fwd + side * math.sin(max_off)
            look.normalize()
        return look

    @staticmethod
    def _compose_matrix(
        position: Vector,
        tangent: Vector,
        up: Vector,
        right: Vector,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> Matrix:
        """Build the 4×4 camera world matrix.

        Base columns (Blender camera convention):
            col0 = right      → local +X
            col1 = up         → local +Y
            col2 = −tangent   → local +Z  (looks down −Z = +tangent)

        Jitter is applied in *camera* space so yaw scans left/right about the
        head's up axis, pitch nods about the ear-to-ear axis, and roll tilts
        about the gaze axis — i.e. Euler XYZ in the camera frame.
        """
        rot_base = Matrix((
            (right.x, up.x, -tangent.x),
            (right.y, up.y, -tangent.y),
            (right.z, up.z, -tangent.z),
        ))
        # mathutils Matrix constructed from rows; the tuple-of-tuples above
        # *is* row-major, and the columns are exactly (right, up, −tangent).
        rot_jitter = Euler((pitch, yaw, roll), "XYZ").to_matrix()
        rot = rot_base @ rot_jitter

        mat = Matrix.Identity(4)
        for i in range(3):
            for j in range(3):
                mat[i][j] = rot[i][j]
        mat.translation = position
        return mat

    @staticmethod
    def _apply_camera_matrix(cam_obj: Any, matrix: Matrix) -> None:
        """Write the 4×4 through loc/rot so Blender 5.x actually uses it.

        Assigning ``matrix_world`` alone can be ignored when the object's
        rotation mode is Euler (the default for a newly created camera).
        """
        loc, rot, _scale = matrix.decompose()
        cam_obj.rotation_mode = "QUATERNION"
        cam_obj.location = loc
        cam_obj.rotation_quaternion = rot
        cam_obj.scale = (1.0, 1.0, 1.0)
        cam_obj.matrix_world = matrix
        cam_obj.hide_render = False


def finite_difference_velocity(
    prev: Sequence[float],
    curr: Sequence[float],
    dt: float,
) -> Vector:
    if dt <= 1e-8:
        return Vector((0.0, 0.0, 0.0))
    return (Vector(curr) - Vector(prev)) / dt


def _self_test() -> None:
    g = halt_speed_gain
    assert g(0.0, None, None) == 1.0
    assert abs(g(0.0, 2.0, 4.0, 0.40) - 1.0) < 1e-12
    assert abs(g(2.0, 2.0, 4.0, 0.40) - 1.0) < 1e-9  # ramp starts here
    assert g(3.0, 2.0, 4.0, 0.40) == 0.0
    assert abs(g(4.0, 2.0, 4.0, 0.40) - 1.0) < 1e-9
    assert abs(g(5.0, 2.0, 4.0, 0.40) - 1.0) < 1e-12
    # Mid-ramp is ½; ends of the cosine have zero derivative.
    mid = g(2.20, 2.0, 4.0, 0.40)
    assert abs(mid - 0.5) < 1e-9, mid
    dt = 1e-4
    d0 = (g(2.0 + dt, 2.0, 4.0, 0.40) - g(2.0 - dt, 2.0, 4.0, 0.40)) / (2.0 * dt)
    d1 = (g(2.4 + dt, 2.0, 4.0, 0.40) - g(2.4 - dt, 2.0, 4.0, 0.40)) / (2.0 * dt)
    assert abs(d0) < 0.02, d0
    assert abs(d1) < 0.02, d1
    assert _look_blends_lateral("crosswalk") and _look_blends_lateral("diagonal_cross")
    assert not _look_blends_lateral("erratic")
    assert not _look_blends_lateral("walk")
    assert not _look_blends_lateral("hasty")
    print("camera_kinematics self-test: OK")


if __name__ == "__main__":
    _self_test()
