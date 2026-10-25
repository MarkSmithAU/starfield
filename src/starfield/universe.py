"""Procedural star systems: suns, planets, moons, and the maths helpers they need."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- math helpers


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = normalize(np.asarray(axis, dtype=np.float64))
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Re-orthonormalise a camera frame (columns right, up, forward), keeping forward."""
    f = normalize(R[:, 2])
    u = normalize(R[:, 1] - f * np.dot(R[:, 1], f))
    r = np.cross(u, f)
    return np.column_stack([r, u, f])


def frame_with_up(n: np.ndarray) -> np.ndarray:
    """Rotation matrix whose local Y axis is ``n``."""
    n = normalize(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    x = normalize(np.cross(n, a))
    z = np.cross(x, n)
    return np.column_stack([x, n, z])


def random_unit(rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    v = rng.normal(size=(3,) if n is None else (n, 3))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def perpendicular_basis(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    s1 = normalize(np.cross(a, f))
    s2 = np.cross(f, s1)
    return s1, s2


# ---------------------------------------------------------------- star colours


def kelvin_to_rgb(T: np.ndarray, desaturate: float = 0.25) -> np.ndarray:
    """Approximate black-body colour (Tanner Helland fit), linear, max channel = 1."""
    t = np.asarray(T, dtype=np.float64) / 100.0
    r = np.where(t <= 66, 255.0, 329.698727446 * np.power(np.maximum(t - 60, 1e-6), -0.1332047592))
    g = np.where(
        t <= 66,
        99.4708025861 * np.log(np.maximum(t, 1e-6)) - 161.1195681661,
        288.1221695283 * np.power(np.maximum(t - 60, 1e-6), -0.0755148492),
    )
    b = np.where(
        t >= 66, 255.0,
        np.where(t <= 19, 0.0, 138.5177312231 * np.log(np.maximum(t - 10, 1e-6)) - 305.0447927307),
    )
    rgb = np.clip(np.stack([r, g, b], axis=-1) / 255.0, 0.0, 1.0) ** 2.2
    rgb /= np.maximum(rgb.max(axis=-1, keepdims=True), 1e-6)
    return rgb * (1.0 - desaturate) + desaturate


_SKY_CLASSES = [  # weight, Tmin, Tmax  (as seen in the sky: hot stars over-represented)
    (0.24, 2600, 3800),
    (0.30, 3800, 5200),
    (0.18, 5200, 6000),
    (0.12, 6000, 7500),
    (0.09, 7500, 10000),
    (0.07, 10000, 28000),
]


def sample_star_temperatures(rng: np.random.Generator, n: int) -> np.ndarray:
    w = np.array([c[0] for c in _SKY_CLASSES])
    idx = rng.choice(len(_SKY_CLASSES), size=n, p=w / w.sum())
    lo = np.array([c[1] for c in _SKY_CLASSES])[idx]
    hi = np.array([c[2] for c in _SKY_CLASSES])[idx]
    return rng.uniform(lo, hi)


# ---------------------------------------------------------------- bodies

SUN, TERRAN, GAS, ICE, LAVA, ROCKY, MOON = range(7)
KIND_NAMES = {
    SUN: "star", TERRAN: "terrestrial world", GAS: "gas giant", ICE: "ice world",
    LAVA: "volcanic world", ROCKY: "rocky world", MOON: "moon",
}


@dataclass
class Body:
    name: str
    type_id: int
    pos: np.ndarray
    radius: float
    seed: float
    c0: tuple = (0.5, 0.5, 0.5, 0.0)
    c1: tuple = (0.5, 0.5, 0.5, 0.0)
    c2: tuple = (0.5, 0.5, 0.5, 0.0)
    atm: tuple = (0.0, 0.0, 0.0, 0.0)
    misc: tuple = (1.0, 0.0, 0.0, 0.0)
    ring: tuple = (0.0, 0.0, 0.0, 0.0)
    ring_inner: float = 0.0
    ring_outer: float = 0.0
    tilt: np.ndarray = field(default_factory=lambda: np.eye(3))
    spin: float = 0.0
    phase: float = 0.0
    light: tuple | None = None        # irradiance colour for suns
    flyby: np.ndarray | None = None   # autopilot waypoint for path planets
    passed: bool = False
    departing: bool = False
    fade: float = 0.0
    fade_target: float = 1.0
    fade_rate: float = 0.7
    description: str = ""

    @property
    def is_sun(self) -> bool:
        return self.type_id == SUN

    @property
    def kind(self) -> str:
        return self.description or KIND_NAMES[self.type_id]

    def orientation(self, t: float) -> np.ndarray:
        return self.tilt @ rot_y(self.spin * t + self.phase)


@dataclass
class SystemInfo:
    name: str
    star_label: str
    n_planets: int
    binary: bool


GREEK = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa",
         "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho", "Sigma", "Tau", "Upsilon", "Phi",
         "Chi", "Psi", "Omega"]
CONSTELLATIONS = ["Ceti", "Eridani", "Draconis", "Cygni", "Lyrae", "Pavonis", "Centauri", "Aquilae",
                  "Hydrae", "Tauri", "Orionis", "Pegasi", "Reticuli", "Leonis", "Serpentis", "Velorum",
                  "Carinae", "Andromedae", "Persei", "Sagittarii", "Indi", "Tucanae", "Ursae Majoris",
                  "Cassiopeiae", "Aurigae", "Columbae", "Fornacis", "Lupi"]


def make_system_name(rng: np.random.Generator) -> str:
    r = rng.random()
    if r < 0.35:
        return f"{rng.choice(GREEK)} {rng.choice(CONSTELLATIONS)}"
    if r < 0.7:
        cat = rng.choice(["HD", "HIP", "Gliese", "GJ", "LHS", "Ross", "Wolf", "Lalande"])
        return f"{cat} {rng.integers(100, 99999)}"
    cat = rng.choice(["Kepler", "TOI", "K2", "WASP", "TRAPPIST", "KELT", "HAT-P", "CoRoT"])
    return f"{cat}-{rng.integers(1, 2400)}"


STAR_TYPES = [  # class, weight, Tmin, Tmax, size multiplier, description, luminosity class
    ("O", 0.02, 30000, 42000, 1.9, "blue giant", "III"),
    ("B", 0.05, 11000, 28000, 1.6, "blue-white star", "IV"),
    ("A", 0.09, 7600, 10000, 1.3, "white star", "V"),
    ("F", 0.15, 6100, 7400, 1.1, "yellow-white dwarf", "V"),
    ("G", 0.23, 5300, 6000, 1.0, "yellow dwarf", "V"),
    ("K", 0.26, 3900, 5200, 0.85, "orange dwarf", "V"),
    ("M", 0.20, 2500, 3800, 0.7, "red dwarf", "V"),
]

PLANET_WEIGHTS = {  # terran, gas, ice, lava, rocky
    "hot": (0.0, 0.40, 0.10, 0.25, 0.25),
    "sunlike": (0.27, 0.30, 0.12, 0.09, 0.22),
    "cool": (0.20, 0.30, 0.15, 0.10, 0.25),
    "red": (0.10, 0.25, 0.25, 0.15, 0.25),
}
PLANET_TYPES = (TERRAN, GAS, ICE, LAVA, ROCKY)

GAS_PALETTES = [
    ((0.78, 0.62, 0.45), (0.52, 0.33, 0.20), (0.92, 0.86, 0.74)),
    ((0.86, 0.76, 0.52), (0.70, 0.58, 0.38), (0.95, 0.90, 0.72)),
    ((0.16, 0.32, 0.80), (0.28, 0.48, 0.92), (0.65, 0.78, 1.00)),
    ((0.50, 0.78, 0.84), (0.40, 0.66, 0.74), (0.78, 0.92, 0.95)),
    ((0.45, 0.26, 0.56), (0.22, 0.44, 0.52), (0.82, 0.62, 0.86)),
    ((0.85, 0.45, 0.25), (0.55, 0.20, 0.12), (0.98, 0.80, 0.55)),
    ((0.35, 0.55, 0.35), (0.55, 0.65, 0.40), (0.85, 0.88, 0.70)),
]
OCEANS = [(0.015, 0.06, 0.18), (0.01, 0.09, 0.15), (0.03, 0.05, 0.16), (0.02, 0.10, 0.12)]
LANDS = [(0.10, 0.20, 0.06), (0.20, 0.20, 0.08), (0.32, 0.20, 0.10), (0.12, 0.14, 0.10),
         (0.22, 0.08, 0.20), (0.30, 0.10, 0.06)]
HIGHLANDS = [(0.32, 0.28, 0.24), (0.40, 0.36, 0.32), (0.28, 0.26, 0.26)]
SKIES = [(0.30, 0.55, 1.0), (0.45, 0.68, 1.0), (0.60, 0.50, 1.0), (0.35, 0.75, 0.85)]
ROCKY_PALETTES = [
    ((0.42, 0.20, 0.10), (0.60, 0.36, 0.20), (0.30, 0.15, 0.09)),
    ((0.30, 0.27, 0.24), (0.45, 0.42, 0.38), (0.22, 0.20, 0.18)),
    ((0.55, 0.48, 0.20), (0.70, 0.62, 0.30), (0.35, 0.28, 0.12)),
    ((0.38, 0.30, 0.26), (0.55, 0.46, 0.40), (0.24, 0.20, 0.18)),
]
RING_COLOURS = [(0.75, 0.68, 0.55), (0.60, 0.60, 0.62), (0.80, 0.76, 0.70), (0.58, 0.47, 0.38)]


def _jit(rng, c, amt=0.08):
    return np.clip(np.asarray(c) * (1.0 + rng.uniform(-amt, amt, 3)), 0.0, 1.0)


def _c4(c, w=0.0):
    return (float(c[0]), float(c[1]), float(c[2]), float(w))


def _make_planet(rng, ptype, name, center, radius, f, ringed=False, ring_outer=0.0) -> Body:
    seed = float(rng.uniform(0, 60))
    b = Body(name=name, type_id=ptype, pos=center, radius=radius, seed=seed)
    b.phase = float(rng.uniform(0, 2 * math.pi))
    # orientation: for ringed planets keep the ring plane oblique to the approach direction
    s1, s2 = perpendicular_basis(f)
    phi = rng.uniform(0, 2 * math.pi)
    a = math.radians(rng.uniform(35, 72) if ringed else rng.uniform(0, 90))
    up = f * math.cos(a) + (s1 * math.cos(phi) + s2 * math.sin(phi)) * math.sin(a)
    b.tilt = frame_with_up(up)

    if ptype == TERRAN:
        b.c0 = _c4(_jit(rng, OCEANS[rng.integers(len(OCEANS))]))
        b.c1 = _c4(_jit(rng, LANDS[rng.integers(len(LANDS))]))
        b.c2 = _c4(_jit(rng, HIGHLANDS[rng.integers(len(HIGHLANDS))]))
        b.atm = _c4(SKIES[rng.integers(len(SKIES))], rng.uniform(0.035, 0.055))
        city = rng.uniform(0.6, 1.0) if rng.random() < 0.55 else 0.0
        b.misc = (1.0, float(rng.uniform(-0.08, 0.12)), float(rng.uniform(0.35, 0.6)), float(city))
        b.spin = 0.015
        if city > 0:
            b.description = "inhabited terrestrial world"
    elif ptype == GAS:
        pal = GAS_PALETTES[rng.integers(len(GAS_PALETTES))]
        b.c0, b.c1, b.c2 = (_c4(_jit(rng, c)) for c in pal)
        atm = np.clip(np.asarray(pal[2]) * 0.8 + 0.1, 0, 1)
        b.atm = _c4(atm, rng.uniform(0.02, 0.03))
        b.misc = (1.0, float(rng.uniform(12, 26)), float(rng.uniform(-0.45, 0.45)), float(rng.uniform(0.6, 1.5)))
        b.spin = 0.03
        if ringed:
            b.ring_inner = float(rng.uniform(1.25, 1.5))
            b.ring_outer = ring_outer
            b.ring = _c4(_jit(rng, RING_COLOURS[rng.integers(len(RING_COLOURS))], 0.1), rng.uniform(0.7, 0.95))
            b.description = "ringed gas giant"
    elif ptype == ICE:
        b.c0 = _c4(_jit(rng, (0.82, 0.88, 0.94), 0.05))
        b.c1 = _c4(_jit(rng, (0.25, 0.45, 0.70)))
        b.c2 = _c4(_jit(rng, (0.70, 0.78, 0.88), 0.05))
        if rng.random() < 0.5:
            b.atm = _c4((0.55, 0.75, 1.0), 0.018)
        b.spin = 0.01
    elif ptype == LAVA:
        b.c0 = _c4((0.035, 0.03, 0.03))
        b.c1 = _c4(_jit(rng, (0.13, 0.09, 0.07)))
        b.c2 = _c4(_jit(rng, (1.0, 0.32, 0.06), 0.1))
        b.atm = _c4((1.0, 0.45, 0.2), 0.02)
        b.spin = 0.01
    elif ptype == ROCKY:
        pal = ROCKY_PALETTES[rng.integers(len(ROCKY_PALETTES))]
        b.c0, b.c1, b.c2 = (_c4(_jit(rng, c)) for c in pal)
        if rng.random() < 0.4:
            b.atm = _c4((0.9, 0.6, 0.4), 0.015)
        b.misc = (1.0, 1.0 if rng.random() < 0.5 else 0.0, 0.0, 0.0)
        b.spin = 0.01
    else:  # MOON
        g = rng.uniform(0.8, 1.15)
        b.c0 = _c4(np.array((0.28, 0.27, 0.26)) * g)
        b.c1 = _c4(np.array((0.42, 0.41, 0.40)) * g)
        b.c2 = _c4(np.array((0.18, 0.18, 0.18)) * g)
        b.spin = 0.005
    return b


def _planet_radius(rng, ptype) -> float:
    lo, hi = {TERRAN: (260, 420), GAS: (650, 1100), ICE: (200, 380), LAVA: (180, 320),
              ROCKY: (180, 360), MOON: (50, 130)}[ptype]
    return float(rng.uniform(lo, hi))


def _make_sun(rng, name, pos, radius, T, intensity, label) -> Body:
    colour = kelvin_to_rgb(np.array([T]), desaturate=0.15)[0]
    b = Body(name=name, type_id=SUN, pos=pos, radius=radius, seed=float(rng.uniform(0, 60)))
    b.c0 = _c4(colour, 28.0)
    b.light = tuple(float(x) for x in colour * intensity)
    b.tilt = frame_with_up(random_unit(rng))
    b.spin = 0.003
    b.description = label
    return b


def generate_system(rng: np.random.Generator, origin: np.ndarray, R: np.ndarray,
                    first_distance: float = 16000.0) -> tuple[SystemInfo, list[Body]]:
    """Build a star system laid out along the ship's current heading."""
    f, s1, s2 = R[:, 2], R[:, 0], R[:, 1]
    weights = np.array([t[1] for t in STAR_TYPES])
    cls, _, tmin, tmax, size, desc, lum = STAR_TYPES[rng.choice(len(STAR_TYPES), p=weights / weights.sum())]
    T = rng.uniform(tmin, tmax)
    sub = min(9, int(10 * (tmax - T) / (tmax - tmin)))
    star_label = f"{cls}{sub}{lum} {desc}"
    name = make_system_name(rng)

    bodies: list[Body] = []
    a = math.radians(rng.uniform(18, 48))
    b = rng.uniform(0, 2 * math.pi)
    sdir = f * math.cos(a) + (s1 * math.cos(b) + s2 * math.sin(b)) * math.sin(a)
    dist = rng.uniform(90000, 140000)
    radius = dist * math.tan(math.radians(rng.uniform(0.9, 1.6) * size))
    binary = rng.random() < 0.18
    sun = _make_sun(rng, name + (" A" if binary else ""), origin + sdir * dist, radius, T,
                    2.2 if binary else 2.8, star_label)
    bodies.append(sun)
    if binary:
        cls2, _, tmin2, tmax2, size2, desc2, lum2 = STAR_TYPES[rng.integers(3, len(STAR_TYPES))]
        T2 = rng.uniform(tmin2, tmax2)
        off = rot_axis(random_unit(rng), math.radians(rng.uniform(4, 9))) @ sdir
        d2 = dist * rng.uniform(0.9, 1.1)
        r2 = d2 * math.tan(math.radians(rng.uniform(0.5, 1.0) * size2))
        sub2 = min(9, int(10 * (tmax2 - T2) / (tmax2 - tmin2)))
        bodies.append(_make_sun(rng, name + " B", origin + off * d2, r2, T2, 1.2,
                                f"{cls2}{sub2}{lum2} {desc2}"))

    group = "hot" if cls in "OBA" else "sunlike" if cls in "FG" else "cool" if cls == "K" else "red"
    pw = np.array(PLANET_WEIGHTS[group])
    n = int(rng.integers(2, 5))
    d = first_distance
    letters = "bcdefgh"
    for k in range(n):
        ptype = PLANET_TYPES[rng.choice(len(PLANET_TYPES), p=pw / pw.sum())]
        pr = _planet_radius(rng, ptype)
        ringed = ptype == GAS and rng.random() < 0.55
        ring_outer = float(rng.uniform(1.9, 2.5)) if ringed else 0.0
        phi = rng.uniform(0, 2 * math.pi)
        u = s1 * math.cos(phi) + s2 * math.sin(phi)
        clearance = pr * (ring_outer * rng.uniform(1.06, 1.25) if ringed else rng.uniform(1.3, 2.3))
        path_pt = origin + f * d
        center = path_pt + u * clearance
        pname = f"{name} {letters[k]}"
        planet = _make_planet(rng, ptype, pname, center, pr, f, ringed, ring_outer)
        planet.flyby = path_pt
        bodies.append(planet)

        n_moons = int(rng.integers(0, 3)) if ptype == GAS else int(rng.random() < 0.4)
        for m in range(n_moons):
            mr = max(45.0, pr * rng.uniform(0.08, 0.22)) if ptype == GAS else max(45.0, pr * rng.uniform(0.18, 0.3))
            dphi = phi + rng.uniform(-1.0, 1.0)
            um = s1 * math.cos(dphi) + s2 * math.sin(dphi)
            mdist = pr * max(rng.uniform(2.4, 4.0), ring_outer + 0.6)
            mcenter = center + um * mdist + f * rng.uniform(-1.0, 1.0) * pr
            bodies.append(_make_planet(rng, MOON, f"{pname} {'I' * (m + 1)}", mcenter, mr, f))
        d += rng.uniform(9000, 14000) + pr * 3

    # a huge planet far from the flight path, part of the scenery
    if rng.random() < 0.45:
        va = math.radians(rng.uniform(25, 55))
        vb = rng.uniform(0, 2 * math.pi)
        vdir = f * math.cos(va) + (s1 * math.cos(vb) + s2 * math.sin(vb)) * math.sin(va)
        vdist = rng.uniform(60000, 90000)
        vr = rng.uniform(4000, 8000)
        ringed = rng.random() < 0.6
        vplanet = _make_planet(rng, GAS, f"{name} {letters[n]}", origin + vdir * vdist, vr, vdir,
                               ringed, float(rng.uniform(1.9, 2.5)) if ringed else 0.0)
        bodies.append(vplanet)
        n += 1

    return SystemInfo(name, star_label, n, binary), bodies
