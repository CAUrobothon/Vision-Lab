"""Synthetic overhead scenes of the competition arena, for tests, samples and the benchmark.

The renderer uses the real camera geometry (pinhole, f = 1410.6 px at 1280x720, lens 601 mm
above the board, straight down, principal point at the image centre) and the arena from the rule
drawings (arena frame: origin = robot axis, +x right, +y towards the pillar, millimetres).
Cubes are true 3D boxes, so their side faces show up by perspective exactly as in the real
camera. Lighting, shadows, noise, blur and JPEG are added on top.

Synthetic data proves the plumbing and catches geometry bugs. It does NOT prove accuracy on real
photographs: printed paper, real light and the real lens are only imitated here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from cubevision.postprocess import order_quad, quad_angle

# --- camera and arena (rule drawings 1/3-3/3) -------------------------------------------------
FOCAL_PX = 1410.6  # at 1280x720 (55 deg diagonal field of view)
CAMERA_MM = 601.0  # lens height above the board
CUBE_MM = 25.0
CAMERA_XY_MM = (-13.1, 131.4)  # point under the lens in the arena frame
ARENA_R_MM = 250.5
ZONE_R_MM = (90.0, 250.0)  # placement ring
PILLAR = (-50.15, 50.15, 250.0, 315.3, 201.0)  # x0, x1, y0, y1, height
BASKETS = (("blue", 0.0, 20.0), ("green", 20.0, 50.0), ("red", 50.0, 90.0))  # mission B, degrees
BASKET_WALL_MM, BASKET_RIM_MM, BASKET_FLOOR_MM = 7.0, 32.0, 4.0

# Base sRGB colours (printed paper is duller than screen primaries). Jittered per scene.
PAPER_RGB = {
    "red": (196, 46, 56),
    "green": (58, 148, 80),
    "blue": (44, 80, 170),
    "wood": (204, 170, 126),
}
BOARD_RGB = (228, 228, 224)
SS = 2  # supersampling factor of the render canvas (anti-aliasing and sub-pixel edges)

Mission = Literal["A", "B"]

# make_scene(seed, **CLEAN): no noise, blur, JPEG, light gradient, shadows, marks or arm.
CLEAN: dict[str, Any] = dict(
    noise=0, blur=0, jpeg=0, light=0, vignette=0, shadows=False, marks=False, arm=False, exposure=1.0, wb=(1, 1, 1)
)


@dataclass
class Camera:
    width: int = 1280
    height: int = 720

    @property
    def f(self) -> float:
        return FOCAL_PX * self.width / 1280.0

    @property
    def pp(self) -> tuple[float, float]:
        return self.width / 2.0, self.height / 2.0

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """(N, 3) arena points in mm -> (N, 2) image pixels."""
        p = np.asarray(xyz, np.float64).reshape(-1, 3)
        s = self.f / (CAMERA_MM - p[:, 2])
        u = self.pp[0] + s * (p[:, 0] - CAMERA_XY_MM[0])
        v = self.pp[1] - s * (p[:, 1] - CAMERA_XY_MM[1])
        return np.stack([u, v], axis=1)


@dataclass
class Placement:
    """One cube in the arena frame. ``level`` 1 = on the board, k = on top of k-1 cubes."""

    color: str
    x: float
    y: float
    angle: float  # degrees, rotation about the vertical axis in the arena frame
    level: int = 1
    stack_xy: tuple[float, float] | None = None  # base of the tower this cube belongs to

    def corners(self, z: float) -> np.ndarray:
        a = math.radians(self.angle)
        c, s = math.cos(a), math.sin(a)
        h = CUBE_MM / 2.0
        local = np.array([(-h, -h), (h, -h), (h, h), (-h, h)])
        xy = local @ np.array([[c, s], [-s, c]]) + (self.x, self.y)
        return np.column_stack([xy, np.full(4, z)])


@dataclass
class Look:
    """Photometric settings of one scene (all drawn from the seed)."""

    paper: str  # "top" or "all": which faces carry the coloured paper
    colors: dict[str, np.ndarray]  # class -> BGR float
    board: np.ndarray
    table: np.ndarray
    light_dir: tuple[float, float]  # shadow offset per mm of height (arena frame)
    shadow_strength: float
    shadow_blur: float
    exposure: float
    wb: np.ndarray  # BGR gains
    gradient: float
    vignette: float
    noise: float
    blur: float
    jpeg: int
    marks: bool
    shadows: bool
    seed: int = 0
    grad_dir: float = 0.0


@dataclass
class Arm:
    target: tuple[float, float]  # gripper tip in the arena frame (mm)
    height: float = 110.0
    color: tuple[int, int, int] = (52, 52, 58)


@dataclass
class SceneSpec:
    mission: Mission
    cubes: list[Placement]
    look: Look
    arm: Arm | None = None
    camera: Camera = field(default_factory=Camera)


# --- random scene layout -----------------------------------------------------------------------
def _ring_point(rng: np.random.Generator, deg_lo: float, deg_hi: float, r_lo: float, r_hi: float) -> tuple[float, float]:
    a = math.radians(rng.uniform(deg_lo, deg_hi))
    r = math.sqrt(rng.uniform(r_lo**2, r_hi**2))
    return r * math.cos(a), r * math.sin(a)


def _in_frame(cam: Camera, p: Placement, margin: float = 4.0) -> bool:
    q = cam.project(p.corners(CUBE_MM * p.level))
    b = cam.project(p.corners(CUBE_MM * (p.level - 1)))
    pts = np.vstack([q, b])
    return bool(
        pts[:, 0].min() >= margin and pts[:, 1].min() >= margin
        and pts[:, 0].max() <= cam.width - 1 - margin and pts[:, 1].max() <= cam.height - 1 - margin
    )


def _in_zone(p: Placement, sectors: list[tuple[float, float]]) -> bool:
    """All four base corners inside the placement ring and inside one of the angular sectors."""
    xy = p.corners(0)[:, :2]
    r = np.hypot(xy[:, 0], xy[:, 1])
    deg = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    if r.min() < ZONE_R_MM[0] + 1 or r.max() > ZONE_R_MM[1] - 1:
        return False
    return any(bool(np.all((deg >= lo) & (deg <= hi))) for lo, hi in sectors)


def _scatter(
    rng: np.random.Generator,
    cam: Camera,
    colors: list[str],
    sectors: list[tuple[float, float]],
    placed: list[Placement],
    touching: bool,
) -> list[Placement]:
    """Place cubes in the given angular sectors of the placement ring without overlaps.

    With ``touching`` the first few cubes are laid flush against an earlier one (same angle,
    one face apart, sometimes shifted half a face), which is the case splitting must handle.
    """
    out: list[Placement] = []
    min_gap = CUBE_MM * math.sqrt(2) + 6.0
    for i, col in enumerate(colors):
        for attempt in range(600):
            want_flush = touching and i % 3 == 1 and attempt < 200 and bool(out or placed)
            anchor = (out + placed)[-1] if want_flush else None
            if anchor is not None and anchor.level == 1:
                a = math.radians(anchor.angle)
                axis = rng.integers(4)
                ux, uy = math.cos(a + axis * math.pi / 2), math.sin(a + axis * math.pi / 2)
                shift = rng.choice([0.0, 0.0, 0.5, -0.5]) * CUBE_MM
                x = anchor.x + ux * CUBE_MM - uy * shift
                y = anchor.y + uy * CUBE_MM + ux * shift
                cand = Placement(col, x, y, anchor.angle)
                skip = anchor
            else:
                lo, hi = sectors[rng.integers(len(sectors))]
                x, y = _ring_point(rng, lo, hi, ZONE_R_MM[0] + 18, ZONE_R_MM[1] - 18)
                cand = Placement(col, x, y, float(rng.uniform(0, 90)))
                skip = None
            if not _in_zone(cand, sectors) or not _in_frame(cam, cand):
                continue
            ok = all(
                math.hypot(cand.x - o.x, cand.y - o.y) >= (CUBE_MM - 0.01 if o is skip else min_gap)
                for o in out + placed
            )
            if ok:
                out.append(cand)
                break
    return out


def _random_look(rng: np.random.Generator, seed: int, **v: Any) -> Look:
    def jitter(rgb: tuple[int, int, int], amount: float) -> np.ndarray:
        return np.clip(np.array(rgb[::-1], np.float64) + rng.uniform(-amount, amount, 3), 0, 255)

    colors = {k: jitter(c, 12) for k, c in PAPER_RGB.items()}
    ang = rng.uniform(0, 2 * math.pi)
    length = rng.uniform(0.15, 0.6)
    return Look(
        paper=v.get("paper") or str(rng.choice(["top", "all"])),
        colors=colors,
        board=jitter(BOARD_RGB, 5),
        table=np.clip(np.array([62.0, 66.0, 70.0]) + rng.uniform(-15, 15) + rng.uniform(-4, 4, 3), 0, 255),
        light_dir=(length * math.cos(ang), length * math.sin(ang)),
        shadow_strength=float(v.get("shadow_strength", rng.uniform(0.15, 0.35))),
        shadow_blur=float(rng.uniform(2.0, 6.0)),
        exposure=float(v.get("exposure", rng.uniform(0.88, 1.06))),
        wb=np.array(v.get("wb", rng.uniform(0.95, 1.05, 3)), np.float64),
        gradient=float(v.get("light", rng.uniform(0.04, 0.14))),
        vignette=float(v.get("vignette", rng.uniform(0.0, 0.12))),
        noise=float(v.get("noise", rng.uniform(1.5, 3.5))),
        blur=float(v.get("blur", rng.uniform(0.5, 0.9))),
        jpeg=int(v.get("jpeg", rng.integers(72, 93))),
        marks=bool(v.get("marks", rng.random() < 0.6)),
        shadows=bool(v.get("shadows", True)),
        seed=seed,
        grad_dir=float(rng.uniform(0, 2 * math.pi)),
    )


def random_spec(seed: int, mission: Mission = "A", size: tuple[int, int] = (1280, 720), **v: Any) -> SceneSpec:
    """Draw a scene layout and look from ``seed`` (same seed and arguments -> same scene)."""
    rng = np.random.default_rng(seed)
    cam = Camera(*size)
    look = _random_look(rng, seed, **v)
    touching = bool(v.get("touching", rng.random() < 0.5))
    cubes: list[Placement] = []
    if v.get("cubes") is not None:
        for c in v["cubes"]:
            cubes.append(Placement(c["color"], float(c["x"]), float(c["y"]), float(c.get("angle", 0.0)), int(c.get("level", 1))))
    elif mission == "A":
        levels = int(v.get("tower", rng.integers(0, 5)))
        levels = max(0, min(levels, 9))
        colors = ["red", "green", "blue"] * 3
        rng.shuffle(colors)
        if levels:
            # Zone 2 (60-120 deg) holds the tower; keep it near the point under the camera.
            tx, ty = _ring_point(rng, 75, 105, 105, 170)
            ta = float(rng.uniform(0, 90))
            for k in range(levels):
                cubes.append(Placement(colors[k], tx + rng.normal(0, 0.6), ty + rng.normal(0, 0.6), ta + rng.normal(0, 2), k + 1, (tx, ty)))
        cubes += _scatter(rng, cam, colors[levels:], [(0, 60), (120, 180)], cubes, touching)
    else:
        colors = ["red", "green", "blue", "wood"] * 3
        rng.shuffle(colors)
        cubes += _scatter(rng, cam, colors, [(92, 178)], cubes, touching)
    arm = None
    if v.get("arm", rng.random() < 0.3):
        tgt = cubes[int(rng.integers(len(cubes)))] if cubes else Placement("red", -120.0, 120.0, 0.0)
        pull = 1.0 - rng.uniform(15, 30) / max(math.hypot(tgt.x, tgt.y), 1.0)  # hover just inside the cube
        arm = Arm(target=(tgt.x * pull, tgt.y * pull))
    return SceneSpec(mission, cubes, look, arm, cam)


# --- drawing helpers ---------------------------------------------------------------------------
def _fill(canvas: np.ndarray, pts_px: np.ndarray, color: Any) -> None:
    """Fill a polygon given in 1x image pixels onto the supersampled canvas (sub-pixel exact)."""
    p = np.round((np.asarray(pts_px, np.float64) * SS + (SS - 1) / 2.0) * 16).astype(np.int32)
    cv2.fillPoly(canvas, [p.reshape(-1, 1, 2)], color, cv2.LINE_8, shift=4)


def _arc(r: float, deg_lo: float, deg_hi: float, n: int = 48) -> np.ndarray:
    t = np.radians(np.linspace(deg_lo, deg_hi, n))
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _at(xy: np.ndarray, z: float) -> np.ndarray:
    return np.column_stack([xy, np.full(len(xy), z)])


def _sector(r0: float, r1: float, d0: float, d1: float, wall: float = 0.0) -> np.ndarray:
    """Annular sector in the arena frame, shrunk by ``wall`` mm on all four sides."""
    ri, ro = r0 + wall, r1 - wall
    pts = []
    for r, lo, hi in ((ro, d0, d1), (ri, d1, d0)):
        lo2 = lo + math.degrees(math.asin(wall / r)) * (1 if lo < hi else -1)
        hi2 = hi - math.degrees(math.asin(wall / r)) * (1 if lo < hi else -1)
        pts.append(_arc(r, lo2, hi2, 40))
    return np.vstack(pts)


def _board_outline() -> tuple[np.ndarray, np.ndarray]:
    half = _arc(ARENA_R_MM, 0, 180, 120)
    mount = np.array([[-62.0, 230.0], [-62.0, 345.0], [62.0, 345.0], [62.0, 230.0]])
    bottom = np.array([[-ARENA_R_MM, -10.0], [ARENA_R_MM, -10.0]])
    return np.vstack([bottom[1:], half, bottom[:1]]), mount


def _render_background(spec: SceneSpec) -> np.ndarray:
    """Board, grooves, marks, pillar, robot base and (mission B) baskets on the 2x canvas."""
    cam, look = spec.camera, spec.look
    canvas = np.empty((cam.height * SS, cam.width * SS, 3), np.uint8)
    canvas[:] = np.round(look.table).astype(np.uint8)
    outline, mount = _board_outline()
    board = tuple(int(c) for c in np.round(look.board))
    _fill(canvas, cam.project(_at(outline, 0)), board)
    _fill(canvas, cam.project(_at(mount, 0)), board)

    groove = tuple(int(c) for c in np.round(look.board * 0.86))
    lines: list[np.ndarray] = [_arc(ZONE_R_MM[0], 0, 180), _arc(ZONE_R_MM[1], 0, 180, 120), _arc(ARENA_R_MM - 0.3, 0, 180, 120)]
    radial = (60.0, 120.0) if spec.mission == "A" else (90.0,)
    for d in radial:
        t = math.radians(d)
        lines.append(np.array([[ZONE_R_MM[0] * math.cos(t), ZONE_R_MM[0] * math.sin(t)], [ZONE_R_MM[1] * math.cos(t), ZONE_R_MM[1] * math.sin(t)]]))
    for ln in lines:
        p = np.round(cam.project(_at(ln, 0)) * SS * 16).astype(np.int32)
        cv2.polylines(canvas, [p.reshape(-1, 1, 2)], False, groove, 2, cv2.LINE_AA, shift=4)

    if look.marks:
        _draw_marks(canvas, spec)

    # Robot base (dark, about 40 mm tall) at the bottom centre.
    base = np.array([[-53.0, 0.0], [53.0, 0.0], [32.0, 26.1], [-32.0, 26.1]])
    _fill(canvas, cam.project(_at(base, 40.0)), (38, 38, 42))

    # Pillar: only its front face (facing the camera) is in view.
    x0, x1, y0, _, hgt = PILLAR
    front = np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y0, hgt], [x0, y0, hgt]])
    _fill(canvas, cam.project(front), (128, 130, 132))

    if spec.mission == "B":
        for name, d0, d1 in BASKETS:
            _draw_basket(canvas, cam, look, name, d0, d1)
    return canvas


def _draw_marks(canvas: np.ndarray, spec: SceneSpec) -> None:
    """Placement marks 1-20: thin printed square outlines with a number (their form is a guess)."""
    cam = spec.camera
    rng = np.random.default_rng(spec.look.seed + 7)
    grey = tuple(int(c) for c in np.round(spec.look.board * 0.78))
    spots = [(c.x, c.y) for c in spec.cubes if c.level == 1] + [
        _ring_point(rng, 95, 175, 110, 230) for _ in range(20)
    ]
    for i, (x, y) in enumerate(spots[:20]):
        sq = Placement("wood", x, y, 0.0).corners(0)
        sq[:, :2] = (sq[:, :2] - (x, y)) * (30.0 / CUBE_MM) + (x, y)
        p = np.round(cam.project(sq) * SS * 16).astype(np.int32)
        cv2.polylines(canvas, [p.reshape(-1, 1, 2)], True, grey, 2, cv2.LINE_AA, shift=4)
        org = cam.project(np.array([[x - 13, y - 13, 0]]))[0] * SS
        cv2.putText(canvas, str(i + 1), (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * SS / 2, grey, SS, cv2.LINE_AA)


def _draw_basket(canvas: np.ndarray, cam: Camera, look: Look, name: str, d0: float, d1: float) -> None:
    """A basket: walls up to 32 mm, floor at 4 mm covered with the class paper."""
    outer = _sector(ZONE_R_MM[0], ZONE_R_MM[1], d0, d1)
    inner = _sector(ZONE_R_MM[0], ZONE_R_MM[1], d0, d1, BASKET_WALL_MM)
    wall_side = tuple(int(c) for c in np.round(look.board * 0.82))
    rim = tuple(int(c) for c in np.round(look.board * 0.97))
    inner_wall = tuple(int(c) for c in np.round(look.board * 0.7))
    _fill(canvas, cam.project(_at(outer, 0)), wall_side)
    _fill(canvas, cam.project(_at(outer, BASKET_RIM_MM)), rim)
    _fill(canvas, cam.project(_at(inner, BASKET_RIM_MM)), inner_wall)
    # The floor is visible only through the rim's inner opening.
    h, w = canvas.shape[:2]
    m1 = np.zeros((h, w), np.uint8)
    m2 = np.zeros((h, w), np.uint8)
    _fill(m1, cam.project(_at(inner, BASKET_FLOOR_MM)), 255)
    _fill(m2, cam.project(_at(inner, BASKET_RIM_MM)), 255)
    floor = cv2.bitwise_and(m1, m2) > 0
    canvas[floor] = np.round(np.clip(look.colors[name] * 1.04, 0, 255)).astype(np.uint8)


def _shadow_polys(spec: SceneSpec) -> list[np.ndarray]:
    """Shadows of the cubes on the board (z = 0), as image polygons."""
    cam, (lx, ly) = spec.camera, spec.look.light_dir
    out = []
    for c in spec.cubes:
        z0, z1 = CUBE_MM * (c.level - 1), CUBE_MM * c.level
        pts = np.vstack([c.corners(z0)[:, :2] + (lx * z0, ly * z0), c.corners(z1)[:, :2] + (lx * z1, ly * z1)])
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        out.append(cam.project(_at(hull, 0)))
    return out


def _arm_polys(arm: Arm) -> list[tuple[np.ndarray, float]]:
    """Arm link and two gripper fingers in the arena frame with their heights (mm)."""
    tx, ty = arm.target
    r = max(math.hypot(tx, ty), 1.0)
    ux, uy = tx / r, ty / r
    nx, ny = -uy, ux
    link_end = (tx - ux * 30, ty - uy * 30)

    def bar(p0: tuple[float, float], p1: tuple[float, float], half: float) -> np.ndarray:
        return np.array([
            [p0[0] + nx * half, p0[1] + ny * half], [p1[0] + nx * half, p1[1] + ny * half],
            [p1[0] - nx * half, p1[1] - ny * half], [p0[0] - nx * half, p0[1] - ny * half],
        ])

    polys = [(bar((ux * 20, uy * 20), link_end, 22.0), arm.height)]
    for side in (-1, 1):
        f0 = (link_end[0] + nx * side * 14, link_end[1] + ny * side * 14)
        f1 = (tx + nx * side * 14, ty + ny * side * 14)
        polys.append((bar(f0, f1, 5.0), arm.height - 40))
    return polys


def _photometric(img: np.ndarray, look: Look, rng: np.random.Generator) -> np.ndarray:
    """Exposure, white balance, light gradient, vignette, blur, sensor noise, JPEG."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    xn = (np.arange(w, dtype=np.float32) - w / 2) / (w / 2)
    yn = (np.arange(h, dtype=np.float32) - h / 2) / (h / 2)
    field_ = 1.0 + look.gradient * (math.cos(look.grad_dir) * xn[None, :] + math.sin(look.grad_dir) * yn[:, None])
    field_ -= look.vignette * (xn[None, :] ** 2 + yn[:, None] ** 2) / 2.0
    gains = (look.exposure * look.wb).astype(np.float32)
    out *= field_[..., None] * gains[None, None, :]
    if look.blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), look.blur)
    if look.noise > 0:
        out += rng.standard_normal(out.shape, dtype=np.float32) * np.float32(look.noise)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if look.jpeg > 0:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(look.jpeg)])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def render(
    spec: SceneSpec, background: np.ndarray | None = None, noise_seed: int | None = None, *, with_gt: bool = True
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Render ``spec``. ``background`` (from ``_render_background``) may be reused across frames."""
    cam, look = spec.camera, spec.look
    canvas = (background if background is not None else _render_background(spec)).copy()

    if look.shadows:
        sh = np.zeros((cam.height, cam.width), np.float32)
        for poly in _shadow_polys(spec):
            cv2.fillPoly(sh, [np.round(poly * 16).astype(np.int32).reshape(-1, 1, 2)], 1.0, cv2.LINE_AA, shift=4)
        _apply_shadow(canvas, sh, look)

    # Cubes, farthest from the camera axis first; a tower bottom-up. Track visible top faces.
    ids = np.zeros((cam.height * SS, cam.width * SS), np.int16)
    cx, cy = CAMERA_XY_MM
    def paint_key(i: int) -> tuple[float, int]:
        c = spec.cubes[i]
        bx, by = c.stack_xy or (c.x, c.y)
        return -math.hypot(bx - cx, by - cy), c.level

    order = sorted(range(len(spec.cubes)), key=paint_key)
    for i in order:
        c = spec.cubes[i]
        z0, z1 = CUBE_MM * (c.level - 1), CUBE_MM * c.level
        base, top = c.corners(z0), c.corners(z1)
        face_rgb = look.colors[c.color]
        side_rgb = face_rgb if (look.paper == "all" or c.color == "wood") else look.colors["wood"]
        for k in range(4):
            a, b = base[k, :2], base[(k + 1) % 4, :2]
            mid = (a + b) / 2
            normal = mid - (c.x, c.y)
            if np.dot((cx, cy) - mid, normal) <= 0:
                continue  # faces away from the camera
            quad = cam.project(np.array([base[k], base[(k + 1) % 4], top[(k + 1) % 4], top[k]]))
            shade = _side_shade(normal, look.light_dir)
            _fill(canvas, quad, tuple(float(v) for v in np.clip(side_rgb * shade, 0, 255)))
            _fill(ids, quad, 0)
        tq = cam.project(top)
        _fill(canvas, tq, tuple(float(v) for v in face_rgb))
        _fill(ids, tq, i + 1)

    if spec.arm is not None:
        polys = _arm_polys(spec.arm)
        if look.shadows:
            sh = np.zeros((cam.height, cam.width), np.float32)
            lx, ly = look.light_dir
            for poly, z in polys:
                p = cam.project(_at(poly + (lx * z, ly * z), 0))
                cv2.fillPoly(sh, [np.round(p * 16).astype(np.int32).reshape(-1, 1, 2)], 1.0, cv2.LINE_AA, shift=4)
            _apply_shadow(canvas, sh, look, extra_blur=4.0)
        for poly, z in polys:
            p = cam.project(_at(poly, z))
            _fill(canvas, p, spec.arm.color)
            _fill(ids, p, 0)

    img = cv2.resize(canvas, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(look.seed * 7919 + 13 if noise_seed is None else noise_seed)
    img = _photometric(img, look, rng)
    return img, (_ground_truth(spec, ids) if with_gt else [])


def _side_shade(normal: np.ndarray, light_dir: tuple[float, float]) -> float:
    """Brightness of a vertical face: faces turned towards the light are brighter."""
    n = normal / max(float(np.linalg.norm(normal)), 1e-9)
    lx, ly = light_dir
    ll = max(math.hypot(lx, ly), 1e-9)
    return 0.66 + 0.16 * float(np.dot(n, (-lx / ll, -ly / ll)))


def _apply_shadow(canvas: np.ndarray, sh: np.ndarray, look: Look, extra_blur: float = 0.0) -> None:
    """Darken the canvas by a soft shadow map ``sh`` (1x, values 0..1), in place."""
    sh = cv2.GaussianBlur(sh, (0, 0), look.shadow_blur + extra_blur)
    keep = np.clip(255.0 * (1.0 - look.shadow_strength * sh), 0, 255).astype(np.uint8)
    keep = cv2.resize(keep, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_LINEAR)
    cv2.multiply(canvas, cv2.merge([keep, keep, keep]), dst=canvas, scale=1.0 / 255.0)


def _ground_truth(spec: SceneSpec, ids: np.ndarray) -> list[dict[str, Any]]:
    """One entry per cube whose top face is in view (the top cube of a tower only)."""
    cam = spec.camera
    counts = np.bincount(ids.ravel().astype(np.int64), minlength=len(spec.cubes) + 1)
    gt = []
    for i, c in enumerate(spec.cubes):
        quad = order_quad(cam.project(c.corners(CUBE_MM * c.level)))
        area = float(cv2.contourArea(np.asarray(quad, np.float32)))
        visible = float(counts[i + 1]) / (area * SS * SS) if area > 0 else 0.0
        if visible < 0.02:
            continue
        qa = np.asarray(quad)
        gt.append({
            "color": c.color,
            "quad": [(round(x, 3), round(y, 3)) for x, y in quad],
            "center": (round(float(qa[:, 0].mean()), 3), round(float(qa[:, 1].mean()), 3)),
            "angle": round(quad_angle(quad), 3),
            "level": c.level,
            "visible": round(min(visible, 1.0), 3),
            "side_px": round(math.sqrt(area), 2),
            "arena_mm": (round(c.x, 2), round(c.y, 2)),
        })
    return gt


def make_scene(seed: int, *, mission: Mission = "A", size: tuple[int, int] = (1280, 720), **variation: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Render a seeded scene: returns (BGR image, ground truth per visible cube).

    Ground truth per cube: ``color``, top-face ``quad`` (clockwise, first = smallest x+y),
    ``center``, ``angle`` (degrees in [0, 90)), plus ``level`` (tower height), ``visible``
    (fraction of the top face not hidden by the arm or other cubes) and ``arena_mm``.

    Variations (all optional; anything not given is drawn from the seed):
    ``paper`` "top" | "all"; ``touching`` bool; ``tower`` levels 0-9 (mission A); ``arm`` bool;
    ``marks`` bool; ``shadows`` bool; ``noise``, ``blur`` (sigma), ``jpeg`` (quality, 0 = off),
    ``light`` (gradient strength), ``vignette``, ``exposure``, ``wb`` (BGR gains);
    ``cubes`` = explicit list of {"color", "x", "y", "angle", "level"} in arena mm.
    """
    return render(random_spec(seed, mission, size, **variation))


# --- sample assets for the lab ("합성 샘플 넣기") -------------------------------------------------
def sample_assets(out_dir: Path) -> list[Path]:
    """Write two stills (mission A and B, JPEG) and a 5 s, 30 fps video (mp4v) into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, seed, mission in (("synth_mission_a.jpg", 11, "A"), ("synth_mission_b.jpg", 12, "B")):
        img, _ = make_scene(seed, mission=mission, arm=False, jpeg=0)  # type: ignore[arg-type]
        p = out_dir / name
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(p)
    paths.append(_sample_video(out_dir / "synth_moves.mp4"))
    return paths


def _sample_video(path: Path, seconds: float = 5.0, fps: int = 30) -> Path:
    """Mission A layout; three cubes slide to new places one at a time, the arm follows them."""
    spec = random_spec(21, "A", tower=0, arm=False, touching=False, jpeg=0, marks=True)
    background = _render_background(spec)
    rng = np.random.default_rng(5)
    movers = [int(i) for i in rng.choice(len(spec.cubes), size=3, replace=False)]
    goals = []
    for i in movers:
        c = spec.cubes[i]
        for _ in range(200):
            x, y = _ring_point(rng, 70, 110, 110, 200)
            if all(math.hypot(x - o.x, y - o.y) > 45 for o in spec.cubes) and all(math.hypot(x - g[0], y - g[1]) > 45 for g in goals):
                break
        goals.append((x, y, c.angle + float(rng.uniform(-35, 35))))
    n = int(seconds * fps)
    seg = n // 3
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (spec.camera.width, spec.camera.height))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter(mp4v) could not be opened")
    starts = {i: (spec.cubes[i].x, spec.cubes[i].y, spec.cubes[i].angle) for i in movers}
    try:
        for f in range(n):
            k = min(f // seg, 2)
            t = (f - k * seg) / seg
            # 0-0.2 still, 0.2-0.8 moving (smooth step), 0.8-1 still.
            s = float(np.clip((t - 0.2) / 0.6, 0, 1))
            s = s * s * (3 - 2 * s)
            for j, i in enumerate(movers):
                x0, y0, a0 = starts[i]
                x1, y1, a1 = goals[j]
                w = 1.0 if j < k else (s if j == k else 0.0)
                spec.cubes[i].x, spec.cubes[i].y, spec.cubes[i].angle = x0 + (x1 - x0) * w, y0 + (y1 - y0) * w, a0 + (a1 - a0) * w
            moving = spec.cubes[movers[k]]
            spec.arm = Arm(target=(moving.x * 0.9, moving.y * 0.9)) if 0.1 < t < 0.9 else None
            frame, _ = render(spec, background, noise_seed=1000 + f, with_gt=False)
            writer.write(frame)
    finally:
        writer.release()
    return path
