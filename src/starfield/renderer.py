"""OpenGL rendering: sky, stars, bodies, bloom and the final composite."""

from __future__ import annotations

import math
from pathlib import Path

import moderngl
import numpy as np

from .sim import STAR_BOX, Simulation
from .universe import kelvin_to_rgb, perpendicular_basis, random_unit, sample_star_temperatures, smoothstep

SHADER_DIR = Path(__file__).with_name("shaders")
MAXB = 16
LOG_FAR = math.log2(1.0 + 1.0e7)
SKY_FACE = 2048
BLOOM_LEVELS = 7
GL_TEXTURE_CUBE_MAP_SEAMLESS = 0x884F

NEB_PALETTES = [
    ((0.90, 0.25, 0.55), (0.15, 0.45, 0.90), (1.00, 0.70, 0.40)),
    ((0.20, 0.80, 0.70), (0.50, 0.20, 0.80), (1.00, 0.85, 0.60)),
    ((0.90, 0.30, 0.15), (0.60, 0.15, 0.40), (1.00, 0.80, 0.50)),
    ((0.30, 0.50, 1.00), (0.80, 0.30, 0.70), (0.70, 0.95, 1.00)),
]


def load_shader(name: str) -> str:
    src = (SHADER_DIR / name).read_text()
    if "//#include common" in src:
        src = src.replace("//#include common", (SHADER_DIR / "common.glsl").read_text())
    return src


def mat3_bytes(m: np.ndarray) -> bytes:
    """Column-major bytes for a GLSL mat3 uniform."""
    return np.asarray(m, dtype="f4").T.tobytes()


def setu(prog: moderngl.Program, name: str, value) -> None:
    u = prog.get(name, None)
    if u is None:
        return
    if isinstance(value, (bytes, bytearray)):
        u.write(value)
    else:
        u.value = value


class Renderer:
    def __init__(self, ctx: moderngl.Context, size: tuple[int, int], seed: int | None = None,
                 star_count: int = 400_000):
        self.ctx = ctx
        ctx.enable_direct(GL_TEXTURE_CUBE_MAP_SEAMLESS)
        self.rng = np.random.default_rng(seed)

        def prog(vs, fs):
            return ctx.program(vertex_shader=load_shader(vs), fragment_shader=load_shader(fs))

        self.p_sky = prog("fullscreen.vert", "sky.frag")
        self.p_bake = prog("fullscreen.vert", "sky_bake.frag")
        self.p_stars = prog("stars.vert", "stars.frag")
        self.p_bodies = prog("fullscreen.vert", "bodies.frag")
        self.p_down = prog("fullscreen.vert", "downsample.frag")
        self.p_up = prog("fullscreen.vert", "upsample.frag")
        self.p_comp = prog("fullscreen.vert", "composite.frag")
        self._fs_vaos: dict[int, moderngl.VertexArray] = {}

        self.quad_vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._new_sky_params()
        self._build_near_stars(star_count)
        self._build_far_stars(45_000)
        self.sky_tex: moderngl.TextureCube | None = None
        self.bake_sky()

        self.size = (0, 0)
        self.hdr_tex = self.hdr_fbo = self.depth_rb = None
        self.mips: list[tuple[moderngl.Texture, moderngl.Framebuffer]] = []
        self.resize(size)

    # ------------------------------------------------------------ setup
    def _fs(self, prog: moderngl.Program) -> None:
        vao = self._fs_vaos.get(id(prog))
        if vao is None:
            vao = self._fs_vaos[id(prog)] = self.ctx.vertex_array(prog, [])
        vao.render(moderngl.TRIANGLES, vertices=3)

    def _new_sky_params(self) -> None:
        rng = self.rng
        self.gal_n = random_unit(rng)
        s1, s2 = perpendicular_basis(self.gal_n)
        a = rng.uniform(0, 2 * math.pi)
        self.gal_c = s1 * math.cos(a) + s2 * math.sin(a)
        self.sky_seed = float(rng.uniform(0, 100))
        self.neb = NEB_PALETTES[rng.integers(len(NEB_PALETTES))]

    def _build_near_stars(self, n: int) -> None:
        rng = self.rng
        pos = rng.uniform(0, STAR_BOX, (n, 3))
        T = sample_star_temperatures(rng, n)
        col = kelvin_to_rgb(T)
        lum = np.exp(rng.normal(0.0, 0.9, n)) * (T / 5800.0) ** 0.5
        lum = np.clip(lum, 0.05, 40.0)
        radius = 0.035 * np.sqrt(lum)
        data = np.hstack([pos, lum[:, None], col, radius[:, None]]).astype("f4")
        self.n_near = n
        self.near_vbo = self.ctx.buffer(data.tobytes())
        self.vao_near = self.ctx.vertex_array(
            self.p_stars, [(self.quad_vbo, "2f", "in_corner"), (self.near_vbo, "4f 4f/i", "in_star", "in_color")])

    def _build_far_stars(self, n: int) -> None:
        rng = self.rng
        n_band = int(n * 0.45)
        uni = random_unit(rng, n - n_band)
        e1 = self.gal_c
        e2 = np.cross(self.gal_n, e1)
        lon = rng.uniform(0, 2 * math.pi, n_band)[:, None]
        lat = rng.normal(0.0, 0.11, n_band)[:, None]
        band = np.cos(lat) * (np.cos(lon) * e1 + np.sin(lon) * e2) + np.sin(lat) * self.gal_n
        dirs = np.vstack([uni, band])
        T = sample_star_temperatures(rng, n)
        col = kelvin_to_rgb(T)
        u = rng.uniform(0.0, 1.0, n)
        lum = np.clip(0.22 * np.power(np.maximum(u, 1e-4), -1.15), 0.0, 30.0)
        sigma = 0.75 + 0.22 * np.log10(1.0 + lum)
        data = np.hstack([dirs, lum[:, None], col, sigma[:, None]]).astype("f4")
        if getattr(self, "far_vbo", None) is not None:
            self.vao_far.release()
            self.far_vbo.release()
        self.n_far = n
        self.far_vbo = self.ctx.buffer(data.tobytes())
        self.vao_far = self.ctx.vertex_array(
            self.p_stars, [(self.quad_vbo, "2f", "in_corner"), (self.far_vbo, "4f 4f/i", "in_star", "in_color")])

    def bake_sky(self, regenerate: bool = False) -> None:
        if regenerate:
            self._new_sky_params()
            self._build_far_stars(self.n_far)
        ctx = self.ctx
        N = SKY_FACE
        tex = ctx.texture((N, N), 4, dtype="f2")
        fbo = ctx.framebuffer(color_attachments=[tex])
        cube = ctx.texture_cube((N, N), 4, dtype="f2")
        p = self.p_bake
        setu(p, "u_galN", tuple(self.gal_n))
        setu(p, "u_galC", tuple(self.gal_c))
        setu(p, "u_seed", self.sky_seed)
        setu(p, "u_nebA", self.neb[0])
        setu(p, "u_nebB", self.neb[1])
        setu(p, "u_nebC", self.neb[2])
        setu(p, "u_faceRes", float(N))
        ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        for face in range(6):
            fbo.use()
            setu(p, "u_face", face)
            self._fs(p)
            cube.write(face, tex.read())
        cube.filter = (moderngl.LINEAR, moderngl.LINEAR)
        fbo.release()
        tex.release()
        if self.sky_tex is not None:
            self.sky_tex.release()
        self.sky_tex = cube

    def resize(self, size: tuple[int, int]) -> None:
        if size == self.size:
            return
        self.size = size
        for obj in (self.hdr_fbo, self.hdr_tex, self.depth_rb):
            if obj is not None:
                obj.release()
        for tex, fbo in self.mips:
            fbo.release()
            tex.release()
        ctx = self.ctx
        w, h = size
        self.hdr_tex = ctx.texture((w, h), 4, dtype="f2")
        self.hdr_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hdr_tex.repeat_x = self.hdr_tex.repeat_y = False
        self.depth_rb = ctx.depth_renderbuffer((w, h))
        self.hdr_fbo = ctx.framebuffer(color_attachments=[self.hdr_tex], depth_attachment=self.depth_rb)
        self.mips = []
        mw, mh = w // 2, h // 2
        for _ in range(BLOOM_LEVELS):
            if mw < 4 or mh < 4:
                break
            t = ctx.texture((mw, mh), 4, dtype="f2")
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            self.mips.append((t, ctx.framebuffer(color_attachments=[t])))
            mw, mh = mw // 2, mh // 2

    # ------------------------------------------------------------ per-frame
    def _upload_bodies(self, sim: Simulation, view: np.ndarray) -> list:
        bodies = sorted((b for b in sim.bodies if b.fade > 0.001), key=lambda b: not b.is_sun)[:MAXB]
        pos = np.zeros((MAXB, 4), "f4")
        rot = np.zeros((MAXB, 9), "f4")
        typ = np.zeros((MAXB, 4), "f4")
        c0, c1, c2, atm, misc, ring = (np.zeros((MAXB, 4), "f4") for _ in range(6))
        rels = []
        for k, b in enumerate(bodies):
            rel = view @ (b.pos - sim.pos)
            rels.append(rel)
            pos[k] = (*rel, b.radius)
            rot[k] = (view @ b.orientation(sim.time)).T.ravel()
            typ[k] = (b.type_id, b.seed, b.ring_inner, b.ring_outer)
            c0[k], c1[k], c2[k], atm[k], ring[k] = b.c0, b.c1, b.c2, b.atm, b.ring
            misc[k] = (b.fade, *b.misc[1:])
        sun_pos = np.zeros((2, 4), "f4")
        sun_col = np.zeros((2, 4), "f4")
        suns = [(b, r) for b, r in zip(bodies, rels) if b.is_sun][:2]
        for j, (b, r) in enumerate(suns):
            sun_pos[j] = (*r, b.radius)
            sun_col[j, :3] = np.asarray(b.light) * b.fade
        p = self.p_bodies
        setu(p, "u_count", len(bodies))
        for name, arr in (("u_pos", pos), ("u_rot", rot), ("u_type", typ), ("u_c0", c0), ("u_c1", c1),
                          ("u_c2", c2), ("u_atm", atm), ("u_misc", misc), ("u_ring", ring),
                          ("u_sunPos", sun_pos), ("u_sunCol", sun_col)):
            setu(p, name, arr.tobytes())
        setu(p, "u_nsun", len(suns))
        occluders = [(r, b.radius) for b, r in zip(bodies, rels) if not b.is_sun and b.fade > 0.5]
        return [(b, r, occluders) for b, r in suns]

    @staticmethod
    def _sun_visibility(rel: np.ndarray, radius: float, occluders) -> float:
        if not occluders:
            return 1.0
        d = rel / np.linalg.norm(rel)
        e1, e2 = perpendicular_basis(d)
        g = np.linspace(-1, 1, 7)
        u, v = np.meshgrid(g, g)
        m = u * u + v * v <= 1.0
        pts = rel + radius * (u[m][:, None] * e1 + v[m][:, None] * e2)
        dist = np.linalg.norm(pts, axis=1)
        dirs = pts / dist[:, None]
        vis = np.ones(len(pts), bool)
        for c, r in occluders:
            tc = dirs @ c
            perp2 = np.dot(c, c) - tc * tc
            vis &= ~((tc > 0) & (perp2 < r * r) & (tc < dist))
        return float(vis.mean())

    def render(self, sim: Simulation, paused: bool = False) -> None:
        ctx = self.ctx
        w, h = self.size
        Rv = sim.R_view
        view = Rv.T
        tany = math.tan(sim.fov * 0.5)
        tanx = tany * w / h
        fy = (h * 0.5) / tany
        px_scale = h / 2160.0
        vel = (0.0, 0.0, 0.0) if paused else tuple(float(x) for x in sim.v_cam_star)
        omega = (0.0, 0.0, 0.0) if paused else tuple(float(x) for x in sim.omega_cam)

        hdr = self.hdr_fbo
        hdr.depth_mask = True
        hdr.use()
        hdr.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)

        # sky
        p = self.p_sky
        self.sky_tex.use(0)
        setu(p, "u_sky", 0)
        setu(p, "u_camR", mat3_bytes(Rv))
        setu(p, "u_tan", (tanx, tany))
        setu(p, "u_bright", 1.0 - 0.5 * sim.warp)
        self._fs(p)

        # stars common uniforms
        p = self.p_stars
        setu(p, "u_view", mat3_bytes(view))
        setu(p, "u_vel", vel)
        setu(p, "u_angvel", omega)
        setu(p, "u_shutter", 1.0 / 60.0)
        setu(p, "u_focal", (fy, fy))
        setu(p, "u_res", (float(w), float(h)))
        setu(p, "u_logfar", LOG_FAR)
        setu(p, "u_doppler", 0.6 * sim.warp)
        setu(p, "u_pxScale", px_scale)

        # distant stars (rotation only)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        setu(p, "u_infinite", 1)
        setu(p, "u_bright", 0.9 * (1.0 - 0.6 * sim.warp))
        setu(p, "u_streakK", 1.0)
        self.vao_far.render(moderngl.TRIANGLE_STRIP, instances=self.n_far)

        # suns, planets, rings, atmospheres
        suns = self._upload_bodies(sim, view)
        pb = self.p_bodies
        setu(pb, "u_tan", (tanx, tany))
        setu(pb, "u_res", (float(w), float(h)))
        setu(pb, "u_time", float(sim.time))
        setu(pb, "u_logfar", LOG_FAR)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "1"
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self._fs(pb)

        # near stars (3D, parallax, motion streaks), occluded by bodies
        ctx.depth_func = "<"
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        hdr.depth_mask = False
        hdr.use()
        setu(p, "u_infinite", 0)
        setu(p, "u_campos", tuple(float(x) for x in sim.star_pos))
        setu(p, "u_L", STAR_BOX)
        setu(p, "u_bright", 22.0)
        setu(p, "u_d0", 250.0)
        setu(p, "u_streakK", 0.3)
        self.vao_near.render(moderngl.TRIANGLE_STRIP, instances=self.n_near)
        hdr.depth_mask = True
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)

        # bloom
        src = self.hdr_tex
        for i, (tex, fbo) in enumerate(self.mips):
            fbo.use()
            src.use(0)
            setu(self.p_down, "u_src", 0)
            setu(self.p_down, "u_srcTexel", (1.0 / src.width, 1.0 / src.height))
            setu(self.p_down, "u_first", 1 if i == 0 else 0)
            self._fs(self.p_down)
            src = tex
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        for i in range(len(self.mips) - 1, 0, -1):
            tex_src = self.mips[i][0]
            self.mips[i - 1][1].use()
            tex_src.use(0)
            setu(self.p_up, "u_src", 0)
            setu(self.p_up, "u_texel", (1.0 / tex_src.width, 1.0 / tex_src.height))
            self._fs(self.p_up)
        ctx.disable(moderngl.BLEND)

        # lens flares
        flare = np.zeros((2, 4), "f4")
        flare_col = np.zeros((2, 3), "f4")
        nf = 0
        for b, rel, occ in suns:
            if rel[2] <= 0.0:
                continue
            ndc = (rel[0] / rel[2] / tanx, rel[1] / rel[2] / tany)
            edge = max(abs(ndc[0]), abs(ndc[1]))
            if edge > 1.6:
                continue
            vis = self._sun_visibility(rel, b.radius, occ)
            I = vis * b.fade * smoothstep(1.6, 0.95, edge)
            if I <= 0.001:
                continue
            flare[nf] = (ndc[0], ndc[1], I, 0.0)
            flare_col[nf] = np.asarray(b.c0[:3])
            nf += 1

        # composite to screen
        ctx.screen.viewport = (0, 0, w, h)
        ctx.screen.use()
        pc = self.p_comp
        self.hdr_tex.use(0)
        self.mips[0][0].use(1)
        setu(pc, "u_hdr", 0)
        setu(pc, "u_bloom", 1)
        setu(pc, "u_bloomStrength", 0.06)
        setu(pc, "u_exposure", 1.0)
        setu(pc, "u_warp", float(sim.warp))
        setu(pc, "u_time", float(sim.time))
        setu(pc, "u_res", (float(w), float(h)))
        setu(pc, "u_nflare", nf)
        setu(pc, "u_flare", flare.tobytes())
        setu(pc, "u_flareCol", flare_col.tobytes())
        self._fs(pc)
