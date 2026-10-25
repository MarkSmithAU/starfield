"""On-screen text: status line, system titles, flyby labels and help."""

from __future__ import annotations

import math

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .renderer import load_shader, setu
from .sim import Simulation
from .universe import smoothstep

FONTS = ["bahnschrift.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
TEXT = (205, 225, 255, 255)
DIM = (150, 170, 205, 255)
ACCENT = (255, 205, 140, 255)

HELP = [
    ("CONTROLS", ACCENT),
    ("W / S  or  Up / Down        pitch", TEXT),
    ("A / D  or  Left / Right     yaw", TEXT),
    ("Q / E                       roll", TEXT),
    ("Left-drag mouse             steer", TEXT),
    ("Mouse wheel  /  + -         cruise speed", TEXT),
    ("Space (hold)                warp", TEXT),
    ("N                           jump to next system", TEXT),
    ("Tab                         toggle autopilot", TEXT),
    ("R                           new galaxy backdrop", TEXT),
    ("P                           pause", TEXT),
    ("H                           hide HUD", TEXT),
    ("F11                         fullscreen / window", TEXT),
    ("F12                         screenshot", TEXT),
    ("Esc                         quit", TEXT),
]


def load_font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont:
    for name in (["consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"] if mono else []) + FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


class TextSprite:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.tex: moderngl.Texture | None = None
        self.key = None

    @property
    def size(self) -> tuple[int, int]:
        return self.tex.size if self.tex else (0, 0)

    def set(self, lines: list[tuple[str, tuple]], font, align: str = "left", gap: int = 4) -> None:
        key = (tuple(lines), id(font), align)
        if key == self.key:
            return
        self.key = key
        pad = max(4, font.size // 6)
        asc, desc = font.getmetrics()
        lh = asc + desc
        widths = [int(math.ceil(font.getlength(t))) for t, _ in lines] or [1]
        w = max(widths) + pad * 2
        h = lh * len(lines) + gap * max(len(lines) - 1, 0) + pad * 2
        img = Image.new("RGBA", (max(w, 1), max(h, 1)), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        off = max(1, font.size // 18)
        for i, (text, colour) in enumerate(lines):
            x = pad + ((max(widths) - widths[i]) // 2 if align == "center" else 0)
            y = pad + i * (lh + gap)
            d.text((x + off, y + off), text, font=font, fill=(0, 0, 0, 190))
            d.text((x, y), text, font=font, fill=colour)
        if self.tex is None or self.tex.size != img.size:
            if self.tex is not None:
                self.tex.release()
            self.tex = self.ctx.texture(img.size, 4, img.tobytes())
            self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        else:
            self.tex.write(img.tobytes())


class Hud:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=load_shader("hud.vert"), fragment_shader=load_shader("hud.frag"))
        self.vbo = ctx.buffer(np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype="f4").tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_pos")])
        self.visible = True
        self.show_help = False
        self.size = (0, 0)
        self.status = TextSprite(ctx)
        self.stats = TextSprite(ctx)
        self.title = TextSprite(ctx)
        self.label = TextSprite(ctx)
        self.help = TextSprite(ctx)
        self.hint = TextSprite(ctx)
        self.title_t = 1e9
        self.label_alpha = 0.0
        self.refresh_t = 0.0
        self.hint_t = 0.0

    def resize(self, size: tuple[int, int]) -> None:
        if size == self.size:
            return
        self.size = size
        s = max(size[1] / 2160.0, 0.45)
        self.f_small = load_font(int(30 * s))
        self.f_mid = load_font(int(40 * s))
        self.f_big = load_font(int(96 * s))
        self.f_help = load_font(int(32 * s), mono=True)
        self.scale = s
        for sp in (self.status, self.stats, self.title, self.label, self.help, self.hint):
            sp.key = None

    def on_event(self, ev: tuple) -> None:
        if ev[0] == "arrive":
            info = ev[1]
            sub = f"{info.star_label}   ·   {info.n_planets} planets" + ("   ·   binary system" if info.binary else "")
            self.title.key = None
            self._title_lines = [(info.name.upper(), TEXT), (sub, DIM)]
            self.title_t = 0.0

    def update(self, sim: Simulation, fps: float, dt: float, paused: bool) -> None:
        self.title_t += dt
        self.hint_t = max(self.hint_t - dt, 0.0)
        self.refresh_t -= dt
        if self.refresh_t > 0.0:
            return
        self.refresh_t = 0.2
        w, h = self.size
        mode = "AUTOPILOT" if sim.autopilot else "MANUAL"
        if sim.state in ("spool", "warp"):
            mode += "   ·   WARP"
        if paused:
            mode += "   ·   PAUSED"
        lines = []
        if sim.system:
            lines.append((f"{sim.system.name}   ·   {sim.system.star_label}", TEXT))
        lines.append((f"{mode}   ·   {sim.speed:,.0f} u/s", ACCENT))
        body, dist = sim.nearest()
        if body is not None:
            lines.append((f"Nearest: {body.name}  ({body.kind})   {max(dist, 0):,.0f} u", DIM))
        self.status.set(lines, self.f_small)
        self.stats.set([(f"{fps:5.0f} fps   ·   {w}×{h}   ·   F1 help", DIM)], self.f_small)
        if self.title_t < 9.0 and hasattr(self, "_title_lines"):
            self.title.set(self._title_lines, self.f_big, align="center", gap=int(10 * self.scale))
        if sim.focus is not None:
            self.label.set([(sim.focus.name, TEXT), (sim.focus.kind, DIM)], self.f_mid, align="center")
        self.label_alpha = smoothstep(0.15, 0.6, sim.focus_w) if sim.focus is not None else 0.0
        if not sim.autopilot and sim.state == "cruise" and not sim.path_remaining:
            self.hint.set([("Hold SPACE to warp to the next system", ACCENT)], self.f_mid, align="center")
            self.hint_t = 0.5
        self.help.set(HELP, self.f_help)

    def _draw(self, sp: TextSprite, x: float, y: float, alpha: float) -> None:
        if sp.tex is None or alpha <= 0.0:
            return
        sw, sh = sp.size
        setu(self.prog, "u_rect", (float(round(x)), float(round(y)), float(sw), float(sh)))
        setu(self.prog, "u_alpha", float(alpha))
        sp.tex.use(0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def draw(self) -> None:
        if not self.visible:
            return
        ctx = self.ctx
        w, h = self.size
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        setu(self.prog, "u_res", (float(w), float(h)))
        setu(self.prog, "u_tex", 0)
        m = 40 * self.scale
        self._draw(self.status, m, h - m - self.status.size[1], 0.9)
        self._draw(self.stats, w - m - self.stats.size[1] * 0 - self.stats.size[0], m, 0.7)
        t = self.title_t
        ta = smoothstep(0.5, 2.0, t) * (1.0 - smoothstep(6.0, 8.5, t))
        self._draw(self.title, (w - self.title.size[0]) / 2, h * 0.2, ta)
        self._draw(self.label, (w - self.label.size[0]) / 2, h * 0.8, self.label_alpha * 0.9)
        self._draw(self.hint, (w - self.hint.size[0]) / 2, h * 0.72, min(self.hint_t * 4, 1.0))
        if self.show_help:
            self._draw(self.help, m, m, 0.95)
        ctx.disable(moderngl.BLEND)
