"""Window, input and main loop."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import glfw
import moderngl
import numpy as np
from PIL import Image

from .hud import Hud
from .renderer import Renderer
from .sim import Controls, Simulation


class App:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        if not glfw.init():
            sys.exit("Could not initialise GLFW")
        monitors = glfw.get_monitors()
        self.monitor = monitors[args.monitor] if 0 <= args.monitor < len(monitors) else glfw.get_primary_monitor()
        mode = glfw.get_video_mode(self.monitor)

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        glfw.window_hint(glfw.SAMPLES, 0)
        glfw.window_hint(glfw.DEPTH_BITS, 0)

        if args.windowed:
            x, y, w, h = glfw.get_monitor_workarea(self.monitor)
            glfw.window_hint(glfw.VISIBLE, False)
            self.win = glfw.create_window(w * 3 // 4, h * 3 // 4, "Starfield", None, None)
            glfw.set_window_pos(self.win, x + w // 8, y + h // 8)
            glfw.maximize_window(self.win)
            glfw.show_window(self.win)
        else:
            glfw.window_hint(glfw.REFRESH_RATE, mode.refresh_rate)
            self.win = glfw.create_window(mode.size.width, mode.size.height, "Starfield", self.monitor, None)
        if not self.win:
            glfw.terminate()
            sys.exit("Could not create an OpenGL 3.3 window")
        glfw.make_context_current(self.win)
        glfw.swap_interval(0 if args.no_vsync else 1)

        self.ctx = moderngl.create_context(require=330)
        print(f"OpenGL: {self.ctx.info['GL_RENDERER']}  ({self.ctx.info['GL_VERSION']})")
        print(f"Display: {mode.size.width}x{mode.size.height} @ {mode.refresh_rate} Hz, "
              f"vsync {'off' if args.no_vsync else 'on'}")

        size = glfw.get_framebuffer_size(self.win)
        t0 = time.perf_counter()
        self.renderer = Renderer(self.ctx, size, seed=args.seed, star_count=args.stars)
        print(f"Sky baked in {time.perf_counter() - t0:.2f}s")
        self.hud = Hud(self.ctx)
        self.sim = Simulation(seed=args.seed)

        self.paused = False
        self.fps = 0.0
        self.drag_start: tuple[float, float] | None = None
        self.last_mouse_move = 0.0
        self.cursor_hidden = False
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_scroll_callback(self.win, self.on_scroll)
        glfw.set_mouse_button_callback(self.win, self.on_mouse_button)
        glfw.set_cursor_pos_callback(self.win, self.on_cursor)

    # ------------------------------------------------------------ input
    def on_key(self, win, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        sim = self.sim
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_TAB:
            sim.autopilot = not sim.autopilot
        elif key == glfw.KEY_N:
            sim.jump()
        elif key == glfw.KEY_H:
            self.hud.visible = not self.hud.visible
        elif key == glfw.KEY_F1:
            self.hud.show_help = not self.hud.show_help
        elif key == glfw.KEY_P:
            self.paused = not self.paused
        elif key == glfw.KEY_R:
            self.renderer.bake_sky(regenerate=True)
        elif key == glfw.KEY_F11:
            self.toggle_fullscreen()
        elif key == glfw.KEY_F12:
            self.screenshot(Path("screenshots"))
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            self.on_scroll(win, 0, 1)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self.on_scroll(win, 0, -1)

    def on_scroll(self, win, dx, dy):
        self.sim.cruise_speed = float(np.clip(self.sim.cruise_speed * 1.2 ** dy, 50.0, 6000.0))

    def on_mouse_button(self, win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.drag_start = glfw.get_cursor_pos(win) if action == glfw.PRESS else None

    def on_cursor(self, win, x, y):
        self.last_mouse_move = glfw.get_time()

    def read_controls(self) -> Controls:
        k = lambda *keys: any(glfw.get_key(self.win, key) == glfw.PRESS for key in keys)
        ctl = Controls(
            pitch=float(k(glfw.KEY_S, glfw.KEY_DOWN)) - float(k(glfw.KEY_W, glfw.KEY_UP)),
            yaw=float(k(glfw.KEY_D, glfw.KEY_RIGHT)) - float(k(glfw.KEY_A, glfw.KEY_LEFT)),
            roll=float(k(glfw.KEY_Q)) - float(k(glfw.KEY_E)),
            warp=k(glfw.KEY_SPACE),
        )
        if self.drag_start is not None:
            x, y = glfw.get_cursor_pos(self.win)
            _, h = glfw.get_window_size(self.win)
            s = max(h * 0.2, 1.0)
            ctl.yaw = float(np.clip(ctl.yaw + (x - self.drag_start[0]) / s, -1.0, 1.0))
            ctl.pitch = float(np.clip(ctl.pitch + (y - self.drag_start[1]) / s, -1.0, 1.0))
            if abs(ctl.yaw) < 0.03:
                ctl.yaw = 0.0
            if abs(ctl.pitch) < 0.03:
                ctl.pitch = 0.0
        return ctl

    # ------------------------------------------------------------ window
    def toggle_fullscreen(self):
        mode = glfw.get_video_mode(self.monitor)
        if glfw.get_window_monitor(self.win):
            x, y, w, h = glfw.get_monitor_workarea(self.monitor)
            glfw.set_window_monitor(self.win, None, x + w // 8, y + h // 8, w * 3 // 4, h * 3 // 4, 0)
            glfw.maximize_window(self.win)
        else:
            glfw.set_window_monitor(self.win, self.monitor, 0, 0, mode.size.width, mode.size.height,
                                    mode.refresh_rate)
        glfw.swap_interval(0 if self.args.no_vsync else 1)

    def screenshot(self, folder: Path, name: str | None = None) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        w, h = self.renderer.size
        data = self.ctx.screen.read(viewport=(0, 0, w, h), components=3)
        img = Image.frombytes("RGB", (w, h), data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        path = folder / (name or time.strftime("starfield_%Y%m%d_%H%M%S.png"))
        img.save(path)
        print(f"Saved {path}")
        return path

    # ------------------------------------------------------------ loop
    def run(self) -> None:
        args = self.args
        captures = sorted(float(x) for x in args.capture_times.split(",")) if args.capture_dir else []
        start = glfw.get_time()
        last = start
        frame_times: list[float] = []
        avg_dt = 1.0 / 60.0
        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            now = glfw.get_time()
            raw_dt = now - last
            last = now
            dt = min(raw_dt, 0.1)
            frame_times.append(raw_dt)
            avg_dt += (raw_dt - avg_dt) * 0.05
            self.fps = 1.0 / max(avg_dt, 1e-6)

            fb = glfw.get_framebuffer_size(self.win)
            if fb[0] == 0 or fb[1] == 0:
                glfw.wait_events_timeout(0.1)
                continue
            self.renderer.resize(fb)
            self.hud.resize(fb)

            hide = glfw.get_window_monitor(self.win) and now - self.last_mouse_move > 2.0
            if hide != self.cursor_hidden:
                glfw.set_input_mode(self.win, glfw.CURSOR, glfw.CURSOR_HIDDEN if hide else glfw.CURSOR_NORMAL)
                self.cursor_hidden = hide

            ctl = self.read_controls()
            if not self.paused:
                self.sim.update(dt, ctl)
            for ev in self.sim.events:
                self.hud.on_event(ev)
            self.sim.events.clear()

            self.renderer.render(self.sim, self.paused)
            self.hud.update(self.sim, self.fps, dt, self.paused)
            self.hud.draw()

            if captures and now - start >= captures[0]:
                self.screenshot(Path(args.capture_dir), f"capture_{now - start:06.1f}s.png")
                captures.pop(0)
                if not captures:
                    glfw.set_window_should_close(self.win, True)

            glfw.swap_buffers(self.win)

        if len(frame_times) > 30:
            ft = np.array(frame_times[30:])
            print(f"Frames: {len(ft)}   mean {1.0 / ft.mean():.1f} fps   "
                  f"p99 frame {np.percentile(ft, 99) * 1000:.2f} ms")
        glfw.terminate()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Cinematic GPU starfield simulator")
    ap.add_argument("--windowed", action="store_true", help="maximised window instead of fullscreen")
    ap.add_argument("--monitor", type=int, default=-1, help="monitor index (default: primary)")
    ap.add_argument("--seed", type=int, default=None, help="random seed for the universe")
    ap.add_argument("--stars", type=int, default=300_000, help="number of 3D stars")
    ap.add_argument("--no-vsync", action="store_true", help="disable vsync (benchmarking)")
    ap.add_argument("--capture-dir", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--capture-times", default="3,6,10", help=argparse.SUPPRESS)
    App(ap.parse_args(argv)).run()


if __name__ == "__main__":
    main()
