"""Ship flight model, autopilot, warp sequencing and system management."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .universe import Body, SystemInfo, generate_system, orthonormalize, rot_axis, rot_x, rot_y, smoothstep

STAR_BOX = 8000.0
CRUISE_SPEED = 2500.0
WARP_SPEED = 9000.0
BASE_FOV = math.radians(52.0)


@dataclass
class Controls:
    pitch: float = 0.0   # +1 = nose down
    yaw: float = 0.0     # +1 = turn right
    roll: float = 0.0    # +1 = roll left
    warp: bool = False


class Simulation:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.pos = np.zeros(3)
        self.R = np.eye(3)               # ship frame, columns: right, up, forward
        self.speed = 900.0
        self.angvel = np.zeros(3)
        self.cruise_speed = CRUISE_SPEED
        self.autopilot = True
        self.state = "cruise"            # cruise | spool | warp | arrive
        self.state_t = 0.0
        self.idle_t = 0.0
        self.warp_hold_t = 0.0
        self.look = np.zeros(2)
        self.R_view = np.eye(3)
        self.omega_cam = np.zeros(3)
        self.star_pos = np.zeros(3)
        self.v_cam_star = np.zeros(3)
        self.warp = 0.0
        self.fov = BASE_FOV
        self.bodies: list[Body] = []
        self.system: SystemInfo | None = None
        self.focus: Body | None = None
        self.focus_w = 0.0
        self.events: list[tuple] = []
        self._spawn_system(first_distance=9000.0)

    # ------------------------------------------------------------ helpers
    @property
    def forward(self) -> np.ndarray:
        return self.R[:, 2]

    @property
    def path_remaining(self) -> list[Body]:
        return [b for b in self.bodies if b.flyby is not None and not b.departing and not b.passed]

    def nearest(self) -> tuple[Body | None, float]:
        best, best_d = None, math.inf
        for b in self.bodies:
            if b.departing:
                continue
            d = float(np.linalg.norm(b.pos - self.pos)) - b.radius
            if d < best_d:
                best, best_d = b, d
        return best, best_d

    def _spawn_system(self, first_distance: float = 16000.0) -> None:
        info, bodies = generate_system(self.rng, self.pos.copy(), self.R.copy(), first_distance)
        for b in bodies:
            b.fade, b.fade_target, b.fade_rate = 0.0, 1.0, 0.6
        self.bodies.extend(bodies)
        self.system = info
        self.events.append(("arrive", info))

    def jump(self) -> None:
        if self.state != "cruise":
            return
        self.state, self.state_t = "spool", 0.0
        for b in self.bodies:
            b.departing, b.fade_target, b.fade_rate = True, 0.0, 0.8
        self.events.append(("jump",))

    def _speed_cap(self) -> float:
        cap = math.inf
        fwd = self.forward
        for b in self.bodies:
            if b.departing or b.fade < 0.05:
                continue
            rel = b.pos - self.pos
            d = float(np.linalg.norm(rel))
            margin = b.radius * (2.0 if b.is_sun else 1.0 + b.atm[3])
            surf = max(d - margin, 0.0)
            heading = float(np.dot(rel, fwd)) / max(d, 1e-6)
            cap = min(cap, 0.7 * surf + 60.0 * (1.0 - max(heading, 0.0)))
        return max(cap, 2.0)

    # ------------------------------------------------------------ update
    def update(self, dt: float, ctl: Controls) -> None:
        if dt <= 0.0:
            return
        self.time += dt
        self.state_t += dt
        t = self.time

        if self.autopilot and (ctl.pitch or ctl.yaw or ctl.roll):
            self.autopilot = False
            self.events.append(("mode", False))

        fwd = self.forward
        for b in self.bodies:
            if b.flyby is not None and not b.passed and np.dot(b.flyby - self.pos, fwd) < 0.0:
                b.passed = True
        remaining = self.path_remaining

        # ---- state machine / target speed
        if self.state == "cruise":
            self.idle_t = 0.0 if remaining else self.idle_t + dt
            if self.autopilot:
                target = self.cruise_speed
                if self.idle_t > 4.0:
                    self.jump()
            else:
                target = WARP_SPEED if ctl.warp else self.cruise_speed
                self.warp_hold_t = self.warp_hold_t + dt if (ctl.warp and not remaining) else 0.0
                if self.warp_hold_t > 1.5:
                    self.jump()
        if self.state == "spool":
            target = WARP_SPEED
            if self.state_t > 1.2:
                self.state, self.state_t = "warp", 0.0
        elif self.state == "warp":
            target = WARP_SPEED
            if self.state_t > 4.0:
                self.bodies = [b for b in self.bodies if not b.departing]
                self._spawn_system()
                self.state, self.state_t = "arrive", 0.0
        elif self.state == "arrive":
            target = self.cruise_speed
            if self.speed < self.cruise_speed * 1.3 or self.state_t > 6.0:
                self.state, self.state_t = "cruise", 0.0

        cap = math.inf if self.state in ("spool", "warp") else self._speed_cap()
        goal = min(target, cap)
        if goal < self.speed:
            tau = 0.45
        else:
            tau = 0.9 if self.state in ("spool", "warp") else 1.6
        self.speed += (goal - self.speed) * (1.0 - math.exp(-dt / tau))

        # ---- steering
        ang_t = np.zeros(3)
        if self.autopilot:
            tgt = remaining[0].flyby if (remaining and self.state in ("cruise", "arrive")) else None
            if tgt is not None:
                local = self.R.T @ (tgt - self.pos)
                yaw_err = math.atan2(local[0], local[2])
                pitch_err = -math.atan2(local[1], local[2])
                gain = 0.9 * smoothstep(300.0, 3000.0, local[2])
                ang_t[0] = float(np.clip(pitch_err * gain, -0.3, 0.3))
                ang_t[1] = float(np.clip(yaw_err * gain, -0.3, 0.3))
            else:
                ang_t[0] = 0.03 * math.sin(t * 0.037 + 1.0)
                ang_t[1] = 0.04 * math.sin(t * 0.05)
            ang_t[2] = 0.025 * math.sin(t * 0.07)
        else:
            ang_t[:] = (ctl.pitch * 0.6, ctl.yaw * 0.6, ctl.roll * 0.9)
        self.angvel += (ang_t - self.angvel) * (1.0 - math.exp(-dt * 3.0))
        w = float(np.linalg.norm(self.angvel))
        if w > 1e-9:
            self.R = orthonormalize(self.R @ rot_axis(self.angvel / w, w * dt))
        fwd = self.forward
        self.pos = self.pos + fwd * self.speed * dt

        # ---- bodies: fades and clean-up
        for b in self.bodies:
            step = b.fade_rate * dt
            b.fade += float(np.clip(b.fade_target - b.fade, -step, step))
        keep = []
        for b in self.bodies:
            if b.departing and b.fade <= 0.0:
                continue
            if not b.is_sun and np.linalg.norm(b.pos - self.pos) > 250000.0:
                continue
            keep.append(b)
        self.bodies = keep

        # ---- cinematic look-at during flybys
        look_t = np.zeros(2)
        self.focus, self.focus_w = None, 0.0
        if self.state in ("cruise", "arrive"):
            for b in self.bodies:
                if b.departing or b.is_sun or b.flyby is None:
                    continue
                rel = b.pos - self.pos
                d = float(np.linalg.norm(rel))
                wgt = smoothstep(10.0 * b.radius, 3.0 * b.radius, d) * b.fade
                if wgt > self.focus_w:
                    self.focus, self.focus_w = b, wgt
                    if self.autopilot:
                        local = self.R.T @ (rel / d)
                        yaw = math.atan2(local[0], local[2])
                        pitch = -math.atan2(local[1], math.hypot(local[0], local[2]))
                        look_t = np.clip(np.array([yaw, pitch]) * 0.8 * wgt, -1.9, 1.9)
        self.look += (look_t - self.look) * (1.0 - math.exp(-dt * 2.2))

        # ---- warp visuals and camera
        prev_warp = self.warp
        self.warp = smoothstep(3500.0, 8500.0, self.speed)
        dwarp = abs(self.warp - prev_warp) / dt
        amp = min(0.0012 * self.warp + 0.004 * dwarp, 0.005)
        sx = amp * (math.sin(t * 37.1) + 0.5 * math.sin(t * 71.3 + 1.2))
        sy = amp * (math.sin(t * 41.7 + 2.0) + 0.5 * math.sin(t * 83.9))
        R_view = self.R @ rot_y(self.look[0]) @ rot_x(self.look[1]) @ rot_x(sx) @ rot_y(sy)

        dR = self.R_view.T @ R_view
        self.omega_cam = 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / dt
        self.R_view = R_view
        self.fov = BASE_FOV + math.radians(16.0) * self.warp

        # star layer moves at a compressed speed so cruising looks calm and warp looks fast
        vs = self.speed if self.speed < 250.0 else 250.0 + (self.speed - 250.0) * 0.06
        vs *= 1.0 + 16.0 * self.warp
        self.star_pos = (self.star_pos + fwd * vs * dt) % STAR_BOX
        self.v_cam_star = self.R_view.T @ (fwd * vs)
