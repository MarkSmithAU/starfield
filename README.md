# Starfield

A cinematic, GPU-rendered space-flight simulator. Everything is procedural and drawn by
OpenGL shaders at native resolution (built for 4K), with vsync.

- **400,000 stars** in 3D with parallax, black-body colours, and motion-blur streaks.
- **Distant sky**: a baked Milky Way band with dust lanes, nebulae, and 45k background stars.
- **Star systems**: O–M class suns (some binary) with granulation, sunspots, and an animated corona.
- **Planets**: terrestrial worlds with oceans, clouds, ice caps and city lights on the night side,
  banded gas giants with storms and rings, ice worlds, volcanic worlds, cratered rocky worlds, and moons.
  Rings and moons cast shadows, and atmospheres glow at the limb.
- **Autopilot**: a cinematic tour that flies past each planet, turns the camera to follow it,
  then warps to a new procedurally generated system.
- **Post-processing**: HDR, bloom, lens flares that planets can block, warp radial blur,
  blueshift and chromatic aberration, and ACES tonemapping.

## Run

```
uv run starfield               # fullscreen at the monitor's native resolution and refresh rate
uv run starfield --windowed    # maximised window
```

Options: `--monitor N`, `--seed N`, `--stars N`, `--no-vsync` (for benchmarking).

## Controls

| Key | Action |
| --- | --- |
| W/S, ↑/↓ | pitch |
| A/D, ←/→ | yaw |
| Q/E | roll |
| Left-drag | steer |
| Mouse wheel, + / - | cruise speed |
| Space (hold) | warp (keep holding after the last planet to jump systems) |
| N | jump to the next system |
| Tab | toggle autopilot (any steering switches to manual) |
| R | new galaxy backdrop |
| P | pause |
| H | hide HUD |
| F1 | help |
| F11 | toggle fullscreen / window |
| F12 | screenshot (saved to `screenshots/`) |
| Esc | quit |
