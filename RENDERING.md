# Sight visualization

From the project folder:

```bash
python renderer.py --stage 2 --visibility
```

The blue-gray area shows unobstructed 360° wall sensing up to 600 world units.
The warm sector shows the potential 60° target field of view. Walls cast sharp
shadows, while range and sector edges fade gently. All arena geometry is drawn
at twice the display resolution and downsampled for smooth edges.

Legacy policies receive 32 wall rays and five target rays; angular policies use
six target regions plus a center ray. Continuous sight shading is optional and
off by default; press V or pass `--visibility` to enable it. Angular-region fills
follow the R sensor toggle. Display shading adds no policy input. The contact
badge and bright target indicate an actual sensor detection. Undetected targets
remain dimly visible for an observer, including behind walls.

Controls: WASD move, Q/E or arrows aim, Space fires, R toggles discrete sensor
rays, V toggles sight shading, T toggles movement trails, F toggles the frame
cap, Escape closes the window. Player-versus-agent mode keeps sight shading
disabled and uses its existing mouse controls and H visibility toggle.

Capture an initial scene without opening a window:

```bash
python renderer.py --visibility \
  --headless --seed 7 --screenshot /tmp/shooter-sight/seed-7.png
```

Capture contact, wall occlusion, range, sensor rays, and movement examples:

```bash
python render_preview.py --output-dir /tmp/shooter-sight
```

Run the geometry, controls, and environment parity checks from the project folder:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -m unittest discover -s tests -v
```

The renderer uses existing NumPy and Pygame dependencies. Headless captures run
the same Pygame drawing path as the desktop renderer using SDL's dummy driver.
