#!/usr/bin/env python3
"""Writes scenes/marbles.toml: a cluster of marbles lit by fireflies.

    python3 scripts/make_marbles.py > scenes/marbles.toml

The marbles are placed at random, none touching, from a fixed seed, so the
same script writes the same file on any machine (Python's random.Random is
specified to give the same sequence for the same seed). Edit the scene here,
not in the file it writes.
"""
import math
import random
import sys
rng = random.Random(7)
TABLE = 0.75
CAM_Z = 0.32

# (kind, count): each kind's radius range in meters
kinds = (["clear"] * 4 + ["tint"] * 9 + ["cats"] * 9 + ["porcelain"] * 10 + ["metal"] * 4)
rng.shuffle(kinds)
radius_of = lambda: rng.choice([0.006, 0.007, 0.008, 0.008, 0.008, 0.009, 0.0095, 0.0105, 0.0125])

placed = []
for kind in kinds:
    for attempt in range(20000):
        r = 0.0125 if (kind == "clear" and not any(k == "clear" and rr == 0.0125 for k, _, _, rr in placed)) else radius_of()
        z = rng.gauss(-0.06, 0.085)
        if not (-0.30 < z < 0.16):
            continue
        d = CAM_Z - z
        x = rng.gauss(0.0, 0.33 * d * 0.386)
        if abs(x) > 0.36 * d:
            continue
        if all(math.hypot(x - px, z - pz) > r + pr + 0.0015 for _, px, pz, pr in placed):
            placed.append((kind, x, z, r))
            break
    else:
        sys.exit(f"no room for a {kind}")

tints = [("ruby", [0.95, 0.12, 0.18]), ("cobalt", [0.15, 0.32, 0.95]), ("emerald", [0.18, 0.85, 0.32]),
         ("amber", [0.95, 0.58, 0.12]), ("violet", [0.6, 0.28, 0.92]), ("aqua", [0.25, 0.85, 0.85])]
swirls = [("orange_swirl", [0.9, 0.3, 0.03], [0.92, 0.88, 0.78], 3, 0.6, 1),
          ("blue_swirl", [0.04, 0.2, 0.75], [0.9, 0.92, 0.95], 4, -0.4, 2),
          ("red_swirl", [0.7, 0.03, 0.03], [0.95, 0.8, 0.1], 2, 0.8, 3),
          ("green_swirl", [0.05, 0.55, 0.15], [0.95, 0.95, 0.85], 3, -0.7, 4),
          ("gold_swirl", [0.95, 0.7, 0.1], [0.55, 0.08, 0.05], 5, 0.5, 5)]
porcelains = [("red_porcelain", [0.6, 0.04, 0.03]), ("white_porcelain", [0.8, 0.78, 0.74]),
              ("blue_porcelain", [0.03, 0.08, 0.45]), ("jade_porcelain", [0.12, 0.5, 0.3]),
              ("coral_porcelain", [0.88, 0.38, 0.28]), ("black_porcelain", [0.015, 0.015, 0.018]),
              ("yellow_porcelain", [0.88, 0.7, 0.08])]
metals = [("brass", [0.91, 0.78, 0.42], 0.05), ("steel", [0.56, 0.57, 0.58], 0.03), ("copper", [0.95, 0.64, 0.54], 0.06)]

out = []
w = out.append
w('''# The marbles at night: the scene the renderer is for (docs/process/
# MEMORY.md, brief: the look of NVIDIA's Marbles at Night). A cluster of
# real marbles, 1.2 to 2.5 cm across, on a plank table, seen from 13 cm above
# it through a lens focused on them, under a black sky, lit by fireflies
# alone: the scene's only light.
#
# Thirty-six marbles, each of a kind that shows its own part of the light:
#
#   clear glass   refracts the night upside down, and focuses the fireflies'
#                 light on the table (caustics);
#   tinted glass  ruby, cobalt, emerald, amber, violet and aqua, deeper where
#                 thicker: glass filled with an absorbing medium
#                 (core/media/absorbing.h);
#   cat's-eyes    a clear shell round an opaque swirled core
#                 (core/textures/swirl.h), lit only through the glass;
#   porcelain     opaque glossy color under a clear coat
#                 (core/materials/coated.h): the color catches every firefly,
#                 the coat shows each as a sharp point;
#   metal         brass, steel and copper, mirrors.
#
# Through graphs/path.toml, the naive path tracer: one light of 616 chosen
# uniformly at each bounce. The porcelain shows that choice's noise most
# plainly, the speckle ReSTIR DI is for; the cores, lit only by paths that
# happen through the glass to a firefly, show the noise manifold next event
# estimation is for; the clear and tinted glass, the caustics. The preview
# counts every light at every pixel and is not for this scene.
#
# It opens in the dark (docs/research/2026-10-10-dark-opening.md): no light
# at all until the first firefly wakes. One swarm rests on the table and the
# marbles, in the frame; its fireflies wake one by one, light what they rest
# on, linger, and rise. The other holds above the frame, dark, and wakes
# later, coming down to circle the marbles. The share of each awake grows
# as the square of the time, so the first come one at a time and the last
# in a flood. The opening is the naive estimator's worst case: one light of
# 616 chosen uniformly, nearly all of them dark.
#
# Scale: meters, real ones. The table top is at 0.75 m; each marble rests on
# it, its center its radius above. Written by a script, the marbles placed
# at random without touching, seeded, by scripts/make_marbles.py: edit the
# scene there and write this file again.

[camera]
position = [0.0, 0.88, 0.3]
look_at = [0.0, 0.757, -0.07]
vertical_fov_degrees = 28
# Focused on the cluster's middle 39 cm away: the marbles near it sharp, the
# nearest and farthest soft, the fireflies beyond them soft discs.
lens = { radius = 0.003, focus = 0.39 }

[environment]
kind = "gradient"
# Black: the fireflies are the only light, so the frame is black until the
# first wakes.
zenith = [0.0, 0.0, 0.0]
horizon = [0.0, 0.0, 0.0]

[textures.walnut]
kind = "wood"
light = [0.32, 0.17, 0.07]
dark = [0.1, 0.05, 0.018]
ring = 0.01
board = 0.12
seed = 3
''')
for name, a, b, vanes, twist, seed in swirls:
    w(f'''[textures.{name}]
kind = "swirl"
a = {a}
b = {b}
vanes = {vanes}
twist = {twist}
seed = {seed}
''')
w('''[materials.table]
kind = "rough"
texture = "walnut"

[materials.grass]
kind = "rough"
color = [0.015, 0.02, 0.012]

[materials.glass]
kind = "dielectric"
ior = 1.5
''')
for name, tint in tints:
    w(f'''[media.{name}_tint]
kind = "absorbing"
tint = {tint}
tint_distance = 0.01
''')
for name, *_ in swirls:
    w(f'''[materials.{name.replace("_swirl", "_core")}]
kind = "rough"
texture = "{name}"
''')
for name, color in porcelains:
    w(f'''[materials.{name}]
kind = "coated"
color = {color}
ior = 1.5
''')
for name, f0, rough in metals:
    w(f'''[materials.{name}]
kind = "conductor"
f0 = {f0}
roughness = {rough}
''')
w('''[materials.firefly_gold]
kind = "emissive"
radiance = [396.0, 246.0, 60.0]

[materials.firefly_white]
kind = "emissive"
radiance = [306.0, 260.0, 184.0]

# The table: a top 2.8 m by 2.6 m, 3 cm thick, on four legs.
[[shapes]]
kind = "box"
min = [-1.4, 0.72, -1.6]
max = [1.4, 0.75, 1.0]
material = "table"

[[shapes]]
kind = "box"
min = [-1.34, 0.0, -1.54]
max = [-1.28, 0.72, -1.48]
material = "table"

[[shapes]]
kind = "box"
min = [1.28, 0.0, -1.54]
max = [1.34, 0.72, -1.48]
material = "table"

[[shapes]]
kind = "box"
min = [-1.34, 0.0, 0.88]
max = [-1.28, 0.72, 0.94]
material = "table"

[[shapes]]
kind = "box"
min = [1.28, 0.0, 0.88]
max = [1.34, 0.72, 0.94]
material = "table"

# The ground, to the horizon.
[[shapes]]
kind = "box"
min = [-200.0, -0.1, -200.0]
max = [200.0, 0.0, 200.0]
material = "grass"
''')
names = []
counters = {"tint": 0, "cats": 0, "porcelain": 0, "metal": 0, "clear": 0}
f = lambda v: f"{v:.4f}".rstrip("0").rstrip(".") if v != 0 else "0.0"
for i, (kind, x, z, r) in enumerate(placed):
    y = TABLE + r
    c = f"[{f(x)}, {f(y)}, {f(z)}]"
    name = f"marble_{i:02d}"
    names.append(name)
    k = counters[kind]; counters[kind] += 1
    if kind == "clear":
        w(f'[[shapes]]\nkind = "sphere"\nname = "{name}"\ncenter = {c}\nradius = {f(r)}\nmaterial = "glass"\n')
    elif kind == "tint":
        w(f'[[shapes]]\nkind = "sphere"\nname = "{name}"\ncenter = {c}\nradius = {f(r)}\nmaterial = "glass"\ninterior = "{tints[k % len(tints)][0]}_tint"\n')
    elif kind == "cats":
        core = swirls[k % len(swirls)][0].replace("_swirl", "_core")
        w(f'[[shapes]]\nkind = "sphere"\nname = "{name}"\ncenter = {c}\nradius = {f(r)}\nmaterial = "glass"\n')
        w(f'[[shapes]]\nkind = "sphere"\ncenter = {c}\nradius = {f(0.6 * r)}\nmaterial = "{core}"\n')
    elif kind == "porcelain":
        w(f'[[shapes]]\nkind = "sphere"\nname = "{name}"\ncenter = {c}\nradius = {f(r)}\nmaterial = "{porcelains[k % len(porcelains)][0]}"\n')
    else:
        w(f'[[shapes]]\nkind = "sphere"\nname = "{name}"\ncenter = {c}\nradius = {f(r)}\nmaterial = "{metals[k % len(metals)][0]}"\n')
targets = ", ".join(f'"{n}"' for n in names)
# The held swarm, above the frame, wakes from 10 to 45 s and comes down; the
# perched swarm, on the table and the marbles in the frame (the box is the
# camera's view of the table, the marbles' tops inside it), wakes from 1 to
# 20 s, lingers 3 to 12 s glowing, then rises.
openings = {
    "firefly_gold": ('start = { kind = "above", depth = 0.15 }',
                     "wake = { from = 10, to = 45, power = 2, ramp = 2 }"),
    "firefly_white": ('start = { kind = "perch", min = [-0.16, 0.75, -0.3], max = [0.16, 0.8, 0.09], '
                      'linger = [3, 12] }',
                      "wake = { from = 1, to = 20, power = 2, ramp = 1.5 }"),
}
for material, count, seed in (("firefly_gold", 480, 1), ("firefly_white", 136, 2)):
    start, wake = openings[material]
    w(f'''[[swarms]]
count = {count}
radius = 0.0015
material = "{material}"
min = [-0.6, 0.795, -0.75]
max = [0.6, 1.25, 0.22]
targets = [{targets}]
# Calm, and mostly circling the marbles, where a firefly flashes most.
speed = 0.05
clearance = 0.008
circle = 6
swoop = 0
drift = 1
flash = 0.9
dim = 0.1
seed = {seed}
{start}
{wake}
''')
sys.stdout.write("\n".join(out))
