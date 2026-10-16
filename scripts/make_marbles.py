#!/usr/bin/env python3
"""Writes scenes/marbles.toml: a cluster of marbles lit by fireflies.

    python3 scripts/make_marbles.py > scenes/marbles.toml
    python3 scripts/make_marbles.py --seed N [--out FILE]

The marbles are placed at random, none touching, from a fixed seed, so the
same script writes the same file on any machine (Python's random.Random is
specified to give the same sequence for the same seed). Edit the scene here,
not in the file it writes.

With no options it writes the real marbles, scenes/marbles.toml, byte for
byte. --seed N writes generated scene N instead, for the denoiser's training
data (docs/research/2026-10-14-denoiser-training-design.md, "The data"),
which never trains on the real marbles: from N alone, the marbles' kinds,
counts, sizes, colors and layout, the firefly swarms' seeds, and the
camera's place and focus, moved about the real framing and drawn again
until every marble is in frame. The same N writes the same file. --out FILE
writes to FILE rather than to standard output.
"""
import argparse
from collections.abc import Sequence
import math
import random
import sys

parser = argparse.ArgumentParser(description="Writes the marbles scene, or a generated one (--seed).")
parser.add_argument("--seed", type=int, help="write generated scene SEED rather than the real marbles")
parser.add_argument("--out", help="the file to write (default: standard output)")
args = parser.parse_args()
generated = args.seed is not None

# The real marbles' layout seed, 7; a generated scene's own stream, keyed
# by its seed (a string seed is hashed by SHA-512, the same on any machine).
rng = random.Random(7 if not generated else f"marbles-{args.seed}")
TABLE = 0.75
CAM_Z = 0.32

# The kinds and how many of each, and the radii drawn from, in meters: the
# real marbles' 36; a generated scene's 28 to 40, every kind present, its
# radii scaled by 0.85 to 1.2, and its layout's spread (the mean and spread
# in z, and the spread in x as a share of the view) moved about the real
# one's.
counts = {"clear": 4, "tint": 9, "cats": 9, "porcelain": 10, "metal": 4}
radii = [0.006, 0.007, 0.008, 0.008, 0.008, 0.009, 0.0095, 0.0105, 0.0125]
spread = (-0.06, 0.085, 0.33)
if generated:
    while True:
        counts = {"clear": rng.randint(2, 6), "tint": rng.randint(4, 12), "cats": rng.randint(4, 12),
                  "porcelain": rng.randint(5, 13), "metal": rng.randint(2, 6)}
        if 28 <= sum(counts.values()) <= 40:
            break
    scale = rng.uniform(0.85, 1.2)
    radii = [round(r * scale, 5) for r in radii]
    spread = (rng.uniform(-0.09, -0.03), rng.uniform(0.07, 0.095), rng.uniform(0.28, 0.36))
kinds = [kind for kind, n in counts.items() for _ in range(n)]
rng.shuffle(kinds)
radius_of = lambda: rng.choice(radii)

# The frame a camera sees (16:9, 28 degrees high), for a generated scene's
# marbles, each kept inside the real camera's as it is placed, and its
# camera, drawn until it frames them all.
FOV = 28.0
AIM = (0.0, 0.757, -0.07)
EYE = (0.0, 0.88, 0.3)


def framed(position: Sequence[float], look_at: Sequence[float], marbles: list[tuple[float, float, float]]) -> bool:
    """Whether every (x, z, r) of `marbles`, whole, is inside the frame of a camera at `position` aimed at `look_at`."""
    forward = [a - b for a, b in zip(look_at, position)]
    forward = [v / math.hypot(*forward) for v in forward]
    right = [-forward[2], 0.0, forward[0]]  # forward x (0, 1, 0)
    right = [v / math.hypot(*right) for v in right]
    up = [right[1] * forward[2] - right[2] * forward[1], right[2] * forward[0] - right[0] * forward[2],
          right[0] * forward[1] - right[1] * forward[0]]
    half_h = math.tan(math.radians(FOV / 2))
    half_w = half_h * 16 / 9
    for x, z, r in marbles:
        v = [x - position[0], TABLE + r - position[1], z - position[2]]
        depth = sum(a * b for a, b in zip(v, forward))
        if depth <= 0.05:
            return False
        # Its center's place, and its radius seen from there, with a margin.
        if abs(sum(a * b for a, b in zip(v, right))) + 1.2 * r > 0.92 * half_w * depth:
            return False
        if abs(sum(a * b for a, b in zip(v, up))) + 1.2 * r > 0.92 * half_h * depth:
            return False
    return True


placed = []
for kind in kinds:
    for attempt in range(20000):
        r = radii[-1] if (kind == "clear" and not any(k == "clear" and rr == radii[-1] for k, _, _, rr in placed)) else radius_of()
        z = rng.gauss(spread[0], spread[1])
        if not (-0.30 < z < 0.16):
            continue
        d = CAM_Z - z
        x = rng.gauss(0.0, spread[2] * d * 0.386)
        if abs(x) > 0.36 * d:
            continue
        if generated and not framed(EYE, AIM, [(x, z, r)]):
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
if generated:
    # Which colors a scene's marbles of each kind wear, in its own order.
    for palette in (tints, swirls, porcelains, metals):
        rng.shuffle(palette)

# The camera: the real marbles', or, for a generated scene, moved about it
# (up to 12 degrees around its aim, 10% nearer or farther, 2 cm higher or
# lower, the aim moved up to 2 cm on the table) and focused on its aim;
# drawn again until every marble is inside its frame. The real camera
# frames them all (each was placed so), so a draw near it always does.
position, look_at, focus = list(EYE), list(AIM), 0.39
if generated:
    back = [e - a for e, a in zip(EYE, AIM)]
    for _ in range(1000):
        turn = math.radians(rng.uniform(-12.0, 12.0))
        reach = rng.uniform(0.9, 1.1)
        lift = rng.uniform(-0.02, 0.02)
        aim = [AIM[0] + rng.uniform(-0.02, 0.02), AIM[1], AIM[2] + rng.uniform(-0.02, 0.02)]
        eye = [aim[0] + reach * (back[0] * math.cos(turn) + back[2] * math.sin(turn)),
               aim[1] + reach * back[1] + lift,
               aim[2] + reach * (-back[0] * math.sin(turn) + back[2] * math.cos(turn))]
        eye, aim = [round(v, 4) for v in eye], [round(v, 4) for v in aim]
        if framed(eye, aim, [(x, z, r) for _, x, z, r in placed]):
            position, look_at, focus = eye, aim, round(math.dist(eye, aim), 4)
            break
    else:
        sys.exit("no camera frames every marble")
swarm_seeds = (1, 2) if not generated else (rng.randrange(1, 2 ** 31), rng.randrange(1, 2 ** 31))

f = lambda v: f"{v:.4f}".rstrip("0").rstrip(".") if v != 0 else "0.0"
out = []
w = out.append
real = '''# The marbles at night: the scene the renderer is for (docs/process/
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

'''
made = f'''# Generated marbles, scene {args.seed}: written by scripts/make_marbles.py
# --seed {args.seed}, for the denoiser's training data (docs/research/
# 2026-10-14-denoiser-training-design.md, "The data"); never the real
# marbles (scenes/marbles.toml), which the network is tested on. The real
# scene's materials, table, sky and fireflies; {sum(counts.values())} marbles ({counts["clear"]} clear,
# {counts["tint"]} tinted, {counts["cats"]} cat's-eyes, {counts["porcelain"]} porcelain, {counts["metal"]} metal); their sizes,
# colors and layout, the swarms' seeds and the camera drawn from the seed.

[camera]
position = [{f(position[0])}, {f(position[1])}, {f(position[2])}]
look_at = [{f(look_at[0])}, {f(look_at[1])}, {f(look_at[2])}]
vertical_fov_degrees = 28
# Focused on its aim, the cluster's middle.
lens = {{ radius = 0.003, focus = {f(focus)} }}

'''
w((made if generated else real) + '''[environment]
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
# Paced to be full within some 20 s: a viewer decides in the first few
# seconds. The perched swarm, on the table and the marbles in the frame (the
# box is the camera's view of the table, the marbles' tops inside it, its
# near edge where a loop can start above a perch: the volume's near face
# less the start's margin, some 11 cm), wakes from 0.5 to 8 s, lingers 10
# to 20 s glowing, then rises; the held swarm, above the frame, wakes from 1
# to 12 s and comes down while the perched still glow, so the frame holds
# some 100 to 140 lit fireflies from 8 s to 24 s, and the perched rise past
# the descending as it thins to the loops' some 40.
openings = {
    "firefly_gold": ('start = { kind = "above", depth = 0.15 }',
                     "wake = { from = 1, to = 12, power = 2, ramp = 1 }"),
    "firefly_white": ('start = { kind = "perch", min = [-0.16, 0.75, -0.3], max = [0.16, 0.8, 0.0], '
                      'linger = [10, 20] }',
                      "wake = { from = 0.5, to = 8, power = 2, ramp = 1 }"),
}
for (material, count), seed in zip((("firefly_gold", 480), ("firefly_white", 136)), swarm_seeds):
    start, wake = openings[material]
    w(f'''[[swarms]]
count = {count}
radius = 0.0015
material = "{material}"
# Its near face 20 cm in front of the lens: nearer, a firefly blurs to a
# disc a tenth of the frame tall (focused at 39 cm); here, some 30 pixels
# of 1080 at most.
min = [-0.6, 0.795, -0.75]
max = [0.6, 1.25, 0.1]
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
if args.out:
    with open(args.out, "w", encoding="utf-8", newline="") as file:
        file.write("\n".join(out))
else:
    sys.stdout.write("\n".join(out))
