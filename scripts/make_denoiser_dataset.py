#!/usr/bin/env python3
"""Renders the denoiser's data set, denoiser-v1, into a directory outside the repo.

    python3 scripts/make_denoiser_dataset.py --out DIR            render it, or resume
    python3 scripts/make_denoiser_dataset.py --out DIR --estimate  the first input and target, timed; stop
    python3 scripts/make_denoiser_dataset.py --out DIR --verify    check every file against the manifest
    python3 scripts/make_denoiser_dataset.py --print-plan          the plan as JSON; renders nothing

What it is (docs/research/2026-10-14-denoiser-training-design.md, "The data"
and "Splits"; what was rendered, docs/research/2026-10-14-denoiser-dataset.md):

  - 40 generated scenes, scripts/make_marbles.py --seed 101 .. 140, never
    the real marbles (scenes/marbles.toml, the layout of seed 7); 36 train,
    4 validation (seeds 105, 115, 125, 135), fixed here. The brass sphere
    scenes train too, as other geometry at another scale. The real marbles
    are the test set alone.
  - Per scene one sequence of 8 consecutive frames at 60 frames a second,
    its first frame's time drawn per scene from its stratum: the opening
    (1 to 8 s, fireflies waking in the dark), the flood (8 to 24 s, the
    most fireflies lit and the perched ones rising past the descending),
    awake (24 to 120 s). The test: the real marbles from t = 2, 4, 6 and 8
    (the opening; t = 8, its end, is labeled the flood in the plan, as the
    strata are half-open) and t = 30, 60 and 120 (t = 8 is asked for twice
    and rendered once).
  - Per frame, the input: graphs/restir_pt_features.toml (ReSTIR PT + MNEE
    and the features) at 960 x 540, one sample, its linear radiance and its
    albedo, normal, depth and motion. One run a sequence, frames f .. f + 7.
    serenity-headless replays a reusing graph's frames 0 .. f - 1 before
    frame f (core/frame/history.h), so each frame is the very frame a run
    from the start shows: its history full, at least 60 frames (t >= 1 s),
    more than the design's 30 of warm-up.
  - Per frame, the target: graphs/path.toml at 1920 x 1080, TARGET_SAMPLES
    samples, time frozen at the frame's instant, linear; and the albedo at
    1920 x 1080 from graphs/features.toml (one run a sequence, the input's
    frame indices, so its rays are the ones the runtime's full-size albedo
    would take) to multiply the lighting back.

Random numbers. The shaders key every one by pixel, frame index and
dimension. The input's indices are the timeline's, 0 .. f + 7 (under 7,300).
A target of frame i is rendered as headless frame TARGET_FIRST + i, its
samples the indices (TARGET_FIRST + i) x TARGET_SAMPLES and on, each target
its own block, all far past the inputs' and past the 65,536 a reference
uses from index 0 (Makefile, reference). So no target shares a number with
its input or with another target.

The plan is a function of the constants below alone, the same on any machine
(random.Random of a string seed hashes it by SHA-512); tests/
test_make_denoiser_dataset.py pins it. A changed plan is another data set:
name it anew.

Resuming. Work is done in units: a scene's file, a sequence's input run, its
albedo run, and each target frame. Each renders into DIR/.staging, its
files are checked finite, moved into place, hashed (SHA-256) and recorded in
DIR/manifest.json, written whole and renamed, so a unit is in the manifest
only if all its files are. A rerun verifies every recorded unit's files
against their checksums, renders again any unit whose files are missing or
changed, and skips the rest. A manifest made by another plan or another
renderer commit is refused: the data set would mix two.

The renderer: build/native-release/serenity-headless, built from this
checkout first; its commit, whether the tree was clean, its build flags and
the binary's checksum go in the manifest. A dirty tree is refused unless
--allow-dirty (for trial runs; recorded).

Failure. A run that fails (serenity-headless fails a frame whose samples
were not finite, naming the count) is logged with its message and recorded
in the manifest's failures; the other units go on; the script then exits
non-zero naming them. Every PFM written is also scanned for non-finite
floats, and the count recorded (it should be 0).

The GPU. Rendering is not timed for a result, but another serenity process
shares the GPU and slows it: each sequence logs any serenity process not
ours and the GPU's utilization (ioreg), and goes on.

Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATASET = 'denoiser-v1'

# The scenes and splits.
GENERATED_SEEDS = tuple(range(101, 141))
VALIDATION_SEEDS = (105, 115, 125, 135)
REAL_MARBLES_SEED = 7  # the real marbles' layout (make_marbles.py): never a generated scene
BRASS_SCENES = ('brass_sphere', 'brass_sphere_flight', 'brass_sphere_wander')
TEST_SCENE = 'marbles'
TEST_STARTS = (2, 4, 6, 8, 30, 60, 120)  # seconds: the opening, then t = 30, 60 and 120

# The strata a generated scene's start time is drawn from, by seed: one each
# of the 10 opening and 10 flood seeds is validation, and two of the awake.
STRATA = (
    ('opening', range(101, 111), 1.0, 8.0),
    ('flood', range(111, 121), 8.0, 24.0),
    ('awake', range(121, 141), 24.0, 120.0),
)
BRASS_STRATUM = ('brass', 1.0, 120.0)

# The frames.
FPS = 60
STEP = 1.0 / FPS  # serenity-headless's default step, given explicitly
KEPT = 8          # frames a sequence keeps
WARMUP = 30       # at least this many frames of history before the first kept one

# The renders.
INPUT_GRAPH = 'graphs/restir_pt_features.toml'
INPUT_SIZE = (960, 540)
TARGET_GRAPH = 'graphs/path.toml'
ALBEDO_GRAPH = 'graphs/features.toml'
TARGET_SIZE = (1920, 1080)
TARGET_SAMPLES = 4096
TARGET_FIRST = 100_000  # target of frame i: headless frame TARGET_FIRST + i
INPUT_FEATURES = ('albedo', 'normal', 'depth', 'motion')
RENDERER = ROOT / 'build/native-release/serenity-headless'


def frame_time(index: int) -> float:
    """Frame `index`'s time, as serenity-headless computes it (step x index, in double)."""
    return STEP * index


def stratum_of(seconds: float) -> str:
    """The generated strata's name for a time: a test sequence's, by where it starts."""
    return next((name for name, _, _, high in STRATA if seconds < high), STRATA[-1][0])


def drawn_start(name: str, low: float, high: float) -> int:
    """A sequence's first frame, drawn from [low, high) seconds by `name` alone."""
    t = random.Random(f'{DATASET}-time-{name}').uniform(low, high)
    return round(t * FPS)


def plan(sequences: int | None = None, kept: int = KEPT, target_samples: int = TARGET_SAMPLES) -> dict:
    """The data set: its sequences, each with its scene, split, frames and renders. Pure.

    `sequences`, `kept` and `target_samples` other than their defaults make a
    trial: the first `sequences` of the plan, `kept` frames each.
    """
    seqs = []
    for stratum, seeds, low, high in STRATA:
        for seed in seeds:
            name = f'gen-{seed:04d}'
            seqs.append({'sequence': name, 'scene': name, 'seed': seed,
                         'scene_source': f'scripts/make_marbles.py --seed {seed}',
                         'split': 'validation' if seed in VALIDATION_SEEDS else 'train',
                         'stratum': stratum, 'start': drawn_start(name, low, high)})
    stratum, low, high = BRASS_STRATUM
    for scene in BRASS_SCENES:
        seqs.append({'sequence': scene, 'scene': scene, 'seed': None, 'scene_source': f'scenes/{scene}.toml',
                     'split': 'train', 'stratum': stratum, 'start': drawn_start(scene, low, high)})
    for t in TEST_STARTS:
        seqs.append({'sequence': f'{TEST_SCENE}-t{t:03d}', 'scene': TEST_SCENE, 'seed': None,
                     'scene_source': f'scenes/{TEST_SCENE}.toml', 'split': 'test',
                     'stratum': stratum_of(t), 'start': t * FPS})
    order = {'train': 0, 'validation': 1, 'test': 2}
    seqs.sort(key=lambda s: order[s['split']])
    if sequences is not None:
        seqs = seqs[:sequences]
    for s in seqs:
        s['frames'] = list(range(s['start'], s['start'] + kept))
        s['times'] = [frame_time(i) for i in s['frames']]
        s['target_first'] = [TARGET_FIRST + i for i in s['frames']]
    return {
        'dataset': DATASET,
        'trial': (sequences, kept, target_samples) != (None, KEPT, TARGET_SAMPLES),
        'fps': FPS, 'step': STEP, 'kept': kept, 'warmup_at_least': WARMUP,
        'input': {'graph': INPUT_GRAPH, 'size': list(INPUT_SIZE), 'samples': 1,
                  'features': list(INPUT_FEATURES)},
        'target': {'graph': TARGET_GRAPH, 'size': list(TARGET_SIZE), 'samples': target_samples,
                   'first_offset': TARGET_FIRST, 'albedo_graph': ALBEDO_GRAPH},
        'sequences': seqs,
    }


# ---------------------------------------------------------------- files

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


# A float32 is not finite when its exponent's 8 bits are all ones: bits 23
# to 30, the high byte's low 7 and the next byte's top bit (little-endian).
_HIGH = bytes(1 if (b & 0x7F) == 0x7F else 0 for b in range(256))
_NEXT = bytes(1 if b & 0x80 else 0 for b in range(256))


def non_finite_floats(path: Path) -> int:
    """How many of a little-endian PFM's floats are NaN or infinite."""
    data = Path(path).read_bytes()
    lines = data.split(b'\n', 3)
    if len(lines) < 4 or lines[0] != b'PF' or float(lines[2]) >= 0:
        raise ValueError(f'{path}: not a little-endian colour PFM')
    body = memoryview(data)[len(data) - len(lines[3]):]
    high = int.from_bytes(bytes(body[3::4]).translate(_HIGH), 'little')
    nxt = int.from_bytes(bytes(body[2::4]).translate(_NEXT), 'little')
    return (high & nxt).bit_count()


def pfm_bytes(size: tuple[int, int]) -> int:
    w, h = size
    return len(f'PF\n{w} {h}\n-1.0\n') + w * h * 12


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=1, sort_keys=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


# ---------------------------------------------------------------- the renderer

def git(*args: str) -> str:
    return subprocess.run(['git', '-C', str(ROOT), *args], check=True, capture_output=True, text=True).stdout


def build_renderer() -> None:
    for cmd in (['cmake', '--preset', 'native-release', '--log-level=NOTICE'],
                ['cmake', '--build', '--preset', 'native-release', '--target', 'serenity-headless']):
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f'building the renderer failed: {" ".join(cmd)}\n{result.stdout}{result.stderr}')


def renderer_record(allow_dirty: bool) -> dict:
    dirty = [line for line in git('status', '--porcelain').splitlines() if line]
    if dirty and not allow_dirty:
        sys.exit('the checkout has uncommitted changes, so no commit names the renderer; commit them, '
                 'or pass --allow-dirty for a trial:\n' + '\n'.join(dirty))
    cache = (ROOT / 'build/native-release/CMakeCache.txt').read_text(encoding='utf-8')
    cached = dict(re.findall(r'^([A-Z_]+):[A-Z]+=(.*)$', cache, re.M))
    metal = (ROOT / 'cmake/MetalLibrary.cmake').read_text(encoding='utf-8')
    return {
        'commit': git('rev-parse', 'HEAD').strip(),
        'dirty': dirty,
        'binary': str(RENDERER.relative_to(ROOT)),
        'binary_sha256': sha256(RENDERER),
        'build': {
            'preset': 'native-release',
            'build_type': cached.get('CMAKE_BUILD_TYPE'),
            'cxx_compiler': cached.get('CMAKE_CXX_COMPILER'),
            'cxx_flags': ' '.join(filter(None, (cached.get('CMAKE_CXX_FLAGS'), cached.get('CMAKE_CXX_FLAGS_RELEASE')))),
            'cxx_standard': 20,
            'osx_deployment_target': cached.get('CMAKE_OSX_DEPLOYMENT_TARGET'),
            'metal_flags': ' '.join(re.findall(r'set\(SERENITY_METAL_\w+ "([^"]*)"\)', metal)),
            'toolchain': json.loads((ROOT / 'cmake/toolchain.json').read_text(encoding='utf-8')),
        },
    }


def gpu_note(utilization: bool) -> str:
    """Any serenity process running (none of ours is, between units), and, if asked, the GPU's
    utilization: asked before the first render only, since ioreg's lags behind our own last one."""
    others = subprocess.run(['pgrep', '-l', 'serenity'], capture_output=True, text=True).stdout.split('\n')
    others = [line for line in others if line.strip()]
    notes = []
    if utilization:
        ioreg = subprocess.run(['ioreg', '-r', '-d', '1', '-c', 'IOAccelerator'], capture_output=True,
                               text=True).stdout
        util = re.search(r'"Device Utilization %"=(\d+)', ioreg)
        notes.append(f'GPU {util.group(1) if util else "?"}% busy')
    notes.append(f'other serenity processes: {", ".join(others)}' if others else 'no other serenity process')
    return '; '.join(notes)


class RenderFailed(Exception):
    pass


def headless(args: list, staging: Path) -> float:
    """Runs serenity-headless with `args` into a fresh `staging`: its seconds, or RenderFailed."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(RENDERER), *[str(a) for a in args], '--out', str(staging)]
    start = time.monotonic()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.monotonic() - start
    if result.returncode != 0:
        raise RenderFailed(f'{" ".join(cmd)}\n{result.stderr.strip()}')
    return elapsed


# ---------------------------------------------------------------- the work

class Dataset:
    def __init__(self, out, the_plan, renderer):
        self.out = out
        self.plan = the_plan
        self.renderer = renderer
        self.path = out / 'manifest.json'
        self.staging = out / '.staging'
        self.units = {}
        self.failures = {}

    # The manifest.

    def load(self, same_renderer):
        if not self.path.exists():
            return
        old = json.loads(self.path.read_text(encoding='utf-8'))
        if old.get('plan') != self.plan:
            sys.exit(f'{self.path} was made by another plan: this is another data set; use another --out')
        if same_renderer and (old['renderer']['commit'] != self.renderer['commit']
                              or old['renderer']['binary_sha256'] != self.renderer['binary_sha256']):
            sys.exit(f'{self.path} was rendered by {old["renderer"]["commit"]} (binary '
                     f'{old["renderer"]["binary_sha256"][:12]}), this checkout is {self.renderer["commit"]} '
                     f'({self.renderer["binary_sha256"][:12]}): the data set would mix two renderers')
        self.units = old.get('units', {})
        self.failures = old.get('failures', {})
        if not same_renderer:
            self.renderer = old['renderer']  # verifying: the manifest's renderer stays its own

    def save(self):
        write_json(self.path, {
            'dataset': DATASET,
            'made_by': 'scripts/make_denoiser_dataset.py',
            'renderer': self.renderer,
            'plan': self.plan,
            'splits': {split: [s['sequence'] for s in self.plan['sequences'] if s['split'] == split]
                       for split in ('train', 'validation', 'test')},
            'scenes': {u.split('/', 1)[1]: rec for u, rec in self.units.items() if u.startswith('scene/')},
            'frames': self.frames(),
            'units': self.units,
            'failures': self.failures,
        })

    def frames(self):
        """One entry per frame whose input, albedo and target are all done."""
        entries = []
        for s in self.plan['sequences']:
            seq_dir = f'{s["split"]}/{s["sequence"]}'
            inp, alb = self.units.get(f'input/{s["sequence"]}'), self.units.get(f'albedo/{s["sequence"]}')
            for i, t, first in zip(s['frames'], s['times'], s['target_first']):
                tgt = self.units.get(f'target/{s["sequence"]}/{i:06d}')
                if not (inp and alb and tgt):
                    continue
                n = self.plan['target']['samples']
                inputs = {'radiance': f'{seq_dir}/input/frame-{i:06d}.pfm',
                          **{k: f'{seq_dir}/input/frame-{i:06d}-{k}.pfm' for k in INPUT_FEATURES}}
                targets = {'radiance': f'{seq_dir}/target/frame-{i:06d}.pfm',
                           'albedo': f'{seq_dir}/target/frame-{i:06d}-albedo.pfm'}
                entries.append({
                    'sequence': s['sequence'], 'scene': s['scene'], 'scene_file': f'scenes/{s["scene"]}.toml',
                    'seed': s['seed'], 'split': s['split'], 'stratum': s['stratum'],
                    'frame': i, 'time': t,
                    'input': {'graph': INPUT_GRAPH, 'size': list(INPUT_SIZE), 'samples': 1, 'frame_index': i,
                              'history_from_frame': 0,
                              'files': {k: {'path': p, 'sha256': inp['files'][p]} for k, p in inputs.items()}},
                    'target': {'graph': TARGET_GRAPH, 'size': list(TARGET_SIZE), 'samples': n,
                               'time_frozen': t, 'headless_frame': first,
                               'sample_indices': [first * n, first * n + n - 1],
                               'albedo': {'graph': ALBEDO_GRAPH, 'size': list(TARGET_SIZE), 'frame_index': i},
                               'files': {'radiance': {'path': targets['radiance'],
                                                      'sha256': tgt['files'][targets['radiance']]},
                                         'albedo': {'path': targets['albedo'],
                                                    'sha256': alb['files'][targets['albedo']]}}},
                    'non_finite_floats': sum(inp['non_finite'][p] for p in inputs.values())
                    + alb['non_finite'][targets['albedo']] + tgt['non_finite'][targets['radiance']],
                })
        return entries

    def verify(self):
        """Drops every recorded unit whose files are missing or changed; returns their ids."""
        bad = []
        for unit, rec in list(self.units.items()):
            for rel, digest in rec['files'].items():
                p = self.out / rel
                if not p.is_file() or sha256(p) != digest:
                    bad.append(unit)
                    del self.units[unit]
                    break
        return bad

    # The units.

    def done(self, unit):
        return unit in self.units

    def record(self, unit, files, seconds, non_finite):
        self.units[unit] = {'files': {str(p.relative_to(self.out)): sha256(p) for p in files},
                            'non_finite': {str(p.relative_to(self.out)): n for p, n in zip(files, non_finite)},
                            'seconds': round(seconds, 2)}
        self.failures.pop(unit, None)
        self.save()

    def place(self, moves):
        """Moves staged files into place; returns their final paths and each one's non-finite floats."""
        placed, bad = [], []
        for src, dst in moves:
            bad.append(non_finite_floats(src))
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
            placed.append(dst)
        return placed, bad

    def scene(self, s):
        unit = f'scene/{s["scene"]}'
        if self.done(unit):
            return self.out / f'scenes/{s["scene"]}.toml'
        dst = self.out / f'scenes/{s["scene"]}.toml'
        dst.parent.mkdir(parents=True, exist_ok=True)
        if s['seed'] is not None:
            subprocess.run([sys.executable, str(ROOT / 'scripts/make_marbles.py'), '--seed', str(s['seed']),
                            '--out', str(dst)], check=True)
        else:
            shutil.copyfile(ROOT / f'scenes/{s["scene"]}.toml', dst)
        self.units[unit] = {'files': {str(dst.relative_to(self.out)): sha256(dst)}, 'non_finite': {},
                            'source': s['scene_source'], 'seconds': 0}
        self.save()
        return dst

    def input(self, s, scene):
        unit = f'input/{s["sequence"]}'
        if self.done(unit):
            return None
        stage = self.staging / unit
        first, kept = s['frames'][0], len(s['frames'])
        seconds = headless(['--graph', INPUT_GRAPH, '--scene', scene, '--first', first, '--frames', kept,
                            '--step', repr(STEP), '--size', 'x'.join(map(str, INPUT_SIZE)), '--samples', 1,
                            '--format', 'pfm', '--write', 'all', '--features'], stage)
        seq_dir = self.out / s['split'] / s['sequence'] / 'input'
        names = [f'frame-{i:06d}.pfm' for i in s['frames']] + \
                [f'frame-{i:06d}-{k}.pfm' for i in s['frames'] for k in INPUT_FEATURES]
        files, bad = self.place([(stage / n, seq_dir / n) for n in names])
        self.record(unit, files, seconds, bad)
        return seconds

    def albedo(self, s, scene):
        unit = f'albedo/{s["sequence"]}'
        if self.done(unit):
            return None
        stage = self.staging / unit
        seconds = headless(['--graph', ALBEDO_GRAPH, '--scene', scene, '--first', s['frames'][0],
                            '--frames', len(s['frames']), '--step', repr(STEP),
                            '--size', 'x'.join(map(str, TARGET_SIZE)), '--write', 'all', '--features'], stage)
        seq_dir = self.out / s['split'] / s['sequence'] / 'target'
        # The albedo only: the network reads normal, depth and motion at the input's size.
        files, bad = self.place([(stage / f'frame-{i:06d}-albedo.pfm', seq_dir / f'frame-{i:06d}-albedo.pfm')
                                 for i in s['frames']])
        self.record(unit, files, seconds, bad)
        return seconds

    def target(self, s, scene, i, first):
        unit = f'target/{s["sequence"]}/{i:06d}'
        if self.done(unit):
            return None
        stage = self.staging / unit
        seconds = headless(['--graph', TARGET_GRAPH, '--scene', scene, '--first', first, '--frames', 1,
                            '--time', repr(frame_time(i)), '--size', 'x'.join(map(str, TARGET_SIZE)),
                            '--samples', self.plan['target']['samples'], '--format', 'pfm', '--write', 'last'],
                           stage)
        dst = self.out / s['split'] / s['sequence'] / 'target' / f'frame-{i:06d}.pfm'
        files, bad = self.place([(stage / f'frame-{first:06d}.pfm', dst)])
        self.record(unit, files, seconds, bad)
        return seconds


def estimate(the_plan: dict, input_seconds: float, input_frames: int, target_seconds: float,
             albedo_seconds: float) -> dict:
    """Hours for the whole plan from one input run, one target and one albedo run, and the bytes it writes."""
    seqs = the_plan['sequences']
    rate = input_seconds / input_frames  # a timeline frame, replayed or kept, start-up included
    timeline = sum(s['frames'][-1] + 1 for s in seqs)
    targets = sum(len(s['frames']) for s in seqs)
    total = timeline * rate + targets * target_seconds + len(seqs) * albedo_seconds
    per_frame = (1 + len(INPUT_FEATURES)) * pfm_bytes(INPUT_SIZE) + 2 * pfm_bytes(TARGET_SIZE)
    return {'hours': total / 3600, 'input_hours': timeline * rate / 3600,
            'target_hours': targets * target_seconds / 3600, 'timeline_frames': timeline,
            'targets': targets, 'bytes': targets * per_frame}


def main() -> int:
    parser = argparse.ArgumentParser(description='Renders the denoiser data set (see the docstring).')
    parser.add_argument('--out', type=Path, help='the data set directory (outside the repo)')
    parser.add_argument('--print-plan', action='store_true', help='print the plan as JSON and stop')
    parser.add_argument('--estimate', action='store_true', help='render the first input and target, estimate, stop')
    parser.add_argument('--verify', action='store_true', help='verify every planned file against the manifest')
    parser.add_argument('--sequences', type=int, help='trial: the first N sequences only')
    parser.add_argument('--frames', type=int, default=KEPT, help='trial: N frames a sequence')
    parser.add_argument('--target-samples', type=int, default=TARGET_SAMPLES, help='trial: target samples')
    parser.add_argument('--allow-dirty', action='store_true', help='trial: render from an uncommitted tree')
    args = parser.parse_args()

    the_plan = plan(args.sequences, args.frames, args.target_samples)
    if args.print_plan:
        print(json.dumps(the_plan, indent=1))
        return 0
    if args.out is None:
        parser.error('--out is required')
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error('--out must be outside the repository: the data is never committed')

    if not args.verify:
        build_renderer()
    ds = Dataset(out, the_plan, renderer_record(args.allow_dirty or args.verify))
    out.mkdir(parents=True, exist_ok=True)
    ds.load(same_renderer=not args.verify)
    t0 = time.monotonic()
    bad = ds.verify()
    print(f'verified {len(ds.units)} recorded units in {time.monotonic() - t0:.0f} s'
          + (f'; {len(bad)} missing or changed, to render again: {", ".join(bad)}' if bad else ''), flush=True)

    planned = []
    for s in the_plan['sequences']:
        planned += [f'scene/{s["scene"]}', f'input/{s["sequence"]}', f'albedo/{s["sequence"]}']
        planned += [f'target/{s["sequence"]}/{i:06d}' for i in s['frames']]
    planned = list(dict.fromkeys(planned))
    if args.verify:
        missing = [u for u in planned if u not in ds.units]
        non_finite = sum(n for rec in ds.units.values() for n in rec['non_finite'].values())
        files = sum(len(rec['files']) for rec in ds.units.values())
        size = sum((out / rel).stat().st_size for rec in ds.units.values() for rel in rec['files'])
        print(f'{len(planned) - len(missing)} of {len(planned)} planned units present, {files} files, '
              f'{size / 1e9:.1f} GB, checksums match; non-finite floats: {non_finite}')
        if missing or bad:
            print('missing: ' + ', '.join(missing))
            return 1
        return 0

    print(f'{gpu_note(True)}; {len(planned) - sum(u in ds.units for u in planned)} of {len(planned)} units to render',
          flush=True)
    ds.save()
    measured = {}
    run_start = time.monotonic()
    for n, s in enumerate(the_plan['sequences']):
        print(f'[{n + 1}/{len(the_plan["sequences"])}] {s["split"]} {s["sequence"]} '
              f'from frame {s["frames"][0]} (t = {s["times"][0]:.3f} s); {gpu_note(False)}', flush=True)
        try:
            scene = ds.scene(s)
            secs = ds.input(s, scene)
            if secs is not None:
                measured.setdefault('input', (secs, s['frames'][-1] + 1))
                print(f'  input: {secs:.1f} s ({s["frames"][-1] + 1} frames from 0)', flush=True)
            secs = ds.albedo(s, scene)
            if secs is not None:
                measured.setdefault('albedo', secs)
        except RenderFailed as failure:
            print(f'  FAILED: {failure}', flush=True)
            ds.failures[f'sequence/{s["sequence"]}'] = str(failure)
            ds.save()
            continue
        for i, first in zip(s['frames'], s['target_first']):
            try:
                secs = ds.target(s, scene, i, first)
            except RenderFailed as failure:
                print(f'  FAILED target {i}: {failure}', flush=True)
                ds.failures[f'target/{s["sequence"]}/{i:06d}'] = str(failure)
                ds.save()
                continue
            if secs is not None:
                measured.setdefault('target', secs)
                print(f'  target {i}: {secs:.1f} s', flush=True)
            if 'estimate' not in measured and {'input', 'albedo', 'target'} <= measured.keys():
                e = estimate(the_plan, *measured['input'], measured['target'], measured['albedo'])
                measured['estimate'] = e
                print(f'ESTIMATE: {e["hours"]:.2f} h in all ({e["input_hours"]:.2f} h of inputs, '
                      f'{e["timeline_frames"]} timeline frames; {e["target_hours"]:.2f} h of {e["targets"]} '
                      f'targets at {measured["target"]:.1f} s); {e["bytes"] / 1e9:.1f} GB', flush=True)
                if args.estimate:
                    return 0
    if ds.staging.exists():
        shutil.rmtree(ds.staging)
    ds.save()
    print(f'done in {(time.monotonic() - run_start) / 3600:.2f} h; {len(ds.frames())} frames complete; '
          f'{len(ds.failures)} failures', flush=True)
    for unit, message in ds.failures.items():
        print(f'FAILED {unit}: {message}', file=sys.stderr)
    return 1 if ds.failures else 0


if __name__ == '__main__':
    sys.exit(main())
