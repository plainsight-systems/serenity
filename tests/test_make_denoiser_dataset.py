"""scripts/make_denoiser_dataset.py's plan, run as its user runs it (--print-plan).

Contract: the plan is the same on every run, pinned by its SHA-256, so the
data set on disk (denoiser-v2, once rendered) is the one the script describes; a changed
plan fails here and is another data set, named anew. The splits: 36
generated scenes train and 4 validate, the brass sphere scenes train, the
real marbles are the test and nothing else, and no generated scene is the
real marbles' seed, 7. Each sequence has at least 30 frames of history
before its first kept frame, several start in the opening and several in
the flood, and no target's sample indices meet an input's. A data set made
--targets-from another takes its scenes, albedo and targets and never its
inputs, and refuses one made by another plan. Every scene is written
through a pinhole: its lens line, and only it, taken off; a lens in a form
the script does not know is refused. Rendering is the
renderer's to show (docs/research/2026-10-14-denoiser-dataset.md).
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/make_denoiser_dataset.py'
sys.path.insert(0, str(SCRIPT.parent))
import make_denoiser_dataset  # noqa: E402
PLAN_SHA256 = 'ef6356c5134b0e2ef71ad1308f1f8aa22b7b25e726e7e4bee2223357f863f5d7'


def printed_plan():
    return subprocess.run([sys.executable, str(SCRIPT), '--print-plan'], check=True, capture_output=True).stdout


class MakeDenoiserDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = printed_plan()
        cls.plan = json.loads(cls.text)
        cls.seqs = cls.plan['sequences']

    def test_the_plan_is_the_same_every_run(self):
        self.assertEqual(self.text, printed_plan())
        self.assertEqual(hashlib.sha256(self.text).hexdigest(), PLAN_SHA256)
        self.assertFalse(self.plan['trial'])

    def test_the_splits(self):
        by = lambda split: [s for s in self.seqs if s['split'] == split]
        generated = lambda split: [s['seed'] for s in by(split) if s['seed'] is not None]
        self.assertEqual(len(generated('train')), 36)
        self.assertEqual(len(generated('validation')), 4)
        self.assertEqual(generated('test'), [])
        self.assertFalse(set(generated('train')) & set(generated('validation')))
        self.assertNotIn(7, generated('train') + generated('validation'))
        self.assertEqual(sorted(s['scene'] for s in by('train') if s['seed'] is None),
                         ['brass_sphere', 'brass_sphere_flight', 'brass_sphere_wander'])
        self.assertTrue(all(s['scene'] == 'marbles' for s in by('test')))
        self.assertNotIn('marbles', [s['scene'] for s in by('train') + by('validation')])
        self.assertEqual(sorted(s['start'] // 60 for s in by('test')), [2, 4, 6, 8, 30, 60, 120])

    def test_the_times(self):
        generated = [s for s in self.seqs if s['seed'] is not None]
        starts = [s['times'][0] for s in generated]
        self.assertTrue(all(1.0 <= t <= 120.0 for t in starts))
        self.assertGreaterEqual(sum(t < 8.0 for t in starts), 8)          # the opening
        self.assertGreaterEqual(sum(8.0 <= t < 24.0 for t in starts), 8)  # the flood, fireflies in flight
        for s in self.seqs:
            self.assertGreaterEqual(s['start'], self.plan['warmup_at_least'])
            self.assertEqual(s['frames'], list(range(s['start'], s['start'] + 8)))
            self.assertEqual(s['times'], [i * (1.0 / 60.0) for i in s['frames']])

    def test_no_target_shares_a_sample_index(self):
        n = self.plan['target']['samples']
        last_input = max(s['frames'][-1] for s in self.seqs)
        blocks = sorted((first * n, first * n + n - 1) for s in self.seqs for first in s['target_first'])
        self.assertGreater(blocks[0][0], max(last_input, 65535))  # past the inputs and a reference's
        self.assertLess(blocks[-1][1], 2 ** 32)
        # Within one scene, each target its own block (scenes share nothing).
        for s in self.seqs:
            own = sorted(s['target_first'])
            self.assertEqual(len(set(own)), len(own))

    def test_reuse_takes_scenes_albedo_and_targets_and_never_inputs(self):
        old = {'plan': self.plan, 'units': {
            'scene/gen-0101': {'files': {}}, 'input/gen-0101': {'files': {}},
            'albedo/gen-0101': {'files': {}}, 'target/gen-0101/000100': {'files': {}}}}
        taken = make_denoiser_dataset.reusable_units(old, self.plan)
        self.assertEqual(sorted(taken), ['albedo/gen-0101', 'scene/gen-0101', 'target/gen-0101/000100'])

    def test_reuse_refuses_another_plan(self):
        old = {'plan': make_denoiser_dataset.plan(sequences=1), 'units': {}}
        with self.assertRaises(SystemExit):
            make_denoiser_dataset.reusable_units(old, self.plan)

    def test_each_frame_has_the_renderers_jitter(self):
        """The renderer's (src/core/sampling/jitter.h), pinned to the table its test and training's hold
        (tests/jitter_test.cpp, training/test_training.py)."""
        table = {0: (0.0, -1.0 / 6.0), 1: (-0.25, 1.0 / 6.0), 2: (0.25, -7.0 / 18.0),
                 31: (-0.484375, 0.2901234567901234), 32: (0.0, -1.0 / 6.0), 7200: (0.0, -1.0 / 6.0),
                 7201: (-0.25, 1.0 / 6.0)}
        for index, (x, y) in table.items():
            jx, jy = make_denoiser_dataset.camera_jitter(index)
            self.assertAlmostEqual(jx, x, places=12, msg=index)
            self.assertAlmostEqual(jy, y, places=12, msg=index)
        for s in self.seqs:
            self.assertEqual(s['jitters'], [make_denoiser_dataset.camera_jitter(i) for i in s['frames']])

    def test_the_input_graph_renders_what_the_demo_hands_its_network(self):
        """The input graph's ReSTIR PT passes and settings are the demo's, and it has the features; a graph
        that differs in any is refused before anything renders."""
        read = lambda g: (ROOT / g).read_text(encoding='utf-8')
        inp, demo = read(make_denoiser_dataset.INPUT_GRAPH), read(make_denoiser_dataset.DEMO_GRAPH)
        self.assertIsNone(make_denoiser_dataset.input_graph_mismatch(inp, demo))
        self.assertIn('[restir_pt] settings',
                      make_denoiser_dataset.input_graph_mismatch(inp.replace('sigma = 16.0', 'sigma = 12.0'), demo))
        self.assertIn('no features', make_denoiser_dataset.input_graph_mismatch(inp.replace('"features", ', ''), demo))
        self.assertIn('passes', make_denoiser_dataset.input_graph_mismatch(
            inp.replace('"restir_pt_spatial", ', ''), demo))

    def test_every_scene_is_written_through_a_pinhole(self):
        for text in ((ROOT / 'scenes/marbles.toml').read_text(encoding='utf-8'),
                     subprocess.run([sys.executable, str(ROOT / 'scripts/make_marbles.py'), '--seed', '101'],
                                    check=True, capture_output=True, text=True).stdout):
            pinhole = make_denoiser_dataset.pinhole_scene(text)
            camera = lambda scene: scene.split('[camera]', 1)[1].split('\n[', 1)[0]
            self.assertIn('lens', camera(text))
            self.assertNotIn('lens', camera(pinhole))
            # Nothing else changed: the lens's one line is the difference.
            lines = text.splitlines()
            self.assertEqual([line for line in lines if not line.startswith('lens')], pinhole.splitlines())
        scene = '[camera]\nvertical_fov_degrees = 30\n'
        self.assertEqual(make_denoiser_dataset.pinhole_scene(scene), scene)
        with self.assertRaises(SystemExit):
            make_denoiser_dataset.pinhole_scene('[camera]\nlens = {\n radius = 0.1 }\n')


if __name__ == '__main__':
    unittest.main()
