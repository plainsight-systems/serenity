"""scripts/make_denoiser_dataset.py's plan, run as its user runs it (--print-plan).

Contract: the plan is the same on every run, pinned by its SHA-256, so the
data set on disk (denoiser-v1) is the one the script describes; a changed
plan fails here and is another data set, named anew. The splits: 36
generated scenes train and 4 validate, the brass sphere scenes train, the
real marbles are the test and nothing else, and no generated scene is the
real marbles' seed, 7. Each sequence has at least 30 frames of history
before its first kept frame, several start in the opening and several in
the flood, and no target's sample indices meet an input's. Rendering is the
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
PLAN_SHA256 = '604d537e1b06a7a67f117d4fc9dc2527af1f31400840073e0ad3b04b16f12d6f'


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


if __name__ == '__main__':
    unittest.main()
