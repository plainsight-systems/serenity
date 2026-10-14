"""scripts/make_marbles.py, run as its user runs it.

Contract: with no options it writes the real marbles, scenes/marbles.toml,
byte for byte; --seed N writes generated scene N, the same bytes for the same
N and other bytes for another; --out FILE writes to FILE what standard
output would hold. That a generated scene loads and frames its marbles is
the renderer's to show (docs/research/2026-10-14-denoiser-features.md).
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/make_marbles.py'


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], check=True, capture_output=True).stdout


class MakeMarblesTests(unittest.TestCase):
    def test_default_is_the_real_marbles_byte_for_byte(self):
        self.assertEqual(run(), (ROOT / 'scenes/marbles.toml').read_bytes())

    def test_a_seed_is_reproducible_and_seeds_differ(self):
        three = run('--seed', '3')
        self.assertEqual(three, run('--seed', '3'))
        self.assertNotEqual(three, run('--seed', '4'))
        self.assertNotEqual(three, run())
        self.assertIn(b'# Generated marbles, scene 3', three)

    def test_out_writes_what_standard_output_would(self):
        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch) / 'scene.toml'
            self.assertEqual(run('--seed', '5', '--out', str(out)), b'')
            self.assertEqual(out.read_bytes(), run('--seed', '5'))


if __name__ == '__main__':
    unittest.main()
