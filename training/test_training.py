"""The toolkit's conventions, checked on synthetic data (no data set, no GPU).

    uv run python -m unittest test_training -v

Each test names the convention it holds; each was seen to fail with the
convention broken (docs/research/2026-10-15-denoiser-training.md, "Tests").
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import Augmentation, albedo_factor, augment, validation_tiles
from loss import frame_weights, ms_ssim, pu
from model import SCORES, SUBPIXELS, TAPS, Denoiser, Widths, warp_history
from pfm import read_pfm, write_pfm
from train import pick_device

TINY = Widths(enc=[8, 8, 8, 8], bottleneck=8, dec=[8, 8, 8, 8])


def moving_scene(t: int, height: int, width: int) -> torch.Tensor:
    """(3, H, W): a smooth pattern moving +2 output pixels in x and -2 in y per frame."""
    y = torch.arange(height, dtype=torch.float64)[:, None] + 2.0 * t
    x = torch.arange(width, dtype=torch.float64)[None, :] - 2.0 * t
    return torch.stack([torch.sin(0.21 * x + 0.13 * y), torch.cos(0.17 * x - 0.11 * y), torch.sin(0.05 * x * y / 40)])


class PfmTest(unittest.TestCase):
    def test_round_trip_top_row_first(self):
        image = np.random.default_rng(0).random((5, 7, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            write_pfm(Path(d) / "a.pfm", image)
            raw = (Path(d) / "a.pfm").read_bytes()
            self.assertTrue(raw.startswith(b"PF\n7 5\n-1.0\n"))
            # The file's first row is the image's last (PFM stores rows bottom first).
            first = np.frombuffer(raw[len(b"PF\n7 5\n-1.0\n"):][:7 * 12], dtype="<f4").reshape(7, 3)
            np.testing.assert_array_equal(first, image[-1])
            np.testing.assert_array_equal(read_pfm(Path(d) / "a.pfm"), image)


class WarpTest(unittest.TestCase):
    def test_old_equals_new_plus_motion(self):
        """NRD's convention: last frame's point of output pixel X is X + 2 MV (MV in input pixels)."""
        h, w = 16, 24
        prev = moving_scene(0, 2 * h, 2 * w)[None].float()
        cur = moving_scene(1, 2 * h, 2 * w)[None].float()
        # The pattern moved +2 output px in x and -2 in y, so a point now at X was at X - (2, -2):
        # MV = (-1, +1) input pixels.
        motion = torch.zeros(1, 2, h, w)
        motion[:, 0], motion[:, 1] = -1.0, 1.0
        moved, valid = warp_history(prev, motion)
        inner = (slice(None), slice(None), slice(4, -4), slice(4, -4))
        torch.testing.assert_close(moved[inner], cur[inner], atol=1e-5, rtol=0)
        self.assertEqual(float(valid[inner].min()), 1.0)
        self.assertEqual(float(valid[0, 0, :, 0].max()), 0.0)  # read from left of the left edge
        self.assertEqual(float(valid[0, 0, -1, :].max()), 0.0)  # and from below the bottom
        self.assertEqual(float(valid[0, 0, 0, 2:].min()), 1.0)  # but the top row reads inside

    def test_augmentation_keeps_motion_true(self):
        """Flips and the transpose carry into the motion vectors: the warp still lands."""
        h, w = 16, 16
        prev = moving_scene(0, 2 * h, 2 * w).float()
        cur = moving_scene(1, 2 * h, 2 * w).float()
        inp = torch.zeros(2, 12, h, w)
        inp[:, 10], inp[:, 11] = -1.0, 1.0
        tgt = torch.zeros(2, 6, 2 * h, 2 * w)
        tgt[0, 0:3], tgt[1, 0:3] = prev, cur
        for hflip in (False, True):
            for vflip in (False, True):
                for transpose in (False, True):
                    a = Augmentation(hflip, vflip, transpose, (0, 1, 2), (0, 1, 2), False)
                    i, t = augment(inp, tgt, a)
                    moved, _ = warp_history(t[0:1, 0:3], i[1:2, 10:12])
                    inner = (slice(None), slice(None), slice(4, -4), slice(4, -4))
                    torch.testing.assert_close(moved[inner], t[1:2, 0:3][inner], atol=1e-5, rtol=0,
                                               msg=f"hflip {hflip} vflip {vflip} transpose {transpose}")

    def test_augmentation_mirrors_normals(self):
        """A view-space normal pointing right points left after a horizontal flip; after a transpose,
        a normal pointing right points down (-up), one pointing up points left (-right)."""
        inp = torch.zeros(1, 12, 4, 4)
        inp[:, 6] = 1.0  # right
        i, _ = augment(inp, torch.zeros(1, 6, 8, 8), Augmentation(True, False, False, (0, 1, 2), (0, 1, 2), False))
        self.assertEqual(float(i[0, 6, 0, 0]), -1.0)
        i, _ = augment(inp, torch.zeros(1, 6, 8, 8), Augmentation(False, False, True, (0, 1, 2), (0, 1, 2), False))
        self.assertEqual((float(i[0, 6, 0, 0]), float(i[0, 7, 0, 0])), (0.0, -1.0))

    def test_color_permutation_is_shared(self):
        inp = torch.zeros(1, 12, 2, 2)
        tgt = torch.zeros(1, 6, 4, 4)
        inp[:, 0], inp[:, 3], tgt[:, 0], tgt[:, 3] = 1.0, 2.0, 3.0, 4.0  # red everywhere
        i, t = augment(inp, tgt, Augmentation(False, False, False, (2, 0, 1), (0, 1, 2), False))
        self.assertEqual([float(i[0, c, 0, 0]) for c in (0, 1, 2)], [0.0, 1.0, 0.0])
        self.assertEqual([float(i[0, c, 0, 0]) for c in (3, 4, 5)], [0.0, 2.0, 0.0])
        self.assertEqual([float(t[0, c, 0, 0]) for c in (0, 1, 2)], [0.0, 3.0, 0.0])
        self.assertEqual([float(t[0, c, 0, 0]) for c in (3, 4, 5)], [0.0, 4.0, 0.0])


class KernelTest(unittest.TestCase):
    def scored(self, bias: torch.Tensor) -> Denoiser:
        model = Denoiser(TINY)
        with torch.no_grad():
            model.unet.head.bias.copy_(bias)
        return model

    def test_center_tap_is_nearest_upsampling_times_albedo(self):
        """Scores all on spacing 1's center tap, no history: each output pixel is its input pixel's
        demodulated radiance times the full-size albedo factor."""
        bias = torch.zeros(SCORES, SUBPIXELS)
        bias[4] = 50.0                      # spacing 1, center tap (index 4 of 3 x 3)
        bias[3 * TAPS + 0] = 50.0           # all weight on spacing 1
        model = self.scored(bias.reshape(-1))
        inp = torch.rand(1, 12, 16, 16)
        albedo = torch.rand(1, 3, 32, 32)
        out, _, _ = model.frame(inp, albedo, None)
        expected = F.interpolate(inp[:, 0:3], scale_factor=2, mode="nearest") * albedo_factor(albedo)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_taps_outside_the_image_get_no_weight(self):
        """Uniform scores: at a corner only the 4 (of 9) spacing-1 taps inside the image count."""
        bias = torch.zeros(SCORES, SUBPIXELS)
        bias[3 * TAPS + 0] = 50.0
        model = self.scored(bias.reshape(-1))
        inp = torch.zeros(1, 12, 16, 16)
        inp[:, 0:3] = torch.rand(1, 3, 16, 16)
        _, _, parts = model.frame(inp, torch.ones(1, 3, 32, 32), None)
        corner = inp[0, 0:3, 0:2, 0:2].mean(dim=(1, 2))
        torch.testing.assert_close(parts["filtered"][0, :, 0, 0], corner, atol=1e-5, rtol=1e-5)

    def test_subpixels_get_their_own_kernels(self):
        """The four output pixels over one input pixel can each choose a different tap."""
        bias = torch.zeros(SCORES, SUBPIXELS)
        bias[3 * TAPS + 0] = 50.0
        bias[5, 1] = 50.0   # top-right output pixel: the tap to the right
        bias[4, 0] = bias[4, 2] = bias[4, 3] = 50.0
        model = self.scored(bias.reshape(-1))
        inp = torch.zeros(1, 12, 16, 16)
        inp[:, 0:3] = torch.rand(1, 3, 16, 16)
        _, _, parts = model.frame(inp, torch.ones(1, 3, 32, 32), None)
        f = parts["filtered"][0]
        torch.testing.assert_close(f[:, 10, 11], inp[0, 0:3, 5, 6], atol=1e-5, rtol=1e-5)  # (2*5, 2*5+1)
        torch.testing.assert_close(f[:, 10, 10], inp[0, 0:3, 5, 5], atol=1e-5, rtol=1e-5)

    def test_history_blend_only_where_valid(self):
        bias = torch.zeros(SCORES, SUBPIXELS)
        bias[SCORES - 1] = 50.0  # alpha -> 1
        model = self.scored(bias.reshape(-1))
        inp = torch.zeros(1, 12, 16, 16)
        inp[:, 0:3] = 1.0
        history = torch.full((1, 3, 32, 32), 5.0)
        _, demod_out, _ = model.frame(inp, torch.ones(1, 3, 32, 32), history)
        torch.testing.assert_close(demod_out, history, atol=1e-4, rtol=0)
        _, demod_out, _ = model.frame(inp, torch.ones(1, 3, 32, 32), None)
        torch.testing.assert_close(demod_out, torch.ones_like(history), atol=1e-6, rtol=0)


class LossTest(unittest.TestCase):
    def test_pu_is_monotone_and_normalized(self):
        y = torch.logspace(-8, 4.8, 200, dtype=torch.float64)
        v = pu(y)
        self.assertTrue(bool((v[1:] > v[:-1]).all()))
        self.assertAlmostEqual(float(pu(torch.tensor(65504.0, dtype=torch.float64))), 1.0, places=6)

    def test_ms_ssim_of_an_image_with_itself_is_one(self):
        x = torch.rand(2, 3, 192, 192)
        self.assertAlmostEqual(float(ms_ssim(x, x)), 1.0, places=5)
        self.assertLess(float(ms_ssim(x, torch.rand_like(x))), 0.5)

    def test_frame_weights_are_chaitanyas(self):
        w = frame_weights(7)
        np.testing.assert_allclose(w, [0.011, 0.044, 0.135, 0.325, 0.607, 0.882, 1.0], atol=6e-4)


class TilingTest(unittest.TestCase):
    def test_validation_tiles_are_whole_and_centered(self):
        tiles = validation_tiles(1)
        self.assertEqual(len(tiles), 28)
        self.assertEqual(tiles[0], (0, 14, 32))
        self.assertEqual(tiles[-1], (0, 14 + 3 * 128, 32 + 6 * 128))



class DeviceTest(unittest.TestCase):
    """Training runs on CUDA or the Mac's GPU: the one asked for if the machine has it, else CUDA
    where there is one, else MPS; never a device the machine lacks, never without a GPU."""

    def test_the_device_asked_for_or_the_one_there_is(self):
        self.assertEqual(pick_device(None, cuda=True, mps=False), "cuda")
        self.assertEqual(pick_device(None, cuda=False, mps=True), "mps")
        self.assertEqual(pick_device("mps", cuda=True, mps=True), "mps")
        with self.assertRaises(SystemExit):
            pick_device("cuda", cuda=False, mps=True)
        with self.assertRaises(SystemExit):
            pick_device(None, cuda=False, mps=False)


if __name__ == "__main__":
    unittest.main()
