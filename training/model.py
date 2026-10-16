"""The denoiser: a kernel-predicting network that also upscales 2x each way.

docs/research/2026-10-14-denoiser-training-design.md, "What the network
sees" and "Output". One frame, 960 x 540 in, 1920 x 1080 out:

1. History. Last frame's output, kept demodulated (before the albedo
   multiply) at 1920 x 1080, is moved by this frame's motion vectors: the
   output pixel X (center X + 1/2) reads last frame's output at X + 1/2 +
   2 MV(p), bilinearly, where p = floor(X / 2) is the input pixel whose
   footprint holds X's center and MV(p) its motion in input pixels ("old =
   new + MV"; x 2 for output pixels). The motion is upsampled by nearest
   neighbor: a primary surface's vector, never a blend of two surfaces'
   across an edge. A read whose point falls outside the last frame is not
   history: `valid` is 0 there, and on a sequence's first frame everywhere.

2. Features. A U-Net (OIDN's layout: 3 x 3 convolutions and ReLUs, 2 x 2
   max pools, nearest upsampling, skip connections by concatenation, four
   levels below the input's) at 960 x 540 reads 28 channels: the noisy
   demodulated radiance through OIDN's PU transfer (loss.pu), the albedo,
   the view-space normal, the depth and the motion (data.py), and the moved
   history, PU-transferred, and its validity, each 2 x 2 block of output
   pixels folded into channels at the input pixel it covers (pixel
   unshuffle: 12 + 4 channels, nothing averaged away). A 1 x 1 convolution
   then gives each input pixel 4 x 31 scores: 31 for each of the four
   output pixels it covers.

3. Kernels, per output pixel X, over the input pixels around p = floor(X /
   2): three 3 x 3 sets at spacings 1, 2 and 4 (taps p + s (i, j), i, j in
   {-1, 0, 1}), each set's 9 scores softmax-normalized and used to average
   the noisy demodulated radiance; a softmax over 3 more scores blends the
   three averages. Taps outside the image get no weight (their scores are
   masked before the softmax), so nothing is read that is not there.
   Sub-pixel position: X's center lies a quarter of an input pixel from p's
   center, toward one of p's four corners; the four output pixels sharing p
   each get their own 31 scores (step 2), so each kernel is free to lean
   toward its own quarter. No fixed bilinear weight is imposed.

4. Blend. One more score, through a sigmoid and multiplied by `valid`,
   weighs the moved history against the filtered frame:
   demod_out = alpha history + (1 - alpha) filtered. demod_out is the
   history the next frame reads.

5. Albedo. The output is demod_out x factor(full-size albedo), the
   demodulation's rule (data.albedo_factor), so texture comes from the
   full-size albedo, at full resolution.

Every weight is a softmax or a sigmoid of something computed from this
frame and the last, so the output is a convex combination of noisy inputs
and history, never negative and never brighter than its brightest input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import INPUT_CHANNELS, SCALE, albedo_factor
from loss import pu

SPACINGS = (1, 2, 4)
TAPS = 9
SCORES = len(SPACINGS) * TAPS + len(SPACINGS) + 1  # 27 kernel taps, 3 scale weights, 1 history weight
SUBPIXELS = SCALE * SCALE
NETWORK_INPUTS = INPUT_CHANNELS + 3 * SUBPIXELS + SUBPIXELS  # 12 + 12 history + 4 validity = 28
MASKED = -1.0e4  # a score no softmax gives weight to


@dataclass
class Widths:
    """Channels per U-Net level (OIDN's naming: enc at each level, dec on the way up)."""
    enc: list[int] = field(default_factory=lambda: [32, 48, 64, 80])
    bottleneck: int = 96
    dec: list[int] = field(default_factory=lambda: [80, 64, 48, 32])  # levels 3, 2, 1, 0


def conv(cin: int, cout: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, padding=1)


class UNet(nn.Module):
    """Features at the input's resolution; inputs a multiple of 16 on each side."""

    alignment = 16

    def __init__(self, cin: int, cout: int, w: Widths):
        super().__init__()
        e = w.enc
        self.enc = nn.ModuleList([conv(cin, e[0])] + [conv(e[i - 1], e[i]) for i in range(1, 4)])
        self.bottleneck_a = conv(e[3], w.bottleneck)
        self.bottleneck_b = conv(w.bottleneck, w.bottleneck)
        d = w.dec
        ins = [w.bottleneck + e[3], d[0] + e[2], d[1] + e[1], d[2] + e[0]]
        self.dec = nn.ModuleList([conv(ins[i], d[i]) for i in range(4)])
        self.dec_final = conv(d[3], d[3])
        self.head = nn.Conv2d(d[3], cout, 1)
        nn.init.zeros_(self.head.weight)  # first kernels uniform, first blend 1/2
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for layer in self.enc:
            x = F.relu(layer(x))
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = F.relu(self.bottleneck_a(x))
        x = F.relu(self.bottleneck_b(x))
        for i, layer in enumerate(self.dec):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = F.relu(layer(torch.cat((x, skips[3 - i]), dim=1)))
        x = F.relu(self.dec_final(x))
        return self.head(x)


def warp_history(history: torch.Tensor, motion: torch.Tensor,
                 extent: tuple[int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Last frame's (B, 3, H, W) output moved by this frame's (B, 2, h, w) motion: (moved, valid (B, 1, H, W)).

    `extent`, the image's (H, W) inside a padded tensor: reads past it are not history.
    """
    _, _, height, width = history.shape
    if extent is not None:
        height, width = extent
    mv = F.interpolate(motion, scale_factor=SCALE, mode="nearest") * SCALE  # output pixels, nearest
    ys = torch.arange(history.shape[2], device=history.device, dtype=history.dtype) + 0.5
    xs = torch.arange(history.shape[3], device=history.device, dtype=history.dtype) + 0.5
    sx = xs[None, None, :] + mv[:, 0]
    sy = ys[None, :, None] + mv[:, 1]
    valid = ((sx >= 0) & (sx <= width) & (sy >= 0) & (sy <= height)).to(history.dtype)[:, None]
    grid = torch.stack((sx / history.shape[3] * 2 - 1, sy / history.shape[2] * 2 - 1), dim=-1)
    moved = F.grid_sample(history, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return moved * valid, valid


def tap_validity(height: int, width: int, device: torch.device, dtype: torch.dtype,
                 extent: tuple[int, int] | None = None) -> list[torch.Tensor]:
    """Per spacing, (1, 9, 1, h, w): 1 where a tap lands inside the image (its (h, w) `extent` if padded)."""
    inside = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
    eh, ew = extent if extent is not None else (height, width)
    inside[:, :, :eh, :ew] = 1.0
    return [F.unfold(inside, 3, dilation=s, padding=s).view(1, TAPS, 1, height, width) for s in SPACINGS]


class Denoiser(nn.Module):
    def __init__(self, widths: Widths | None = None):
        super().__init__()
        self.widths = widths or Widths()
        self.unet = UNet(NETWORK_INPUTS, SCORES * SUBPIXELS, self.widths)

    def frame(self, inp: torch.Tensor, albedo_full: torch.Tensor, history: torch.Tensor | None,
              extent: tuple[int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """One frame. inp (B, 12, h, w), albedo_full (B, 3, 2h, 2w), history (B, 3, 2h, 2w) or None.

        Returns (radiance (B, 3, 2h, 2w), demodulated output: the next history, parts for inspection).
        h and w must be multiples of UNet.alignment; a frame that is not is zero-padded (evaluate.py)
        and its own input (h, w) given as `extent`, so no tap and no history read lands in the padding.
        """
        out_extent = None if extent is None else (SCALE * extent[0], SCALE * extent[1])
        b, _, h, w = inp.shape
        demod = inp[:, 0:3]
        motion = inp[:, 10:12]
        if history is None:
            moved = torch.zeros(b, 3, SCALE * h, SCALE * w, device=inp.device, dtype=inp.dtype)
            valid = torch.zeros(b, 1, SCALE * h, SCALE * w, device=inp.device, dtype=inp.dtype)
        else:
            moved, valid = warp_history(history, motion, out_extent)
        net_in = torch.cat((pu(demod), inp[:, 3:12], F.pixel_unshuffle(pu(moved), SCALE),
                            F.pixel_unshuffle(valid, SCALE)), dim=1)
        scores = self.unet(net_in).view(b, SCORES, SUBPIXELS, h, w)

        filtered = []
        for k, (s, inside) in enumerate(zip(SPACINGS, tap_validity(h, w, inp.device, inp.dtype, extent))):
            patches = F.unfold(demod, 3, dilation=s, padding=s).view(b, 3, TAPS, h, w)
            logits = scores[:, k * TAPS:(k + 1) * TAPS]  # (B, 9, 4, h, w)
            logits = torch.where(inside > 0, logits, torch.full_like(logits, MASKED))
            weights = torch.softmax(logits, dim=1)
            filtered.append(torch.einsum("bcthw,btqhw->bcqhw", patches, weights))  # (B, 3, 4, h, w)
        scale_weights = torch.softmax(scores[:, 3 * TAPS:3 * TAPS + 3], dim=1)  # (B, 3, 4, h, w)
        mixed = sum(scale_weights[:, k:k + 1] * filtered[k] for k in range(len(SPACINGS)))
        mixed = F.pixel_shuffle(mixed.reshape(b, 3 * SUBPIXELS, h, w), SCALE)
        alpha = torch.sigmoid(F.pixel_shuffle(scores[:, SCORES - 1], SCALE)) * valid
        demod_out = alpha * moved + (1.0 - alpha) * mixed
        radiance = demod_out * albedo_factor(albedo_full)
        return radiance, demod_out, {"alpha": alpha, "valid": valid, "filtered": mixed}

    def forward(self, inputs: torch.Tensor, albedo_full: torch.Tensor,
                history: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """A sequence, recurrently: inputs (B, T, 12, h, w), albedo_full (B, T, 3, H, W) -> (B, T, 3, H, W).

        Each frame reads the frame before's own output (its demodulated
        output), from `history` for the first, if given. Returns the outputs
        and the last history.
        """
        outputs = []
        for t in range(inputs.shape[1]):
            radiance, history, _ = self.frame(inputs[:, t], albedo_full[:, t], history)
            outputs.append(radiance)
        return torch.stack(outputs, dim=1), history


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def macs_per_input_pixel(widths: Widths) -> float:
    """Multiply-adds of the U-Net per 960 x 540 input pixel (the kernels' application not counted)."""
    e, bn, d = widths.enc, widths.bottleneck, widths.dec
    level = [1.0, 1 / 4, 1 / 16, 1 / 64, 1 / 256]
    total = 9 * NETWORK_INPUTS * e[0] * level[0]
    total += sum(9 * e[i - 1] * e[i] * level[i] for i in range(1, 4))
    total += 9 * (e[3] * bn + bn * bn) * level[4]
    ins = [bn + e[3], d[0] + e[2], d[1] + e[1], d[2] + e[0]]
    total += sum(9 * ins[i] * d[i] * level[3 - i] for i in range(4))
    total += 9 * d[3] * d[3] + d[3] * SCORES * SUBPIXELS
    return total
