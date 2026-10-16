"""The renderer's tone map, in PyTorch, for scoring what the viewer sees.

The computation is the core's (src/core/passes/tone_map.h, steps 1 to 6),
with its constants, as src/metal/passes/tone_map/tone_map.metal carries it
out: exposure, a six-level bloom pyramid (Jimenez's 13-tap down filter, a
3 x 3 tent up), the composite, Khronos' PBR Neutral, the sRGB transfer, and
8 bits, as the PNG the renderer writes. Bilinear reads at texel centers,
clamped to the edge, normalized coordinates: grid_sample with
align_corners=False and border padding is that sampler.

Not bit-exact with the GPU: the renderer stores the pyramid as half floats
and its sampler filters with its own weight precision; this keeps floats.
How close it comes was measured once against the renderer's own PNG
(docs/research/2026-10-15-denoiser-training.md, "Displayed MSE"; no test
holds it): the displayed MSE here is only as exact as that.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

BLOOM_LEVELS = 6
BLOOM_CEILING = 65504.0
DOWN_MIDDLE = 0.5
DOWN_CORNER = 0.125
NEUTRAL_F90 = 0.04
NEUTRAL_TOE = 0.08
NEUTRAL_TOE_CURVE = 6.25
NEUTRAL_START = 0.76
NEUTRAL_DESATURATION = 0.15

# graphs/path.toml's and graphs/restir_pt.toml's settings, Milestone 2's.
EXPOSURE = 0.0
BLOOM = 0.04


def _sample(image: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinear reads of (N, C, H, W) `image` at normalized `uv` (N, h, w, 2), clamped to the edge."""
    return F.grid_sample(image, uv * 2.0 - 1.0, mode="bilinear", padding_mode="border", align_corners=False)


def _centers(height: int, width: int, device: torch.device) -> torch.Tensor:
    """(1, h, w, 2): each texel's center, (i + 0.5) / size, as x, y."""
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx, gy), dim=-1)[None]


def _down13(read: Callable[[torch.Tensor], torch.Tensor], source_hw: tuple[int, int], target_hw: tuple[int, int],
            device: torch.device) -> torch.Tensor:
    """Step 2's filter: five 2 x 2 boxes from 13 bilinear reads about each target texel's center."""
    uv = _centers(*target_hw, device)
    texel = torch.tensor([1.0 / source_hw[1], 1.0 / source_hw[0]], device=device)

    def s(dx: float, dy: float) -> torch.Tensor:
        return read(uv + torch.tensor([dx, dy], device=device) * texel)

    nw, n, ne = s(-2, -2), s(0, -2), s(2, -2)
    w, c, e = s(-2, 0), s(0, 0), s(2, 0)
    sw, so, se = s(-2, 2), s(0, 2), s(2, 2)
    middle = 0.25 * (s(-1, -1) + s(1, -1) + s(-1, 1) + s(1, 1))
    corners = 0.25 * ((nw + n + w + c) + (n + ne + c + e) + (w + c + sw + so) + (c + e + so + se))
    return DOWN_MIDDLE * middle + DOWN_CORNER * corners


def _tent(level: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Step 3's 3 x 3 tent of `level`, read at `target_hw`'s texel centers, a texel of `level` apart."""
    device = level.device
    uv = _centers(*target_hw, device)
    texel = torch.tensor([1.0 / level.shape[3], 1.0 / level.shape[2]], device=device)

    def s(dx: float, dy: float) -> torch.Tensor:
        return _sample(level, uv + torch.tensor([dx, dy], device=device) * texel)

    return (s(-1, -1) + 2 * s(0, -1) + s(1, -1) + 2 * s(-1, 0) + 4 * s(0, 0) + 2 * s(1, 0)
            + s(-1, 1) + 2 * s(0, 1) + s(1, 1)) / 16.0


def _neutral(color: torch.Tensor) -> torch.Tensor:
    """Step 5: Khronos' PBR Neutral, per pixel of (N, 3, H, W)."""
    x = color.min(dim=1, keepdim=True).values
    offset = torch.where(x < NEUTRAL_TOE, x - NEUTRAL_TOE_CURVE * x * x, torch.full_like(x, NEUTRAL_F90))
    color = color - offset
    peak = color.max(dim=1, keepdim=True).values
    span = 1.0 - NEUTRAL_START
    rolled = 1.0 - span * span / (peak + span - NEUTRAL_START)
    scaled = color * (rolled / torch.clamp(peak, min=1e-30))
    g = 1.0 - 1.0 / (NEUTRAL_DESATURATION * (peak - rolled) + 1.0)
    rolled_color = scaled * (1.0 - g) + rolled * g
    return torch.where(peak < NEUTRAL_START, color, rolled_color)


def _srgb(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * torch.pow(torch.clamp(c, min=0.0031308), 1.0 / 2.4) - 0.055)


def displayed(radiance: torch.Tensor, exposure: float = EXPOSURE, bloom: float = BLOOM) -> torch.Tensor:
    """(N, 3, H, W) linear radiance -> (N, 3, H, W) 8-bit display values as floats 0..255."""
    device = radiance.device
    radiance = radiance.float()
    scale = 2.0 ** exposure
    height, width = radiance.shape[2], radiance.shape[3]

    # Steps 1 and 2.
    sizes = []
    h, w = height, width
    for _ in range(BLOOM_LEVELS):
        h, w = max(1, (h + 1) // 2), max(1, (w + 1) // 2)
        sizes.append((h, w))
    levels = [
        _down13(lambda uv: torch.clamp(_sample(radiance, uv) * scale, max=BLOOM_CEILING), (height, width), sizes[0],
                device) / BLOOM_LEVELS
    ]
    for k in range(1, BLOOM_LEVELS):
        source = levels[-1]
        levels.append(_down13(lambda uv, src=source: _sample(src, uv), sizes[k - 1], sizes[k], device))
    # Step 3.
    for k in range(BLOOM_LEVELS - 2, -1, -1):
        levels[k] = levels[k] + _tent(levels[k + 1], sizes[k])
    # Steps 1 and 4.
    exposed = torch.clamp(radiance * scale, max=BLOOM_CEILING)
    composite = (1.0 - bloom) * exposed + bloom * _tent(levels[0], (height, width))
    # Steps 5 and 6, then 8 bits as the target texture stores them.
    encoded = _srgb(torch.clamp(_neutral(composite), 0.0, 1.0))
    return torch.round(encoded * 255.0)
