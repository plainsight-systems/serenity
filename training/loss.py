"""The training loss (docs/research/2026-10-14-denoiser-training-design.md,
"The loss").

Per frame t of a sequence, on PU-transferred radiance P = pu(output),
T = pu(target):

    spatial(t)  = 0.16 mean|P - T| + 0.84 (1 - MS-SSIM(P, T))
    temporal(t) = mean|(P_t - P_t-1) - (T_t - T_t-1)|          (t >= 1)

    loss = sum_t w_t (spatial(t) + 0.125 temporal(t)) / sum_t w_t

- The spatial term is Open Image Denoise's default, l1_msssim (training/
  loss.py: MixLoss([L1Loss(), MSSSIMLoss()], [0.16, 0.84]), after Zhao et
  al. 2018), with OIDN's MS-SSIM for its RT filter: 5 scales weighted 0.2
  each (config.py), an 11-tap Gaussian window of sigma 1.5, K1 = 0.01,
  K2 = 0.03, data range 1, 2 x 2 average pooling between scales, contrast
  and structure at the four finer scales and luminance too at the coarsest
  (training/ssim.py). Written again here from that definition.
- PU is OIDN's "pu" transfer (training/color.py): the PU2 fit of Aydin et
  al. 2008, normalized so 65504 maps to 1. Its default for HDR color. OIDN
  first scales each image by an autoexposure; this does not, because the
  renderer has no autoexposure (src/core/passes/tone_map.h: exposure fixed,
  "a flash shows as a flash"), so the loss weighs error at the brightness
  the viewer sees it at.
- The temporal term is Chaitanya et al. 2017's (Interactive Reconstruction
  of Monte Carlo Image Sequences using a Recurrent Denoising Autoencoder,
  section 4.4): the L1 difference of the temporal derivatives, consecutive
  frames finite-differenced, unwarped. Its weight is theirs relative to
  the spatial term: w_s / w_t = 0.8 / 0.1, "to approximately equalize the
  scales" (their third term, a gradient-domain L1 at 0.1, is not in the
  design and not used). So 0.125 against a spatial weight of 1.
- The frame weights w_t are theirs too: "higher weight to the loss functions
  of frames later in the sequence to amplify temporal gradients", a
  Gaussian, (0.011, 0.044, 0.135, 0.325, 0.607, 0.882, 1) for 7 frames,
  which is exp(-(T - 1 - t)^2 / 8); for 8 frames the same curve, one value
  longer: w_0 = 0.0022 ... w_7 = 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# PU2, normalized at 100 cd/m^2 [Aydin et al. 2008], as OIDN fits it (color.py).
PU_A, PU_B, PU_C, PU_D = 1.41283765e03, 1.64593172e00, 4.31384981e-01, -2.94139609e-03
PU_E, PU_F, PU_G = 1.92653254e-01, 6.26026094e-03, 9.98620152e-01
PU_Y0, PU_Y1 = 1.57945760e-06, 3.22087631e-02
HDR_Y_MAX = 65504.0


def _pu_raw(y: torch.Tensor) -> torch.Tensor:
    return torch.where(y <= PU_Y0, PU_A * y,
                       torch.where(y <= PU_Y1, PU_B * torch.pow(torch.clamp(y, min=PU_Y0), PU_C) + PU_D,
                                   PU_E * torch.log(torch.clamp(y, min=PU_Y1) + PU_F) + PU_G))


PU_NORM_SCALE = 1.0 / float(_pu_raw(torch.tensor(HDR_Y_MAX, dtype=torch.float64)))


def pu(y: torch.Tensor) -> torch.Tensor:
    """OIDN's PU transfer: linear radiance >= 0 to roughly perceptually uniform [0, 1]."""
    return _pu_raw(torch.clamp(y, min=0.0)) * PU_NORM_SCALE


SPATIAL_L1 = 0.16
SPATIAL_MSSSIM = 0.84
TEMPORAL = 0.1 / 0.8
MSSSIM_WEIGHTS = (0.2, 0.2, 0.2, 0.2, 0.2)
WINDOW, SIGMA, K1, K2 = 11, 1.5, 0.01, 0.03


def frame_weights(frames: int) -> list[float]:
    """Chaitanya et al.'s Gaussian frame weights, the last frame 1."""
    return [math.exp(-((frames - 1 - t) ** 2) / 8.0) for t in range(frames)]


def _gaussian(channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.arange(WINDOW, device=device, dtype=dtype) - WINDOW // 2
    g = torch.exp(-(x ** 2) / (2 * SIGMA ** 2))
    g = g / g.sum()
    return g.view(1, 1, 1, WINDOW).repeat(channels, 1, 1, 1)


def _blur(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    x = F.conv2d(x, g, groups=c)
    return F.conv2d(x, g.transpose(2, 3), groups=c)


def _ssim_parts(x: torch.Tensor, y: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(luminance x contrast-structure, contrast-structure), each averaged over the image: (B, C)."""
    c1, c2 = K1 ** 2, K2 ** 2
    mx, my = _blur(x, g), _blur(y, g)
    sxx = _blur(x * x, g) - mx * mx
    syy = _blur(y * y, g) - my * my
    sxy = _blur(x * y, g) - mx * my
    cs = (2 * sxy + c2) / (sxx + syy + c2)
    lum = (2 * mx * my + c1) / (mx * mx + my * my + c1)
    return (lum * cs).mean(dim=(-2, -1)), cs.mean(dim=(-2, -1))


def ms_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """OIDN's MS-SSIM of (B, C, H, W) images in [0, 1], averaged over batch and channels."""
    if min(x.shape[-2:]) <= (WINDOW - 1) * 2 ** (len(MSSSIM_WEIGHTS) - 1):
        raise ValueError(f"MS-SSIM needs images larger than {(WINDOW - 1) * 16} on each side, not {x.shape[-2:]}")
    g = _gaussian(x.shape[1], x.device, x.dtype)
    weights = torch.tensor(MSSSIM_WEIGHTS, device=x.device, dtype=x.dtype)
    values = []
    for i in range(len(MSSSIM_WEIGHTS)):
        if i > 0:
            pad = (x.shape[2] % 2, x.shape[3] % 2)
            x = F.avg_pool2d(x, 2, padding=pad)
            y = F.avg_pool2d(y, 2, padding=pad)
        ssim, cs = _ssim_parts(x, y, g)
        values.append(ssim if i == len(MSSSIM_WEIGHTS) - 1 else cs)
    stacked = torch.stack(values, dim=-1)  # (B, C, scales)
    return torch.prod(F.relu(stacked) ** weights, dim=-1).mean()


def spatial_loss(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """OIDN's l1_msssim on PU values (B, 3, H, W)."""
    return SPATIAL_L1 * (p - t).abs().mean() + SPATIAL_MSSSIM * (1.0 - ms_ssim(p, t))


def sequence_loss(outputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The loss of (B, T, 3, H, W) outputs against targets, both linear radiance, and its weighted
    spatial and temporal parts (detached, for the log)."""
    frames = outputs.shape[1]
    w = frame_weights(frames)
    p = pu(outputs)
    t = pu(targets)
    total = outputs.new_zeros(())
    spatial_sum = outputs.new_zeros(())
    temporal_sum = outputs.new_zeros(())
    for i in range(frames):
        s = spatial_loss(p[:, i], t[:, i])
        term = s
        spatial_sum = spatial_sum + w[i] * s.detach()
        if i > 0:
            d = ((p[:, i] - p[:, i - 1]) - (t[:, i] - t[:, i - 1])).abs().mean()
            term = term + TEMPORAL * d
            temporal_sum = temporal_sum + w[i] * d.detach()
        total = total + w[i] * term
    norm = sum(w)
    return total / norm, {"spatial": spatial_sum / norm, "temporal": temporal_sum / norm}
