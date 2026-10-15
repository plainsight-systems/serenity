"""A rough inference time per frame on the M3 Max's GPU through PyTorch's MPS backend.

    uv run --locked python benchmark.py --checkpoint RUN/best.pt [--frames 50]

One 960 x 540 frame (padded to 544) to 1920 x 1080, with history, batch 1,
in float32 and float16: the whole frame, and the U-Net alone. The median of
`--frames` frames after 10 of warm-up, each synchronized. Measure with the
GPU otherwise idle (no serenity window, no training).

Not the runtime's cost: PyTorch dispatches each operation as its own kernel
(the kernels' softmax, gathers and the history warp are dozens of small
dispatches), and MPS's convolutions are not tuned for this network. Phase 4's
Metal kernels are the measurement that counts; this bounds what a naive
implementation costs and splits it between the features and the rest.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from data import INPUT_CHANNELS
from evaluate import load_network
from model import NETWORK_INPUTS, macs_per_input_pixel, parameter_count


def timed(fn, frames: int) -> float:
    for _ in range(10):
        fn()
    torch.mps.synchronize()
    samples = []
    for _ in range(frames):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--frames", type=int, default=50)
    args = p.parse_args()
    device = torch.device("mps")
    model, _ = load_network(args.checkpoint, device)
    print(f"{args.checkpoint}: {parameter_count(model):,} parameters, "
          f"{macs_per_input_pixel(model.widths) * 960 * 540 / 1e9:.1f} G multiply-adds a frame in the U-Net")
    g = torch.Generator(device="cpu").manual_seed(0)
    for dtype in (torch.float32, torch.float16):
        m = model.to(dtype)
        x = torch.rand(1, INPUT_CHANNELS, 544, 960, generator=g).to(device, dtype)
        a = torch.rand(1, 3, 1088, 1920, generator=g).to(device, dtype)
        h = torch.rand(1, 3, 1088, 1920, generator=g).to(device, dtype)
        u = torch.rand(1, NETWORK_INPUTS, 544, 960, generator=g).to(device, dtype)
        with torch.no_grad():
            whole = timed(lambda: m.frame(x, a, h, extent=(540, 960)), args.frames)
            unet = timed(lambda: m.unet(u), args.frames)
        print(f"{str(dtype).replace('torch.', '')}: whole frame {whole:.1f} ms, U-Net alone {unet:.1f} ms "
              f"(median of {args.frames})")


if __name__ == "__main__":
    main()
