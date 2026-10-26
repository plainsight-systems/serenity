"""The trained network's U-Net, for Metal 4's machine-learning path: a Metal package beside its weights.

    uv run python export_metal.py --model MODEL_DIR

Reads MODEL_DIR's layout.json and weights.bin (export.py's) and writes, into MODEL_DIR:

  unet.mtlpackage   the U-Net's features (model.py, UNet.features: the 30 inputs to the
                    last convolution's features, before the 1 x 1 head, which the renderer's
                    own output kernel computes), as Apple's workflow makes one: an ML program
                    converted by Core ML Tools, its tensors and weights 16-bit (Open Image
                    Denoise's precision on Metal), its height and width flexible (the
                    renderer sets them for its frame's size), then Xcode's
                    metal-package-builder (WWDC25 session 262). Its input `features_in`,
                    (1, 30, H, W); its output `features_out`, (1, last width, H, W).

Before writing it, checks the ML program computes what PyTorch computes in 32 bits on a random
frame, within what 16-bit tensors round to; the export is refused otherwise
(docs/research/2026-10-23-apple-silicon-design.md, section 6, change 3).

The program targets Core ML's iOS 18 operation set, not the newest: metal-package-builder of
Xcode 26 rejects the newest set's concatenation ("Missing pattern to rewrite MIL op
ios19.concat"), which every decoder's skip uses.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from export import read_back
from model import NETWORK_INPUTS, UNet

PACKAGE = "unet.mtlpackage"
# The sides Metal may be asked for: a multiple of UNet.alignment, up to the largest light the
# renderer traces (3456 x 2234's, padded) and past it.
SIDE_RANGE = (UNet.alignment, 4096)
# The check's frame: random inputs in [0, 1) at a size the U-Net takes.
CHECK_SIZE = (96, 160)
# 16-bit tensors round each value to 2^-11 of itself; through 11 convolutions, measured some 1e-4
# of the largest feature on random weights. Refused past a hundredth.
TOLERANCE = 1e-2


class Features(torch.nn.Module):
    """UNet.features alone, for tracing."""

    def __init__(self, unet: UNet):
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet.features(x)


def convert(model: torch.nn.Module) -> ct.models.MLModel:
    traced = torch.jit.trace(model, torch.rand(1, NETWORK_INPUTS, *CHECK_SIZE))
    height = ct.RangeDim(lower_bound=SIDE_RANGE[0], upper_bound=SIDE_RANGE[1], default=CHECK_SIZE[0])
    width = ct.RangeDim(lower_bound=SIDE_RANGE[0], upper_bound=SIDE_RANGE[1], default=CHECK_SIZE[1])
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name="features_in", shape=(1, NETWORK_INPUTS, height, width), dtype=np.float16)],
        outputs=[ct.TensorType(name="features_out", dtype=np.float16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )


def checked(program: ct.models.MLModel, model: torch.nn.Module) -> float:
    """The largest difference from PyTorch's 32-bit features, as a fraction of their largest."""
    x = torch.rand(1, NETWORK_INPUTS, *CHECK_SIZE, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = model(x).numpy()
    found = program.predict({"features_in": x.numpy().astype(np.float16)})["features_out"]
    return float(np.abs(found.astype(np.float32) - expected).max() / max(np.abs(expected).max(), 1e-6))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", type=Path, required=True, help="export.py's MODEL_DIR")
    args = p.parse_args()
    doc = json.loads((args.model / "layout.json").read_text())
    network = read_back(doc, (args.model / "weights.bin").read_bytes())
    model = Features(network.unet).eval()
    program = convert(model)
    error = checked(program, model)
    print(f"ML program against PyTorch: {error:.2e} of the largest feature")
    if not error <= TOLERANCE:
        print(f"refused: past {TOLERANCE}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as work:
        source = Path(work) / "unet.mlpackage"
        program.save(str(source))
        built = Path(work) / PACKAGE
        subprocess.run(["xcrun", "metal-package-builder", str(source), "-ml", "-o", str(built)], check=True)
        target = args.model / PACKAGE
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(built, target)
    print(f"wrote {args.model / PACKAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
