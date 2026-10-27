"""The trained network's U-Net, for the Neural Engine through Core ML: a compiled model beside its weights.

    uv run python export_coreml.py --model MODEL_DIR [--rungs HxW ...]

Reads MODEL_DIR's layout.json and weights.bin (export.py's) and writes, into MODEL_DIR:

  unet.mlmodelc   the U-Net's features (model.py, UNet.features: the 30 inputs to the last
                  convolution's features, before the 1 x 1 head, which the renderer's own output
                  kernel computes), an ML program converted by Core ML Tools, its tensors and
                  weights 16-bit, compiled by `xcrun coremlcompiler` so the renderer loads it
                  without compiling. Its input `features_in`, (1, 30, H, W); its output
                  `features_out`, (1, last width, H, W); both float16.

The Neural Engine runs only shapes known when the model is compiled: a flexible height and width
runs 5 times slower and falls partly off it (docs/research/2026-10-26-neural-engine-denoiser.md,
section 1). So the model enumerates its shapes: every pair of LADDER's sides, 121 of Core ML's
128. The renderer pads a light to the smallest rung that holds it and reads the rungs from the
model itself, so the ladder is written once, here. --rungs gives a model its own shapes instead:
the test fixture's (tests/data/denoise/network), whose lights are tests' sizes.

The first rung is the model's default shape, and the one the check runs. Core ML on macOS 26.6
traps in its first prediction when the default is small (measured: 32 x 32, 32 x 48 and 64 x 64
do, with 96 x 128 the other rung; with 256 x 256 the default, every one of those runs). So a
model's first rung is 256 x 256 or larger: LADDER's is, and the fixture's is given first.

Each of LADDER's sides is a multiple of 32 halves, so a plane's row is a multiple of 64 bytes and a
pixel buffer over it has no padding between rows (Core Video aligns rows to 64 bytes, contract 18's
tensor pitch; the renderer refuses a pitch that is not that). The design's 336 and 672 became 320
and 640 for this.

Before writing it, checks the model computes what PyTorch computes in 32 bits on a random frame,
within what 16-bit tensors round to, on the Neural Engine at one rung; the export is refused
otherwise, and if Core ML's compute plan puts any operation off the Neural Engine.
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

MODEL = "unet.mlmodelc"
# The ladder's sides (above); 2048 x 2048 is the largest light.
LADDER = (256, 320, 448, 544, 640, 832, 960, 1120, 1376, 1728, 2048)
# The check's frame, one of LADDER's rungs; a model of its own --rungs is checked at its first.
CHECK_SIZE = (256, 320)
# 16-bit tensors round each value to 2^-11 of itself; through 11 convolutions, measured some 4e-3
# of the largest feature on the trained weights (the note's section 1). Refused past a hundredth.
TOLERANCE = 1e-2


class Features(torch.nn.Module):
    """UNet.features alone, for tracing."""

    def __init__(self, unet: UNet):
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet.features(x)


def ladder() -> list[tuple[int, int]]:
    """LADDER's rungs, (height, width)."""
    return [(h, w) for h in LADDER for w in LADDER]


def convert(model: torch.nn.Module, rungs: list[tuple[int, int]]) -> ct.models.MLModel:
    for h, w in rungs:
        if h % UNet.alignment or w % UNet.alignment:
            raise ValueError(f"{h} x {w} is not a multiple of the U-Net's alignment, {UNet.alignment}")
    check = rungs[0]
    traced = torch.jit.trace(model, torch.rand(1, NETWORK_INPUTS, *check))
    shapes = ct.EnumeratedShapes(
        shapes=[(1, NETWORK_INPUTS, h, w) for h, w in rungs], default=(1, NETWORK_INPUTS, *check)
    )
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name="features_in", shape=shapes, dtype=np.float16)],
        outputs=[ct.TensorType(name="features_out", dtype=np.float16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )


def checked(program: ct.models.MLModel, model: torch.nn.Module, size: tuple[int, int]) -> float:
    """The largest difference from PyTorch's 32-bit features at `size`, as a fraction of their largest."""
    x = torch.rand(1, NETWORK_INPUTS, *size, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = model(x).numpy()
    found = program.predict({"features_in": x.numpy().astype(np.float16)})["features_out"]
    return float(np.abs(found.astype(np.float32) - expected).max() / max(np.abs(expected).max(), 1e-6))


def off_engine(compiled: Path) -> list[str]:
    """The operations the compute plan does not put on the Neural Engine."""
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    found = []
    for function in plan.model_structure.program.functions.values():
        for op in function.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                continue  # constants carry no device
            device = type(usage.preferred_compute_device).__name__
            if "NeuralEngine" not in device:
                found.append(f"{op.operator_name} on {device}")
    return found


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", type=Path, required=True, help="export.py's MODEL_DIR")
    p.add_argument("--rungs", nargs="+", metavar="HxW", help="the model's own shapes instead of LADDER's")
    args = p.parse_args()
    rungs = [tuple(int(v) for v in r.split("x")) for r in args.rungs] if args.rungs else ladder()
    doc = json.loads((args.model / "layout.json").read_text())
    network = read_back(doc, (args.model / "weights.bin").read_bytes())
    model = Features(network.unet).eval()
    program = convert(model, rungs)
    error = checked(program, model, CHECK_SIZE if not args.rungs else rungs[0])
    print(f"Core ML on the Neural Engine against PyTorch: {error:.2e} of the largest feature")
    if not error <= TOLERANCE:
        print(f"refused: past {TOLERANCE}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as work:
        source = Path(work) / "unet.mlpackage"
        program.save(str(source))
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(source), work], check=True)
        built = Path(work) / MODEL
        off = off_engine(built)
        if off:
            print("refused: off the Neural Engine:\n  " + "\n  ".join(off), file=sys.stderr)
            return 1
        target = args.model / MODEL
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(built, target)
    print(f"wrote {args.model / MODEL}: {len(rungs)} shapes, every operation on the Neural Engine")
    return 0


if __name__ == "__main__":
    sys.exit(main())
