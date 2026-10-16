"""The renderer's denoise pass against PyTorch: inputs, the expected outputs, and the comparison.

    # a check of real frames, with an exported network (not committed):
    uv run --locked python denoise_check.py fixture --network MODEL_DIR --data DATA \\
        --sequence gen-0105 --frames 8 --out CHECK_DIR
    # the committed test fixture: a random network of small widths, a crop
    uv run --locked python denoise_check.py fixture --random-network --data DATA \\
        --sequence gen-0101 --frames 2 --crop 600 300 40 22 --out ../tests/data/denoise
    # after the GPU has run a fixture (tests/gpu/denoise_check, or the test):
    uv run --locked python denoise_check.py compare --fixture CHECK_DIR --gpu GPU_DIR

docs/research/2026-10-15-denoiser-live.md, "The match against PyTorch".

A fixture is a directory the GPU side reads without Python: the network
(network/weights.bin, network/layout.json, export.py's format), and per frame
k the renderer's own images, as the denoise pass is given them at its light
size, cropped if asked: frame-K-radiance.pfm (the light pass's accumulated
image: the data set's noisy input), -albedo, -normal (world space), -depth,
-motion, and -full_albedo at twice the size; the camera's view axes and
the sizes in fixture.json; and -expected.pfm, what model.Denoiser.frame
computes from them, in float32 on the CPU, frame after frame with its own
history, the input zero-padded to a multiple of 16 as evaluate.py pads it.
The input channels are data.input_frame's, so the depth's median is over
the (cropped) frame, as the GPU computes it over its light.

The data set is read through its manifest, every file checked (data.py). A
sequence of the train or validation split: this is a numerical check of an
implementation, not a score, and the test split stays unread.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from data import SCALE, Manifest, input_frame, read_checked
from export import layout, read_back, serialize
from model import Denoiser, UNet, Widths
from pfm import read_pfm, write_pfm
from regions import view_axes
from tonemap import displayed

# The committed fixture's network: every width a multiple of 8, as the
# kernels need, and small, so the fixture stays some hundreds of KB; its
# widths' blocks (1 to 5) exercise every conv3x3 chunk size but 6's.
RANDOM_WIDTHS = Widths(enc=[8, 16, 24, 32], bottleneck=40, dec=[32, 24, 16, 8])


def random_network(seed: int) -> Denoiser:
    """A network of RANDOM_WIDTHS, PyTorch's default initialization, and the head (zeroed in training's
    init) drawn too, so the kernels and the blend are not uniform."""
    torch.manual_seed(seed)
    model = Denoiser(RANDOM_WIDTHS)
    with torch.no_grad():
        model.unet.head.weight.normal_(0.0, 0.5)
        model.unet.head.bias.normal_(0.0, 0.5)
    return model.eval()


def export_network(model: Denoiser, out: Path, provenance: dict) -> None:
    """export.py's two files for `model`."""
    out.mkdir(parents=True)
    tensors, weights = serialize(model)
    (out / "weights.bin").write_bytes(weights)
    w = model.widths
    state = dict(provenance, config={"widths": {"enc": w.enc, "bottleneck": w.bottleneck, "dec": w.dec}})
    (out / "layout.json").write_text(json.dumps(layout(model, tensors, state), indent=1))


def expected_outputs(model: Denoiser, frames: list[np.ndarray], albedo_full: list[np.ndarray]) -> list[np.ndarray]:
    """evaluate.run_network's computation, on the CPU in float32: each frame's (H, W, 3) output."""
    h, w = frames[0].shape[1:]
    ph = -(-h // UNet.alignment) * UNet.alignment
    pw = -(-w // UNet.alignment) * UNet.alignment
    history, outputs = None, []
    with torch.no_grad():
        for frame, albedo in zip(frames, albedo_full):
            x = torch.zeros(1, frame.shape[0], ph, pw)
            x[0, :, :h, :w] = torch.from_numpy(frame)
            a = torch.zeros(1, 3, SCALE * ph, SCALE * pw)
            a[0, :, :SCALE * h, :SCALE * w] = torch.from_numpy(np.ascontiguousarray(albedo.transpose(2, 0, 1)))
            out, history, _ = model.frame(x, a, history, extent=(h, w))
            outputs.append(out[0, :, :SCALE * h, :SCALE * w].permute(1, 2, 0).numpy())
    return outputs


def fixture(args: argparse.Namespace) -> int:
    if args.out.exists():
        print(f"{args.out} exists", file=sys.stderr)
        return 1
    if (args.network is None) == (not args.random_network):
        print("give --network DIR or --random-network", file=sys.stderr)
        return 1
    manifest = Manifest(args.data)
    seqs = {s.name: s for sp in ("train", "validation") for s in manifest.sequences(sp)}
    if args.sequence not in seqs:
        print(f"{args.sequence} is in neither the train nor the validation split", file=sys.stderr)
        return 1
    seq = seqs[args.sequence]
    axes = view_axes(manifest.scene_file(seq.scene))
    if args.random_network:
        model = random_network(args.seed)
    else:
        model = network_from_export(args.network)
    args.out.mkdir(parents=True)
    if args.random_network:
        export_network(model, args.out / "network", {"epoch": 0, "valid_loss": 0.0, "git_commit": "none",
                                                     "random_seed": args.seed})
    else:
        shutil.copytree(args.network, args.out / "network")

    crop = args.crop  # x, y, w, h at the input's size
    frames, albedos = [], []
    for k, i in enumerate(seq.frames[:args.frames]):
        f = manifest.frames[(seq.name, i)]
        parts = [read_checked(manifest.root, f["input"]["files"][n])
                 for n in ("radiance", "albedo", "normal", "depth", "motion")]
        full = read_checked(manifest.root, f["target"]["files"]["albedo"])
        if crop:
            x, y, w, h = crop
            parts = [p[y:y + h, x:x + w] for p in parts]
            full = full[SCALE * y:SCALE * (y + h), SCALE * x:SCALE * (x + w)]
        for name, image in zip(("radiance", "albedo", "normal", "depth", "motion"), parts):
            write_pfm(args.out / f"frame-{k}-{name}.pfm", image)
        write_pfm(args.out / f"frame-{k}-full_albedo.pfm", full)
        frames.append(input_frame(*parts, axes))
        albedos.append(np.clip(full, 0.0, 1.0))
    for k, out in enumerate(expected_outputs(model, frames, albedos)):
        write_pfm(args.out / f"frame-{k}-expected.pfm", out)
    h, w = frames[0].shape[1:]
    (args.out / "fixture.json").write_text(json.dumps({
        "frames": len(frames), "light": [w, h],
        "axes": {"right": axes[0].tolist(), "up": axes[1].tolist(), "backward": axes[2].tolist()},
        "sequence": seq.name, "data_frames": seq.frames[:args.frames], "crop": crop,
        "made_by": "training/denoise_check.py fixture",
    }, indent=1))
    print(args.out)
    return 0


def network_from_export(directory: Path) -> Denoiser:
    """A network read back from export.py's files alone, as export.py checks it."""
    return read_back(json.loads((directory / "layout.json").read_text()), (directory / "weights.bin").read_bytes())


def _displayed(image: np.ndarray) -> torch.Tensor:
    """(H, W, 3) linear -> (3, H, W) 8-bit display values, as floats 0..255."""
    return displayed(torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None])[0]


def compare(args: argparse.Namespace) -> int:
    meta = json.loads((args.fixture / "fixture.json").read_text())
    rows = []
    for k in range(meta["frames"]):
        expected = read_pfm(args.fixture / f"frame-{k}-expected.pfm")
        gpu = read_pfm(args.gpu / f"frame-{k}-gpu.pfm")
        linear = np.abs(gpu - expected)
        relative = linear / (np.abs(expected) + 1e-3)
        shown = (_displayed(gpu) - _displayed(expected)).abs()
        rows.append({"frame": k, "max_abs_linear": float(linear.max()), "max_relative": float(relative.max()),
                     "mean_relative": float(relative.mean()),
                     "displayed_max_8bit": float(shown.max()), "displayed_mean_8bit": float(shown.mean()),
                     "displayed_pixels_off": float((shown.amax(0) > 0).float().mean())})
    for r in rows:
        print(json.dumps(r))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fixture")
    f.add_argument("--network", type=Path, help="export.py's directory")
    f.add_argument("--random-network", action="store_true")
    f.add_argument("--seed", type=int, default=20261015)
    f.add_argument("--data", type=Path, required=True)
    f.add_argument("--sequence", required=True)
    f.add_argument("--frames", type=int, default=2)
    f.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    f.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--fixture", type=Path, required=True)
    c.add_argument("--gpu", type=Path, required=True)
    args = p.parse_args()
    return fixture(args) if args.command == "fixture" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
