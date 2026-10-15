"""The trained weights, for Phase 4's Metal kernels: a plain binary and its layout.

    uv run python export.py --checkpoint RUN/best.pt --out MODEL_DIR

Writes, into a new MODEL_DIR (outside the repository, beside the data):

  weights.bin   every parameter as little-endian float32, one after another,
                each in PyTorch's own order (a convolution's weight is
                [out][in][ky][kx], its bias [out]);
  layout.json   each tensor's name, shape, byte offset and count; the
                network's structure, layer by layer; what each input channel
                is and how it is made; what each output score is; the
                constants the pre- and post-processing use; and where the
                weights came from (checkpoint, epoch, validation loss, git
                commit, data set manifest);
  SHA256SUMS    the SHA-256 of weights.bin and layout.json, in shasum's format.

Then reads weights.bin back by layout.json alone into a fresh network and
checks it computes what the checkpoint computes, to the bit, on a random
frame; the export is refused otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from data import ALBEDO_FLOOR, INPUT_CHANNELS, INPUT_SIZE, SCALE
from evaluate import load_network
from loss import HDR_Y_MAX, PU_A, PU_B, PU_C, PU_D, PU_E, PU_F, PU_G, PU_NORM_SCALE, PU_Y0, PU_Y1
from model import MASKED, NETWORK_INPUTS, SCORES, SPACINGS, SUBPIXELS, TAPS, Denoiser, Widths


def structure(w: Widths) -> list[dict]:
    e, bn, d = w.enc, w.bottleneck, w.dec
    layers = []
    for i in range(4):
        layers.append({"op": "conv3x3_relu", "weights": f"unet.enc.{i}", "level": i,
                       "in": NETWORK_INPUTS if i == 0 else e[i - 1], "out": e[i], "keep_as_skip": i})
        layers.append({"op": "maxpool2x2", "level": i})
    layers.append({"op": "conv3x3_relu", "weights": "unet.bottleneck_a", "level": 4, "in": e[3], "out": bn})
    layers.append({"op": "conv3x3_relu", "weights": "unet.bottleneck_b", "level": 4, "in": bn, "out": bn})
    prev = bn
    for i in range(4):
        level = 3 - i
        layers.append({"op": "upsample_nearest2x", "level": level})
        layers.append({"op": "concat", "order": ["upsampled", f"skip {level}"], "channels": prev + e[level]})
        layers.append({"op": "conv3x3_relu", "weights": f"unet.dec.{i}", "level": level,
                       "in": prev + e[level], "out": d[i]})
        prev = d[i]
    layers.append({"op": "conv3x3_relu", "weights": "unet.dec_final", "level": 0, "in": d[3], "out": d[3]})
    layers.append({"op": "conv1x1", "weights": "unet.head", "level": 0, "in": d[3], "out": SCORES * SUBPIXELS,
                   "no_relu": True})
    return layers


def layout(model: Denoiser, tensors: list[dict], state: dict) -> dict:
    return {
        "format": "serenity-denoiser-weights/1",
        "dtype": "float32", "byte_order": "little", "tensor_order": "PyTorch: conv weight [out][in][ky][kx], bias [out]",
        "tensors": tensors,
        "padding": "3x3 convolutions zero-pad by 1; levels are 1, 1/2, 1/4, 1/8, 1/16 of the input size each way; "
                   "the input is zero-padded to a multiple of 16 (960 x 540 -> 960 x 544)",
        "structure": structure(model.widths),
        "input_size": [INPUT_SIZE[1], INPUT_SIZE[0]], "output_scale": SCALE,
        "inputs": {
            "channels": NETWORK_INPUTS,
            "order": [
                {"channels": [0, 3], "what": "pu(demod)", "demod": "radiance / (floor + (1 - floor) clamp(albedo, 0, 1))"},
                {"channels": [3, 6], "what": "albedo, clamped to [0, 1]"},
                {"channels": [6, 9], "what": "shading normal in view space (right, up, backward): dot(n, axis) for the "
                                             "camera's unit right, up and -forward axes; sky (0, 0, 0)"},
                {"channels": [9, 10], "what": "log(1 + z / median z over the frame's pixels with z > 0); 0 where z = 0"},
                {"channels": [10, 12], "what": "motion dx, dy in input pixels, NRD IN_MV (old = new + MV)"},
                {"channels": [12, 24], "what": "pixel_unshuffle(pu(moved history), 2): channel 12 + c * 4 + dy * 2 + dx "
                                               "is color c of output pixel (2y + dy, 2x + dx)"},
                {"channels": [24, 28], "what": "pixel_unshuffle(history validity, 2), same order"},
            ],
            "albedo_floor": ALBEDO_FLOOR,
            "pu": {"y0": PU_Y0, "y1": PU_Y1, "a": PU_A, "b": PU_B, "c": PU_C, "d": PU_D, "e": PU_E, "f": PU_F,
                   "g": PU_G, "norm_scale": PU_NORM_SCALE, "hdr_y_max": HDR_Y_MAX,
                   "formula": "y <= y0: a y; y <= y1: b y^c + d; else: e log(y + f) + g; times norm_scale; y clamped >= 0"},
        },
        "scores": {
            "channels": SCORES * SUBPIXELS,
            "order": "head channel k * 4 + dy * 2 + dx is score k of output pixel (2y + dy, 2x + dx)",
            "k": {f"0..{TAPS * len(SPACINGS) - 1}": "taps: k = spacing_index * 9 + (j + 1) * 3 + (i + 1) for the tap at "
                                                   "input pixel (x + s i, y + s j), s in spacings, i, j in -1, 0, 1",
                  f"{TAPS * len(SPACINGS)}..{TAPS * len(SPACINGS) + 2}": "one weight per spacing",
                  f"{SCORES - 1}": "history blend"},
            "spacings": list(SPACINGS),
            "masked_tap_score": MASKED,
        },
        "output": [
            "per spacing: softmax over its 9 tap scores (a tap outside the image scored as masked_tap_score); "
            "filtered_s = sum of weight x demod at the tap",
            "filtered = sum over spacings of softmax(spacing scores)_s x filtered_s",
            "history: last frame's demod_out read bilinearly (clamped to edge) at the output pixel's center + 2 MV of "
            "input pixel (x / 2, y / 2); valid = that point inside the frame; on the first frame valid = 0, history 0",
            "alpha = sigmoid(history score) x valid; demod_out = alpha history + (1 - alpha) filtered (kept for the next frame)",
            "radiance = demod_out x (floor + (1 - floor) clamp(full-size albedo, 0, 1))",
        ],
        "provenance": {k: state[k] for k in ("epoch", "valid_loss", "train_loss", "git_commit", "manifest_sha256",
                                             "dataset", "torch") if k in state},
        "config": state["config"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        print(f"{args.out} is not empty", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    model, state = load_network(args.checkpoint, torch.device("cpu"))

    tensors, blobs, offset = [], [], 0
    for name, t in model.state_dict().items():
        a = t.detach().cpu().numpy().astype("<f4")
        tensors.append({"name": name, "shape": list(a.shape), "offset": offset, "count": int(a.size)})
        blobs.append(a.tobytes())
        offset += a.nbytes
    weights = b"".join(blobs)
    (args.out / "weights.bin").write_bytes(weights)
    doc = layout(model, tensors, state)
    doc["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    (args.out / "layout.json").write_text(json.dumps(doc, indent=1))

    # Read back by the layout alone; the same output, bit for bit.
    fresh = Denoiser(Widths(**doc["config"]["widths"]))
    loaded = {}
    raw = (args.out / "weights.bin").read_bytes()
    for t in doc["tensors"]:
        a = np.frombuffer(raw, dtype="<f4", count=t["count"], offset=t["offset"]).reshape(t["shape"])
        loaded[t["name"]] = torch.from_numpy(a.copy())
    fresh.load_state_dict(loaded)
    fresh.eval()
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, INPUT_CHANNELS, 64, 64, generator=g)
    a = torch.rand(1, 3, 128, 128, generator=g)
    with torch.no_grad():
        if not torch.equal(model.frame(x, a, None)[0], fresh.frame(x, a, None)[0]):
            print("weights.bin read back by layout.json computes something else; refused", file=sys.stderr)
            return 1
    sums = "".join(f"{hashlib.sha256((args.out / n).read_bytes()).hexdigest()}  {n}\n"
                   for n in ("weights.bin", "layout.json"))
    (args.out / "SHA256SUMS").write_text(sums)
    print(sums, end="")
    print(f"{len(weights):,} bytes, {sum(t['count'] for t in tensors):,} parameters; read back and checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
