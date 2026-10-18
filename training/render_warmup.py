"""Warm-up frames for the test sequences: the 8 input frames before each.

Why: the network is recurrent, and a test sequence's first frame is where
the 65,536-sample references sit (t = 8 and t = 120 are frames 480 and 7200,
each the first of its sequence). Scored there with no history, the network
would be judged on a cold start the renderer never shows after its first
frame. So each test sequence is run again from 8 frames earlier, 16 frames
in all, and the evaluation (evaluate.py) feeds the network the 8 before it
first, scoring only the data set's 8.

Inputs only (ReSTIR PT + MNEE and its features at 960 x 540, as the data
set's), no targets: nothing here is trained on or scored. Rendered by the
data set's own renderer binary, checked by its SHA-256 against the
manifest's, from the data set's copy of the scene, checked likewise. The
renderer replays frames 0 .. f - 1 before frame f (core/frame/history.h), so
the 8 frames the run shares with the data set must match its recorded
checksums; they are checked, and the run refused if any differs beyond
RADIANCE_TOLERANCE. Not every byte: ReSTIR PT's radiance is not
bit-reproducible on the GPU (found here, 2026-10-15: two reruns of
marbles-t006 with the data set's own arguments each differ from it, and
from each other, in 1 or 2 pixels of some frames, by at most 6e-8, float
rounding); the features are, and must match exactly. The data set itself
is only read.

    uv run python render_warmup.py --data DIR --checkout SERENITY_CHECKOUT --out WARMUP_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from pfm import read_pfm

INPUT_GRAPH = "graphs/restir_pt_features.toml"
WARMUP = 8
# What a shared frame's radiance may differ by: float rounding in a few
# pixels, not a different frame (which differs in most pixels).
RADIANCE_TOLERANCE = {"pixels": 16, "absolute": 1e-5}
FEATURES = ("albedo", "normal", "depth", "motion")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="the denoiser-v1 data set (read only)")
    parser.add_argument("--checkout", type=Path, required=True,
                        help="a Serenity checkout whose build/native-release/serenity-headless is the data set's binary")
    parser.add_argument("--out", type=Path, required=True, help="where the warm-up frames go (new or empty)")
    args = parser.parse_args()
    # Absolute, as the renderer runs from the checkout: a relative path would be read from there.
    args.data, args.checkout, args.out = args.data.resolve(), args.checkout.resolve(), args.out.resolve()

    manifest = json.loads((args.data / "manifest.json").read_text())
    renderer = args.checkout / "build/native-release/serenity-headless"
    if sha256(renderer) != manifest["renderer"]["binary_sha256"]:
        print(f"{renderer} is not the data set's renderer (SHA-256 differs)", file=sys.stderr)
        return 1

    sequences = [s for s in manifest["plan"]["sequences"] if s["split"] == "test"]
    frames_by_index = {(f["sequence"], f["frame"]): f for f in manifest["frames"]}
    if args.out.exists() and any(args.out.iterdir()):
        print(f"--out {args.out} is not empty", file=sys.stderr)
        return 1
    record: dict = {"renderer_sha256": manifest["renderer"]["binary_sha256"], "graph": INPUT_GRAPH,
                    "warmup": WARMUP, "sequences": {}}
    for s in sequences:
        scene_rel, scene_hash = next(iter(manifest["scenes"][s["scene"]]["files"].items()))
        scene = args.data / scene_rel
        if sha256(scene) != scene_hash:
            print(f"{scene}: checksum differs from the manifest", file=sys.stderr)
            return 1
        first = s["frames"][0] - WARMUP
        out = args.out / s["sequence"]
        stage = args.out / ".staging"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        cmd = [str(renderer), "--graph", INPUT_GRAPH, "--scene", str(scene), "--first", str(first),
               "--frames", str(WARMUP + len(s["frames"])), "--step", repr(manifest["plan"]["step"]),
               "--size", "960x540", "--samples", "1", "--format", "pfm", "--write", "all", "--features",
               "--out", str(stage)]
        print(" ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=args.checkout, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return 1
        # The shared frames: features byte for byte; radiance byte for byte
        # or within float rounding (see RADIANCE_TOLERANCE).
        shared: dict = {}
        for i in s["frames"]:
            files = frames_by_index[(s["sequence"], i)]["input"]["files"]
            for kind, entry in files.items():
                name = Path(entry["path"]).name
                if sha256(stage / name) == entry["sha256"]:
                    continue
                if kind != "radiance":
                    print(f"{s['sequence']} frame {i} {kind}: differs from the data set; refused", file=sys.stderr)
                    return 1
                diff = np.abs(read_pfm(stage / name) - read_pfm(args.data / entry["path"]))
                pixels = int((diff.max(axis=2) > 0).sum())
                largest = float(diff.max())
                if pixels > RADIANCE_TOLERANCE["pixels"] or largest > RADIANCE_TOLERANCE["absolute"]:
                    print(f"{s['sequence']} frame {i} radiance: {pixels} pixels differ, by up to {largest}; refused",
                          file=sys.stderr)
                    return 1
                shared[name] = {"pixels_differing": pixels, "largest_difference": largest}
        out.mkdir(parents=True)
        kept = {}
        for i in range(first, first + WARMUP):
            for suffix in ("",) + tuple(f"-{k}" for k in FEATURES):
                name = f"frame-{i:06d}{suffix}.pfm"
                shutil.move(stage / name, out / name)
                kept[name] = sha256(out / name)
        shutil.rmtree(stage)
        record["sequences"][s["sequence"]] = {"first": first, "frames": list(range(first, first + WARMUP)),
                                              "shared_radiance_not_bitwise": shared, "files": kept}
        print(f"{s['sequence']}: frames {first}..{first + WARMUP - 1} kept; shared frames match "
              f"(radiance within rounding in {len(shared)} files: {shared})", flush=True)
    (args.out / "warmup.json").write_text(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
