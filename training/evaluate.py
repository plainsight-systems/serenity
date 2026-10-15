"""The test, once, after training: the real marbles, never trained on.

    uv run python evaluate.py --checkpoint RUN/best.pt --data DATA --warmup WARMUP \\
        --references REFS --out RESULTS [--oidn /opt/homebrew/bin/oidnDenoise] [--images DIR]

The design's "Test once" (docs/research/2026-10-14-denoiser-training-design.md):
this reads the test split, and refuses an --out that already holds results.
`--dry-run-on-validation` runs the same code on the validation split
instead (no warm-up frames, no references: each sequence's first frame has
no history), to check the script itself before the test; its numbers are
validation numbers, labeled so in results.json ("split").

For each test sequence, the network runs over 16 frames: the 8 warm-up
frames rendered before it (render_warmup.py; their outputs are only
history and are not scored) and the data set's 8. Frames are 960 x 540,
zero-padded to 960 x 544 for the U-Net, the padding outside every tap and
every history read (model.Denoiser.frame's `extent`).

Scored, per sequence and per region (regions.py: glass and metal, porcelain,
table, whole), in displayed MSE: both images through the renderer's tone map
(tonemap.py: exposure 0, bloom 0.04, 8-bit sRGB, as Milestone 2), the MSE
of the 8-bit values over 255, averaged over the region's pixels and the
three channels:

- against the 4,096-sample targets, every frame, averaged over the 8;
- at t = 8 and t = 120 (the first frames of marbles-t008 and marbles-t120,
  where the 65,536-sample references were rendered), against the
  references; and the targets' own error against them, beside;
- flicker: the mean absolute change of the displayed image (0..1) from one
  frame to the next, over the 7 pairs, for each image kind and the targets.

Compared with: the ReSTIR PT input upscaled bilinearly (the raw input, no
denoising), and, if --oidn is given, Open Image Denoise 2.5 (its RT filter,
"high" quality, HDR color with albedo and normal, run at 960 x 540 on each
frame alone, then upscaled bilinearly): a single-frame quality yardstick.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import (INPUT_SIZE, SCALE, Manifest, _read_checked, file_sha256, input_frame)
from model import Denoiser, UNet, Widths
from pfm import read_pfm, write_pfm
from regions import REGIONS, masks, view_axes
from tonemap import displayed

REFERENCE_FRAMES = {"marbles-t008": ("t8", "marbles-t8-1920x1080-64x1024.pfm"),
                    "marbles-t120": ("t120", "marbles-t120-1920x1080-64x1024.pfm")}
IMAGES = [("marbles-t004", 240, "opening-t4"), ("marbles-t030", 1800, "t30"), ("marbles-t120", 7200, "t120")]
VIDEO_SEQUENCE = "marbles-t120"


def load_network(checkpoint: Path, device: torch.device) -> tuple[Denoiser, dict]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Denoiser(Widths(**state["config"]["widths"]))
    model.load_state_dict(state["model"])
    model.eval().to(device)
    return model, state


def warmup_inputs(warmup: Path, record: dict, sequence: str, axes: np.ndarray) -> list[np.ndarray]:
    entry = record["sequences"][sequence]
    frames = []
    for i in entry["frames"]:
        parts = []
        for suffix in ("", "-albedo", "-normal", "-depth", "-motion"):
            name = f"frame-{i:06d}{suffix}.pfm"
            parts.append(_read_checked(warmup / sequence, {"path": name, "sha256": entry["files"][name]}))
        frames.append(input_frame(*parts, axes))
    return frames


def run_network(model: Denoiser, frames: list[np.ndarray], albedo_full: list[torch.Tensor],
                device: torch.device) -> list[torch.Tensor]:
    """Each frame's (3, 1080, 1920) output, in order, the network's own history carried."""
    h, w = INPUT_SIZE
    ph = -(-h // UNet.alignment) * UNet.alignment
    pw = -(-w // UNet.alignment) * UNet.alignment
    history, outputs = None, []
    with torch.no_grad():
        for frame, albedo in zip(frames, albedo_full):
            x = torch.zeros(1, frame.shape[0], ph, pw, device=device)
            x[0, :, :h, :w] = torch.from_numpy(frame).to(device)
            a = torch.zeros(1, 3, SCALE * ph, SCALE * pw, device=device)
            a[0, :, :SCALE * h, :SCALE * w] = albedo.to(device)
            out, history, _ = model.frame(x, a, history, extent=(h, w))
            outputs.append(out[0, :, :SCALE * h, :SCALE * w].cpu())
    return outputs


def bilinear(image_hwc: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(image_hwc.transpose(2, 0, 1)))[None]
    return F.interpolate(x, scale_factor=SCALE, mode="bilinear", align_corners=False)[0]


def oidn(binary: Path, radiance: np.ndarray, albedo: np.ndarray, normal: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        write_pfm(d / "c.pfm", radiance)
        write_pfm(d / "a.pfm", albedo)
        write_pfm(d / "n.pfm", normal)
        subprocess.run([str(binary), "-f", "RT", "-q", "high", "--hdr", str(d / "c.pfm"), "--alb", str(d / "a.pfm"),
                        "--nrm", str(d / "n.pfm"), "-o", str(d / "o.pfm")], check=True, capture_output=True)
        return read_pfm(d / "o.pfm")


def to_display(radiance: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(3, H, W) linear -> (3, H, W) displayed, 0..1 (8-bit values over 255), on the CPU."""
    return (displayed(radiance[None].to(device))[0] / 255.0).cpu()


def region_mse(a: torch.Tensor, b: torch.Tensor, region_masks: dict) -> dict:
    sq = ((a - b) ** 2).mean(dim=0)
    return {r: float(sq[m].mean()) for r, m in region_masks.items()}


def region_flicker(frames: list[torch.Tensor], region_masks: dict) -> dict:
    change = torch.stack([(frames[i] - frames[i - 1]).abs().mean(dim=0) for i in range(1, len(frames))]).mean(0)
    return {r: float(change[m].mean()) for r, m in region_masks.items()}


def save_side_by_side(path: Path, panels: list[torch.Tensor]) -> None:
    image = torch.cat(panels, dim=2).clamp(0, 1)
    Image.fromarray((image.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)).save(path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--warmup", type=Path, help="the test sequences' warm-up frames (render_warmup.py)")
    p.add_argument("--references", type=Path, help="the 65,536-sample references")
    p.add_argument("--dry-run-on-validation", action="store_true",
                   help="run this script on the validation split instead, without warm-up or references: "
                        "a check of the script before the test, not a result")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--images", type=Path, help="where the side-by-side PNGs and the MP4 go")
    p.add_argument("--oidn", type=Path, help="oidnDenoise, to compare with Open Image Denoise")
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    split = "validation" if args.dry_run_on_validation else "test"
    if split == "test" and (args.warmup is None or args.references is None):
        print("the test needs --warmup and --references", file=sys.stderr)
        return 1
    if args.out.exists() and any(args.out.iterdir()):
        print(f"{args.out} already holds results: the test is run once", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, state = load_network(args.checkpoint, device)
    manifest = Manifest(args.data)
    if manifest.sha256 != state["manifest_sha256"]:
        print("the checkpoint was trained on another data set", file=sys.stderr)
        return 1
    warmup_record = json.loads((args.warmup / "warmup.json").read_text()) if split == "test" else None
    kinds = ["network", "input_bilinear"] + (["oidn_bilinear"] if args.oidn else [])
    results: dict = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": file_sha256(args.checkpoint),
                     "epoch": state["epoch"], "valid_loss": state["valid_loss"], "git_commit": state["git_commit"],
                     "manifest_sha256": manifest.sha256, "split": split, "sequences": {}}
    if args.images:
        args.images.mkdir(parents=True, exist_ok=True)

    for seq in manifest.sequences(split):
        scene = manifest.scene_file(seq.scene)
        axes = view_axes(scene)
        region_masks = {k: torch.from_numpy(v) for k, v in
                        masks(scene, INPUT_SIZE[1] * SCALE, INPUT_SIZE[0] * SCALE).items()}
        inputs, targets = manifest.load_sequence(seq)
        warm = warmup_inputs(args.warmup, warmup_record, seq.name, axes) if warmup_record else []
        # Warm-up frames have no full-size albedo; their radiance output is not used, only their
        # demodulated output (the history), which the albedo does not enter.
        warm_albedo = [bilinear(np.ascontiguousarray(f[3:6].transpose(1, 2, 0))) for f in warm]
        target_rad = [torch.from_numpy(t[0:3]) for t in targets]
        target_alb = [torch.from_numpy(t[3:6]) for t in targets]
        outputs = run_network(model, warm + list(inputs), warm_albedo + target_alb, device)[len(warm):]

        raw = {k: [] for k in ("radiance", "albedo", "normal")}
        for i in seq.frames:
            files = manifest.frames[(seq.name, i)]["input"]["files"]
            for k in raw:
                raw[k].append(_read_checked(manifest.root, files[k]))
        images = {"network": outputs, "input_bilinear": [bilinear(r) for r in raw["radiance"]]}
        if args.oidn:
            images["oidn_bilinear"] = [bilinear(oidn(args.oidn, r, a, n))
                                       for r, a, n in zip(raw["radiance"], raw["albedo"], raw["normal"])]
        shown = {k: [to_display(x, device) for x in v] for k, v in images.items()}
        shown_target = [to_display(t, device) for t in target_rad]

        entry: dict = {"frames": seq.frames, "warmup_frames": len(warm), "vs_targets": {}, "flicker": {},
                       "region_pixels": {r: int(m.sum()) for r, m in region_masks.items()}}
        for k in kinds:
            per_frame = [region_mse(a, b, region_masks) for a, b in zip(shown[k], shown_target)]
            entry["vs_targets"][k] = {r: float(np.mean([f[r] for f in per_frame])) for r in REGIONS}
            entry["vs_targets"][k + "_per_frame"] = per_frame
            entry["flicker"][k] = region_flicker(shown[k], region_masks)
        entry["flicker"]["targets"] = region_flicker(shown_target, region_masks)

        if split == "test" and seq.name in REFERENCE_FRAMES:
            tag, ref_name = REFERENCE_FRAMES[seq.name]
            reference = torch.from_numpy(read_pfm(args.references / ref_name).transpose(2, 0, 1).copy())
            ref_shown = to_display(reference, device)
            entry["vs_reference"] = {"frame": seq.frames[0], "reference": ref_name}
            for k in kinds:
                entry["vs_reference"][k] = region_mse(shown[k][0], ref_shown, region_masks)
            entry["vs_reference"]["targets"] = region_mse(shown_target[0], ref_shown, region_masks)

        if args.images:
            for name, frame, label in IMAGES:
                if name == seq.name:
                    j = seq.frames.index(frame)
                    save_side_by_side(args.images / f"{label}-input-denoised-target.png",
                                      [shown["input_bilinear"][j], shown["network"][j], shown_target[j]])
            if seq.name == VIDEO_SEQUENCE:
                frames_dir = args.out / "video-frames"
                frames_dir.mkdir()
                for j in range(len(seq.frames)):
                    panels = [F.interpolate(x[j][None], size=(720, 1280), mode="area")[0]
                              for x in (shown["input_bilinear"], shown["network"], shown_target)]
                    save_side_by_side(frames_dir / f"{j:02d}.png", panels)
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "8", "-i",
                                str(frames_dir / "%02d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                                str(args.images / f"{seq.name}-input-denoised-target-8fps.mp4")], check=True)
        results["sequences"][seq.name] = entry
        summary = {k: round(entry["vs_targets"][k]["whole"], 6) for k in kinds}
        print(f"{seq.name}: displayed MSE vs targets (whole) {summary}", flush=True)

    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
