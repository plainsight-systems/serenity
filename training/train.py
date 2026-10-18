"""Training, and OIDN's learning-rate finder.

    uv run python train.py find-lr --config config-wide.json --out RUN_DIR
    uv run python train.py train   --config config-wide.json --out RUN_DIR

Open Image Denoise's schedule (training/train.py, config.py), where the
design does not say otherwise: Adam; batch 16; a one-cycle learning rate,
cosine, 15% warm-up, from max_lr / 25 up to max_lr and down to max_lr / 25e4
(PyTorch's OneCycleLR with OIDN's div_factor and final_div_factor), stepped
once an epoch as OIDN steps it; float32 (OIDN's default off CUDA). An epoch
here is `crops_per_sequence` random crops of every training sequence (32 x
39 = 1,248 sequences of 8 frames, 78 steps of 16): about the training
frames' area once (a 1920 x 1080 target holds 31.6 crops of 256 x 256).

The validation loss is computed every epoch on OIDN's fixed tiling of the
validation sequences (data.validation_tiles: 28 tiles a sequence, 112 in
all, no augmentation), with the same loss. The checkpoint with the lowest
validation loss is kept as best.pt; the run's every epoch is in log.csv.

Reproducible: one seed (the config's) seeds PyTorch, and each epoch's crops
and augmentations come from numpy's generator seeded by (seed, epoch), so a
resumed run draws what an unbroken one would. The MPS backend's own
reductions are not promised bit-exact by PyTorch; how far two runs from
one seed agree is measured, not assumed (docs/research/
2026-10-15-denoiser-training.md). Each checkpoint records the config, the
data set's manifest SHA-256, the git commit of this code and whether the
training/ folder had uncommitted changes, and the versions.

A run directory is new or a run's own: `train` resumes from last.pt if one
is there, and refuses a directory whose config differs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from data import (CROP, FRAMES, SCALE, Augmentation, Manifest, Split, augment, epoch_crops, validation_tiles)
from loss import sequence_loss
from model import Denoiser, Widths, macs_per_input_pixel, parameter_count

HERE = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def pick_device(asked: str | None, cuda: bool, mps: bool) -> str:
    """The device a run trains on: `asked`, if this machine has it, else CUDA where there is
    one, else the Mac's GPU (MPS). Pure; exits naming what is missing."""
    have = {"cuda": cuda, "mps": mps}
    if asked is not None:
        if not have[asked]:
            raise SystemExit(f"--device {asked}: this machine has no {asked} device")
        return asked
    if cuda or mps:
        return "cuda" if cuda else "mps"
    raise SystemExit("no GPU: training needs CUDA or the Mac's GPU (MPS)")


def hold_on_device(data_bytes: int, free_bytes: int, working_bytes: int) -> bool:
    """Whether the splits are held on the GPU: if they fit in its free memory beside training's own.
    Pure. Where not, they are held in the host's memory and each batch copied over."""
    return data_bytes + working_bytes <= free_bytes


def working_device_bytes(model: Denoiser, batch_size: int, device: torch.device) -> int:
    """Training's own GPU memory, measured: one step (forward, backward, Adam) of a copy of `model`
    on a zero batch of the run's shape. The copy and its optimizer are thrown away, so the run's
    network, its gradients and every random stream are untouched."""
    probe = copy.deepcopy(model)
    optimizer = torch.optim.Adam(probe.parameters())
    inputs = torch.zeros(batch_size, FRAMES, 12, CROP, CROP, device=device)
    targets = torch.zeros(batch_size, FRAMES, 6, SCALE * CROP, SCALE * CROP, device=device)
    targets[:, :, 3:6] = 1.0  # a white albedo
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss, _ = run_batch(probe, inputs, targets)
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        used = torch.cuda.max_memory_reserved(device)
    else:
        torch.mps.synchronize()
        used = torch.mps.driver_allocated_memory()  # the allocator's cache keeps the step's peak
    del probe, optimizer, inputs, targets, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    else:
        torch.mps.empty_cache()
    return used


def free_device_bytes(device: torch.device) -> int:
    """The GPU memory a run may use: CUDA's (and ROCm's) free memory; on the Mac, Metal's recommended
    working set, the unified memory the GPU may take."""
    if device.type == "cuda":
        return torch.cuda.mem_get_info(device)[0]
    return torch.mps.recommended_max_memory() - torch.mps.current_allocated_memory()


def setup(config: dict, need_validation: bool) -> tuple[torch.device, Manifest, Split,
                                                       Split | None, Denoiser]:
    device = torch.device(config["device"])
    torch.manual_seed(config["seed"])
    model = Denoiser(Widths(**config["widths"])).to(device)
    print(f"network: {parameter_count(model):,} parameters, "
          f"{macs_per_input_pixel(model.widths):,.0f} multiply-adds per input pixel", flush=True)
    manifest = Manifest(Path(config["data"]))
    data_bytes = manifest.split_bytes("train") + (manifest.split_bytes("validation") if need_validation else 0)
    working = working_device_bytes(model, config["batch_size"], device)
    free = free_device_bytes(device)
    on_device = hold_on_device(data_bytes, free, working)
    held = device if on_device else torch.device("cpu")
    print(f"data held {'on the GPU' if on_device else 'in host memory, each batch copied to the GPU'}: "
          f"{data_bytes / 2**30:.1f} GiB of data and {working / 2**30:.1f} GiB of training (measured) "
          f"against {free / 2**30:.1f} GiB free", flush=True)
    t0 = time.time()
    train_split = Split(manifest, "train", device, held)
    valid = Split(manifest, "validation", device, held) if need_validation else None
    print(f"data: {len(train_split)} training sequences"
          + (f", {len(valid)} validation" if valid else "") + f", read and checked in {time.time() - t0:.0f} s",
          flush=True)
    return device, manifest, train_split, valid, model


def find_lr(config: dict, out: Path) -> None:
    """OIDN's find_lr.py: Adam, the learning rate raised exponentially each step from lr_min to lr_max,
    the loss smoothed (beta 0.98, bias-corrected), stopped once it passes 4x its best."""
    out.mkdir(parents=True, exist_ok=False)
    _, manifest, train_split, _, model = setup(config, need_validation=False)
    f = config["find_lr"]
    steps = f["steps"]
    optimizer = torch.optim.Adam(model.parameters(), lr=f["lr_min"])
    gamma = (f["lr_max"] / f["lr_min"]) ** (1.0 / (steps - 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    rng = np.random.default_rng([config["seed"], 999_999])
    picks = epoch_crops(len(train_split), math.ceil(steps * config["batch_size"] / len(train_split)), rng)
    rows = [("step", "learning_rate", "smoothed_loss", "loss")]
    avg, best, beta = 0.0, math.inf, 0.98
    bs = config["batch_size"]
    for step in range(steps):
        batch = picks[step * bs:(step + 1) * bs]
        inputs, targets = make_batch(train_split, batch, [Augmentation.draw(rng) for _ in batch])
        optimizer.zero_grad()
        loss, _ = run_batch(model, inputs, targets)
        loss.backward()
        value = loss.item()
        avg = beta * avg + (1 - beta) * value
        smoothed = avg / (1 - beta ** (step + 1))
        lr = scheduler.get_last_lr()[0]
        rows.append((step, lr, smoothed, value))
        print(f"find-lr {step:4d} lr {lr:.3e} loss {value:.5f} smoothed {smoothed:.5f}", flush=True)
        if smoothed < best:
            best = smoothed
        elif smoothed > 4 * best or not math.isfinite(value):
            break
        optimizer.step()
        scheduler.step()
    with open(out / "find_lr.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    (out / "provenance.json").write_text(json.dumps(provenance(config, manifest), indent=1))


def train(config: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    config_file = out / "config.json"
    if config_file.exists() and load_config(config_file) != config:
        raise SystemExit(f"{out} holds a run of another config; name a new directory")
    config_file.write_text(json.dumps(config, indent=1))

    device, manifest, train_split, valid_split, model = setup(config, need_validation=True)
    epochs, bs = config["epochs"], config["batch_size"]
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["max_lr"], total_steps=epochs, pct_start=config["lr_warmup"],
        anneal_strategy="cos", div_factor=config["div_factor"], final_div_factor=config["final_div_factor"])
    meta = provenance(config, manifest)
    start, best = 1, math.inf
    last = out / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best = state["epoch"] + 1, state["best_valid"]
        torch.manual_seed(config["seed"] + start)
        print(f"resumed after epoch {state['epoch']}", flush=True)

    log = out / "log.csv"
    if not log.exists():
        with open(log, "w", newline="") as fh:
            csv.writer(fh).writerow(["epoch", "learning_rate", "train_loss", "train_spatial", "train_temporal",
                                     "valid_loss", "valid_spatial", "valid_temporal", "seconds"])
    for epoch in range(start, epochs + 1):
        t0 = time.time()
        rng = np.random.default_rng([config["seed"], epoch])
        picks = epoch_crops(len(train_split), config["crops_per_sequence"], rng)
        steps = len(picks) // bs
        totals = torch.zeros(3, device=device)
        lr = scheduler.get_last_lr()[0]
        for step in range(steps):
            batch = picks[step * bs:(step + 1) * bs]
            inputs, targets = make_batch(train_split, batch, [Augmentation.draw(rng) for _ in batch])
            optimizer.zero_grad()
            loss, parts = run_batch(model, inputs, targets)
            loss.backward()
            optimizer.step()
            totals += torch.stack((loss.detach(), parts["spatial"], parts["temporal"]))
        scheduler.step()
        train_loss, train_spatial, train_temporal = (totals / steps).tolist()
        if not math.isfinite(train_loss):
            raise SystemExit(f"epoch {epoch}: the training loss is {train_loss}; stopped")
        valid_loss, valid_spatial, valid_temporal = validate(model, valid_split, bs)
        seconds = time.time() - t0
        with open(log, "a", newline="") as fh:
            csv.writer(fh).writerow([epoch, lr, train_loss, train_spatial, train_temporal,
                                     valid_loss, valid_spatial, valid_temporal, round(seconds, 1)])
        improved = valid_loss < best
        if improved:
            best = valid_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "valid_loss": valid_loss,
                        "train_loss": train_loss, **meta}, out / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best_valid": best, **meta}, last)
        print(f"epoch {epoch:4d}/{epochs} lr {lr:.3e} train {train_loss:.5f} valid {valid_loss:.5f} "
              f"gap {valid_loss - train_loss:+.5f}{' best' if improved else ''} ({seconds:.0f} s)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Trains Serenity's denoiser (training/README.md).")
    parser.add_argument("command", choices=["find-lr", "train"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path,
                        help="the data set, if not where the config's path (relative to the config file) puts it")
    parser.add_argument("--device", choices=["cuda", "mps"],
                        help="the GPU to train on; by default CUDA where there is one, else the Mac's (MPS)")
    args = parser.parse_args()
    config = load_config(args.config)
    # The device is the machine's, not the network's: chosen here and recorded with the run.
    config["device"] = pick_device(args.device, torch.cuda.is_available(), torch.backends.mps.is_available())
    print(f"device: {config['device']}", flush=True)
    # The data set's absolute path is recorded with the run; its identity is the manifest's SHA-256.
    data = args.data if args.data else (args.config.resolve().parent / config["data"])
    config["data"] = str(data.resolve())
    if args.command == "find-lr":
        find_lr(config, args.out)
    else:
        train(config, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
