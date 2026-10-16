"""The denoiser-v1 data set, read from its manifest, as the network sees it.

Contract
--------
- Reads only the splits it is asked for. Training reads "train" and
  "validation"; the test split is read only by evaluate.py, after training
  (docs/research/2026-10-14-denoiser-training-design.md, "Not overtraining").
- Every file is checked against the manifest's SHA-256 as it is read; a
  mismatch is refused, not trained on.
- The data set is never written.

Per frame, at the input's 960 x 540 (INPUT_CHANNELS, 12 channels):

  demod   0:3   noisy radiance / albedo factor, linear (below)
  albedo  3:6   the input albedo, clipped to [0, 1] as OIDN clips it
  normal  6:9   the shading normal in view space: (right, up, backward),
                each the world normal's component along the camera's axis
                (regions.view_axes); the sky's (0, 0, 0) stays 0
  depth   9     log(1 + z / median z), the median over the frame's
                non-sky pixels (z > 0); the sky's 0 stays 0
  motion  10:12 NRD's IN_MV dx, dy in input pixels, "old = new + MV"
                (docs/research/2026-10-14-denoiser-features.md); dz unused

and at the target's 1920 x 1080 (TARGET_CHANNELS, 6 channels): the
4,096-sample radiance (0:3) and the full-size albedo (3:6, clipped).

Demodulation. The albedo factor is NRD's (Shaders/NRD.hlsli,
NRD_MaterialFactors: lerp(NRD_MATERIAL_FACTOR_MIN_SCALE = 0.02, 1, albedo)),
per channel:

    factor(a) = 0.02 + 0.98 a,    demod = radiance / factor(a_input)

so a black albedo divides by 0.02, never by 0; the output is multiplied by
factor(a_full) of the full-size albedo, the same rule (model.py). OIDN takes
albedo only as a guide and does not demodulate; NRD's is the production rule
for dividing by it.

Why view space for the normal. The renderer writes world-space normals;
the augmentation mirrors images (below), and mirroring an image is
mirroring the scene across a plane of the camera, which negates the
normal's component along the camera's right (or up) axis. In view space that
is one channel's sign, exact; in world space it is not an axis at all. Phase
4 does the same 3 x 3 transform with the camera's axes (queued).

Augmentation (training only), OIDN's (training/dataset.py, TrainingDataset)
for a sequence of 8 frames, one draw for all 8 so the sequence stays a
sequence: a random 128 x 128 input crop and the aligned 256 x 256 target
crop (same place, twice the coordinates); a horizontal flip, a vertical
flip and a transpose, each with probability 1/2, each carried into the
vectors (horizontal: normal right and motion dx negated; vertical: normal up
and dy negated; transpose: (dx, dy) swapped and (right, up) becoming (-up,
-right), since image rows run down and view up runs up); a random
permutation of the color channels, the same for the input's radiance and
albedo and the target's radiance and albedo; a random permutation of the
normal's three axes, after the flips; and with probability 0.01 the input
and target radiance zeroed ("prevents ghosting artifacts when the main
feature is entirely black", OIDN), which here is the dark opening. Not
taken from OIDN: its random zero padding of 10% of the crops, there to
match images padded to the network's alignment; here the filter never
reads outside the image (model.py masks its taps), so there is nothing for
it to learn.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch

from pfm import parse_pfm
from regions import view_axes

INPUT_SIZE = (540, 960)  # (H, W)
TARGET_SIZE = (1080, 1920)
SCALE = 2
FRAMES = 8
INPUT_CHANNELS = 12
TARGET_CHANNELS = 6
ALBEDO_FLOOR = 0.02  # NRD_MATERIAL_FACTOR_MIN_SCALE
CROP = 128  # input pixels; the target's is 256

Image = TypeVar("Image", np.ndarray, torch.Tensor)


def albedo_factor(albedo: Image) -> Image:
    """NRD's demodulation factor, per channel: 0.02 + 0.98 albedo (albedo in [0, 1])."""
    return ALBEDO_FLOOR + (1.0 - ALBEDO_FLOOR) * albedo


class ChecksumError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def read_checked(root: Path, entry: dict) -> np.ndarray:
    """The PFM `entry` names, refused if its bytes are not the manifest's."""
    path = root / entry["path"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ChecksumError(f"{path}: SHA-256 differs from the manifest")
    return parse_pfm(data, path)


def input_frame(radiance: np.ndarray, albedo: np.ndarray, normal: np.ndarray, depth: np.ndarray,
                motion: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """One frame's 12 input channels, (12, H, W) float32, from its five PFMs (H, W, 3) and the view axes."""
    albedo = np.clip(albedo, 0.0, 1.0)
    demod = radiance / albedo_factor(albedo)
    normal_view = normal @ axes.T  # (H, W, 3): right, up, backward
    z = depth[..., 0]
    surface = z > 0
    median = float(np.median(z[surface])) if surface.any() else 1.0
    depth_n = np.where(surface, np.log1p(z / median), 0.0)
    out = np.concatenate([demod, albedo, normal_view, depth_n[..., None], motion[..., :2]], axis=2)
    return np.ascontiguousarray(out.transpose(2, 0, 1), dtype=np.float32)


def target_frame(radiance: np.ndarray, albedo: np.ndarray) -> np.ndarray:
    """One frame's 6 target channels, (6, H, W) float32."""
    out = np.concatenate([radiance, np.clip(albedo, 0.0, 1.0)], axis=2)
    return np.ascontiguousarray(out.transpose(2, 0, 1), dtype=np.float32)


@dataclass
class Sequence:
    name: str
    split: str
    frames: list[int]
    scene: str


class Manifest:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "manifest.json"
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.data = json.loads(raw)
        self.frames = {(f["sequence"], f["frame"]): f for f in self.data["frames"]}

    def sequences(self, split: str) -> list[Sequence]:
        names = self.data["splits"][split]
        plan = {s["sequence"]: s for s in self.data["plan"]["sequences"]}
        return [Sequence(n, split, plan[n]["frames"], plan[n]["scene"]) for n in names]

    def scene_file(self, scene: str) -> Path:
        (rel, digest), = self.data["scenes"][scene]["files"].items()
        path = self.root / rel
        if file_sha256(path) != digest:
            raise ChecksumError(f"{path}: SHA-256 differs from the manifest")
        return path

    def load_sequence(self, seq: Sequence, with_target: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        """(8, 12, 540, 960) inputs and (8, 6, 1080, 1920) targets, checked against the manifest."""
        axes = view_axes(self.scene_file(seq.scene))
        inputs, targets = [], []
        for i in seq.frames:
            f = self.frames[(seq.name, i)]
            fi = f["input"]["files"]
            inputs.append(input_frame(*(read_checked(self.root, fi[k])
                                        for k in ("radiance", "albedo", "normal", "depth", "motion")), axes))
            if with_target:
                ft = f["target"]["files"]
                targets.append(target_frame(read_checked(self.root, ft["radiance"]),
                                            read_checked(self.root, ft["albedo"])))
        return np.stack(inputs), (np.stack(targets) if with_target else None)


class SplitOnDevice:
    """A split's sequences, whole, on the device: inputs (S, 8, 12, h, w), targets (S, 8, 6, H, W)."""

    def __init__(self, manifest: Manifest, split: str, device: torch.device, workers: int = 8):
        if split == "test":
            raise ValueError("the test split is read by evaluate.py alone, after training")
        self.sequences = manifest.sequences(split)
        with ThreadPoolExecutor(workers) as pool:
            loaded = list(pool.map(manifest.load_sequence, self.sequences))
        self.inputs = torch.stack([torch.from_numpy(i) for i, _ in loaded]).to(device)
        self.targets = torch.stack([torch.from_numpy(t) for _, t in loaded]).to(device)
        del loaded

    def __len__(self) -> int:
        return len(self.sequences)

    def crop(self, s: int, oy: int, ox: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequence s's (8, 12, 128, 128) input crop at (oy, ox) and its (8, 6, 256, 256) target crop."""
        inp = self.inputs[s, :, :, oy:oy + CROP, ox:ox + CROP]
        tgt = self.targets[s, :, :, SCALE * oy:SCALE * (oy + CROP), SCALE * ox:SCALE * (ox + CROP)]
        return inp, tgt


@dataclass
class Augmentation:
    hflip: bool
    vflip: bool
    transpose: bool
    colors: tuple[int, int, int]
    normal_axes: tuple[int, int, int]
    zero: bool

    @staticmethod
    def draw(rng: np.random.Generator) -> "Augmentation":
        return Augmentation(hflip=bool(rng.random() < 0.5), vflip=bool(rng.random() < 0.5),
                            transpose=bool(rng.random() < 0.5), colors=tuple(int(c) for c in rng.permutation(3)),
                            normal_axes=tuple(int(c) for c in rng.permutation(3)), zero=bool(rng.random() < 0.01))

    @staticmethod
    def none() -> "Augmentation":
        return Augmentation(False, False, False, (0, 1, 2), (0, 1, 2), False)


def augment(inp: torch.Tensor, tgt: torch.Tensor, a: Augmentation) -> tuple[torch.Tensor, torch.Tensor]:
    """`a` applied to one sequence's (T, 12, h, w) input and (T, 6, H, W) target; new tensors."""
    inp = inp.clone()
    tgt = tgt.clone()
    if a.hflip:
        inp, tgt = inp.flip(-1), tgt.flip(-1)
        inp[:, 6] = -inp[:, 6]    # normal: right
        inp[:, 10] = -inp[:, 10]  # motion: dx
    if a.vflip:
        inp, tgt = inp.flip(-2), tgt.flip(-2)
        inp[:, 7] = -inp[:, 7]    # normal: up
        inp[:, 11] = -inp[:, 11]  # motion: dy
    if a.transpose:
        inp, tgt = inp.transpose(-1, -2), tgt.transpose(-1, -2)
        right, up = inp[:, 6].clone(), inp[:, 7].clone()
        inp[:, 6], inp[:, 7] = -up, -right
        inp[:, [10, 11]] = inp[:, [11, 10]]
    c = list(a.colors)
    inp[:, 0:3] = inp[:, c]
    inp[:, 3:6] = inp[:, [3 + i for i in c]]
    tgt[:, 0:3] = tgt[:, c]
    tgt[:, 3:6] = tgt[:, [3 + i for i in c]]
    inp[:, 6:9] = inp[:, [6 + i for i in a.normal_axes]]
    if a.zero:
        inp[:, 0:3] = 0.0
        tgt[:, 0:3] = 0.0
    return inp.contiguous(), tgt.contiguous()


def epoch_crops(num_sequences: int, crops_per_sequence: int, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    """One epoch's (sequence, oy, ox) draws: every sequence crops_per_sequence times, shuffled."""
    picks = np.repeat(np.arange(num_sequences), crops_per_sequence)
    rng.shuffle(picks)
    oy = rng.integers(0, INPUT_SIZE[0] - CROP + 1, size=len(picks))
    ox = rng.integers(0, INPUT_SIZE[1] - CROP + 1, size=len(picks))
    return [(int(s), int(y), int(x)) for s, y, x in zip(picks, oy, ox)]


def validation_tiles(num_sequences: int) -> list[tuple[int, int, int]]:
    """OIDN's validation tiling (ValidationDataset): whole tiles, centered, no augmentation. 7 x 4 a frame."""
    ny, nx = INPUT_SIZE[0] // CROP, INPUT_SIZE[1] // CROP
    sy, sx = (INPUT_SIZE[0] % CROP) // 2, (INPUT_SIZE[1] % CROP) // 2
    return [(s, sy + y * CROP, sx + x * CROP) for s in range(num_sequences) for y in range(ny) for x in range(nx)]
