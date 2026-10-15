"""PFM images, as the renderer writes them (src/core/output/pfm.h).

"PF\\n", "W H\\n", "-1.0\\n", then W x H x 3 little-endian float32, rows stored
bottom first. Images here are top first, (H, W, 3) float32, as the renderer's
are (core/contracts/linear_image.h). Anything else is refused by name rather
than read as something it is not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PfmError(ValueError):
    pass


def _header(data: bytes, path: Path) -> tuple[int, int, int]:
    """(width, height, offset of the first float), or PfmError."""
    lines = data[:64].split(b"\n", 3)
    if len(lines) < 4 or lines[0] != b"PF":
        raise PfmError(f"{path}: not a three-channel PFM (magic {lines[0][:4]!r})")
    try:
        width, height = (int(v) for v in lines[1].split())
        scale = float(lines[2])
    except ValueError as e:
        raise PfmError(f"{path}: malformed header") from e
    if scale != -1.0:
        raise PfmError(f"{path}: scale {scale}, only -1 (little-endian, unscaled) is written by the renderer")
    offset = len(lines[0]) + len(lines[1]) + len(lines[2]) + 3
    return width, height, offset


def read_pfm(path: Path | str) -> np.ndarray:
    """The image at `path`, top row first, (H, W, 3) float32."""
    path = Path(path)
    return parse_pfm(path.read_bytes(), path)


def parse_pfm(data: bytes, path: Path | str = "<bytes>") -> np.ndarray:
    """A PFM file's bytes as an image, top row first, (H, W, 3) float32."""
    path = Path(path)
    width, height, offset = _header(data, path)
    expected = width * height * 3 * 4
    if len(data) - offset != expected:
        raise PfmError(f"{path}: {len(data) - offset} bytes of floats, {expected} expected for {width} x {height}")
    image = np.frombuffer(data, dtype="<f4", offset=offset).reshape(height, width, 3)
    return np.ascontiguousarray(image[::-1])


def write_pfm(path: Path | str, image: np.ndarray) -> None:
    """Writes (H, W, 3) float32 `image`, top row first, as the renderer would."""
    image = np.asarray(image, dtype="<f4")
    if image.ndim != 3 or image.shape[2] != 3:
        raise PfmError(f"write_pfm: shape {image.shape}, (H, W, 3) expected")
    height, width, _ = image.shape
    with open(path, "wb") as f:
        f.write(f"PF\n{width} {height}\n-1.0\n".encode())
        f.write(np.ascontiguousarray(image[::-1]).tobytes())
