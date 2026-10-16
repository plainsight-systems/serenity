"""Markdown tables from evaluate.py's results.json (one or more networks).

    uv run --locked python report.py NAME=RESULTS.json [NAME=RESULTS.json ...]

Displayed MSE (8-bit sRGB after the renderer's tone map, over 255, squared,
averaged over the region's pixels and channels); flicker, the mean absolute
displayed change from frame to frame.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REGIONS = (("whole", "whole"), ("glass_metal", "glass+metal"), ("porcelain", "porcelain"), ("table", "table"))


def fmt(v: float) -> str:
    """Five decimals under 0.01, four above."""
    return f"{v:.5f}" if v < 0.01 else f"{v:.4f}"


def main() -> int:
    if len(sys.argv) < 2 or not all("=" in arg for arg in sys.argv[1:]):
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    runs: dict[str, dict] = {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        runs[name] = json.loads(Path(path).read_text())
    first = next(iter(runs.values()))
    seqs = list(first["sequences"])
    # OIDN's column only if the first run's evaluate.py was given --oidn.
    with_oidn = "oidn_bilinear" in first["sequences"][seqs[0]]["flicker"]
    oidn = [("OIDN, bilinear", None, "oidn_bilinear")] if with_oidn else []
    kinds = [(f"{n} network", n, "network") for n in runs] + [("input, bilinear", None, "input_bilinear")] + oidn

    def value(run_name: str | None, kind: str, seq: str, table: str, region: str) -> float:
        run = runs[run_name] if run_name else first
        return run["sequences"][seq][table][kind][region]

    print("### Displayed MSE against the 4,096-sample targets, mean of each sequence's 8 frames\n")
    for key, label in REGIONS:
        print(f"**{label}**\n")
        print("| sequence | " + " | ".join(k[0] for k in kinds) + f" | {next(iter(runs))} / input |")
        print("|---|" + "---|" * (len(kinds) + 1))
        for s in seqs:
            vals = [value(r, k, s, "vs_targets", key) for _, r, k in kinds]
            print(f"| {s} | " + " | ".join(fmt(v) for v in vals) + f" | {vals[0] / vals[len(runs)]:.2f} |")
        print()

    print("### Displayed MSE against the 65,536-sample references (the sequence's first frame, after 8 warm-up frames)\n")
    for s in seqs:
        if "vs_reference" not in first["sequences"][s]:
            continue
        print(f"**{s}, frame {first['sequences'][s]['vs_reference']['frame']}**\n")
        cols = kinds + [("4,096-sample target", None, "targets")]
        print("| region | " + " | ".join(c[0] for c in cols) + " |")
        print("|---|" + "---|" * len(cols))
        for key, label in REGIONS:
            vals = [(runs[r] if r else first)["sequences"][s]["vs_reference"][k][key] for _, r, k in cols]
            print(f"| {label} | " + " | ".join(fmt(v) for v in vals) + " |")
        print()

    print("### Flicker: mean absolute displayed change, frame to frame (7 pairs a sequence)\n")
    cols = kinds + [("targets", None, "targets")]
    for key, label in REGIONS:
        print(f"**{label}**\n")
        print("| sequence | " + " | ".join(c[0] for c in cols) + " |")
        print("|---|" + "---|" * len(cols))
        for s in seqs:
            vals = [value(r, k, s, "flicker", key) for _, r, k in cols]
            print(f"| {s} | " + " | ".join(fmt(v) for v in vals) + " |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
