#!/usr/bin/env bash
# fetch_denoiser_weights.sh — download the trained denoiser network.
#
# Usage:  ./scripts/fetch_denoiser_weights.sh [directory]
#
# The network the demo runs (training/README.md), Milestone 3's retrained
# in Milestone 4 on denoiser-v2 (a pinhole, the jitter, checkerboard), is
# published on the project's media host, not committed. This writes it into
# <directory>/wide/ (default ../serenity-data/denoiser-v2-model beside the
# checkout, where graphs/restir_pt_denoised.toml reads it) and checks every
# file against the SHA-256 pinned below, so a
# fresh clone runs the same network the write-ups measured. A file already
# present with the pinned checksum is kept; one that differs is refused, not
# overwritten, so a network trained locally is never replaced silently.
#
# Then the network's Core ML model beside the weights, which the Metal
# backend runs its U-Net through on the Neural Engine
# (training/export_coreml.py; Xcode's coremlcompiler and the training
# environment's Core ML Tools), made here from the checked weights rather
# than published: it is the Mac's toolchain's build of them.

set -euo pipefail

base_url="https://media.plainsight-systems.com/denoiser-v2-model"
root="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:-$root/../serenity-data/denoiser-v2-model}"

# network file sha256 (training/export.py's export, 2026-10-19)
pinned="
wide  weights.bin 302a2b5f2ef1594d5991830e79b8259e2c3e9ed6c1ff17fb2049ebde814e5977
wide  layout.json b2d219c8bdaef1feee5d07a46ad0463faaae124cf87f1fca2bde953d2196596d
"

sha256() { shasum -a 256 "$1" | cut -d ' ' -f 1; }

while read -r network file expected; do
    [ -n "$network" ] || continue
    dir="$out/$network"
    path="$dir/$file"
    mkdir -p "$dir"
    if [ -e "$path" ]; then
        if [ "$(sha256 "$path")" = "$expected" ]; then
            echo "kept $path"
            continue
        fi
        echo "fetch_denoiser_weights: $path exists and is not the published file; move it aside to fetch" >&2
        exit 1
    fi
    partial="$path.partial"
    curl -sSfL -o "$partial" "$base_url/$network/$file"
    if [ "$(sha256 "$partial")" != "$expected" ]; then
        rm -f "$partial"
        echo "fetch_denoiser_weights: $base_url/$network/$file does not match its pinned checksum" >&2
        exit 1
    fi
    mv "$partial" "$path"
    echo "fetched $path"
done <<< "$pinned"

if [ "$(uname)" = "Darwin" ]; then
    (cd "$root/training" && uv run --locked python export_coreml.py --model "$out/wide")
fi
