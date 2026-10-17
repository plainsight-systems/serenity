#!/usr/bin/env bash
# fetch_denoiser_weights.sh — download the trained denoiser network.
#
# Usage:  ./scripts/fetch_denoiser_weights.sh [directory]
#
# The network Milestone 3 trained (training/README.md), the demo's, is
# published on the project's media host, not committed. This writes it into
# <directory>/wide/ (default ../serenity-data/denoiser-v1-model beside the
# checkout, where graphs/restir_pt_denoised.toml reads it) and checks every
# file against the SHA-256 pinned below, so a
# fresh clone runs the same network the write-ups measured. A file already
# present with the pinned checksum is kept; one that differs is refused, not
# overwritten, so a network trained locally is never replaced silently.

set -euo pipefail

base_url="https://media.plainsight-systems.com/denoiser-v1-model"
root="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:-$root/../serenity-data/denoiser-v1-model}"

# network file sha256 (training/export.py's export, 2026-10-15)
pinned="
wide  weights.bin 6f31158ceb8ad41527b11472e20079267287be9e6f34cb44b5c01860987dd4af
wide  layout.json 8b8828f261ca6529c7b1d6f75470678d4aad18fbac3cd7e020c4e9155f6c99f9
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
