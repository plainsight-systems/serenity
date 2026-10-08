#!/bin/sh
# Proves tools/check_toolchain.py fails when the machine's toolchain differs
# from the pin, for each pinned tool, and names the tool.
set -eu

cd "$(dirname "$0")/.."
CHECK=./tools/check_toolchain.py
PINS="$(mktemp -t serenity-pins.XXXXXX)"
ERR="$(mktemp -t serenity-pins-err.XXXXXX)"
trap 'rm -f "${PINS}" "${ERR}"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

# Baseline: the real pins match this machine, or the probes prove nothing.
"${CHECK}" >/dev/null 2>&1 || fail "the real pins do not match this machine: run ${CHECK}"

for key in xcode xcode_build macos_sdk metal cxx; do
    python3 - "${key}" "${PINS}" <<'PY'
import json, sys
key, out = sys.argv[1], sys.argv[2]
pins = json.load(open("cmake/toolchain.json"))
pins[key] = pins[key] + " (not this one)"
json.dump(pins, open(out, "w"))
PY
    if "${CHECK}" --pins "${PINS}" >/dev/null 2>"${ERR}"; then
        fail "${key}: a mismatched pin passed"
    fi
    grep -q "^  ${key}:" "${ERR}" || fail "${key}: the failure did not name the tool: $(cat "${ERR}")"
done

echo "check_toolchain: OK (a mismatch in any pin fails, by name)"
