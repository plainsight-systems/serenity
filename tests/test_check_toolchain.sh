#!/bin/sh
# Proves tools/check_toolchain.py fails when the machine's toolchain differs
# from the pin, for each pinned tool, and names the tool.
set -eu

cd "$(dirname "$0")/.."
CHECK=./tools/check_toolchain.py
PINS="$(mktemp -t serenity-pins.XXXXXX)"
ERR="$(mktemp -t serenity-pins-err.XXXXXX)"
FAKE="$(mktemp -t serenity-fake-tool.XXXXXX)"
trap 'rm -f "${PINS}" "${ERR}" "${FAKE}"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

# Baseline: the real pins match this machine, or the probes prove nothing.
"${CHECK}" >/dev/null 2>&1 || fail "the real pins do not match this machine: run ${CHECK}"

for key in xcode xcode_build macos_sdk metal cxx cc objc objcxx cmake; do
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

# The lint tools' pins (LLVM's) are --lint's alone: a mismatch in one fails
# --lint by name, and does not touch the build's check, which a machine
# without LLVM must still pass; a build pin's mismatch does not touch --lint.
"${CHECK}" --lint >/dev/null 2>&1 || fail "the real lint pins do not match this machine: run ${CHECK} --lint"
for key in clang_format clang_tidy clangd cxx; do
    python3 - "${key}" "${PINS}" <<'PY'
import json, sys
key, out = sys.argv[1], sys.argv[2]
pins = json.load(open("cmake/toolchain.json"))
pins[key] = pins[key] + " (not this one)"
json.dump(pins, open(out, "w"))
PY
    case "${key}" in
        cxx)
            "${CHECK}" --lint --pins "${PINS}" >/dev/null 2>&1 ||
                fail "${key}: a build pin's mismatch failed --lint"
            ;;
        *)
            if "${CHECK}" --lint --pins "${PINS}" >/dev/null 2>"${ERR}"; then
                fail "${key}: a mismatched lint pin passed --lint"
            fi
            grep -q "^  ${key}:" "${ERR}" || fail "${key}: the failure did not name the tool: $(cat "${ERR}")"
            "${CHECK}" --pins "${PINS}" >/dev/null 2>&1 ||
                fail "${key}: a lint pin's mismatch failed the build's check"
            ;;
    esac
done

# Lint tools other than the pinned ones fail --lint, each by name: the check
# reads the tools in --llvm-bin, not fixed paths.
FAKE_LLVM="$(mktemp -d -t serenity-fake-llvm.XXXXXX)"
trap 'rm -f "${PINS}" "${ERR}" "${FAKE}"; rm -rf "${FAKE_LLVM}"' EXIT
for tool in clang-format clang-tidy clangd; do
    printf '#!/bin/sh\necho "Homebrew %s version 99.0.0"\n' "${tool}" > "${FAKE_LLVM}/${tool}"
    chmod +x "${FAKE_LLVM}/${tool}"
done
if "${CHECK}" --lint --llvm-bin "${FAKE_LLVM}" >/dev/null 2>"${ERR}"; then
    fail "lint: unpinned lint tools passed"
fi
for key in clang_format clang_tidy clangd; do
    grep -q "^  ${key}:" "${ERR}" || fail "${key}: the failure did not name it: $(cat "${ERR}")"
done

# A tool other than the pinned one fails, even with every pin intact: the
# check reads the tools CMake resolved, not fixed paths.
printf '#!/bin/sh\necho "Apple clang version 99.0.0 (not the pinned one)"\n' > "${FAKE}"
chmod +x "${FAKE}"
for tool in cxx cc objc objcxx cmake; do
    if "${CHECK}" "--${tool}" "${FAKE}" >/dev/null 2>"${ERR}"; then
        fail "${tool}: an unpinned one passed"
    fi
    grep -q "^  ${tool}:" "${ERR}" || fail "${tool}: the failure did not name it: $(cat "${ERR}")"
done

# An SDK root other than the pinned SDK's fails, and so does none at all (a
# configure without the presets, which name it).
if "${CHECK}" --sdk /nonexistent/MacOSX.sdk >/dev/null 2>"${ERR}"; then
    fail "macos_sdk: an SDK root that is not the pinned SDK passed"
fi
grep -q "^  macos_sdk:" "${ERR}" || fail "macos_sdk: the failure did not name it: $(cat "${ERR}")"
if "${CHECK}" --sdk "" >/dev/null 2>&1; then
    fail "macos_sdk: no SDK root passed"
fi
"${CHECK}" --sdk "$(xcrun --sdk macosx --show-sdk-path)" >/dev/null 2>&1 ||
    fail "macos_sdk: the pinned SDK, named by its root, was refused"

# A pins file that is missing or not JSON fails by name, not by a traceback.
if "${CHECK}" --pins /nonexistent/toolchain.json >/dev/null 2>"${ERR}"; then
    fail "pins: a missing pins file passed"
fi
grep -q "^TOOLCHAIN PINS UNREADABLE: /nonexistent/toolchain.json: " "${ERR}" ||
    fail "pins: a missing pins file was not named: $(cat "${ERR}")"
printf '{ not json' > "${PINS}"
if "${CHECK}" --pins "${PINS}" >/dev/null 2>"${ERR}"; then
    fail "pins: a pins file that is not JSON passed"
fi
grep -q "^TOOLCHAIN PINS UNREADABLE: ${PINS}: " "${ERR}" ||
    fail "pins: a pins file that is not JSON was not named: $(cat "${ERR}")"
grep -q "Traceback" "${ERR}" && fail "pins: a traceback, not a named failure: $(cat "${ERR}")"

echo "check_toolchain: OK (a mismatch in any pin fails, by name; the lint pins only --lint)"
