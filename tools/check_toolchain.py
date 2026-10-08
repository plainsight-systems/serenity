#!/usr/bin/env python3
"""Checks that this machine's toolchain is the one cmake/toolchain.json pins.

    tools/check_toolchain.py [--pins <path>] [--cxx <compiler>]

Metal cannot run in a container, so the toolchain cannot be pinned by an
image. It is pinned by version instead, and this check is what makes the pin
binding: CMake runs it at configure time and stops on a mismatch. Each
mismatch is named with both versions, so the failure says what to install or
which pin to move. Moving a pin is a commit of its own (cmake/toolchain.json).

Exit status: 0 when every version matches, 1 on any mismatch or when a tool
cannot be run. Covered by tests/test_check_toolchain.sh, which proves a
mismatch fails.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# What each pin is compared with: the first line of each command's output.
# The compiler checked is the one CMake resolved (--cxx), so a build configured
# with another compiler is caught; run by hand, it is /usr/bin/clang++, the
# one the presets select. The one on PATH may be another clang entirely
# (Homebrew's, on the development machine).
COMMANDS = {
    "xcode": ["xcodebuild", "-version"],
    "xcode_build": ["xcodebuild", "-version"],
    "macos_sdk": ["xcrun", "--sdk", "macosx", "--show-sdk-version"],
    "metal": ["xcrun", "-sdk", "macosx", "metal", "--version"],
    "cxx": ["/usr/bin/clang++", "--version"],
}
# xcodebuild prints two lines; each pin takes one.
LINE = {"xcode_build": 1}


def installed(key):
    try:
        out = subprocess.run(COMMANDS[key], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return None, f"could not run {' '.join(COMMANDS[key])}: {error}"
    lines = out.splitlines()
    index = LINE.get(key, 0)
    if index >= len(lines):
        return None, f"{' '.join(COMMANDS[key])} printed {len(lines)} line(s), expected more than {index}"
    return lines[index].strip(), None


def main(argv):
    pins_path = ROOT / "cmake" / "toolchain.json"
    args = argv[1:]
    while args:
        if len(args) >= 2 and args[0] == "--pins":
            pins_path = pathlib.Path(args[1])
        elif len(args) >= 2 and args[0] == "--cxx":
            COMMANDS["cxx"] = [args[1], "--version"]
        else:
            print(__doc__, file=sys.stderr)
            return 1
        args = args[2:]

    pins = json.loads(pins_path.read_text())
    failures = []
    for key in COMMANDS:
        if key not in pins:
            failures.append(f"{key}: no pin in {pins_path}")
            continue
        found, error = installed(key)
        if error:
            failures.append(f"{key}: {error}")
        elif found != pins[key]:
            failures.append(f"{key}: pinned '{pins[key]}', found '{found}'")

    if failures:
        print(f"TOOLCHAIN MISMATCH against {pins_path}:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("toolchain OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
