#!/usr/bin/env python3
"""Checks that this machine's toolchain is the one cmake/toolchain.json pins.

    tools/check_toolchain.py [--pins <path>] [--cxx <compiler>] [--cc <compiler>]
                             [--objc <compiler>] [--objcxx <compiler>]
                             [--sdk <name or path>] [--cmake <cmake>]
    tools/check_toolchain.py --lint [--pins <path>] [--llvm-bin <directory>]

Metal cannot run in a container, so the toolchain cannot be pinned by an
image. It is pinned by version instead, and this check is what makes the pin
binding: CMake runs it at configure time and stops on a mismatch. Each
mismatch is named with both versions, so the failure says what to install or
which pin to move. Moving a pin is a commit of its own (cmake/toolchain.json).

What is checked is what the build uses, as CMake resolved it: the C++, C,
Objective-C and Objective-C++ compilers (SDL is built with C and
Objective-C, the display link with Objective-C++), the SDK the build
compiles against (its root, CMAKE_OSX_SYSROOT), and CMake itself, whose
version decides the compile lines. Run by hand, each defaults to what the
presets select: /usr/bin/clang++ (C++ and Objective-C++), /usr/bin/clang
(C and Objective-C), the macosx SDK, and the
cmake on PATH. The one on PATH may be another clang entirely (Homebrew's, on
the development machine). Python, which runs only this check and the
review scripts and builds nothing, is found by CMake (find_package) and
given a floor there, not pinned.

--lint checks the other pins instead, and only them: LLVM's clang-format,
clang-tidy and clangd, in --llvm-bin (default Homebrew's,
/opt/homebrew/opt/llvm/bin). The build runs none of them, so configuring
does not ask for them and a machine without them still builds; make
format, format-check and tidy do, since the formatter's output depends on
its version.

Exit status: 0 when every version matches, 1 on any mismatch, when a tool
cannot be run, or when the pins file cannot be read as JSON (named, with
why). Covered by tests/test_check_toolchain.sh, which proves a
mismatch in each pin fails.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# xcodebuild prints two lines; each pin takes one.
LINE = {"xcode_build": 1}


def lint_commands(llvm_bin):
    """What each of --lint's pins is compared with: the first line of each
    tool's --version, the tools in `llvm_bin`."""
    return {
        "clang_format": [f"{llvm_bin}/clang-format", "--version"],
        "clang_tidy": [f"{llvm_bin}/clang-tidy", "--version"],
        "clangd": [f"{llvm_bin}/clangd", "--version"],
    }


def commands(tools):
    """What each pin is compared with: the first line (or LINE's) of each
    command's output, for the tools given."""
    return {
        "xcode": ["xcodebuild", "-version"],
        "xcode_build": ["xcodebuild", "-version"],
        "macos_sdk": ["xcrun", "--sdk", tools["sdk"], "--show-sdk-version"],
        "metal": ["xcrun", "-sdk", "macosx", "metal", "--version"],
        "cxx": [tools["cxx"], "--version"],
        "cc": [tools["cc"], "--version"],
        "objc": [tools["objc"], "--version"],
        "objcxx": [tools["objcxx"], "--version"],
        "cmake": [tools["cmake"], "--version"],
    }


def installed(command, index):
    try:
        out = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return None, f"could not run {' '.join(command)}: {error}"
    lines = out.splitlines()
    if index >= len(lines):
        return None, f"{' '.join(command)} printed {len(lines)} line(s), expected more than {index}"
    return lines[index].strip(), None


def main(argv):
    pins_path = ROOT / "cmake" / "toolchain.json"
    tools = {
        "cxx": "/usr/bin/clang++",
        "cc": "/usr/bin/clang",
        "objc": "/usr/bin/clang",
        "objcxx": "/usr/bin/clang++",
        "sdk": "macosx",
        "cmake": "cmake",
    }
    lint = False
    llvm_bin = "/opt/homebrew/opt/llvm/bin"
    args = argv[1:]
    while args:
        if args[0] == "--lint":
            lint = True
            args = args[1:]
            continue
        if len(args) >= 2 and args[0] == "--llvm-bin":
            llvm_bin = args[1]
        elif len(args) >= 2 and args[0] == "--pins":
            pins_path = pathlib.Path(args[1])
        elif len(args) >= 2 and args[0].startswith("--") and args[0][2:] in tools:
            if not args[1]:
                print(f"{args[0]} is empty: configure through a preset (CMakePresets.json)", file=sys.stderr)
                return 1
            tools[args[0][2:]] = args[1]
        else:
            print(__doc__, file=sys.stderr)
            return 1
        args = args[2:]

    # A pins file that cannot be read, or is not JSON, is a named failure
    # like any mismatch, not a traceback.
    try:
        pins = json.loads(pins_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"TOOLCHAIN PINS UNREADABLE: {pins_path}: {error}", file=sys.stderr)
        return 1
    failures = []
    checked = lint_commands(llvm_bin) if lint else commands(tools)
    for key, command in checked.items():
        if key not in pins:
            failures.append(f"{key}: no pin in {pins_path}")
            continue
        found, error = installed(command, LINE.get(key, 0))
        if error:
            failures.append(f"{key}: {error}")
        elif found != pins[key]:
            failures.append(f"{key}: pinned '{pins[key]}', found '{found}'")

    if failures:
        print(f"TOOLCHAIN MISMATCH against {pins_path}:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("lint tools OK" if lint else "toolchain OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
