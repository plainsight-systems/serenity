#!/bin/sh
# Proves each boundary rule in tools/check_boundaries.sh fires.
#
# A checker nobody has seen fail is not evidence of anything. Each probe
# writes one violating file, expects the checker to fail, and removes it.
set -eu

cd "$(dirname "$0")/.."
CHECK=./tools/check_boundaries.sh
CREATED=""

cleanup() { for f in ${CREATED}; do rm -f "$f"; done; rmdir src/metal_extra 2>/dev/null || true; }
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

expect_violation() {
    if "${CHECK}" >/dev/null 2>&1; then
        fail "$1: checker passed but should have failed"
    fi
}

probe() {
    desc="$1"; file="$2"; content="$3"
    CREATED="${CREATED} ${file}"
    printf '%b' "${content}" > "${file}"
    expect_violation "${desc}"
    rm -f "${file}"
}

# Baseline: the real tree must pass, or every assertion below is meaningless.
"${CHECK}" >/dev/null 2>&1 || fail "clean tree does not pass the checker"

# 1. Platform API in the core: an include, an angled framework include other
#    than Metal's, and a metal-cpp name with no include at all.
probe "Metal host include in core" src/core/_probe1.h '#pragma once\n#include <Metal/Metal.hpp>\n'
probe "QuartzCore include in core" src/core/_probe2.h '#pragma once\n#include <QuartzCore/QuartzCore.hpp>\n'
probe "Vulkan include in core" src/core/_probe3.h '#pragma once\n#include <vulkan/vulkan.h>\n'
probe "metal-cpp name in core" src/core/_probe4.h '#pragma once\nvoid f(MTL::Device* d);\n'

# 2. The core including a backend, by quoted and by relative path.
probe "core includes the Metal backend" src/core/_probe5.h '#pragma once\n#include "metal/device.h"\n'
probe "core includes a backend by relative path" src/core/_probe6.h '#pragma once\n#include "../metal/device.h"\n'

# 3. The Metal host API outside src/metal/, including a directory whose name
#    merely starts with it.
mkdir -p src/metal_extra
probe "Metal host API outside src/metal/" src/metal_extra/_probe7.cpp '#include <Metal/Metal.hpp>\n'

# The shading language's own headers in the core are allowed.
F=src/core/_probe8.h; CREATED="${CREATED} ${F}"
printf '#pragma once\n#if defined(__METAL_VERSION__)\n#include <metal_stdlib>\n#endif\n' > "${F}"
"${CHECK}" >/dev/null 2>&1 || fail "<metal_stdlib> in core was refused; it is the shading language, not the host API"
rm -f "${F}"

# Tree must be clean again afterwards.
"${CHECK}" >/dev/null 2>&1 || fail "checker still failing after probes removed"

echo "check_boundaries: OK (every rule fires)"
