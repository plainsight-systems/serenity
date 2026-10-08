#!/bin/sh
# Enforces the boundary that keeps the core platform-neutral.
#
# src/core/ is the scene, the sampling and the math every backend shares. It
# must build and be tested with no GPU API at all, so that a second backend
# (Vulkan, for the AMD machine) renders the same scene from the same code.
# Only src/metal/ may touch Metal's host API. A reviewer noticing a stray
# include is not a control; this is. Covered by tests/test_check_boundaries.sh,
# which proves each rule fires.
#
# The Metal shading language's own headers (<metal_stdlib> and the other
# <metal_*> headers) are not the host API: a core header that a shader also
# includes may name them, behind __METAL_VERSION__ (core/portable_math.h).
set -eu

cd "$(dirname "$0")/.."

# The host APIs of Apple's GPU and windowing frameworks, and Vulkan's: any
# include from them, and metal-cpp's namespaces.
PLATFORM_INCLUDES='#[[:space:]]*(include|import)[[:space:]]*[<"](Metal|MetalKit|MetalFX|Foundation|QuartzCore|AppKit|Cocoa|objc|dispatch|vulkan)/'
PLATFORM_TOKENS='(^|[^[:alnum:]_])(MTL|NS|CA|MTL4)::'
PLATFORM="${PLATFORM_INCLUDES}|${PLATFORM_TOKENS}"

# The directory permitted to use Metal's host API.
METAL_DIR=src/metal/

status=0
fail() { echo "BOUNDARY VIOLATION: $1" >&2; status=1; }

# 1. The core uses no platform API.
hits="$(grep -rnE "${PLATFORM}" src/core/ 2>/dev/null || true)"
if [ -n "${hits}" ]; then
    echo "${hits}" | sed 's/^/  /' >&2
    fail "src/core/ uses a platform GPU or windowing API. The core must build with none."
fi

# 2. The core does not depend outward on a backend or the app. Quoted and
#    angled forms alike, and any relative path into them.
hits="$(grep -rnE '#[[:space:]]*include[[:space:]]*[<"](\.\./)*(metal|vulkan|app)/' src/core/ 2>/dev/null || true)"
if [ -n "${hits}" ]; then
    echo "${hits}" | sed 's/^/  /' >&2
    fail "src/core/ includes from a backend or the app. Dependencies point into the core only."
fi

# 3. Within src/, only src/metal/ uses Metal's host API. An exact directory
#    prefix, not a pattern: src/metal_extra/ is not src/metal/.
offenders="$(grep -rlE "${PLATFORM}" src/ 2>/dev/null | grep -v "^${METAL_DIR}" || true)"
if [ -n "${offenders}" ]; then
    echo "${offenders}" | sed 's/^/  /' >&2
    fail "only ${METAL_DIR} may use Metal's host API."
fi

if [ "${status}" -eq 0 ]; then
    echo "boundaries OK"
fi
exit "${status}"
