#!/bin/sh
# Enforces the boundary that keeps the core platform-neutral.
#
# src/core/ is the scene, the sampling and the math every backend shares. It
# must build and be tested with no GPU API at all, so that a second backend
# (Vulkan, for the AMD machine) renders the same scene from the same code.
# Only src/metal/ may touch Metal's host API. A reviewer noticing a stray
# include is not a control; this is. Covered by tests/test_check_boundaries.sh,
# which proves each rule fires; both run under ctest (tests/CMakeLists.txt).
#
# The Metal shading language's own headers (<metal_stdlib> and the other
# <metal_*> headers) are not the host API: a core header that a shader also
# includes may name them, behind __METAL_VERSION__.
#
# A search that cannot run (a directory missing, a file unreadable: grep's
# status 2) fails the check: a rule that searched nothing proved nothing.
set -eu

cd "$(dirname "$0")/.."

# The host APIs of Apple's GPU and windowing frameworks, and Vulkan's: any
# include or import from them, an Objective-C module import of one, and
# metal-cpp's namespaces (MTL, MTL4, MTLFX, MTL4FX and any MTL-prefixed one,
# NS, CA). Every Metal framework (Metal, MetalKit, MetalFX,
# MetalPerformanceShaders, MetalPerformanceShadersGraph ...), and the
# surfaces and video frames that hold their images (IOSurface, CoreVideo).
FRAMEWORKS='(Metal[A-Za-z]*|Foundation|QuartzCore|AppKit|Cocoa|IOSurface|CoreVideo|objc|dispatch|vulkan)'
PLATFORM_INCLUDES="#[[:space:]]*(include|import)[[:space:]]*[<\"]${FRAMEWORKS}/"
PLATFORM_MODULES="@import[[:space:]]+${FRAMEWORKS}([[:space:];.]|\$)"
PLATFORM_TOKENS='(^|[^[:alnum:]_])(MTL[A-Z0-9]*|NS|CA)::'
PLATFORM="${PLATFORM_INCLUDES}|${PLATFORM_MODULES}|${PLATFORM_TOKENS}"

# The directory permitted to use Metal's host API.
METAL_DIR=src/metal/

status=0
fail() { echo "BOUNDARY VIOLATION: $1" >&2; status=1; }

for dir in src src/core src/metal; do
    [ -d "${dir}" ] || { echo "check_boundaries: ${dir} is missing; nothing would be checked" >&2; exit 2; }
done

# grep's matches, or the run stopped if grep itself failed (status 2); no
# match (status 1) is no output.
search() {
    if out="$(grep "$@")"; then
        printf '%s\n' "${out}"
    else
        code=$?
        if [ "${code}" -ge 2 ]; then
            echo "check_boundaries: grep $* failed (status ${code})" >&2
            exit 2
        fi
    fi
}

# 1. The core uses no platform API.
hits="$(search -rnE "${PLATFORM}" src/core/)"
if [ -n "${hits}" ]; then
    echo "${hits}" | sed 's/^/  /' >&2
    fail "src/core/ uses a platform GPU or windowing API. The core must build with none."
fi

# 2. The core does not depend outward on a backend or the app. Quoted and
#    angled forms alike, and any relative path into them.
hits="$(search -rnE '#[[:space:]]*include[[:space:]]*[<"](\.\./)*(metal|vulkan|app|headless)/' src/core/)"
if [ -n "${hits}" ]; then
    echo "${hits}" | sed 's/^/  /' >&2
    fail "src/core/ includes from a backend or the app. Dependencies point into the core only."
fi

# 3. Within src/, only src/metal/ uses Metal's host API. An exact directory
#    prefix, not a pattern: src/metal_extra/ is not src/metal/.
users="$(search -rlE "${PLATFORM}" src/)"
offenders="$(printf '%s\n' "${users}" | grep -v -e "^${METAL_DIR}" -e '^$' || true)"
if [ -n "${offenders}" ]; then
    echo "${offenders}" | sed 's/^/  /' >&2
    fail "only ${METAL_DIR} may use Metal's host API."
fi

# 4. Each third-party library is reached from one place only, so swapping it
#    touches only its readers (change-axes.md): SDL from src/app/, toml++
#    from the two file readers (frame graphs and scenes), stb from the PNG
#    writer.
confined() {
    pattern="$1"; allowed="$2"; what="$3"
    users="$(search -rlE "${pattern}" src/)"
    offenders="$(printf '%s\n' "${users}" | grep -vE -e "^${allowed}" -e '^$' || true)"
    if [ -n "${offenders}" ]; then
        echo "${offenders}" | sed 's/^/  /' >&2
        fail "${what} may be included only from ${allowed}."
    fi
}
confined '#[[:space:]]*include[[:space:]]*[<"]SDL3/' 'src/app/' 'SDL'
confined '#[[:space:]]*include[[:space:]]*[<"]toml\+\+/' 'src/core/(frame/graph_file|scene/scene)\.cpp$' 'toml++'
confined '#[[:space:]]*include[[:space:]]*[<"]stb_' 'src/core/output/png\.cpp$' 'stb'

if [ "${status}" -eq 0 ]; then
    echo "boundaries OK"
fi
exit "${status}"
