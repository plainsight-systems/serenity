#!/bin/sh
# Enforces the boundary that keeps the core platform-neutral, the one
# between two shader families that only a Pass composes (rule 5), and the
# one between the core and the shaders (rule 7).
#
# src/core/ is the scene, the sampling and the math every backend shares. It
# must build and be tested with no GPU API at all, so that a second backend
# (CUDA and OptiX, for the RTX 3090 Ti) renders the same scene from the same
# code.
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

# The host APIs of Apple's GPU and windowing frameworks: any
# include or import from them, an Objective-C module import of one, and
# metal-cpp's namespaces (MTL, MTL4, MTLFX, MTL4FX and any MTL-prefixed one,
# NS, CA). Every Metal framework (Metal, MetalKit, MetalFX,
# MetalPerformanceShaders, MetalPerformanceShadersGraph ...), and the
# surfaces and video frames that hold their images (IOSurface, CoreVideo).
FRAMEWORKS='(Metal[A-Za-z]*|Foundation|QuartzCore|AppKit|Cocoa|IOSurface|CoreVideo|objc|dispatch)'
PLATFORM_INCLUDES="#[[:space:]]*(include|import)[[:space:]]*[<\"]${FRAMEWORKS}/"
PLATFORM_MODULES="@import[[:space:]]+${FRAMEWORKS}([[:space:];.]|\$)"
PLATFORM_TOKENS='(^|[^[:alnum:]_])(MTL[A-Z0-9]*|NS|CA)::'
PLATFORM="${PLATFORM_INCLUDES}|${PLATFORM_MODULES}|${PLATFORM_TOKENS}"
# NVIDIA's: CUDA's and OptiX's headers, which the core may not include
# either (rule 1).
CUDA_INCLUDES='#[[:space:]]*include[[:space:]]*[<"](cuda|optix|nvrtc)[A-Za-z_]*\.h[>"]'

# The directory permitted to use Metal's host API.
METAL_DIR=src/metal/

status=0
fail() { echo "BOUNDARY VIOLATION: $1" >&2; status=1; }

for dir in src src/core src/metal src/measure src/metal/reuse src/metal/integrator src/app src/headless; do
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
hits="$(search -rnE "${PLATFORM}|${CUDA_INCLUDES}" src/core/)"
if [ -n "${hits}" ]; then
    echo "${hits}" | sed 's/^/  /' >&2
    fail "src/core/ uses a platform GPU or windowing API. The core must build with none."
fi

# 2. The core does not depend outward on a backend or a program (the app,
#    the headless renderer, the measuring program). Quoted and angled forms
#    alike, and any relative path into them.
hits="$(search -rnE '#[[:space:]]*include[[:space:]]*[<"](\.\./)*(metal|cuda|app|headless|measure)/' src/core/)"
if [ -n "${hits}" ]; then
    echo "${hits}" | sed 's/^/  /' >&2
    fail "src/core/ includes from a backend or a program. Dependencies point into the core only."
fi

# 2b. The measuring program is the core's alone (src/measure/options.h): it
#     includes from no backend and no other program, so a reference is made
#     and judged on any machine the core builds on. Rule 3 keeps the
#     platform APIs out of it, as out of all of src/ but src/metal/.
hits="$(search -rnE '#[[:space:]]*include[[:space:]]*[<"](\.\./)*(metal|cuda|app|headless)/' src/measure/)"
if [ -n "${hits}" ]; then
    echo "${hits}" | sed 's/^/  /' >&2
    fail "src/measure/ includes from a backend or another program. It uses the core alone."
fi

# 3. Within src/, only src/metal/ uses Metal's host API. An exact directory
#    prefix, not a pattern: src/metal_extra/ is not src/metal/.
users="$(search -rlE "${PLATFORM}" src/)"
offenders="$(printf '%s\n' "${users}" | grep -v -e "^${METAL_DIR}" -e '^$' || true)"
if [ -n "${offenders}" ]; then
    echo "${offenders}" | sed 's/^/  /' >&2
    fail "only ${METAL_DIR} may use Metal's host API."
fi

# 3b. Within src/, only src/cuda/ uses CUDA's host API and OptiX's: their
#     headers, by any spelling the include takes.
users="$(search -rlE "${CUDA_INCLUDES}" src/)"
offenders="$(printf '%s\n' "${users}" | grep -v -e '^src/cuda/' -e '^$' || true)"
if [ -n "${offenders}" ]; then
    echo "${offenders}" | sed 's/^/  /' >&2
    fail "only src/cuda/ may use CUDA's host API."
fi

# 4. Each third-party library is reached from one place only, so swapping it
#    touches only its readers (change-axes.md): SDL from src/app/, toml++
#    from the two file readers (frame graphs and scenes), stb from the PNG
#    writer (and the one unit compiling stb's implementation).
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
confined '#[[:space:]]*include[[:space:]]*[<"]stb_' 'src/core/output/(png|stb_image_write)\.cpp$' 'stb'

# 5. Sample reuse and the Integrator meet at contracts, and only a Pass
#    composes them (change-axes.md; metal/passes/restir_pt/
#    restir_pt.metal.h, "Where the families meet"): src/metal/reuse/
#    includes no Integrator header and no Pass header, the backend's or the
#    core's, and src/metal/integrator/ no Sample reuse header and no Pass
#    header. By quoted, angled and relative path.
hits="$(search -rnE '#[[:space:]]*include[[:space:]]*[<"](\.\./)*(metal/|core/)?(integrator|passes)/' src/metal/reuse/)"
if [ -n "${hits}" ]; then
    echo "${hits}" | sed 's/^/  /' >&2
    fail "src/metal/reuse/ includes the Integrator or a Pass. Sample reuse depends on contracts and the Sampler."
fi
hits="$(search -rnE '#[[:space:]]*include[[:space:]]*[<"](\.\./)*(metal/|core/)?(reuse|passes)/' src/metal/integrator/)"
if [ -n "${hits}" ]; then
    echo "${hits}" | sed 's/^/  /' >&2
    fail "src/metal/integrator/ includes Sample reuse or a Pass. The Integrator sees reservoirs as contract 15 alone."
fi

# 6. The window is not shaped by how a backend paces frames
#    (change-axes.md, "a frame source is the seam"): every file of src/app/
#    but main.cpp, which composes the window with a backend, includes from
#    no backend, by quoted, angled or relative path; and no file of
#    src/app/, main.cpp included, includes Metal's display link or names
#    its types, which only Metal's frame source uses.
hits="$(search -rnE '#[[:space:]]*include[[:space:]]*[<"](\.\./)*(metal|cuda)/' --exclude=main.cpp src/app/)"
if [ -n "${hits}" ]; then
    echo "${hits}" | sed 's/^/  /' >&2
    fail "src/app/ includes a backend outside main.cpp. The window meets a backend through the frame source alone."
fi
hits="$(search -rnE 'display_link\.h|(^|[^[:alnum:]_])(DisplayLink|DisplayUpdate)([^[:alnum:]_]|$)' src/app/)"
if [ -n "${hits}" ]; then
    echo "${hits}" | sed 's/^/  /' >&2
    fail "src/app/ reaches Metal's display link. How frames are paced is the backend's frame source's."
fi

# 8. The headless renderer names no backend (core/presentation/
#    offline_renderer.h): no file of src/headless/, main.cpp included,
#    includes from one, by quoted, angled or relative path. The program
#    reaches its platform's backend through the factory it is linked with.
hits="$(search -rnE '#[[:space:]]*include[[:space:]]*[<"](\.\./)*(metal|cuda)/' src/headless/)"
if [ -n "${hits}" ]; then
    echo "${hits}" | sed 's/^/  /' >&2
    fail "src/headless/ includes a backend. It renders through the offline renderer alone."
fi

# 7. A shader includes from the core only its shared layouts
#    (file-mapping.md, "The contracts"): headers that include
#    core/contracts/shared_layout.h, the layouts and their encodings, never
#    the core's math, so the GPU tests check each backend's own code
#    against the core's. Every Metal shader and header and every CUDA
#    kernel source and header of src/ and of both backends' probe kernels.
shaders="$(find src tests/gpu/kernels tests/cuda/kernels -type f \( -name '*.metal' -o -name '*.metal.h' -o -name '*.cu' -o -name '*.cuh' \) | sort)"
for shader in ${shaders}; do
    for header in $(search -hoE '#[[:space:]]*include[[:space:]]*[<"]core/[^">]+' "${shader}" | sed -E 's/.*[<"]//'); do
        [ "${header}" = core/contracts/shared_layout.h ] && continue
        if [ ! -r "src/${header}" ] || ! grep -q 'core/contracts/shared_layout.h' "src/${header}"; then
            echo "  ${shader}: ${header}" >&2
            fail "a shader includes a core header that is not a shared layout. Shaders take the core's layouts, never its math."
        fi
    done
done

if [ "${status}" -eq 0 ]; then
    echo "boundaries OK"
fi
exit "${status}"
