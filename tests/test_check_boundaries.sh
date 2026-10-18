#!/bin/sh
# Proves each boundary rule in tools/check_boundaries.sh fires.
#
# A checker nobody has seen fail is not evidence of anything. Each probe
# writes one violating file, expects the checker to fail with the message
# of the rule the probe breaks, and removes it: a rule gone dead whose probe
# another rule happens to catch fails here, not passes unseen.
set -eu

cd "$(dirname "$0")/.."
CHECK=./tools/check_boundaries.sh
CREATED=""
ERR="$(mktemp -t serenity-boundaries-err.XXXXXX)"

# Each rule's message, as the checker prints it (its fail() calls).
CORE_PLATFORM="src/core/ uses a platform GPU or windowing API"
CORE_OUTWARD="src/core/ includes from a backend or a program"
MEASURE_OUTWARD="src/measure/ includes from a backend or another program"
METAL_ONLY="only src/metal/ may use Metal's host API"
SDL_CONFINED="SDL may be included only from"
TOML_CONFINED="toml++ may be included only from"
STB_CONFINED="stb may be included only from"
REUSE_APART="src/metal/reuse/ includes the Integrator or a Pass"
INTEGRATOR_APART="src/metal/integrator/ includes Sample reuse or a Pass"
APP_NEUTRAL="src/app/ includes a backend outside main.cpp"
APP_PACING="src/app/ reaches Metal's display link"
SEARCH_FAILED="failed (status"

# src/core/ may hold no files yet, and git keeps no empty directory.
CORE_CREATED=0
[ -d src/core ] || { mkdir -p src/core; CORE_CREATED=1; }

cleanup() {
    for f in ${CREATED}; do rm -f "$f"; done
    rm -f "${ERR}"
    rmdir src/metal_extra 2>/dev/null || true
    if [ "${CORE_CREATED}" = 1 ]; then rmdir src/core 2>/dev/null || true; fi
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

# The checker fails, saying `rule`'s message.
expect_violation() {
    desc="$1"; rule="$2"
    if "${CHECK}" >/dev/null 2>"${ERR}"; then
        fail "${desc}: checker passed but should have failed"
    fi
    grep -qF -- "${rule}" "${ERR}" ||
        fail "${desc}: the checker failed, but not by the rule \"${rule}\": $(cat "${ERR}")"
}

probe() {
    desc="$1"; rule="$2"; file="$3"; content="$4"
    CREATED="${CREATED} ${file}"
    printf '%b' "${content}" > "${file}"
    expect_violation "${desc}" "${rule}"
    rm -f "${file}"
}

# Baseline: the real tree must pass, or every assertion below is meaningless.
"${CHECK}" >/dev/null 2>&1 || fail "clean tree does not pass the checker"

# 1. Platform API in the core: an include, an angled framework include other
#    than Metal's, and a metal-cpp name with no include at all.
probe "Metal host include in core" "${CORE_PLATFORM}" src/core/_probe1.h '#pragma once\n#include <Metal/Metal.hpp>\n'
probe "QuartzCore include in core" "${CORE_PLATFORM}" \
    src/core/_probe2.h '#pragma once\n#include <QuartzCore/QuartzCore.hpp>\n'
probe "Vulkan include in core" "${CORE_PLATFORM}" src/core/_probe3.h '#pragma once\n#include <vulkan/vulkan.h>\n'
probe "metal-cpp name in core" "${CORE_PLATFORM}" src/core/_probe4.h '#pragma once\nvoid f(MTL::Device* d);\n'
probe "MetalFX name in core" "${CORE_PLATFORM}" src/core/_probe13.h '#pragma once\nvoid f(MTLFX::SpatialScaler* s);\n'
probe "Metal 4 MetalFX name in core" "${CORE_PLATFORM}" \
    src/core/_probe14.h '#pragma once\nvoid f(MTL4FX::SpatialScaler* s);\n'
probe "MetalPerformanceShaders include in core" "${CORE_PLATFORM}" \
    src/core/_probe15.h '#pragma once\n#include <MetalPerformanceShaders/MetalPerformanceShaders.h>\n'
probe "IOSurface include in core" "${CORE_PLATFORM}" \
    src/core/_probe16.h '#pragma once\n#include <IOSurface/IOSurface.h>\n'
probe "CoreVideo include in core" "${CORE_PLATFORM}" \
    src/core/_probe17.h '#pragma once\n#include <CoreVideo/CoreVideo.h>\n'
probe "Objective-C module import in core" "${CORE_PLATFORM}" src/core/_probe18.mm '@import Metal;\n'

# 2. The core including a backend, by quoted and by relative path.
probe "core includes the Metal backend" "${CORE_OUTWARD}" \
    src/core/_probe5.h '#pragma once\n#include "metal/device/device.h"\n'
probe "core includes a backend by relative path" "${CORE_OUTWARD}" \
    src/core/_probe6.h '#pragma once\n#include "../metal/device/device.h"\n'
probe "core includes the headless renderer" "${CORE_OUTWARD}" \
    src/core/_probe9.h '#pragma once\n#include "headless/render.h"\n'
probe "core includes the measuring program" "${CORE_OUTWARD}" \
    src/core/_probe20.h '#pragma once\n#include "measure/options.h"\n'

# 2b. The measuring program including a backend or another program, by
#     quoted, angled and relative path.
probe "measure includes the Metal backend" "${MEASURE_OUTWARD}" \
    src/measure/_probe21.h '#pragma once\n#include "metal/frame/renderer.h"\n'
probe "measure includes the app by angle" "${MEASURE_OUTWARD}" \
    src/measure/_probe22.h '#pragma once\n#include <app/window.h>\n'
probe "measure includes the headless renderer by relative path" "${MEASURE_OUTWARD}" src/measure/_probe23.h \
    '#pragma once\n#include "../headless/options.h"\n'
# And rule 3 covers it: Metal's host API in src/measure/.
probe "Metal host API in src/measure/" "${METAL_ONLY}" src/measure/_probe24.cpp 'void f(MTL::Device* d);\n'

# 3. The Metal host API outside src/metal/, including a directory whose name
#    merely starts with it.
mkdir -p src/metal_extra
probe "Metal host API outside src/metal/" "${METAL_ONLY}" src/metal_extra/_probe7.cpp '#include <Metal/Metal.hpp>\n'

# 4. Third-party libraries outside their one place.
probe "SDL outside src/app/" "${SDL_CONFINED}" src/core/_probe10.h '#pragma once\n#include <SDL3/SDL.h>\n'
probe "toml++ outside the frame graph reader" "${TOML_CONFINED}" \
    src/core/frame/_probe11.cpp '#include <toml++/toml.hpp>\n'
probe "stb outside the PNG writer" "${STB_CONFINED}" src/core/frame/_probe12.cpp '#include <stb_image_write.h>\n'

# 4b. Sample reuse and the Integrator including each other, or a Pass: the
#     backend's and the core's, by quoted, angled and relative path.
probe "Sample reuse includes the Integrator" "${REUSE_APART}" src/metal/reuse/_probe25.metal.h \
    '#pragma once\n#include "metal/integrator/nee.metal.h"\n'
probe "Sample reuse includes a Pass" "${REUSE_APART}" src/metal/reuse/_probe26.metal.h \
    '#pragma once\n#include "metal/passes/restir_pt/restir_pt.metal.h"\n'
probe "Sample reuse includes the core's Pass settings" "${REUSE_APART}" src/metal/reuse/_probe27.metal.h \
    '#pragma once\n#include <core/passes/restir_pt.h>\n'
probe "Sample reuse includes the Integrator by relative path" "${REUSE_APART}" src/metal/reuse/_probe28.metal.h \
    '#pragma once\n#include "../integrator/path.metal.h"\n'
probe "the Integrator includes Sample reuse" "${INTEGRATOR_APART}" src/metal/integrator/_probe29.metal.h \
    '#pragma once\n#include "metal/reuse/reservoir.metal.h"\n'
probe "the Integrator includes a Pass" "${INTEGRATOR_APART}" src/metal/integrator/_probe30.metal.h \
    '#pragma once\n#include "metal/passes/restir_pt/restir_pt.metal.h"\n'
probe "the Integrator includes the core's Pass settings by relative path" "${INTEGRATOR_APART}" \
    src/metal/integrator/_probe31.metal.h \
    '#pragma once\n#include "../../core/passes/restir_pt.h"\n'
# Contracts are where they meet, and are allowed.
F=src/metal/reuse/_probe32.metal.h; CREATED="${CREATED} ${F}"
printf '#pragma once\n#include "core/contracts/path_reuse_settings.h"\n' > "${F}"
"${CHECK}" >/dev/null 2>&1 || fail "a contract included by Sample reuse was refused"
rm -f "${F}"

# 4c. The window shaped by a backend's pacing: a backend included by an app
#     file but main.cpp, by quoted, angled and relative path; the display
#     link included, or its types named, anywhere in src/app/.
probe "an app file includes the Metal backend" "${APP_NEUTRAL}" src/app/_probe33.h \
    '#pragma once\n#include "metal/device/presenter.h"\n'
probe "an app file includes a backend by angle" "${APP_NEUTRAL}" src/app/_probe34.h \
    '#pragma once\n#include <vulkan/frame/swapchain_frames.h>\n'
probe "an app file includes the Metal backend by relative path" "${APP_NEUTRAL}" src/app/_probe35.cpp \
    '#include "../metal/frame/display_frames.h"\n'
probe "an app file includes the display link" "${APP_PACING}" src/app/_probe36.cpp \
    '#include "metal/device/display_link.h"\n'
probe "an app file names the display link" "${APP_PACING}" src/app/_probe37.cpp \
    'void f(serenity::metal::DisplayLink& link);\n'
probe "an app file names the display link's update" "${APP_PACING}" src/app/_probe38.h \
    '#pragma once\nstruct DisplayUpdate;\n'

# 5. A search that cannot read a file fails the check, rather than finding
#    nothing in it and passing.
F=src/core/_probe19.h; CREATED="${CREATED} ${F}"
printf '#pragma once\n' > "${F}"
chmod 000 "${F}"
expect_violation "an unreadable file in core" "${SEARCH_FAILED}"
chmod 644 "${F}"
rm -f "${F}"

# The shading language's own headers in the core are allowed.
F=src/core/_probe8.h; CREATED="${CREATED} ${F}"
printf '#pragma once\n#if defined(__METAL_VERSION__)\n#include <metal_stdlib>\n#endif\n' > "${F}"
"${CHECK}" >/dev/null 2>&1 || fail "<metal_stdlib> in core was refused; it is the shading language, not the host API"
rm -f "${F}"

# Tree must be clean again afterwards.
"${CHECK}" >/dev/null 2>&1 || fail "checker still failing after probes removed"

echo "check_boundaries: OK (every rule fires)"
