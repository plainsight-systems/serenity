#pragma once

// The path reuse settings. Owned by Sample reuse; read by Pass, the
// Integrator's shift and the Frame graph. Contract 15's sibling
// (contracts/path_reservoir.h), as the reuse settings are contract 14's.
//
// The values ReSTIR PT runs with (docs/research/2026-10-11-restir-pt-
// layout.md, "What is built"), each its meaning and range, the ranges named
// below once (ES.45). The values the graphs are written with are ReSTIR PT
// Enhanced's defaults (core/passes/restir_pt.h); a graph varies them, never
// code (change-axes.md, "Mechanism is code; values are data").
//
//   nee_candidates   the lights NEE weighs by resampled importance sampling
//                    at the primary hit; at the path's vertex B, B counted
//                    from 1 at the primary hit, nee_candidates / B^2 of them,
//                    at least 1 (Enhanced §7). An integer in [1, 32]: a
//                    vertex's NEE numbers are 4 a candidate in a block of
//                    128 (metal/integrator/shift.metal.h, "The numbers").
//   max_surfaces     the surfaces a path scatters from, x1 to x_n; the ray
//                    the last one's BSDF draws still counts the light it
//                    strikes (the 2022 code's maxSurfaceBounces, whose
//                    path tracer traces that last ray for its emission). An
//                    integer in [1, 14]: a path's length, n + 1, fits the
//                    reservoir's 4 bits.
//   footprint        c of the dual footprint test (Enhanced eq. 5): a pair
//                    (x_{k-1}, x_k) may reconnect only where both footprints
//                    are at least c / 100 of the primary ray's. Finite,
//                    above 0 and at most 10.
//   roughness        alpha_min, the least roughness of the lobe sampled at
//                    x_{k-1} (Enhanced §4.2; the 2022 code's
//                    specularRoughnessThreshold, compared with a material's
//                    roughness, a conductor's sqrt(GGX alpha), a diffuse
//                    lobe's 1, a delta lobe's 0). Finite, within [0, 1].
//   neighbors        the paired neighbors the spatial reuse merges, one per
//                    reuse texture (core/sampling/pairing.h). An integer in
//                    [1, 3] with restir_pt_spatial; 0 without it.
//   sigma            the reuse textures' standard deviation in pixels
//                    (Enhanced §3.1, §7: 16 matches a disc of radius 30).
//                    Read on the host only, where the textures are built.
//                    Finite, within [0.8, 32] with restir_pt_spatial (one
//                    shuffle is the least; a texture of 210 wraps links past
//                    105 pixels); 0 without it.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

SERENITY_CONSTANT uint32_t max_nee_candidates = 32;
SERENITY_CONSTANT uint32_t max_path_surfaces = 14;
SERENITY_CONSTANT float max_footprint = 10.0f;
SERENITY_CONSTANT uint32_t max_path_neighbors = 3;
SERENITY_CONSTANT float min_pairing_sigma = 0.8f;
SERENITY_CONSTANT float max_pairing_sigma = 32.0f;

// The reuse textures' widths, one per neighbor (Enhanced §3.2: three
// different sizes, so their repeats do not line up), in the order the
// shaders and the host read them, and where each starts in the one array
// that holds them all, row by row.
SERENITY_CONSTANT uint32_t pairing_widths[max_path_neighbors] = {254u, 230u, 210u};
SERENITY_CONSTANT uint32_t pairing_starts[max_path_neighbors] = {0u, 254u * 254u, 254u * 254u + 230u * 230u};
SERENITY_CONSTANT uint32_t pairing_entries = 254u * 254u + 230u * 230u + 210u * 210u;

struct PathReuseSettings {
    uint32_t nee_candidates;
    uint32_t max_surfaces;
    float footprint;
    float roughness;
    uint32_t neighbors;
    float sigma;
    uint32_t padding[2];
};

static_assert(sizeof(PathReuseSettings) == 32,
              "PathReuseSettings must be the same 32 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<PathReuseSettings>, "PathReuseSettings is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
