#pragma once

// The path reuse settings. Owned by Sample reuse; read by Pass, the
// Integrator's shift and the Frame graph. Contract 15's sibling
// (contracts/path_reservoir.h).
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
//   confidence_cap   c_cap, the most a temporal input counts for, as a
//                    multiple of the pixel's own initial reservoir's count
//                    (Lin et al. 2022's temporal history length, 20; ReSTIR
//                    PT Enhanced §5's c_Cap^Default): a path kept for many
//                    frames weighs as at most this many frames' trees. Finite,
//                    within [1, 60] with restir_pt_temporal (a reservoir's
//                    count is 8 bits, and a spatial merge sums four:
//                    4 (1 + 60) < 256); 0 without it.
//   cap_floor        c_min, the cap where every pixel of a temporal input's
//                    17 x 17 neighborhood holds a copy of its path (Enhanced
//                    §5: the duplication map's D = 1). Finite, within [0,
//                    confidence_cap] with restir_pt_duplication; 0 without it.
//   cap_exponent     alpha, how fast the cap falls from confidence_cap to
//                    cap_floor as D grows: c_cap = lerp(confidence_cap,
//                    cap_floor, D^alpha) (Enhanced §5). Finite, within (0, 1]
//                    with restir_pt_duplication; 0 without it.
//   mnee             1 where the graph sets mnee (core/frame/schedule.h,
//                    Schedule::mnee), else 0: caustics by manifold next
//                    event estimation at every receiver of the path tree,
//                    added to the pixel's light and not resampled, and the
//                    paths it counts left out of the tree and of every shift
//                    (metal/passes/restir_pt/restir_pt.metal.h). Not a key of
//                    [restir_pt]: the renderer copies it from the schedule.
//   checkerboard     1 where the graph sets checkerboard (core/frame/
//                    schedule.h, Schedule::checkerboard), else 0: the
//                    passes trace the pixels contract 20 names for the
//                    frame, half of them, their arrays packed and their
//                    grid the packed one (core/contracts/checkerboard.h).
//                    Not a key of [restir_pt]: copied from the schedule, as
//                    mnee is.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/camera.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

SERENITY_CONSTANT uint32_t max_nee_candidates = 32u;
SERENITY_CONSTANT uint32_t max_path_surfaces = 14u;
SERENITY_CONSTANT float max_footprint = 10.0f;
SERENITY_CONSTANT uint32_t max_path_neighbors = 3u;
SERENITY_CONSTANT float min_pairing_sigma = 0.8f;
SERENITY_CONSTANT float max_pairing_sigma = 32.0f;
SERENITY_CONSTANT float max_confidence_cap = 60.0f;

// The duplication map's threadgroup, pixels on a side (metal/reuse/
// duplication.metal.h), which its pass dispatches whole: named here, where
// the host and the shader both read it.
SERENITY_CONSTANT uint32_t duplication_group = 24u;

// The replays' threadgroup (metal/passes/restir_pt/restir_pt.metal.h, the
// replay grid), which the host checks the replay pipelines take: named
// here, where both read it.
SERENITY_CONSTANT uint32_t replay_group = 64u;

// The reuse textures' widths, one per neighbor (Enhanced §3.2: three
// different sizes, so their repeats do not line up), in the order the
// shaders and the host read them, and where each starts in the one array
// that holds them all, row by row.
SERENITY_CONSTANT uint32_t pairing_widths[max_path_neighbors] = {254u, 230u, 210u};
SERENITY_CONSTANT uint32_t pairing_starts[max_path_neighbors] = {
    0u,
    pairing_widths[0] * pairing_widths[0],
    pairing_widths[0] * pairing_widths[0] + pairing_widths[1] * pairing_widths[1],
};
SERENITY_CONSTANT uint32_t pairing_entries = pairing_starts[2] + pairing_widths[2] * pairing_widths[2];
static_assert(max_path_neighbors == 3u, "pairing_starts is written out for three textures");

struct PathReuseSettings {
    uint32_t nee_candidates;
    uint32_t max_surfaces;
    float footprint;
    float roughness;
    uint32_t neighbors;
    float sigma;
    float confidence_cap;
    float cap_floor;
    float cap_exponent;
    uint32_t mnee;
    uint32_t checkerboard;
    uint32_t padding;
};

static_assert(sizeof(PathReuseSettings) == 48,
              "PathReuseSettings must be the same 48 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<PathReuseSettings>, "PathReuseSettings is written to the GPU as bytes");
#endif

// What ReSTIR PT's temporal reuse is told each frame, beside its settings:
// the previous frame's camera, framed for this frame's size, and whether that
// frame is this one's history (1) or the history starts over here (0).
struct PathReuseConstants {
    CameraData previous_camera;
    uint32_t previous;
    uint32_t padding[3];
};

static_assert(sizeof(PathReuseConstants) == 80,
              "PathReuseConstants must be the same 80 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<PathReuseConstants>, "PathReuseConstants is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
