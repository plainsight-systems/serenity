#pragma once

// Contract 5: film outputs, as ReSTIR DI first needs them. Owned by Film;
// read by Pass and Sample reuse.
//
// What a pixel's camera path found at its first surface that can be lit by
// aiming at a light (contracts/bsdf.h, aims_at_lights): through glass and
// off mirrors to it, as RTXDI's primary surface replacement does, and no
// further. The surfaces pass writes them (metal/passes/surfaces/surfaces.h,
// by metal/integrator/restir_di.metal.h's step 1); every later ReSTIR pass
// reads them. Three arrays, one element a pixel, flat, in row-major pixel
// order (GPU.2: neighbouring lanes, neighbouring pixels, neighbouring
// addresses), each split by who reads it (CACHE.6: hot fields apart from
// cold when the reads are memory-bound, and the merges' are):
//
//   SurfaceRecord   what the target function needs of the surface
//                   (metal/integrator/target.metal.h): where it is, the
//                   direction its light leaves toward, and its resolved
//                   BSDF. Read by every pass after the surfaces pass, and
//                   by the spatial merge at k neighbors a pixel: the record
//                   is most of ReSTIR's memory traffic, so it holds that and
//                   nothing else, in 64 bytes (CACHE.4: the merges read
//                   every field of a record together, so one record is one
//                   structure).
//   PathState       what the path's continuation from the surface needs and
//                   the merges do not (metal/integrator/path.metal.h): its
//                   throughput, the light it found on the way, the medium
//                   it is in, its numbers' counter. Read by the shade alone.
//   previous position  where the surface was in the previous frame, a
//                   Float3: the hit's point on its shape, placed by last
//                   frame's transform of that shape (contract 10). For the
//                   temporal merge's reprojection alone, and written only in
//                   a graph that has one.
//
// The record. A pixel whose camera path reaches no such surface (it left
// the scene, reached a light, or roulette or the safety stop ended it among
// glass and mirrors) has a record of kind BsdfKind::none, which no pass
// aims at a light from or reuses (bsdf.h: none has no lobe), and its path
// state holds what it found.
//
//   position  the surface's point, world space, at full precision: the
//             target's distance to a light is taken from it, and a shadow
//             ray leaves from it.
//   wo        unit, toward where the surface's light goes: the camera, or
//             the mirror or glass the camera's path came through. At full
//             precision, not packed: the shade evaluates the BSDF at it,
//             and an f evaluated at a direction other than the path's would
//             be the image of another surface.
//   kind, normal, color, alpha, ior, escape
//             contract 2's Bsdf, field for field, without its 8 bytes of
//             padding, which is where wo's precision comes from: the
//             record is 64 bytes, not 80. metal/integrator/target.metal.h
//             makes a Bsdf of them again (record_bsdf()). A field added to
//             Bsdf fails the assertion on its size below, so the record is
//             revisited with it, not left behind.
//
// A ray leaves the record's surface offset along `normal`, the BSDF's
// shading normal, where the path tracer offsets along the geometric normal
// (metal/integrator/surface.metal.h): equal for every shape kind and
// material so far (contract 1: only a material that perturbs the normal
// makes them differ, and none does). The first that does moves the
// geometric normal into this record, and pays its bytes then. Every ReSTIR
// pass tests visibility from a record the same way, so the ray that decides
// whether a sample is visible from a surface is one ray, whichever pass
// traces it (metal/reuse/unbiased.metal.h).
//
// The path state. Throughput, eta_scale, radiance, medium and the surface's
// count are the path tracer's own (path.metal.h, steps 1 to 8); the numbers'
// counter is where the path's stream stands (metal/sampler/sampler.metal.h),
// so the shade continues the path's numbers rather than starting another
// stream that could repeat them; geometric_normal, flags and interior are
// contract 1's, the fields of the surface the continuation reads (leaving
// it, and which medium a transmission enters).
//
// Cost, per frame, at P pixels: the surfaces pass writes 64 + 64 bytes a
// pixel, and 12 more in a graph with a temporal merge; then the record is
// read once by the candidates, twice by the temporal merge (this pixel's
// and, last frame's, the one it reprojects to), 1 + k times by the spatial
// merge and once by the shade; the path state once, by the shade. At the
// display's 3456 x 2234 (P = 7.7 M), a record array is 494 MB, as is the
// path states'; the window traces fewer pixels than that
// (logical-overview.md, "The frame"). Totals: metal/integrator/
// restir_di.metal.h.
//
// Layout rules as for every shared contract (contracts/frame_constants.h):
// each struct four groups of a packed triple and a word (CACHE.5).

#include "core/contracts/shared_layout.h"
#include "core/contracts/bsdf.h"
#include "core/contracts/float3.h"

namespace serenity {
namespace contracts {

static_assert(sizeof(Bsdf) == 48, "SurfaceRecord copies Bsdf's fields: a field added to Bsdf is added here too");

struct SurfaceRecord {
    Float3 position;  // world space
    BsdfKind kind;    // none: the pixel has no surface to aim at lights from
    Float3 wo;        // unit, toward where the light goes
    float alpha;      // Bsdf::alpha
    Float3 normal;    // Bsdf::normal: unit, outward, the shading normal
    float ior;        // Bsdf::ior
    Float3 color;     // Bsdf::color
    float escape;     // Bsdf::escape
};

static_assert(sizeof(SurfaceRecord) == 64, "SurfaceRecord must be the same 64 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<SurfaceRecord>, "SurfaceRecord is written to the GPU as bytes");
#endif

struct PathState {
    Float3 throughput;        // beta, through the glass and mirrors before the surface
    float eta_scale;          // what transmissions scaled beta by, undone (path.metal.h, step 6)
    Float3 radiance;          // L: the light the path found before the surface
    uint32_t dimension;       // the next number the path's stream draws
    Float3 geometric_normal;  // contract 1's, unit, outward
    uint32_t surface;         // the surface's count along the path, from 0 (roulette starts at 3)
    uint32_t medium;          // the medium on wo's side (contract 12), or no_medium
    uint32_t flags;           // contract 1's: arrived_from_outside, or 0
    uint32_t interior;        // contract 1's: the medium inside the surface's shape
    uint32_t padding;
};

static_assert(sizeof(PathState) == 64, "PathState must be the same 64 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<PathState>, "PathState is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
