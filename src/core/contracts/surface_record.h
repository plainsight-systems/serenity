#pragma once

// Contract 5: film outputs, as ReSTIR DI first needs them. Owned by Film;
// read by Pass and Sample reuse.
//
// What a pixel's camera path found at its first surface that can be lit by
// aiming at a light (contracts/bsdf.h, aims_at_lights): through glass and
// off mirrors to it, as RTXDI's primary surface replacement does, and no
// further. ReSTIR's initial pass writes them (metal/passes/restir_initial/
// restir_initial.h, by metal/passes/restir_di/restir_di.metal.h's step 1);
// every later ReSTIR pass reads them. Three arrays, one element a pixel,
// flat, in row-major pixel order (GPU.2: neighbouring lanes, neighbouring
// pixels, neighbouring
// addresses), each split by who reads it (CACHE.6: hot fields apart from
// cold when the reads are memory-bound, and the merges' are):
//
//   SurfaceRecord   what the target function needs of the surface
//                   (metal/integrator/target.metal.h): where it is, the
//                   direction its light leaves toward, and its resolved
//                   BSDF. Read by every pass after the initial one, and by
//                   the spatial merge at k neighbors a pixel: the record is
//                   most of ReSTIR's memory traffic, so it holds that and
//                   nothing else, in 80 bytes (CACHE.4: the merges read
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
// glass and mirrors) has a record whose bsdf.kind is none, which no pass
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
//   bsdf      contract 2's Bsdf, whole: the surface's material resolved
//             there, the one definition Material owns (ES.3). A field
//             Material adds to it, in its padding or not, is in the record
//             with no change here, and a change of its size fails the
//             assertion on the record's size below.
//   padding   two words, unread: Bsdf's 48 bytes and two packed triples
//             make 72, and the record is 80, five 16-byte groups, so an
//             array of them keeps every record 16-byte aligned (GPU.2). No
//             layout of one Bsdf definition and the two full-precision
//             triples fits 64: the Bsdf alone is 48.
//
// A ray leaves the record's surface offset along bsdf.normal, the BSDF's
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
// Cost, per frame, at P pixels: the initial pass writes 80 + 64 bytes a
// pixel, and 12 more in a graph with a temporal merge; then the record is
// read twice by the temporal merge (this pixel's and, last frame's, the one
// it reprojects to), up to 1 + 2k times by the spatial merge (its own, and
// each neighbor's in its two walks) and once by the shade; the path state
// once, by the shade. At the display's 3456 x 2234 (P = 7.7 M), a record
// array is 618 MB and the path states' 494 MB; the window traces fewer
// pixels than that (logical-overview.md, "The frame"). Totals:
// metal/passes/restir_di/restir_di.metal.h.
//
// Layout rules as for every shared contract (contracts/frame_constants.h):
// the record the Bsdf's three groups and two groups of a packed triple and
// a word, the path state four such groups (CACHE.5).

#include "core/contracts/shared_layout.h"
#include "core/contracts/bsdf.h"
#include "core/contracts/float3.h"

namespace serenity {
namespace contracts {

struct SurfaceRecord {
    Bsdf bsdf;          // contract 2; kind none: the pixel has no surface to aim at lights from
    Float3 position;    // world space
    uint32_t padding0;
    Float3 wo;          // unit, toward where the light goes
    uint32_t padding1;
};

static_assert(sizeof(SurfaceRecord) == 80, "SurfaceRecord must be the same 80 bytes on the host and in shaders");
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
