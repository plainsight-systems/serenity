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
// pixels, neighbouring addresses), each split by who reads it (CACHE.6: hot
// fields apart from cold when the reads are memory-bound, and the merges'
// are):
//
//   SurfaceRecord      what the target function needs of the surface
//                      (metal/integrator/target.metal.h), as a G-buffer
//                      holds it: where it is, the direction its light leaves
//                      toward, and its BSDF, packed. Read by every pass
//                      after the initial one, and by the merges at the
//                      pixels they reuse from: the record is the largest
//                      part of ReSTIR's memory traffic, so it holds that and
//                      nothing else, in 32 bytes (CACHE.4: the merges read
//                      every field of a record together, so one record is
//                      one structure).
//   PathState          what the path's continuation from the surface needs
//                      and the merges do not (metal/integrator/
//                      path.metal.h): its throughput, the light it found on
//                      the way, the media, its numbers' counter. Read by the
//                      shade alone, 36 bytes.
//   previous position  where the surface was in the previous frame, a
//                      Float3: the hit's point on its shape, placed by last
//                      frame's transform of that shape (contract 10). For the
//                      temporal merge's reprojection alone, and written only
//                      in a graph that has one.
//
// The record. A pixel whose camera path reaches no such surface (it left
// the scene, reached a light, or roulette or the safety stop ended it among
// glass and mirrors) has a record whose bsdf's kind is none, which no pass
// aims at a light from or reuses (bsdf.h: none has no lobe), and its path
// state holds what it found.
//
//   position  the surface's point, world space, at full precision: the
//             target's distance to a light is taken from it, and a shadow
//             ray leaves from it. Not reconstructed from a depth, as a
//             camera-ray G-buffer's can be: through glass and off mirrors
//             the surface is not on the pixel's camera ray, so its depth
//             along that ray does not place it.
//   wo        unit, toward where the surface's light goes: the camera, or
//             the mirror or glass the camera's path came through. The
//             equal-area octahedral map in two 16-bit numbers
//             (contracts/light_sample.h), within 10^-4 radians
//             (unit_round_trip_error; 6.3 x 10^-5 measured).
//   bsdf      contract 2's PackedBsdf: the surface's material resolved
//             there, packed by Material's one definition (contracts/
//             bsdf.h, pack_bsdf; its shading normal is in it). A field
//             Material adds to the Bsdf, it adds to the packed form there,
//             and a change of that form's size fails the assertion on the
//             record's size below.
//
// What the packing changes, stated: every pass that reads a record, the
// shade included, unpacks the same 32 bytes, so the surface the target
// weighs is the surface the shade shades and the path continues from. That
// surface differs from the full-precision one the reference shades by wo's
// and the normal's quantization and the halves of the BSDF's parameters
// (bsdf.h states each): for a glossy lobe, a turn of under 10^-4 radians;
// for a color, under 5 x 10^-4 of it. A bias in the image, small, as a
// G-buffer's is in production; the raytraced mode's check against the
// reference holds ReSTIR within a tolerance that includes it (core/passes/
// restir_di.h). The position is exact.
//
// A ray leaves the record's surface offset along the unpacked shading
// normal, where the path tracer offsets along the geometric normal
// (metal/integrator/surface.metal.h): equal for every shape kind and
// material so far (contract 1: only a material that perturbs the normal
// makes them differ, and none does). The first that does adds the
// geometric normal, packed, to the path state and to the record if a
// merge's ray needs it, and pays its 4 bytes then. Every ReSTIR pass tests
// visibility from a record the same way, so the ray that decides whether a
// sample is visible from a surface is one ray, whichever pass traces it
// (metal/reuse/unbiased.metal.h).
//
// The path state. Throughput, eta_scale, radiance, the media and the
// surface's count are the path tracer's own (path.metal.h, steps 1 to 8),
// at full precision: they scale and add to everything the pixel shows, and
// are read once a pixel, by the shade, so their bytes are cheap and their
// error would not be. The numbers' counter is where the path's stream
// stands (metal/sampler/sampler.metal.h), so the shade continues the path's
// numbers rather than starting another stream that could repeat them. The
// flags and the interior medium are contract 1's, the fields of the surface
// the continuation reads (which side it was arrived at from, and which
// medium a transmission enters). Packed where the values are small:
//
//   counts  the numbers' counter in bits 0 to 15 (a path's segment to its
//           first rough surface draws under 2^11 numbers, the safety stop's
//           bound for a whole path: path.metal.h), the surface's count
//           along the path in bits 16 to 23 (at most 255: the safety stop
//           ends a path at 256 surfaces), contract 1's flags in bits 24 to
//           31.
//   media   the medium on wo's side in bits 0 to 15, the interior medium in
//           bits 16 to 31 (contract 12), each a medium's index or 0xffff
//           for no_medium. A scene has a handful of media; one with 0xffff
//           or more is refused where a ReSTIR graph's renderer is built
//           (metal/frame/renderer.h, I.5).
//
// The geometric normal is not kept: it equals the record's shading normal
// for every surface so far (above).
//
// Cost, per frame, at P pixels: the initial pass writes 32 + 36 bytes a
// pixel, and 12 more in a graph with a temporal merge; then the record is
// read by the temporal merge at the pixel and at the pixel it reprojects
// to (up to 9 there, when a disocclusion makes it search: metal/reuse/
// temporal.metal.h), by the spatial merge at the pixel and at each of k
// neighbors, and at those again in its normalization's walk, and once by
// the shade; the path state once, by the shade. At the display's
// 3456 x 2234 (P = 7.7 M), a record array is 247 MB and the path states'
// 278 MB; the window traces fewer pixels than that (logical-overview.md,
// "The frame"). Totals: metal/passes/restir_di/restir_di.metal.h.
//
// Layout rules as for every shared contract (contracts/frame_constants.h):
// the record a packed triple and a word, then the packed BSDF's four words,
// two 16-byte groups (CACHE.5); the path state two groups of a packed
// triple and a word, and a word.

#include "core/contracts/shared_layout.h"
#include "core/contracts/bsdf.h"
#include "core/contracts/float3.h"

namespace serenity {
namespace contracts {

struct SurfaceRecord {
    Float3 position;  // world space, full precision
    uint32_t wo;      // unit, toward where the light goes; equal-area octahedral
    PackedBsdf bsdf;  // contract 2; kind none: the pixel has no surface to aim at lights from
};

static_assert(sizeof(SurfaceRecord) == 32, "SurfaceRecord must be the same 32 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<SurfaceRecord>, "SurfaceRecord is written to the GPU as bytes");
#endif

// PathState's packed words' bits (above).
SERENITY_CONSTANT uint32_t path_counter_mask = 0xffffu;
SERENITY_CONSTANT uint32_t path_medium_mask = 0xffffu;  // the medium's 16 bits, at bit 0 of its word
SERENITY_CONSTANT uint32_t path_surface_shift = 16u;
SERENITY_CONSTANT uint32_t path_surface_mask = 0xffu;  // after its shift
SERENITY_CONSTANT uint32_t path_flags_shift = 24u;
SERENITY_CONSTANT uint32_t path_interior_shift = 16u;
SERENITY_CONSTANT uint32_t path_no_medium = 0xffffu;  // contract 12's no_medium, in 16 bits

struct PathState {
    Float3 throughput;  // beta, through the glass and mirrors before the surface
    float eta_scale;    // what transmissions scaled beta by, undone (path.metal.h, step 6)
    Float3 radiance;    // L: the light the path found before the surface
    uint32_t counts;    // the numbers' counter (bits 0-15), the surface's count (16-23), contract 1's flags (24-31)
    uint32_t media;     // the medium on wo's side (bits 0-15), the interior (16-31); 0xffff: none
};

static_assert(sizeof(PathState) == 36, "PathState must be the same 36 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<PathState>, "PathState is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
