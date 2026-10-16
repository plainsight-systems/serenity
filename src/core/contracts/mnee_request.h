#pragma once

// Contract 16: manifold next event estimation's work list. Owned by the
// Integrator (metal/integrator/mnee.metal.h, which states the algorithm);
// written by the estimators' loops, read by the MNEE pass's kernels
// (metal/passes/mnee/mnee.h).
//
// Why a list (decided 2026-10-12, by measurement: docs/research/
// 2026-10-12-mnee.md). Solved where the light sample is drawn, inside the
// path loop, MNEE made the path pass 4.1 times as long (30 to 124 ms at
// 3456 x 2234): a tenth of the paths solve, so nearly every SIMD group of 32
// carries one, and its other lanes wait through the walk's iterations and
// its rays (GPU.4). Cycles moved its MNEE out of its surface kernel into a
// kernel of its own for the same reason (intersect_mnee); so here: the loop
// writes a request where the straight segment crosses casters, and the MNEE
// pass solves the requests together, one thread each, grouped by being
// requests (GDSA.13), and adds each one's light to its pixel.
//
// A request: what the solve and its contribution read of the receiver (its
// point the rays leave from, wo, the medium on wo's side, its BSDF packed
// as contract 2 packs it, whose normal is the geometric one for every shape
// so far), of the light sample (direction, distance, radiance, the light's
// shape), the seed's crossings (their t along the segment and their shapes,
// mnee_shape_bits each, two a word), the pixel, and the weight its light is multiplied by: the
// path's throughput over the sample's density (P p_l in the path tracer;
// over survival and times W_RIS in ReSTIR PT). 128 bytes.
//
// The list's header, one small shared buffer of mnee_header_bytes the CPU
// writes once: the count (mnee_count_word, the writers' atomic, set back to
// 0 by the grid kernel), the capacity (mnee_capacity_word), and from
// mnee_grid_offset the MNEE pass's indirect grid (metal/device/
// indirect_grid.h), its threadgroups of mnee_solve_group, which the host
// checks the solve's pipeline takes. Capacity: one request a pixel. On the
// marbles a tenth of the paths write one (1920 x 1080, t = 120); in the
// caustic test scene, a glass sphere filling the image, a quarter of the
// pixels' worth overflowed, so one a pixel, 128 bytes, 988 MB at 3456 x
// 2234. A request past it is not written, and not lost silently: every
// writer adds to the count, so the count passes the capacity by the
// requests that found no room, and the grid kernel adds those to the
// frame's count of lost requests (metal/film/failure_count.h), which the
// window and the headless renderer report apart from samples that are not
// finite. Until 2026-10-16 a lost request made its path's sample a NaN, so
// the frame's count of samples not finite showed it; under the shaders'
// fast math a NaN need not survive (docs/research/
// 2026-10-16-codebase-guidelines-pass.md, C14).
//
// The requests' light reaches a pixel by float atomics in the order the
// solves arrive in: from three requests a pixel the sum's last bits are not
// guaranteed run to run (metal/passes/mnee/mnee.h, "Determinism").
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/bsdf.h"
#include "core/contracts/float3.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

SERENITY_CONSTANT uint32_t mnee_most_crossings = 4u;   // metal/integrator/mnee.metal.h, max_interfaces
SERENITY_CONSTANT uint32_t mnee_shape_bits = 16u;      // a crossing's shape: half a word
SERENITY_CONSTANT uint32_t mnee_largest_shape = (1u << mnee_shape_bits) - 1u;  // 0xffff, and the half's mask
SERENITY_CONSTANT uint32_t mnee_shapes_a_word = 32u / mnee_shape_bits;  // 2

// The header (above): its words, the grid's place, its bytes; the solve's
// threadgroup.
SERENITY_CONSTANT uint32_t mnee_count_word = 0u;
SERENITY_CONSTANT uint32_t mnee_capacity_word = 1u;
SERENITY_CONSTANT uint32_t mnee_grid_offset = 64u;     // bytes into the header
SERENITY_CONSTANT uint32_t mnee_header_bytes = 128u;
SERENITY_CONSTANT uint32_t mnee_solve_group = 64u;     // threads

static_assert(mnee_most_crossings % mnee_shapes_a_word == 0u, "a request's shapes fill whole words");
static_assert((mnee_capacity_word + 1u) * 4u <= mnee_grid_offset, "the header's words lie before its grid");

struct MneeRequest {
    Float3 origin;      // where the receiver's rays leave it toward wo
    uint32_t pixel;     // y * width + x
    Float3 weight;      // the light's multiplier (above)
    uint32_t light;     // the light's shape
    Float3 direction;   // unit, toward the light's point
    float distance;     // to it
    Float3 radiance;    // the light's, as this frame lights it
    uint32_t medium;    // on wo's side: an index into the media, or no_medium
    Float3 wo;          // unit, toward where the light goes
    uint32_t crossings; // how many, 1 to mnee_most_crossings
    float t[mnee_most_crossings];         // each crossing's distance along the segment, nearest first
    uint32_t shapes[mnee_most_crossings / mnee_shapes_a_word];  // each crossing's shape, a half a word, the first low
    uint32_t padding[2];
    PackedBsdf bsdf;    // the receiver's
};

static_assert(sizeof(MneeRequest) == 128, "MneeRequest must be the same 128 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<MneeRequest>, "MneeRequest is written by the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
