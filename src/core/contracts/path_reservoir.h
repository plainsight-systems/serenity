#pragma once

// Contract 15: the path reservoir. Owned by Sample reuse; read by Pass and
// the Integrator's shift.
//
// A path reservoir as ReSTIR PT stores it between its passes (and, from
// Phase 2, from one frame to the next): the one path the pixel's path tree
// resampled to, its contribution weight W, the count M it stands for, and
// what the hybrid shift needs to carry that path to another pixel's primary
// hit (docs/research/2026-10-11-restir-pt-layout.md). ReSTIR PT Enhanced's
// 64-byte reservoir (Lin, Kettunen and Wyman 2026, supplemental §8,
// Algorithm 1), adapted to this scene's shapes: a reconnection vertex is a
// primitive and a point in the shape's own coordinates, not an instance,
// triangle and barycentrics.
//
// The path. Vertices x0 (the camera's lens point), x1 (the pixel's primary
// hit), ..., x_d, where the path ends: on a light, which next event
// estimation (NEE) at x_{d-1} drew or a ray x_{d-1}'s BSDF drew struck, or
// in the sky, which a ray from x_{d-1} escaped to. d >= 2: the light the
// camera sees directly (d = 1) is not resampled, the initial pass adds it
// as it is. The reconnection vertex x_k, 2 <= k <= d, is where the hybrid
// shift stops replaying and reconnects (metal/integrator/shift.metal.h); k
// = 0 when the path has none and is replayed whole. The numbers the path
// drew are a function of its stream, the (pixel, frame) whose path tree it
// came from, and a dimension per vertex fixed by the vertex's index
// (metal/integrator/shift.metal.h, "The numbers"), so any pass can draw
// them again from the counter-based sampler (GDSA.3: numbers keyed by
// address).
//
// The fields, 64 bytes, related ones together (C.1), so a gather reads two
// aligned 32-byte halves (GPU.2):
//
//   weight       W, the unbiased contribution weight of the path in the
//                primary sample space: F x W estimates the pixel's light
//                over every path length and technique the tree could
//                choose. Finite, at least 0; 0 for an empty reservoir.
//   integrand    F, RGB: the path's integrand in the primary sample space,
//                its MIS weight times its contribution over its density,
//                without Russian roulette (which only the source density
//                holds: Enhanced supplemental §6). p̂ = luminance(F).
//   seed_pixel   the stream's pixel key, x + 2^16 y (metal/sampler/
//   seed_frame   sampler.metal.h, path_numbers()), and its frame index.
//                Enhanced keeps a seed for the path and one at the
//                reconnection vertex; here a vertex's dimension is fixed by
//                its index, so the stream and k say both, and the word
//                Algorithm 1 spends on the second seed is unused (padding).
//   flags        M in bits 0 to 7 (Algorithm 1 packs it so); d in 8 to 11;
//                k in 12 to 15; the path's end in 16 to 17 (path_end); the
//                lobe sampled at x_{k-1} in 18 to 22 and at x_k in 23 to 27
//                (contract 2's lobe bits; 0 where the vertex's direction was
//                drawn by NEE); the box face of x_k's point in 28 to 30
//                (shape_point_face below). Bit 31 is 0.
//   primitive    x_k's shape (shapes/primitive.h: primitive i is shape i).
//   point        x_k's point in that shape's own coordinates: on a sphere,
//                its direction from the center by the equal-area
//                octahedral map in two 16-bit numbers, as contract 4 stores
//                a light's point (light_sample.h); on a box, the face in
//                the flags and the point's two coordinates across that face
//                as 16-bit fractions of the box's extent. Decoded where it
//                is read, against where the shape is now (GDSA.18).
//   incoming     at x_k, the direction toward x_{k+1}, octahedral 2 x 16
//                bits (k < d; unread at k = d).
//   radiance     what arrives at x_k along `incoming`, divided by every
//                density after x_k, and with every MIS weight a shift does
//                not change: the path's suffix after x_k's own BSDF, which
//                a reconnection keeps as it is (Enhanced Algorithm 1's
//                rcVertexIrradiance). Unread at k = d.
//   jacobian     the reconnection's denominator, cached: the source path's
//                densities of x_k and of x_k's direction, p_{k-1}(w_{k-1})
//                G(x_{k-1} -> x_k) p_k(w_k), the last factor 1 where the
//                direction at x_k was drawn by NEE or k = d (Enhanced
//                supplemental eq. 4; Algorithm 1 keeps the product).
//   light_density  the NEE density, P p_l in solid angle, of the light the
//                path ends on, as x_{d-1} would draw it: what a shift that
//                changes x_{d-1}'s outgoing direction's BSDF density reweighs
//                MIS with (k = d - 1 only; 0 where unread).
//
// Narrowings, each stated where it is packed: M to 8 bits (at most 255;
// Phase 1's M is 1, Phase 2's cap is 20); directions and a sphere's point to
// the octahedral map's 10^-4 radians (contract 4); a box's point to 1 /
// 131070 of its extent (8 um across a 1 m table; metal/shapes/
// shape_point.metal.h). The source's
// own cached values (jacobian, radiance) were computed at the point before
// it was quantized; a shift reconnects to the quantized one. A path shifted
// to its own pixel therefore keeps its F and a Jacobian of 1 to that
// rounding, not to the bit (tests/gpu/restir_pt_test.cpp holds it).
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/float3.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

// How a path ends (flags bits 16 to 17).
enum class PathEnd : uint32_t {
    none = 0,    // an empty reservoir
    nee = 1,     // on a light NEE at x_{d-1} drew
    struck = 2,  // on a light a ray x_{d-1}'s BSDF drew struck
    sky = 3,     // in the sky, a ray from x_{d-1} escaping
};

// The flags' fields (above), named once (ES.45).
SERENITY_CONSTANT uint32_t path_count_mask = 0xffu;
SERENITY_CONSTANT uint32_t path_length_shift = 8u;
SERENITY_CONSTANT uint32_t path_rc_shift = 12u;
SERENITY_CONSTANT uint32_t path_index_mask = 0xfu;  // d and k, after their shifts
SERENITY_CONSTANT uint32_t path_end_shift = 16u;
SERENITY_CONSTANT uint32_t path_end_mask = 0x3u;
SERENITY_CONSTANT uint32_t path_lobe_before_shift = 18u;
SERENITY_CONSTANT uint32_t path_lobe_at_shift = 23u;
SERENITY_CONSTANT uint32_t path_lobe_mask = 0x1fu;  // contract 2's five lobe bits
SERENITY_CONSTANT uint32_t path_face_shift = 28u;
SERENITY_CONSTANT uint32_t path_face_mask = 0x7u;

// The longest path the flags hold: d in 4 bits.
SERENITY_CONSTANT uint32_t path_max_length = 15u;

// The largest M the flags hold.
SERENITY_CONSTANT uint32_t path_max_count = 255u;

// A point on a shape in the shape's own coordinates (above): the point word,
// and a box's face, 0 to 5 (the axis times 2, plus 1 on the axis's maximum
// side); 0 for a sphere.
struct ShapePoint {
    uint32_t point;
    uint32_t face;
};

struct PathReservoir {
    float weight;            // W
    Float3 integrand;        // F, RGB
    uint32_t seed_pixel;     // the stream: its pixel's key
    uint32_t seed_frame;     // and its frame
    uint32_t flags;          // M, d, k, the end, the lobes at x_{k-1} and x_k, x_k's face
    uint32_t primitive;      // x_k's shape
    uint32_t point;          // x_k's point in that shape's coordinates
    uint32_t incoming;       // at x_k toward x_{k+1}, equal-area octahedral
    Float3 radiance;         // the suffix after x_k's BSDF
    float jacobian;          // p_{k-1} G(x_{k-1} -> x_k) p_k, the source's
    float light_density;     // P p_l of the light the path ends on, from x_{d-1}
    uint32_t padding;        // Algorithm 1's second seed: unused (above)
};

static_assert(sizeof(PathReservoir) == 64, "PathReservoir must be the same 64 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<PathReservoir>, "PathReservoir is written to the GPU as bytes");
#endif

// The fields of what a path is, in the flags' order (C.1), packed by
// pack_path_flags() and read back one by one below.
struct PathShape {
    uint32_t count;        // M, at most path_max_count
    uint32_t length;       // d, at most path_max_length
    uint32_t rc;           // k, or 0
    PathEnd end;
    uint32_t lobe_before;  // at x_{k-1}
    uint32_t lobe_at;      // at x_k
    uint32_t face;         // x_k's box face
};

constexpr uint32_t pack_path_flags(PathShape shape) {
    return (shape.count & path_count_mask) | ((shape.length & path_index_mask) << path_length_shift) |
           ((shape.rc & path_index_mask) << path_rc_shift) |
           ((static_cast<uint32_t>(shape.end) & path_end_mask) << path_end_shift) |
           ((shape.lobe_before & path_lobe_mask) << path_lobe_before_shift) |
           ((shape.lobe_at & path_lobe_mask) << path_lobe_at_shift) |
           ((shape.face & path_face_mask) << path_face_shift);
}

constexpr uint32_t path_count(uint32_t flags) {
    return flags & path_count_mask;
}
constexpr uint32_t path_length(uint32_t flags) {
    return (flags >> path_length_shift) & path_index_mask;
}
constexpr uint32_t path_rc(uint32_t flags) {
    return (flags >> path_rc_shift) & path_index_mask;
}
constexpr PathEnd path_end(uint32_t flags) {
    return static_cast<PathEnd>((flags >> path_end_shift) & path_end_mask);
}
constexpr uint32_t path_lobe_before(uint32_t flags) {
    return (flags >> path_lobe_before_shift) & path_lobe_mask;
}
constexpr uint32_t path_lobe_at(uint32_t flags) {
    return (flags >> path_lobe_at_shift) & path_lobe_mask;
}
constexpr uint32_t path_face(uint32_t flags) {
    return (flags >> path_face_shift) & path_face_mask;
}

}  // namespace contracts
}  // namespace serenity
