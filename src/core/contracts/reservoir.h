#pragma once

// Contract 14: the reservoir. Owned by Sample reuse; read by Pass and the
// Frame graph.
//
// A reservoir as it is stored between ReSTIR DI's passes and from one frame
// to the next: the one light sample it kept, its contribution weight W, and
// the count M of candidates it stands for. Everything resampling streams
// through it is summarized in these 16 bytes (CDSA.23: a fixed-size state
// that update and merge both act on); what is streamed and how two merge is
// Sample reuse's, in registers (metal/reuse/reservoir.metal.h), where the
// running weight sum lives until the reservoir is finished into W.
//
//   sample  the light sample, by identity (contract 4,
//           core/contracts/light_sample.h): evaluated again wherever and
//           whenever the reservoir is read. light no_light when it holds
//           none.
//   weight  W, the contribution weight: for the surface q the reservoir
//           belongs to, W estimates 1 / p(y), the density of the sample in
//           the light's own area, so that f(y) W is an unbiased estimate of
//           the integral of f over the samples q's reservoir could hold
//           (Bitterli et al. 2020, section 4; metal/reuse/unbiased.metal.h).
//           Finite and at least 0. 0 for a reservoir that holds no sample,
//           and for one whose sample is hidden from q: every reservoir any
//           pass writes holds a sample visible from its surface or weighs it
//           0 (metal/reuse/unbiased.metal.h states why that invariant is
//           what makes the merges unbiased).
//   count   M, the candidates the reservoir stands for, kept when W is set
//           to 0 for visibility (the paper's visibility reuse): counted by
//           a later merge's Z wherever the reservoir's surface could have
//           produced that merge's sample. At most (H + 1) x M after the
//           temporal merge (its input capped at H x M, and this frame's M),
//           and (k + 1) (H + 1) M after the spatial merge's sum: under
//           7 x 10^4 at the settings' limits (core/passes/restir_di.h),
//           well within 32 bits, and exact as a float where a merge
//           weighs by it.
//
// Stored flat, one a pixel, in row-major pixel order, 16 bytes aligned to
// 16, so a SIMD group's 32 lanes read 512 contiguous bytes for their own
// pixels (GPU.2); the spatial merge's reads of neighbors' reservoirs are
// gathers, counted in metal/reuse/spatial.metal.h.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/shared_layout.h"
#include "core/contracts/light_sample.h"

namespace serenity {
namespace contracts {

struct Reservoir {
    LightPoint sample;  // y
    float weight;       // W
    uint32_t count;     // M
};

static_assert(sizeof(Reservoir) == 16, "Reservoir must be the same 16 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<Reservoir>, "Reservoir is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
