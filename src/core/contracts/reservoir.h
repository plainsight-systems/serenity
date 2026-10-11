#pragma once

// Contract 14: the reservoir. Owned by Sample reuse; read by Pass and the
// Frame graph.
//
// A reservoir as it is stored between ReSTIR DI's passes and from one frame
// to the next: the one light sample it kept, its contribution weight W, the
// count M of candidates it stands for, the target at its own surface, and
// where and when its sample's visibility was last tested. Everything
// resampling streams through it is summarized in this fixed-size state
// (CDSA.23: one that update and merge both act on); what is streamed and
// how two merge is Sample reuse's, in registers (metal/reuse/
// reservoir.metal.h), where the running weight sum lives until the
// reservoir is finished into W.
//
// RTXDI's packed reservoir, field for field as they apply here, 24 bytes in
// six words:
//
//   sample      the light sample by identity, its point quantized (contract
//               4, core/contracts/light_sample.h, 8 bytes): evaluated again
//               wherever and whenever the reservoir is read. Holds no light
//               (light 0) when the reservoir is empty. Written by every
//               pass that writes a reservoir: the initial pass's survivor of
//               its candidates, each merge's survivor.
//   count       M in bits 0 to 23, the candidates the reservoir stands for;
//               bits 24 to 31 unused, 0. Kept when W is set to 0 for
//               visibility. At most (H + 1) M after the temporal merge (its
//               input capped at H x M, and this frame's M) and (k + 1)(H +
//               1) M after the spatial merge: under 7 x 10^4 at the
//               settings' limits (core/passes/restir_di.h), so 24 bits hold
//               it with room, and every value 24 bits hold is exact as a
//               float, where a merge weighs by it. Written by the initial
//               pass (M) and by each merge (its sum).
//               RTXDI packs a sample's last-tested visibility beside M,
//               three 6-bit channels, for partial shadows. Shadows here are
//               binary, and every test that finds a sample hidden sets its
//               W to 0 (hide(), metal/reuse/unbiased.metal.h: visibility
//               reuse's rule, the shade's too), so a reservoir with W > 0
//               holds a sample found visible where `tested` says; a bit for
//               it would always be 1, and is not kept. The 8 bits are where
//               partial visibility goes if translucent shadows come.
//   tested      where and when the sample's visibility was last tested,
//               relative to this reservoir's pixel and frame: the pixel
//               offset, x in bits 0 to 7 and y in bits 8 to 15, each a
//               signed 8-bit number clamped to [-127, 127]; and the age, in
//               frames, in bits 16 to 23, saturating at 255; bits 24 to 31
//               unused, 0. What the shade's final-visibility reuse decides
//               by (metal/passes/restir_di/restir_di.metal.h, step 6; RTXDI's
//               reuseFinalVisibility, its age and distance): a test within
//               the graph's age and distance is taken for this pixel's.
//               Written 0 (here, now) by the initial pass's visibility test,
//               by a raytraced merge's ray from this pixel's surface, and
//               by the shade when it traces; the temporal merge adds 1
//               to the age, and the reprojection's offset to the distance,
//               when its survivor is last frame's; the spatial merge adds
//               the neighbor's offset when its survivor is a neighbor's;
//               the survivor of the pixel's own input keeps its own.
//   target      p̂ of the sample at the reservoir's own surface, as the
//               frame that wrote it lit it (metal/integrator/
//               target.metal.h): what the pass that chose the sample had
//               just computed. A later pass of the same frame, at the same
//               surface, streams the reservoir with it rather than
//               evaluating the target again: the temporal merge its pixel's
//               initial reservoir, the spatial merge its pixel's own input.
//               And a merge whose survivor came from another input takes
//               that input's stored target as its normalization's pi, the
//               survivor's target at the surface it came from as that
//               surface's frame lit it, which is exactly what it stores
//               (metal/reuse/unbiased.metal.h). As this surface's target it
//               is valid only at this surface in the frame that wrote it,
//               and a pass that needs another surface's or frame's
//               evaluates that. Written by every pass that writes a
//               reservoir, 0 for an empty one.
//   weight      W, the contribution weight: for the surface q the reservoir
//               belongs to, W estimates 1 / p(y), the density of the sample
//               in the light's own area, so that f(y) W estimates the
//               integral of f over the samples q's reservoir could hold
//               (Bitterli et al. 2020, section 4; metal/reuse/
//               unbiased.metal.h says how exactly, mode by mode). Finite
//               and at least 0. 0 for an empty reservoir, and for one whose
//               sample a test found hidden. Written by the initial pass and
//               each merge, and set to 0 by the shade when its ray finds
//               the sample hidden, so the next frame does not carry it.
//
// 24 bytes, not 32 (decided 2026-10-10, against 4463eee's 32, with its
// padding): a quarter less to read and write in every pass, which is every
// pass's traffic, RTXDI's size. Stored flat, one a pixel, in row-major
// pixel order (GPU.2), 4-byte aligned: a SIMD group reading its own pixels'
// reservoirs reads one contiguous run of 32 x 24 bytes. A gathered read,
// the spatial merge's at its neighbors, straddles two cache lines for 1 in
// 8 reservoirs on a 128-byte line (1 in 4 on a 64-byte one), which is the
// price of the 8 bytes saved on every read; at the k = 1 neighbor RTXDI
// ships with, one gather a pixel.
//
// Layout rules as for every shared contract (contracts/frame_constants.h):
// six 4-byte words, no padding (CACHE.5).

#include "core/contracts/shared_layout.h"
#include "core/contracts/light_sample.h"

namespace serenity {
namespace contracts {

// Reservoir::count's and Reservoir::tested's bits (above).
SERENITY_CONSTANT uint32_t count_mask = 0x00ffffffu;
SERENITY_CONSTANT uint32_t tested_offset_x_shift = 0u;
SERENITY_CONSTANT uint32_t tested_offset_y_shift = 8u;
SERENITY_CONSTANT uint32_t tested_age_shift = 16u;
SERENITY_CONSTANT uint32_t tested_field_mask = 0xffu;
SERENITY_CONSTANT int32_t tested_max_offset = 127;
SERENITY_CONSTANT uint32_t tested_max_age = 255u;

struct Reservoir {
    LightPoint sample;  // y
    uint32_t count;     // M, bits 0-23
    uint32_t tested;    // the visibility test's offset (bits 0-15) and age (bits 16-23)
    float target;       // p̂ at its own surface, in the frame that wrote it
    float weight;       // W
};

static_assert(sizeof(Reservoir) == 24, "Reservoir must be the same 24 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<Reservoir>, "Reservoir is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
