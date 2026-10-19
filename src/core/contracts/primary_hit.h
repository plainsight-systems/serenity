#pragma once

// Contract 5: film outputs, as ReSTIR PT needs them. Owned by Film; read by
// Pass and the Integrator's shift.
//
// A pixel's primary hit, y1, the surface its camera ray first meets, as a
// G-buffer holds it: written once a frame by ReSTIR PT's initial pass, which
// finds it anyway as the first vertex of the pixel's path tree, and read by
// every shift whose destination is the pixel (metal/integrator/
// shift.metal.h, step 1) and by the merges' similarity tests. ReSTIR PT
// Enhanced (Lin, Kettunen and Wyman 2026, §7) and the 2022 code read a
// V-buffer of primary hits; Phase 1 traced each shift's destination's
// primary ray again and resolved its BSDF again, three times a pixel
// (docs/research/2026-10-11-restir-pt-phase1.md). Decided 2026-10-11 with
// Phase 2 (docs/research/2026-10-11-restir-pt-phase2.md).
//
// The fields, 48 bytes, three 16-byte groups (CACHE.5), read whole by a
// shift (CACHE.4: every field is read together):
//
//   position   y1, world space, at full precision: rays leave from it.
//   normal     its geometric normal, unit, equal-area octahedral in two
//              16-bit numbers (contracts/light_sample.h): which side a
//              direction is on, which way a ray leaving it is offset
//              (metal/integrator/surface.metal.h), the similarity test.
//   bsdf       contract 2's PackedBsdf, the material resolved at y1 (its
//              shading normal in it), packed by Material's one definition
//              (contracts/bsdf.h, pack_bsdf).
//   wo         unit, toward the camera's lens point, octahedral.
//   distance   from the lens point to y1: the similarity test's. 0 where
//              the pixel has no surface a path starts from: its ray left
//              the scene or met a light (bsdf's kind none then too).
//   threshold  the primary footprint's threshold for the reconnection
//              criteria at this pixel (metal/integrator/reconnection.metal.h,
//              footprint_threshold, from the lens point, y1 and its normal).
//   sides      contract 1's flags in bits 0 to 7 (which side the ray
//              arrived from), and the medium inside y1's shape in bits 16
//              to 31 (contract 12; 0xffff for none): where a transmission
//              at y1 enters. A scene with 0xffff media or more is refused
//              where a ReSTIR PT graph's renderer is built
//              (metal/frame/renderer.h, I.5).
//
// What a shift reads is the full-precision surface the initial pass shaded,
// less the packing: wo and the geometric normal turned by under 10^-4
// radians, the BSDF's parameters to a half's precision (bsdf.h states each:
// for a glossy lobe, a turn of under 10^-4 radians; for a color, under
// 5 x 10^-4 of it; the position is exact). The canonical path's own F at its
// own pixel was computed on the unpacked surface; its shifts to neighbors, and
// theirs to it, on the packed one. A bias of the packing's size, as a
// G-buffer's is in production; tests/gpu/restir_pt_test.cpp holds the shifted F
// to the retraced one's. The shading normal is the geometric one for every
// shape so far (contract 1), so the surface a shift rebuilds takes bsdf's
// normal for both.
//
// Cost, at P pixels: 48 bytes a pixel; two arrays in a graph with temporal
// reuse (this frame's and last frame's, MEM.4), one otherwise: 371 MB each
// at 3456 x 2234. Written once a frame; read three times a pixel by the
// spatial shifts and twice by the temporal (metal/passes/restir_pt/
// restir_pt.metal.h counts them), where Phase 1 traced a ray and resolved a
// material each time.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/bsdf.h"
#include "core/contracts/float3.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

// sides' fields (above), named once (ES.45).
SERENITY_CONSTANT uint32_t primary_flags_mask = 0xffu;
SERENITY_CONSTANT uint32_t primary_interior_shift = 16u;
SERENITY_CONSTANT uint32_t primary_no_medium =
    0xffffu;  // contract 12's no_medium, in 16 bits

struct PrimaryHit {
  Float3 position;  // y1, world space, full precision
  uint32_t normal;  // geometric, equal-area octahedral
  PackedBsdf bsdf;  // contract 2; kind none: no surface a path starts from
  uint32_t wo;      // unit, toward the lens point, equal-area octahedral
  float distance;   // from the lens point; 0: no surface
  float threshold;  // the primary footprint's threshold
  uint32_t sides;  // contract 1's flags (bits 0-7), the interior medium (16-31)
};

static_assert(
    sizeof(PrimaryHit) == 48,
    "PrimaryHit must be the same 48 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<PrimaryHit>,
              "PrimaryHit is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
