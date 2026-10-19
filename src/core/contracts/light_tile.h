#pragma once

// Contract 3's presampled form: ReSTIR PT's light tiles. Owned by Light;
// read by the Integrator's next event estimation (metal/integrator/
// nee.metal.h).
//
// Wyman and Panteleev 2021 (RTXDI's presampling), as ReSTIR PT Enhanced runs
// it (Lin, Kettunen and Wyman 2026, §6.1): once a frame, tiles of lights are
// drawn from the light selection's distribution, each light placed and
// packed small; every 8 x 8 block of pixels draws one tile, and its NEE
// candidates draw lights from that tile alone, uniformly. A candidate's
// light is then a draw from the selection's distribution, as without tiles
// (each entry is one, and the entry is chosen uniformly), so its density is
// the selection's: each light's power over the frame's total, drawn from
// the power pyramid below, as RTXDI's RTXDI_PresampleLocalLights draws from
// its PDF texture's mip chain (metal/light_selection/power_light.metal.h);
// what tiles change is where a candidate's light is read from: one 32-byte
// entry of a 32 KB tile the block's threads share, in place of the light
// record, the sphere, its transform and its glow, four dependent reads from
// four arrays (metal/lights/emitter.metal.h, place_light). The candidates
// of one pixel share a tile, so they are not independent; RIS's estimate is
// unbiased with dependent candidates whose marginal densities are the ones
// it weighs by (each candidate's marginal is the selection's), so this
// changes the noise's correlation between a block's pixels, not the mean.
//
// The entry: a sphere light placed by this frame's transform and lit by its
// glow (metal/lights/sphere_light.metal.h, SphereLight), its center, radius
// and radiance; and in `ids` its light record's index in bits 0 to 15 and
// its shape in bits 16 to 31 (the shadow ray ignores it). A scene with more
// than 65536 lights or shapes is refused where a ReSTIR PT graph's renderer
// is built (metal/frame/renderer.h; I.5). One kind of light so far: a kind
// added adds its tag to the entry, and the emitter's switch that packs an
// entry, which has no default, fails to compile until it does
// (metal/lights/emitter.metal.h, compact_light).
//
// Sizes, Enhanced's (§6.1, from RTXDI's defaults): 128 tiles of 1024 lights,
// 4 MB, drawn by one thread an entry; a block of 8 x 8 pixels a tile.
//
// The power pyramid. RTXDI writes each local light's power into a texture
// at its index's place on a Z-curve (RTXDI_LinearIndexToZCurve), builds
// the texture's mip chain, and draws a light by descending it, choosing
// among each texel's four children in proportion to their values
// (RTXDI_SamplePdfMipmap); a light's probability is its texel over the
// texels' sum (the sample's RAB_EvaluateLocalLightSourcePdf). On a Z-curve
// a texel's four children are four consecutive indices, so the chain is a
// quadtree over the lights' indices, and here it is that, in one buffer of
// floats: level 0, the leaves, 4^L of them, leaf i light record i's power
// (0 past the count), L the least with 4^L at least the count; level l + 1
// each four consecutive values of level l summed; the top, level L, one
// value, the frame's total. Levels in order from the leaves, each after
// the last (light_power_offset()). Sums, not RTXDI's means, so the top is
// the total itself and no level's scale needs undoing. Built once a frame
// by one threadgroup of light_power_threads threads, at its glows' powers,
// in a fixed order of additions, so the same frame builds the same bits
// (GDSA.2); two, this frame's and last frame's, which the temporal reuse
// evaluates last frame's paths by (metal/frame/frame_images.h). Floats a
// scene of 616 lights needs: 4^5 + 4^4 + ... + 1 = 1365, 5.3 KB; at most
// 87381 for 65536 lights, 341 KB.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/float3.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

SERENITY_CONSTANT uint32_t light_tiles = 128u;
SERENITY_CONSTANT uint32_t light_tile_size = 1024u;
SERENITY_CONSTANT uint32_t light_tile_block =
    8u;  // pixels on a side of the block that draws one tile

// ids' fields (above), named once (ES.45).
SERENITY_CONSTANT uint32_t tile_index_mask = 0xffffu;
SERENITY_CONSTANT uint32_t tile_shape_shift = 16u;
SERENITY_CONSTANT uint32_t tile_largest_id = 0xffffu;

struct LightTileEntry {
  Float3 center;
  float radius;
  Float3 radiance;  // the glow's this frame
  uint32_t ids;     // the light record's index (bits 0-15), its shape (16-31)
};

static_assert(
    sizeof(LightTileEntry) == 32,
    "LightTileEntry must be the same 32 bytes on the host and in shaders");

// The power pyramid's one threadgroup (above).
SERENITY_CONSTANT uint32_t light_power_threads = 1024u;

// L, the levels above the leaves, for `count` lights: the least with 4^L
// at least the count; 0 for one light or none.
constexpr uint32_t light_power_levels(uint32_t count) {
  uint32_t levels = 0u;
  while ((1u << (2u * levels)) < count) {
    ++levels;
  }
  return levels;
}

// Where level `level` of a pyramid of `levels` begins, in floats: the
// levels below it, 4^levels + 4^(levels - 1) + ... in all.
constexpr uint32_t light_power_offset(uint32_t levels, uint32_t level) {
  uint32_t offset = 0u;
  for (uint32_t below = 0u; below < level; ++below) {
    offset += 1u << (2u * (levels - below));
  }
  return offset;
}

// The floats in all, the top's place plus it.
constexpr uint32_t light_power_floats(uint32_t levels) {
  return light_power_offset(levels, levels) + 1u;
}
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<LightTileEntry>,
              "LightTileEntry is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
