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
// the selection's (1 / N, uniform: metal/light_selection/uniform_light.metal.h);
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
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/float3.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

SERENITY_CONSTANT uint32_t light_tiles = 128u;
SERENITY_CONSTANT uint32_t light_tile_size = 1024u;
SERENITY_CONSTANT uint32_t light_tile_block = 8u;  // pixels on a side of the block that draws one tile

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

static_assert(sizeof(LightTileEntry) == 32, "LightTileEntry must be the same 32 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<LightTileEntry>, "LightTileEntry is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
