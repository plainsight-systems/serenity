#pragma once

// Axis: Material (rough).
//
// A rough surface: one that light arriving from anywhere scatters, and that
// a path takes light at rather than passes through (logical-overview.md). Its
// color is a constant, or a texture's, named through a texture reference
// (contracts/texture_reference.h). Its BSDF arrives with light; until
// then a preview shows its color, unlit (metal/passes/preview/preview.h).

#include "core/contracts/float3.h"
#include "core/contracts/texture_reference.h"

namespace serenity {
namespace materials {

struct RoughData {
    contracts::Float3 color;              // used when texture.index is no_texture
    contracts::TextureReference texture;
};

static_assert(sizeof(RoughData) == 16, "RoughData must be the same 16 bytes on the host and in shaders");

}  // namespace materials
}  // namespace serenity
