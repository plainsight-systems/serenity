#pragma once

// Axis: Material (rough).
//
// A rough surface: one that light arriving from anywhere scatters, and that
// a path takes light at rather than passes through (logical-overview.md). Its
// color is a constant, or a texture's, named through a texture reference
// (contracts/texture_reference.h). It scatters light by Lambert's law,
// albedo / pi, which the preview evaluates for the light reaching it
// straight from the lights and the sky (metal/passes/preview/preview.h).

#include "core/contracts/shared_layout.h"
#include "core/contracts/float3.h"
#include "core/contracts/texture_reference.h"

namespace serenity {
namespace materials {

struct RoughData {
    contracts::Float3 color;              // used when texture.index is no_texture
    contracts::TextureReference texture;
};

static_assert(sizeof(RoughData) == 16, "RoughData must be the same 16 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<RoughData>, "RoughData is written to the GPU as bytes");
#endif

}  // namespace materials
}  // namespace serenity
