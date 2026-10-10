#pragma once

// Contract 9: a texture reference. Owned by Texture; read by Material.
//
// Which texture a material's parameter takes its value from, or none: an
// index into the texture records (textures/texture.h), or no_texture. A
// material's data holds one where a parameter may be textured
// (materials/rough.h), and the texture family evaluates the one it names, so
// the two families meet here and depend on nothing of each other's.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

struct TextureReference {
    uint32_t index;  // into the texture records, or no_texture
};

static_assert(sizeof(TextureReference) == 4, "TextureReference must be the same 4 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<TextureReference>, "TextureReference is written to the GPU as bytes");
#endif

// The index of a reference to no texture. A constant at namespace scope must
// be in the shading language's constant address space.
SERENITY_CONSTANT uint32_t no_texture = 0xffffffffu;

}  // namespace contracts
}  // namespace serenity
