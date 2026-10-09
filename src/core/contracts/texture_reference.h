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

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

namespace serenity {
namespace contracts {

struct TextureReference {
    uint32_t index;  // into the texture records, or no_texture
};

static_assert(sizeof(TextureReference) == 4, "TextureReference must be the same 4 bytes on the host and in shaders");

// The index of a reference to no texture. A constant at namespace scope must
// be in the shading language's constant address space.
#if defined(__METAL_VERSION__)
constant constexpr uint32_t no_texture = 0xffffffffu;
#else
constexpr uint32_t no_texture = 0xffffffffu;
#endif

}  // namespace contracts
}  // namespace serenity
