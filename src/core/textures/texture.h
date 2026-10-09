#pragma once

// Axis: Texture.
//
// The texture kinds, and the record that says which a texture is, as for
// shapes and materials. Shared with shaders. A texture is evaluated at a
// surface interaction and gives a color. Materials name a texture through
// contracts/texture_reference.h, an index into these records.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

namespace serenity {
namespace textures {

enum class TextureKind : uint32_t {
    checker = 0,
};

struct TextureRecord {
    TextureKind kind;
    uint32_t index;  // into that kind's array
};

static_assert(sizeof(TextureRecord) == 8, "TextureRecord must be the same 8 bytes on the host and in shaders");

}  // namespace textures
}  // namespace serenity
