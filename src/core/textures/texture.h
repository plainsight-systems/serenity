#pragma once

// Axis: Texture.
//
// The texture kinds, and the record that says which a texture is, as for
// shapes and materials. Shared with shaders. A texture is evaluated at a
// surface interaction and gives a color. Materials name a texture through
// contracts/texture_reference.h, an index into these records.
//
// The point it is evaluated at is given twice: in the world, and in the
// shape's own coordinates, the world point taken back through the shape's
// transform at the frame (contract 10, contracts/transform.h), which the
// material's resolving computes (metal/materials/resolve.metal.h). A kind
// laid on the world reads the first: the checker and the wood, a floor's
// squares and a table's planks, which do not move with what wears them. A
// kind laid on its shape reads the second: the swirl, a marble core's vanes,
// which turn, move and scale with the core, so one swirl serves every core
// of its colors.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

namespace serenity {
namespace textures {

enum class TextureKind : uint32_t {
    checker = 0,  // laid on the world (checker.h)
    wood = 1,     // a plank tabletop, laid on the world (wood.h)
    swirl = 2,    // a marble's core, laid on its shape (swirl.h)
};

struct TextureRecord {
    TextureKind kind;
    uint32_t index;  // into that kind's array
};

static_assert(sizeof(TextureRecord) == 8, "TextureRecord must be the same 8 bytes on the host and in shaders");

}  // namespace textures
}  // namespace serenity
