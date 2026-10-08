#pragma once

// Axis: Material.
//
// The material kinds, and the record that says which a material is: a kind
// and an index into that kind's array (Enum.2, C.181), as shapes are
// (shapes/primitive.h). Shared with shaders. Each kind's data and its BSDF
// are its own files; a new kind changes no other.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

namespace serenity {
namespace materials {

enum class MaterialKind : uint32_t {
    rough = 0,       // takes light: lit, once there is light
    dielectric = 1,  // glass: passed through, reflecting and refracting
};

struct MaterialRecord {
    MaterialKind kind;
    uint32_t index;  // into that kind's array
};

static_assert(sizeof(MaterialRecord) == 8, "MaterialRecord must be the same 8 bytes on the host and in shaders");

}  // namespace materials
}  // namespace serenity
