#pragma once

// Axis: Shape.
//
// The shape kinds, and the record that says which shape a primitive in the
// acceleration structure is. Shared with shaders (contracts/frame_constants.h
// gives the layout rules).
//
// Every shape is one bounding box in the acceleration structure, and the
// exact hit is the shape kind's own test (metal/acceleration/primitives.h):
// primitive i is the shape PrimitiveRecord i names, a kind and an index into
// that kind's array (Enum.2; a kind and an index rather than a union, C.181).
// A new kind adds a value here, an array of its data (shapes/shapes.h), its
// bounds, and its test; it changes no other kind, and nothing outside the
// Shape family.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "core/contracts/float3.h"

namespace serenity {
namespace shapes {

enum class ShapeKind : uint32_t {
    sphere = 0,
    box = 1,
};

struct PrimitiveRecord {
    ShapeKind kind;
    uint32_t index;  // into that kind's array
};

static_assert(sizeof(PrimitiveRecord) == 8, "PrimitiveRecord must be the same 8 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
// The axis-aligned box a shape occupies: what the acceleration structure
// holds for it. Each kind's header says how its bounds are computed, and
// shapes/shapes.h gives them for every shape, in primitive order. Two packed
// triples, the layout of Metal's and Vulkan's bounding boxes alike.
struct Bounds {
    contracts::Float3 min;
    contracts::Float3 max;
};

static_assert(sizeof(Bounds) == 24, "Bounds must be two packed triples of floats");
#endif

}  // namespace shapes
}  // namespace serenity
