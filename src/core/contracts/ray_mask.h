#pragma once

// Contract 19: the ray mask. Owned by Acceleration; read by the Integrator
// and the features pass, the passes that trace.
//
// Which shapes a ray finds (architecture/logical-overview.md, principle 11).
// Every shape is in one class: a light, a shape that is one (contract 3,
// light_at: in this renderer, a firefly), or a surface, every other shape.
// A ray carries a mask, the classes it finds, chosen by the question it
// asks:
//
//   everything   what does this ray reach? The camera's rays and the
//                features', a path's bounces, ReSTIR DI's BSDF candidate,
//                a shift's replay: a ray that may strike a light, as MIS's
//                BSDF strategy and the camera do, and must find it.
//   occluders    is this chosen light visible from here? Every shadow ray
//                toward a light sample (next event estimation in the path
//                tracer, ReSTIR DI and ReSTIR PT, and a shift's
//                reconnection to a light), a shift's reconnection check
//                between two vertices, manifold next event estimation's
//                seed and its solved chain's segments, the preview's test
//                of the sky. Surfaces alone: a light casts no shadow.
//
// One mask per question, everywhere the question is asked: a shift tests a
// path's visibility with the mask the path was sampled with, so ReSTIR's
// reuse sees the integrand it drew from. Changing which question a site
// asks, or adding a mask, is the Integrator's change to this contract;
// keeping each mask, the Acceleration's (below).
//
// What a mask guarantees, the contract the Acceleration keeps: a query
// carrying mask m finds exactly the shapes whose class is in m, each at the
// distance and as the shape (its primitive index, contract 1) it would be
// found at unmasked. So a ray carrying everything is the unmasked ray, bit
// for bit, and a ray carrying occluders never stops at, and never reports,
// a light.
//
// The values are bits, a class each, and a mask is the union of its
// classes' bits, as DXR's InstanceMask, Vulkan's cull mask and Metal's
// instance mask test them (an instance is found when its mask and the
// ray's share a bit; 8 bits in each): so a two-level structure gives each
// instance its class's bit and passes the ray's mask to the query as it
// is (Milestone 4's Step 3). Today's structure is one level of boxes,
// which no API masks per primitive; there each mask is a structure of its
// own (metal/acceleration/scene_acceleration.h). Enum.3: scoped, so no
// integer passes for a mask; Enum.7 and Enum.8: the underlying type and the
// values are the bits a shader and an instance descriptor read (and
// AGENTS.md's deviation for enums shared with Metal). Two named masks, not
// any union of classes (I.4): only these are asked for, and only these the
// one-level structure keeps.

#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

// A shape's class: the bit its instance carries in a two-level structure.
enum class ShapeClass : uint32_t {
    surface = 1u << 0u,  // every shape that is not a light
    light = 1u << 1u,    // a shape that is a light (contract 3, light_at)
};

// The classes a ray finds.
enum class RayMask : uint32_t {
    occluders = static_cast<uint32_t>(ShapeClass::surface),
    everything = static_cast<uint32_t>(ShapeClass::surface) | static_cast<uint32_t>(ShapeClass::light),
};

// Whether a ray carrying `mask` finds a shape of class `shape` (Enum.4: the
// one operation on the two, named, so no caller tests bits).
constexpr bool finds(RayMask mask, ShapeClass shape) {
    return (static_cast<uint32_t>(mask) & static_cast<uint32_t>(shape)) != 0u;
}

}  // namespace contracts
}  // namespace serenity
