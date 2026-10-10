#pragma once

// Axis: Light.
//
// The light kinds, and the record that says which light a light is: a kind
// and an index into that kind's array, as for shapes, materials and
// textures (Enum.2, C.181). Shared with shaders. Light selection chooses
// among these records and names no kind; the emitter (contract 3) answers
// for a record by its kind, in the Light family (metal/lights/emitter.metal.h).
// A new kind adds a value here, an array of its data, and its emitter; it
// changes neither selection nor any estimator.

#include "core/contracts/shared_layout.h"

namespace serenity {
namespace lights {

enum class LightKind : uint32_t {
    sphere = 0,  // a glowing sphere: the fireflies (sphere_light.h)
};

struct LightRecord {
    LightKind kind;
    uint32_t index;  // into that kind's array
};

static_assert(sizeof(LightRecord) == 8, "LightRecord must be the same 8 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<LightRecord>, "LightRecord is written to the GPU as bytes");
#endif

// Which light, if any, each shape is: one entry per primitive (shapes/
// primitive.h), the index of its light record, or no_light. A path that
// reaches a shape asks this, through the emitter (contract 3, light_at), and
// so learns it reached a light without naming a material.
SERENITY_CONSTANT uint32_t no_light = 0xffffffffu;

// How many lights a scene has, for a shader to choose among: a count is not
// in an array's address.
struct LightCounts {
    uint32_t lights;   // records, of every kind
    uint32_t spheres;  // sphere lights, the length of their array
    uint32_t padding[2];
};

static_assert(sizeof(LightCounts) == 16, "LightCounts must be the same 16 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<LightCounts>, "LightCounts is written to the GPU as bytes");
#endif

}  // namespace lights
}  // namespace serenity
