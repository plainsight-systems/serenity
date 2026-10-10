#pragma once

// Axis: Material (emissive).
//
// A surface that gives light: the radiance it sends every way, in linear
// RGB, and may exceed 1 (a firefly's glow is far brighter than what it
// lights). Only a sphere may wear it, and each sphere that does is a light
// (lights/sphere_light.h); the scene reader refuses it on any other shape.

#include "core/contracts/shared_layout.h"
#include "core/contracts/float3.h"

namespace serenity {
namespace materials {

struct EmissiveData {
    contracts::Float3 radiance;  // each at least 0
    uint32_t padding;
};

static_assert(sizeof(EmissiveData) == 16, "EmissiveData must be the same 16 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<EmissiveData>, "EmissiveData is written to the GPU as bytes");
#endif

}  // namespace materials
}  // namespace serenity
