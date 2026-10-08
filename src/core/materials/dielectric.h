#pragma once

// Axis: Material (dielectric).
//
// Glass: a smooth boundary that reflects and refracts by the Fresnel
// equations, between air and a medium of the given index of refraction. The
// shader half (metal/materials/dielectric.metal.h) holds the Fresnel term and
// the refracted direction, including total internal reflection.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

namespace serenity {
namespace materials {

struct DielectricData {
    float ior;  // index of refraction inside; outside is 1 (air). Greater than 1.
    uint32_t padding[3];
};

static_assert(sizeof(DielectricData) == 16, "DielectricData must be the same 16 bytes on the host and in shaders");

}  // namespace materials
}  // namespace serenity
