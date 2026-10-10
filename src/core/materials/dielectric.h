#pragma once

// Axis: Material (dielectric).
//
// Glass: a smooth boundary that reflects and refracts by the Fresnel
// equations, between air and a medium of the given index of refraction. The
// shader half (metal/materials/dielectric.metal.h) holds the Fresnel term and
// the refracted direction, including total internal reflection.
//
// It is the boundary alone. What fills the glass behind it, clear or
// tinted, is the shape's interior medium (contract 12, contracts/medium.h;
// media/absorbing.h), not the material's: a tinted marble is a sphere
// wearing this material with an absorbing medium inside.

#include "core/contracts/shared_layout.h"

namespace serenity {
namespace materials {

struct DielectricData {
    float ior;  // index of refraction inside; outside is 1 (air). Greater than 1.
    uint32_t padding[3];
};

static_assert(sizeof(DielectricData) == 16, "DielectricData must be the same 16 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<DielectricData>, "DielectricData is written to the GPU as bytes");
#endif

}  // namespace materials
}  // namespace serenity
