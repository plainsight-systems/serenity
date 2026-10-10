#pragma once

// Axis: Material (dielectric).
//
// Glass: a smooth boundary that reflects and refracts by the Fresnel
// equations, between air and a medium of the given index of refraction. The
// shader half (metal/materials/dielectric.metal.h) holds the Fresnel term and
// the refracted direction, including total internal reflection.
//
// Tinted glass absorbs as light crosses it: by Beer and Lambert, light
// travelling a distance t inside keeps exp(-absorption t) of itself, per
// channel, so thick glass is deeper in color than thin, and a marble's
// middle deeper than its rim. Clear glass absorbs nothing. A scene gives the
// tint as the color white light keeps through `tint_distance` meters of the
// glass (core/scene/scene.h), absorption = -ln(tint) / tint_distance. The
// integrators carry the absorption of the glass a path is inside and apply
// it to every stretch of the path there (metal/integrator/path.metal.h,
// direct.metal.h): one glass at a time, so glass inside glass is not
// modelled, and an opaque core inside glass is (scenes/marbles.toml).

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "core/contracts/float3.h"

namespace serenity {
namespace materials {

struct DielectricData {
    contracts::Float3 absorption;  // per meter, each 0 or more; 0 for clear glass
    float ior;                     // index of refraction inside; outside is 1 (air). Greater than 1.
};

static_assert(sizeof(DielectricData) == 16, "DielectricData must be the same 16 bytes on the host and in shaders");

}  // namespace materials
}  // namespace serenity
