#pragma once

// Axis: Light (the emitter contract's shader side, contracts/emitter.h).
//
// Contract 3, asked of any light record (core/lights/light.h), answered by
// its kind: the one place a shader dispatches on LightKind, so light
// selection and the estimators name none. Each switch has no default, so a
// kind added to LightKind and not here fails to compile (-Werror).

#include <metal_stdlib>

#include "core/contracts/emitter.h"
#include "core/lights/light.h"
#include "core/lights/sphere_light.h"
#include "metal/lights/sphere_light.metal.h"

namespace serenity {
namespace shaders {

// Every light, as a shader reads it: one array per light kind, which a
// record indexes.
struct Lights {
    device const serenity::lights::SphereLightData* spheres;
};

inline serenity::contracts::LightSample sample_light(Lights lights, serenity::lights::LightRecord light, float3 point,
                                                     float2 u) {
    switch (light.kind) {
    case serenity::lights::LightKind::sphere:
        return sphere_sample_light(lights.spheres[light.index], point, u);
    }
    return serenity::contracts::LightSample{};
}

// The radiance `light` sends from a point on it along a direction; the
// sphere's is the same for every point and direction.
inline float3 light_emitted(Lights lights, serenity::lights::LightRecord light, float3, float3) {
    switch (light.kind) {
    case serenity::lights::LightKind::sphere:
        return sphere_emitted(lights.spheres[light.index]);
    }
    return float3(0.0f);
}

inline float light_pdf(Lights lights, serenity::lights::LightRecord light, float3 point, float3 direction) {
    switch (light.kind) {
    case serenity::lights::LightKind::sphere:
        return sphere_light_pdf(lights.spheres[light.index], point, direction);
    }
    return 0.0f;
}

inline serenity::contracts::LightExtent light_extent(Lights lights, serenity::lights::LightRecord light,
                                                     float3 point) {
    switch (light.kind) {
    case serenity::lights::LightKind::sphere:
        return sphere_extent(lights.spheres[light.index], point);
    }
    return serenity::contracts::LightExtent{};
}

inline float3 light_irradiance(Lights lights, serenity::lights::LightRecord light, float3 point, float3 normal) {
    switch (light.kind) {
    case serenity::lights::LightKind::sphere:
        return sphere_irradiance(lights.spheres[light.index], point, normal);
    }
    return float3(0.0f);
}

}  // namespace shaders
}  // namespace serenity
