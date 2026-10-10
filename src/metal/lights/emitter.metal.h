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
#include "core/contracts/transform.h"
#include "metal/lights/sphere_light.metal.h"

namespace serenity {
namespace shaders {

// Every light, as a shader reads it: the records, which light each shape
// is, and one array per light kind, which a record indexes; the frame's
// shape transforms, where a sphere light finds where it is and how big
// (core/lights/sphere_light.h): the same array, at the same address, that
// hits are placed by, so a light is always where its shape is; and the
// frame's glows, one factor per sphere light on its radiance
// (metal/scene/light_glows.h). Every answer below reads a sphere light
// through sphere_light_at(), so none can see it brighter or elsewhere than
// another.
struct Lights {
    constant serenity::lights::LightRecord* records;
    constant uint* shape_lights;
    constant serenity::lights::SphereLightData* spheres;
    constant serenity::contracts::Transform* transforms;
    constant float* sphere_glows;
};

// Sphere light `index`, where its shape is this frame, as bright as its glow.
inline SphereLight sphere_light_at(Lights lights, uint index) {
    const serenity::lights::SphereLightData light = lights.spheres[index];
    return sphere_light(light, lights.transforms[light.shape], lights.sphere_glows[index]);
}

// Whether shape `primitive` is a light, and if so which, in `light`.
inline bool light_at(Lights lights, uint primitive, thread serenity::lights::LightRecord& light) {
    const uint index = lights.shape_lights[primitive];
    if (index == serenity::lights::no_light) {
        return false;
    }
    light = lights.records[index];
    return true;
}

inline serenity::contracts::LightSample sample_light(Lights lights, serenity::lights::LightRecord light, float3 point,
                                                     float2 u) {
    switch (light.kind) {
    case serenity::lights::LightKind::sphere:
        return sphere_sample_light(sphere_light_at(lights, light.index), point, u);
    }
    return serenity::contracts::LightSample{};
}

// The radiance `light` sends from a point on it along a direction; the
// sphere's is the same for every point and direction.
inline float3 light_emitted(Lights lights, serenity::lights::LightRecord light, float3, float3) {
    switch (light.kind) {
    case serenity::lights::LightKind::sphere:
        return sphere_emitted(sphere_light_at(lights, light.index));
    }
    return float3(0.0f);
}

inline float light_pdf(Lights lights, serenity::lights::LightRecord light, float3 point, float3 direction) {
    switch (light.kind) {
    case serenity::lights::LightKind::sphere:
        return sphere_light_pdf(sphere_light_at(lights, light.index), point, direction);
    }
    return 0.0f;
}

}  // namespace shaders
}  // namespace serenity
