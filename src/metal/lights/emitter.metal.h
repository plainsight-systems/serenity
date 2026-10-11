#pragma once

// Axis: Light (the emitter contract's shader side, contracts/emitter.h).
//
// Contract 3, asked of any light record (core/lights/light.h), answered by
// its kind: the one place a shader dispatches on LightKind, so light
// selection and the estimators name none. Each switch has no default, so a
// kind added to LightKind and not here fails to compile (-Werror).
//
// And contract 4, the light sample by identity (contracts/light_sample.h),
// for ReSTIR DI: the same dispatch, declared at the end, beside contract 3's
// functions and changing none of them (sample_light() draws the same
// numbers into the same directions as before). An identity names its light
// by its record's index, which these read the record by, so Sample reuse
// stores 4 bytes of which light rather than the record's 8. Each kind
// places its own point in the identity's two 16-bit numbers: the sphere a
// unit direction by the equal-area octahedral map, a rectangle, when one
// comes, its (u, v).
//
// Cost: drawing a point is sample_light()'s work, one transform of the
// point into the light's space and its encoding (a few dozen flops);
// evaluating one is its decoding (a square root and a dozen flops), a
// transform, a normalization and the reads sample_light() makes: the 8-byte
// light record, the 16-byte sphere light, its 48-byte transform and 4-byte
// glow, 76 bytes of logical loads from small arrays in the constant address
// space (metal/passes/restir_di/restir_di.metal.h counts them a pixel).

#include <metal_stdlib>

#include "core/contracts/emitter.h"
#include "core/contracts/light_sample.h"
#include "core/contracts/transform.h"
#include "core/lights/light.h"
#include "core/lights/sphere_light.h"
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

// Whether a shape is a light, and if so which: one result (F.21).
struct ShapeLight {
    bool is_light;
    serenity::lights::LightRecord light;  // read only where is_light
};

// Whether shape `primitive` is a light, and if so which.
inline ShapeLight light_at(Lights lights, uint primitive) {
    const uint index = lights.shape_lights[primitive];
    if (index == serenity::lights::no_light) {
        return ShapeLight{false, serenity::lights::LightRecord{}};
    }
    return ShapeLight{true, lights.records[index]};
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

// Contract 4 (above). Declared for review; defined with the ReSTIR
// implementation, and called by nothing until then.

// A point on light record `light`, drawn from `point` as sample_light()
// draws a direction with the same `u`; holding no light (contracts/
// light_sample.h, holds_light) where `point` is inside the light.
inline serenity::contracts::LightPoint draw_light_point(Lights lights, uint light, float3 point, float2 u);

// What the light at `sample` gives `point`, as `lights` place and light it:
// this frame's arrays, or last frame's for the temporal merge's previous
// surface. Precondition: holds_light(sample).
inline serenity::contracts::LightReceived evaluate_light_point(Lights lights, serenity::contracts::LightPoint sample,
                                                               float3 point);

}  // namespace shaders
}  // namespace serenity
