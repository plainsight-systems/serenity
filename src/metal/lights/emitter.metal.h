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
// A light is placed once for contract 4's questions (place_light(), below):
// its record, and for a sphere its data, its shape's transform and its
// glow, read into one value, which drawing a point on it, evaluating a
// point and the density then read without loading any of it again. A
// ReSTIR candidate asks all three of one light, so it loads that light's
// data once, not once per question (decided 2026-10-11, by the review of
// f86e7ae: each question had read the record, the sphere, its transform and
// its glow again, and drawing read the transform twice). The answers are
// the ones the per-question reads gave: the same values in, the same
// arithmetic.
//
// Cost: drawing a point is sample_light()'s work, one transform of the
// point into the light's space and its encoding (a few dozen flops);
// evaluating one is its decoding (a square root and a dozen flops), a
// transform and a normalization. Placing a light reads the 8-byte light
// record, the 16-byte sphere light, its 48-byte transform and its 4-byte
// glow: 76 bytes of logical loads from small arrays in the constant address
// space, once per placed light (metal/passes/restir_di/restir_di.metal.h
// counts them a pixel).

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
// through sphere_light() from the same three entries, by sphere_light_at()
// or place_light(), so none can see it brighter or elsewhere than another.
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

// Contract 4 (above).

// A light record placed for contract 4's questions: the record, and the
// kind's own data as a frame places and lights it. A sphere's is what
// sphere_light_at() reads, and the transform it is read through; a kind
// added adds its own fields here and its case to each switch below.
struct PlacedLight {
    serenity::lights::LightRecord record;
    SphereLight sphere;                     // read where record.kind is sphere
    serenity::contracts::Transform placed;  // the sphere's shape's: where it is, and how big
};

// Light record `light`, placed by `lights`: this frame's arrays, or last
// frame's for the temporal merge's previous surface.
inline PlacedLight place_light(Lights lights, uint light) {
    PlacedLight placed{};
    placed.record = lights.records[light];
    switch (placed.record.kind) {
    case serenity::lights::LightKind::sphere: {
        const serenity::lights::SphereLightData data = lights.spheres[placed.record.index];
        placed.placed = lights.transforms[data.shape];
        placed.sphere = sphere_light(data, placed.placed, lights.sphere_glows[placed.record.index]);
        break;
    }
    }
    return placed;
}

// A point on `light`, light record `index`, drawn from `point` as
// sample_light() draws a direction with the same `u`; holding no light
// (contracts/light_sample.h, holds_light) where `point` is inside the light.
inline serenity::contracts::LightPoint draw_light_point(PlacedLight light, uint index, float3 point, float2 u) {
    switch (light.record.kind) {
    case serenity::lights::LightKind::sphere:
        if (metal::distance(light.sphere.center, point) <= light.sphere.radius) {
            return serenity::contracts::LightPoint{0u, 0u};  // inside it: no sample, as sample_light()
        }
        return serenity::contracts::LightPoint{serenity::contracts::light_valid | index,
                                               sphere_draw_point(light.sphere, light.placed, point, u)};
    }
    return serenity::contracts::LightPoint{0u, 0u};
}

// What `light` at `sample`'s point gives `point`. Precondition:
// holds_light(sample), and `light` placed from sample's record.
inline serenity::contracts::LightReceived evaluate_light_point(PlacedLight light,
                                                               serenity::contracts::LightPoint sample, float3 point) {
    switch (light.record.kind) {
    case serenity::lights::LightKind::sphere:
        return sphere_evaluate_point(light.sphere, light.placed, sample, point);
    }
    return serenity::contracts::LightReceived{};
}

// The same, the light placed from `lights` first: for an evaluation that
// asks nothing else of its light (the shade's, and a merge's target).
// Precondition: holds_light(sample).
inline serenity::contracts::LightReceived evaluate_light_point(Lights lights, serenity::contracts::LightPoint sample,
                                                               float3 point) {
    return evaluate_light_point(place_light(lights, serenity::contracts::light_index(sample)), sample, point);
}

// Contract 3's density for a placed light: as light_pdf() above, the
// density with which sample_light() draws `direction` from `point`.
inline float light_pdf(PlacedLight light, float3 point, float3 direction) {
    switch (light.record.kind) {
    case serenity::lights::LightKind::sphere:
        return sphere_light_pdf(light.sphere, point, direction);
    }
    return 0.0f;
}

}  // namespace shaders
}  // namespace serenity
