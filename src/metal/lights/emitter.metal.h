#pragma once

// Axis: Light (the emitter contract's shader side, contracts/emitter.h).
//
// Contract 3, asked of any light record (core/lights/light.h), answered by
// its kind: the one place a shader dispatches on LightKind, so light
// selection and the estimators name none. Each switch has no default, so a
// kind added to LightKind and not here fails to compile (-Werror).
//
// And contract 4, the light sample by identity (contracts/light_sample.h),
// for ReSTIR PT's light vertices: the same dispatch, declared at the end,
// beside contract 3's functions and changing none of them. An identity
// names its light by its record's index, which these read the record by.
// Each kind places its own point in the identity's two 16-bit numbers: the
// sphere a unit direction by the equal-area octahedral map, a rectangle,
// when one comes, its (u, v).
//
// A light is placed once for contract 4's questions (place_light(), below):
// its record, and for a sphere its data, its shape's transform and its
// glow, read into one value, which finding a point on it, evaluating a
// point and the density then read without loading any of it again
// (decided 2026-10-11, by the review of f86e7ae: each question had read the
// record, the sphere, its transform and its glow again).
//
// Cost: keeping a point is one transform of it into the light's space and
// its encoding (a few dozen flops); evaluating one is its decoding (a
// square root and a dozen flops), a transform and a normalization. Placing
// a light reads the 8-byte light record, the 16-byte sphere light, its
// 48-byte transform and its 4-byte glow: 76 bytes of logical loads from
// small arrays in the constant address space, once per placed light.

#include <metal_stdlib>

#include "core/contracts/emitter.h"
#include "core/contracts/light_sample.h"
#include "core/contracts/light_tile.h"
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
// frame's for a shift to last frame's pixel.
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

// The identity of `reached`, a point on `light`, light record `index`: a
// light sample's point, or one a ray struck (metal/integrator/nee.metal.h,
// kept_point). Precondition: `reached` on the light's surface, up to
// rounding.
inline serenity::contracts::LightPoint light_point_at(PlacedLight light, uint index, float3 reached) {
    switch (light.record.kind) {
    case serenity::lights::LightKind::sphere:
        return serenity::contracts::LightPoint{serenity::contracts::light_valid | index,
                                               sphere_point_at(light.placed, reached)};
    }
    return serenity::contracts::LightPoint{0u, 0u};
}

// Which light record shape `primitive` is, or lights::no_light: what a ray
// that struck it found, by index, for light_point_at().
inline uint light_index_at(Lights lights, uint primitive) {
    return lights.shape_lights[primitive];
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

// Contract 3's density for a light point: the density with which
// sample_light() draws, from `point`, the direction toward the point
// `received` describes (evaluate_light_point()), asked of the point rather
// than of a direction, so a point on the light's rim is not lost to the
// rounding of a direction's test against the rim (sphere_point_pdf()).
inline float light_point_pdf(PlacedLight light, float3 point, serenity::contracts::LightReceived received) {
    switch (light.record.kind) {
    case serenity::lights::LightKind::sphere:
        return sphere_point_pdf(light.sphere, point, received);
    }
    return 0.0f;
}

// A placed light's power, what it emits in all, RTXDI's GetPower(): the
// weight ReSTIR PT's selection draws it by (light_selection/
// power_light.metal.h). A sphere's: its surface 4 pi r^2, times pi (a
// Lambertian emitter's radiance to its exitance), times its radiance's
// luminance, at this frame's glow.
inline float light_power(PlacedLight light) {
    switch (light.record.kind) {
    case serenity::lights::LightKind::sphere:
        return sphere_light_power(light.sphere);
    }
    return 0.0f;
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


// Contract 3's presampled form, ReSTIR PT's light tiles (contracts/
// light_tile.h): light record `index` placed and packed as a tile's entry,
// and a direction drawn toward an entry's light from `point` with `u`, as
// sample_light() draws it toward the light the entry was packed from (the
// same SphereLight, so the same numbers give the same direction, distance,
// radiance and density).
inline serenity::contracts::LightTileEntry compact_light(Lights lights, uint index) {
    const PlacedLight placed = place_light(lights, index);
    switch (placed.record.kind) {
    case serenity::lights::LightKind::sphere:
        return serenity::contracts::LightTileEntry{
            to_packed(placed.sphere.center), placed.sphere.radius, to_packed(placed.sphere.radiance),
            (index & serenity::contracts::tile_index_mask) |
                (placed.sphere.primitive << serenity::contracts::tile_shape_shift)};
    }
    return serenity::contracts::LightTileEntry{};
}

inline serenity::contracts::LightSample sample_tile_light(serenity::contracts::LightTileEntry entry, float3 point,
                                                          float2 u) {
    return sphere_sample_light(SphereLight{to_float3(entry.center), entry.radius, to_float3(entry.radiance),
                                           entry.ids >> serenity::contracts::tile_shape_shift},
                               point, u);
}

}  // namespace shaders
}  // namespace serenity
