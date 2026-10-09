// The preview (preview.h): deterministic ray tracing, lit by the scene's
// glowing spheres and its sky.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "core/lights/gradient_sky.h"
#include "core/lights/sphere_light.h"
#include "core/materials/emissive.h"
#include "core/materials/material.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/camera/pinhole.metal.h"
#include "metal/lights/gradient_sky.metal.h"
#include "metal/lights/sphere_light.metal.h"
#include "metal/materials/conductor.metal.h"
#include "metal/materials/dielectric.metal.h"
#include "metal/materials/rough.metal.h"
#include "metal/sampler/sampler.metal.h"
#include "metal/shapes/shapes.metal.h"
#include "metal/textures/textures.metal.h"

using namespace metal;
using namespace metal::raytracing;
using namespace serenity::shaders;
using serenity::lights::SphereLightData;
using serenity::materials::MaterialKind;
using serenity::materials::MaterialRecord;

namespace {

// How far a ray that leaves a surface starts off it, along the normal, so
// it does not hit that surface again: far below the scene's scale (a scene
// unit is about a meter), far above float rounding there.
constant constexpr float offset = 1e-4f;
constant constexpr uint max_glass = 8;

// Samples per pixel, for each estimate (preview.h).
constant constexpr uint pixel_samples = 4;  // positions within the pixel
constant constexpr uint light_samples = 4;  // per position, per light: shadow rays
constant constexpr uint sky_samples = 2;    // per position: rays toward the sky
constant constexpr uint gloss_samples = 4;  // per position: rays a rough metal reflects

// What a sample is for: each purpose draws its own numbers (sampler.metal.h).
enum Purpose : uint {
    purpose_sky = 1,
    purpose_gloss = 2,
    purpose_light = 16,  // + the light's index
};

struct Scene {
    primitive_acceleration_structure structure;
    Shapes shapes;
    Textures textures;
    device const MaterialRecord* materials;
    device const serenity::materials::RoughData* rough;
    device const serenity::materials::DielectricData* dielectrics;
    device const serenity::materials::ConductorData* conductors;
    device const serenity::materials::EmissiveData* emissives;
    device const SphereLightData* lights;
    uint light_count;
    serenity::lights::GradientSkyData sky;
};

// Where a pixel's samples come from: the pixel, and which of its positions
// this is, so each estimate's samples spread over all the pixel's positions.
struct Pixel {
    uint2 pixel;
    uint position;
};

// What a ray reached.
struct Surface {
    bool found;
    float3 point;
    float3 normal;  // outward
    MaterialRecord material;
    uint primitive;
};

Surface reach(Scene scene, float3 origin, float3 direction) {
    const Hit hit = trace(scene.structure, scene.shapes, origin, direction, 0.0f, INFINITY);
    Surface s;
    s.found = hit.found;
    if (!hit.found) {
        return s;
    }
    s.point = origin + hit.t * direction;
    s.normal = shape_normal(scene.shapes, hit.primitive, s.point);
    s.material = scene.materials[shape_material(scene.shapes, hit.primitive)];
    s.primitive = hit.primitive;
    return s;
}

// Whether `target`, a point on light `light`, is visible from `from`: no
// shape but the light lies between them. Glass is opaque to it:
// light that glass would focus is a caustic, which the preview leaves out.
bool sees_light(Scene scene, float3 from, float3 target, SphereLightData light) {
    const float3 to = target - from;
    const float distance = length(to);
    return !occluded(scene.structure, scene.shapes, from, to / distance, 0.0f, distance, light.primitive);
}

// The fraction of `light` visible from `from`, by `count` shadow rays to
// points across its disk.
float visible_fraction(Scene scene, float3 from, SphereLightData light, LightView view, Pixel px, uint light_index,
                       uint count) {
    const float2 offset_2d = sample_offset(px.pixel, purpose_light + light_index);
    uint seen = 0;
    for (uint i = 0; i < count; ++i) {
        const float2 u = sample_2d(offset_2d, px.position * count + i);
        seen += sees_light(scene, from, point_on_light(light, view, u), light) ? 1u : 0u;
    }
    return float(seen) / float(count);
}

float3 surface_color(Scene scene, Surface s) {
    return rough_color(scene.rough[s.material.index], scene.textures, s.point);
}

// A rough surface's light: from each glowing sphere, by the fraction of it
// visible, and from the sky, by rays toward it that nothing stops. What
// other surfaces reflect onto it is not counted.
float3 shade_rough(Scene scene, Surface s, float3 albedo, Pixel px) {
    const float3 from = s.point + offset * s.normal;
    float3 light_in = float3(0.0f);
    for (uint j = 0; j < scene.light_count; ++j) {
        const SphereLightData light = scene.lights[j];
        const LightView view = view_light(light, s.point);
        const float cos_t = dot(s.normal, view.direction);
        if (cos_t <= 0.0f) {
            continue;
        }
        const float seen = visible_fraction(scene, from, light, view, px, j, light_samples);
        // E / pi = L sin^2 cos (lights/sphere_light.h); times albedo below.
        light_in += to_float3(light.radiance) * view.sin2 * cos_t * seen;
    }
    // Cosine-distributed rays toward the sky: each that leaves the scene
    // brings the sky's radiance, and their mean times the albedo is the
    // sky's contribution.
    const float2 sky_offset = sample_offset(px.pixel, purpose_sky);
    float3 sky_in = float3(0.0f);
    for (uint i = 0; i < sky_samples; ++i) {
        const float3 d = cosine_direction(s.normal, sample_2d(sky_offset, px.position * sky_samples + i));
        if (!occluded(scene.structure, scene.shapes, from, d, 0.0f, INFINITY, ~0u)) {
            sky_in += gradient_sky(scene.sky, d);
        }
    }
    return albedo * (light_in + sky_in / float(sky_samples));
}

// What a reflected ray shows, shaded more simply than what the camera sees:
// a rough surface by its lights, one shadow ray each to the center, and the
// unblocked sky above it; a glowing sphere by its glow when `see_glow`;
// metal and glass by the sky in the ray's direction.
float3 shade_reflected(Scene scene, float3 origin, float3 direction, bool see_glow) {
    const Surface s = reach(scene, origin, direction);
    if (!s.found) {
        return gradient_sky(scene.sky, direction);
    }
    switch (s.material.kind) {
    case MaterialKind::emissive:
        return see_glow ? to_float3(scene.emissives[s.material.index].radiance) : float3(0.0f);
    case MaterialKind::rough: {
        const float3 normal = dot(s.normal, direction) > 0.0f ? -s.normal : s.normal;
        const float3 from = s.point + offset * normal;
        float3 light_in = gradient_sky(scene.sky, normal);
        for (uint j = 0; j < scene.light_count; ++j) {
            const SphereLightData light = scene.lights[j];
            const LightView view = view_light(light, s.point);
            const float cos_t = dot(normal, view.direction);
            if (cos_t > 0.0f && sees_light(scene, from, to_float3(light.center), light)) {
                light_in += to_float3(light.radiance) * view.sin2 * cos_t;
            }
        }
        return surface_color(scene, s) * light_in;
    }
    case MaterialKind::conductor:
    case MaterialKind::dielectric:
        return gradient_sky(scene.sky, direction);
    }
    return float3(0.0f);
}

// A metal's light: the glowing spheres' highlights, each through GGX widened
// by the light's size (alpha' = alpha + r / 2d, so a near light's highlight
// is no smaller than the light) and by the fraction of it visible; and the
// rest of the scene, by rays the metal's microfacets reflect.
float3 shade_conductor(Scene scene, Surface s, float3 toward_eye, Pixel px) {
    const serenity::materials::ConductorData conductor = scene.conductors[s.material.index];
    const float alpha = max(conductor.roughness * conductor.roughness, 1e-3f);
    const float3 from = s.point + offset * s.normal;

    float3 color = float3(0.0f);
    for (uint j = 0; j < scene.light_count; ++j) {
        const SphereLightData light = scene.lights[j];
        const LightView view = view_light(light, s.point);
        const float widened = min(1.0f, alpha + light.radius / (2.0f * view.distance));
        const float3 reflectance = conductor_reflectance(conductor, widened, s.normal, toward_eye, view.direction);
        if (all(reflectance == 0.0f)) {
            continue;
        }
        const float seen = visible_fraction(scene, from, light, view, px, j, light_samples);
        color += reflectance * to_float3(light.radiance) * view.solid_angle * seen;
    }

    // The glow is not counted again here: the highlights above are it.
    const float2 gloss_offset = sample_offset(px.pixel, purpose_gloss);
    const float n_v = dot(s.normal, toward_eye);
    float3 reflected = float3(0.0f);
    for (uint i = 0; i < gloss_samples; ++i) {
        const float3 h =
            sample_visible_normal(s.normal, toward_eye, alpha, sample_2d(gloss_offset, px.position * gloss_samples + i));
        const float3 l = reflect(-toward_eye, h);
        const float n_l = dot(s.normal, l);
        if (n_l <= 0.0f) {
            continue;
        }
        const float3 weight = schlick(to_float3(conductor.f0), dot(toward_eye, h)) * smith_g2(n_l, n_v, alpha) /
                              smith_g1(n_v, alpha);
        reflected += weight * shade_reflected(scene, from, l, false);
    }
    return color + reflected / float(gloss_samples);
}

// The light arriving along one camera ray.
float3 radiance(Scene scene, float3 origin, float3 direction, Pixel px) {
    float3 color = float3(0.0f);
    float3 weight = float3(1.0f);
    for (uint glass = 0; glass < max_glass;) {
        const Surface s = reach(scene, origin, direction);
        if (!s.found) {
            return color + weight * gradient_sky(scene.sky, direction);
        }
        switch (s.material.kind) {
        case MaterialKind::emissive:
            return color + weight * to_float3(scene.emissives[s.material.index].radiance);
        case MaterialKind::rough: {
            Surface facing = s;
            facing.normal = dot(s.normal, direction) > 0.0f ? -s.normal : s.normal;
            return color + weight * shade_rough(scene, facing, surface_color(scene, s), px);
        }
        case MaterialKind::conductor:
            return color + weight * shade_conductor(scene, s, -direction, px);
        case MaterialKind::dielectric: {
            const Boundary b = dielectric_boundary(scene.dielectrics[s.material.index], direction, s.normal);
            ++glass;
            if (b.total_internal) {
                origin = s.point + offset * b.facing;
                direction = b.reflected;
                break;
            }
            if (b.entering) {
                color += weight * b.reflectance * shade_reflected(scene, s.point + offset * b.facing, b.reflected, true);
            }
            weight *= 1.0f - b.reflectance;
            origin = s.point - offset * b.facing;
            direction = b.refracted;
            break;
        }
        }
    }
    return color;
}

// For display: a color brighter than the display shows is scaled down by
// its largest channel, so it keeps its hue (a firefly stays yellow rather
// than clipping to white); then linear to sRGB's transfer function, for an
// 8-bit target the display reads as sRGB.
float3 encode_srgb(float3 linear) {
    const float largest = max(max(linear.r, linear.g), linear.b);
    const float3 c = saturate(largest > 1.0f ? linear / largest : linear);
    return select(1.055f * pow(c, 1.0f / 2.4f) - 0.055f, 12.92f * c, c <= 0.0031308f);
}

// The pixel's sample positions: a rotated grid, the four points that
// separate both rows and columns of near-horizontal and near-vertical edges.
constant constexpr float2 positions[pixel_samples] = {
    float2(0.375f, 0.125f), float2(0.875f, 0.375f), float2(0.625f, 0.875f), float2(0.125f, 0.625f)};

}  // namespace

kernel void preview(constant serenity::contracts::FrameConstants& frame [[buffer(0)]],
                    constant serenity::contracts::CameraData& camera [[buffer(1)]],
                    primitive_acceleration_structure structure [[buffer(2)]],
                    constant serenity::lights::GradientSkyData& sky [[buffer(3)]],
                    device const serenity::textures::TextureRecord* texture_records [[buffer(4)]],
                    device const serenity::textures::CheckerData* checkers [[buffer(5)]],
                    device const MaterialRecord* materials [[buffer(6)]],
                    device const serenity::materials::RoughData* rough [[buffer(7)]],
                    device const serenity::materials::DielectricData* dielectrics [[buffer(8)]],
                    device const serenity::materials::ConductorData* conductors [[buffer(9)]],
                    device const serenity::materials::EmissiveData* emissives [[buffer(10)]],
                    device const serenity::shapes::PrimitiveRecord* shape_records [[buffer(11)]],
                    device const serenity::shapes::SphereData* spheres [[buffer(12)]],
                    device const serenity::shapes::BoxData* boxes [[buffer(13)]],
                    device const SphereLightData* lights [[buffer(14)]],
                    constant serenity::lights::LightCounts& light_counts [[buffer(15)]],
                    texture2d<float, access::write> target [[texture(0)]],
                    uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= frame.width || pixel.y >= frame.height) {
        return;
    }
    Scene scene;
    scene.structure = structure;
    scene.shapes = Shapes{shape_records, spheres, boxes};
    scene.textures = Textures{texture_records, checkers};
    scene.materials = materials;
    scene.rough = rough;
    scene.dielectrics = dielectrics;
    scene.conductors = conductors;
    scene.emissives = emissives;
    scene.lights = lights;
    scene.light_count = light_counts.spheres;
    scene.sky = sky;

    float3 color = float3(0.0f);
    for (uint i = 0; i < pixel_samples; ++i) {
        const float3 direction = pinhole_direction(camera, float2(pixel) + positions[i], frame.width, frame.height);
        color += radiance(scene, to_float3(camera.origin), direction, Pixel{pixel, i});
    }
    target.write(float4(encode_srgb(color / float(pixel_samples)), 1.0f), pixel);
}
