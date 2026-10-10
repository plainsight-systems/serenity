#pragma once

// Axis: Integrator (direct light, deterministic).
//
// The light arriving along a camera ray, counting what reaches each surface
// straight from the lights and the sky, and what glossy and delta surfaces
// reflect and refract: the estimator metal/passes/preview/preview.h
// describes, which states what it leaves out. Lights are chosen by the
// every-light selection (light_selection/every_light.metal.h); each sample's
// numbers come from the sampler (sampler/sampler.metal.h), by pixel and
// purpose.
//
// It reads everything through the contracts: surfaces through the surface
// interaction (contract 1) and the BSDF (contract 2, resolve, evaluate,
// sample, lobes), lights through the emitter (contract 3, sample_light,
// emitted, light_at). It names no material kind and no light kind; what it
// does at a surface it decides by the BSDF's lobes alone.
//
// A surface with a delta lobe beside a diffuse or glossy one (a coated
// marble's clear coat over its color) is shaded for its diffuse and glossy
// lobes as any other, and its delta lobe is taken too, as glass's are: the
// BSDF sampled at both ends of u.x, which reach both lobes of a two-lobe
// surface (contract 2), the one whose lobe bits say delta taken, its value
// |cos| its share, shaded as a reflection.
//
// Every stretch the camera ray crosses is dimmed by the medium it is in
// (contract 12), entered through a transmission into a shape and left
// through one out of it, as the path tracer's is (path.metal.h, steps 1 and
// 6). Shadow rays never cross glass (it blocks them), so they cross only
// the medium the surface they leave is in: air, or the medium of a shape
// the light is inside too, dimmed over the shadow ray's length.


#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/emitter.h"
#include "core/contracts/medium.h"
#include "core/contracts/surface_interaction.h"
#include "core/lights/light.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/camera/thin_lens.metal.h"
#include "metal/device/layout.metal.h"
#include "metal/integrator/surface.metal.h"
#include "metal/light_selection/every_light.metal.h"
#include "metal/lights/emitter.metal.h"
#include "metal/lights/gradient_sky.metal.h"
#include "metal/materials/bsdf.metal.h"
#include "metal/materials/resolve.metal.h"
#include "metal/media/media.metal.h"
#include "metal/sampler/sampler.metal.h"
#include "metal/scene/scene_block.metal.h"
#include "metal/shapes/shapes.metal.h"

namespace serenity {
namespace shaders {
namespace direct {

constant constexpr uint max_delta = 8;  // delta surfaces a camera ray passes

// Samples per pixel position, for each estimate.
constant constexpr uint light_samples = 4;    // per light: shadow rays over it
constant constexpr uint diffuse_samples = 2;  // a diffuse lobe's rays toward the sky
constant constexpr uint glossy_samples = 4;   // a glossy lobe's reflected rays

// The far end of [0, 1), 1 - 2^-24: a delta BSDF sampled at u.x = 0 and
// here yields each of its lobes in turn (contract 2: u.x chooses among a
// kind's lobes).
constant constexpr float last_below_one = 0x1.fffffep-1f;

// u.yz for a delta lobe, which has one direction and reads neither: the
// middle of the square.
constant constexpr float2 any_direction = float2(0.5f);

// u for a reflected ray's one shadow ray per light (shade_reflected): (0,
// 0), which the sphere light maps to its middle (sphere_light.metal.h,
// direction_to_light). Contract 3 does not say what u means for a kind.
constant constexpr float2 light_middle = float2(0.0f);

// What a sample is for: each purpose draws its own numbers (sampler.metal.h).
constant constexpr uint purpose_bsdf = 1u;
constant constexpr uint purpose_lobe = 2u;
constant constexpr uint purpose_lights = 16u;  // the first light's; light j's is j after it

inline uint light_purpose(uint j) {
    return purpose_lights + j;
}

// The scene (metal/scene/scene_block.metal.h), its lights chosen among
// all at once.
using Scene = SceneView<EveryLight>;

// Where a sample's numbers come from: the pixel, and which of its positions
// this is, so each estimate's samples spread over all the pixel's positions.
struct PixelSamples {
    uint2 xy;
    uint position;
};

// What a ray reached: where nothing was found, found is false and the
// surface all zeros (ES.20).
struct Reached {
    bool found;
    serenity::contracts::SurfaceInteraction surface;
};

inline Reached reach(Scene scene, metal::raytracing::ray r) {
    const Hit hit = trace(scene.structure, scene.shapes(), r);
    if (!hit.found) {
        return Reached{};
    }
    return Reached{true, surface_interaction(scene.shapes(), hit.primitive, r.origin + hit.t * r.direction,
                                             r.direction)};
}

// Whether a sample is one of a delta lobe.
inline bool is_delta(serenity::contracts::BsdfSample s) {
    return s.pdf > 0.0f && (s.lobe & serenity::contracts::lobe_delta) != 0u;
}

// The light `here` scatters toward wo straight from every light: per light,
// `count` directions the emitter draws over it (contract 3; for a sphere,
// uniformly over its cone), each f(wo, wi) |cos| L / pdf where no shape but
// the light lies before it (glass among them: the light glass would focus is
// a caustic, which this estimator leaves out), averaged, and divided by the
// light's selection probability; each dimmed by here's medium over the
// shadow ray's length (contract 12).
inline float3 from_lights(Scene scene, Shading here, PixelSamples px, uint count) {
    const float3 point = to_float3(here.surface.position);
    const float3 shading = to_float3(here.bsdf.normal);
    float3 total = float3(0.0f);
    for (uint j = 0; j < selected_count(scene.selection()); ++j) {
        const serenity::lights::LightRecord light = selected(scene.selection(), j);
        const float2 offset_2d = sample_offset(px.xy, light_purpose(j));
        float3 sum = float3(0.0f);
        for (uint i = 0; i < count; ++i) {
            const serenity::contracts::LightSample sample =
                sample_light(scene.lights(), light, point, sample_2d(offset_2d, px.position * count + i));
            if (sample.pdf <= 0.0f) {
                continue;
            }
            const float3 wi = to_float3(sample.direction);
            const float3 f = bsdf_evaluate(here.bsdf, here.wo, wi);
            if (!metal::any(f > 0.0f) ||
                occluded(scene.structure, scene.shapes(), leave(here.surface, wi, sample.distance), sample.primitive)) {
                continue;
            }
            sum += f * metal::abs(metal::dot(wi, shading)) * to_float3(sample.radiance) *
                   transmittance(scene.media(), here.medium, sample.distance) / sample.pdf;
        }
        total += sum / float(count) / selection_probability(scene.selection(), j);
    }
    return total;
}

// What a ray reflected from `from` along `direction` shows, shaded more
// simply than what the camera sees (preview.h): a light's glow when
// `see_glow`; a surface with a diffuse lobe by each light toward its
// middle, one shadow ray each, and the sky above it unblocked,
// f(wo, n) pi sky(n); any other surface by the sky along the ray. The ray
// crosses from's medium: what it shows is dimmed over the stretch from
// from's point (contract 12), and the lights its surface sees over theirs.
inline float3 shade_reflected(Scene scene, Shading from, float3 direction, bool see_glow) {
    const Reached at = reach(scene, leave(from.surface, direction));
    if (!at.found) {
        return gradient_sky(scene.sky(), direction);
    }
    const float3 point = to_float3(at.surface.position);
    const float3 dimmed =
        transmittance(scene.media(), from.medium, metal::distance(to_float3(from.surface.position), point));
    float3 color = float3(0.0f);
    if (see_glow) {
        const ShapeLight glowing = light_at(scene.lights(), at.surface.primitive);
        if (glowing.is_light) {
            color += light_emitted(scene.lights(), glowing.light, point, -direction);
        }
    }
    const serenity::contracts::Bsdf bsdf = resolve_bsdf(scene.materials(), scene.textures(), at.surface);
    const uint lobes = bsdf_lobes(bsdf);
    if (lobes == 0u) {
        return dimmed * color;  // scatters nothing: a light's own surface
    }
    if ((lobes & serenity::contracts::lobe_diffuse) == 0u) {
        return dimmed * (color + gradient_sky(scene.sky(), direction));
    }
    const float3 wo = -direction;
    const float3 shading = facing(bsdf, wo);
    for (uint j = 0; j < selected_count(scene.selection()); ++j) {
        const serenity::lights::LightRecord light = selected(scene.selection(), j);
        const serenity::contracts::LightSample sample = sample_light(scene.lights(), light, point, light_middle);
        if (sample.pdf <= 0.0f) {
            continue;
        }
        const float3 wi = to_float3(sample.direction);
        const float3 f = bsdf_evaluate(bsdf, wo, wi);
        if (metal::any(f > 0.0f) &&
            !occluded(scene.structure, scene.shapes(), leave(at.surface, wi, sample.distance), sample.primitive)) {
            color += f * metal::abs(metal::dot(wi, shading)) * to_float3(sample.radiance) *
                     transmittance(scene.media(), from.medium, sample.distance) / sample.pdf /
                     selection_probability(scene.selection(), j);
        }
    }
    return dimmed * (color + bsdf_evaluate(bsdf, wo, shading) * M_PI_F * gradient_sky(scene.sky(), shading));
}

// A delta lobe beside the others (a coat's mirror): found at one end of u.x
// (contract 2), its value |cos| its share, shaded as a reflection; 0 where
// `here` has none.
inline float3 from_delta_lobe(Scene scene, Shading here) {
    if ((bsdf_lobes(here.bsdf) & serenity::contracts::lobe_delta) == 0u) {
        return float3(0.0f);
    }
    const serenity::contracts::BsdfSample first = bsdf_sample(here.bsdf, here.wo, float3(0.0f, any_direction));
    const serenity::contracts::BsdfSample delta =
        is_delta(first) ? first : bsdf_sample(here.bsdf, here.wo, float3(last_below_one, any_direction));
    if (!is_delta(delta)) {
        return float3(0.0f);
    }
    const float3 r = to_float3(delta.direction);
    return to_float3(delta.value) * metal::abs(metal::dot(r, to_float3(here.bsdf.normal))) *
           shade_reflected(scene, here, r, true);
}

// The light `here`, a surface with a lobe that is not delta, sends toward
// wo: from the lights directly, and by rays its BSDF samples (contract 2),
// each weighted value |cos| / pdf. A diffuse lobe's rays bring the sky where
// they escape and nothing where they meet a surface (the preview counts no
// light one rough surface reflects onto another); a glossy lobe's bring what
// they reach, shaded as a reflection, lights not counted again, from_lights
// having counted them. A delta lobe beside these is counted once, by
// from_delta_lobe.
inline float3 shade_scattering(Scene scene, Shading here, PixelSamples px) {
    const uint count =
        (bsdf_lobes(here.bsdf) & serenity::contracts::lobe_glossy) != 0u ? glossy_samples : diffuse_samples;
    const float2 direction_offset = sample_offset(px.xy, purpose_bsdf);
    const float2 lobe_offset = sample_offset(px.xy, purpose_lobe);
    const float3 shading = to_float3(here.bsdf.normal);
    float3 bounced = float3(0.0f);
    for (uint i = 0; i < count; ++i) {
        const uint index = px.position * count + i;
        const float3 u = float3(sample_2d(lobe_offset, index).x, sample_2d(direction_offset, index));
        const serenity::contracts::BsdfSample sample = bsdf_sample(here.bsdf, here.wo, u);
        if (sample.pdf <= 0.0f || (sample.lobe & serenity::contracts::lobe_delta) != 0u) {
            continue;
        }
        const float3 wi = to_float3(sample.direction);
        const float3 weight = to_float3(sample.value) * metal::abs(metal::dot(wi, shading)) / sample.pdf;
        if ((sample.lobe & serenity::contracts::lobe_diffuse) != 0u) {
            if (!occluded(scene.structure, scene.shapes(), leave(here.surface, wi), serenity::contracts::no_primitive)) {
                bounced += weight * gradient_sky(scene.sky(), wi);
            }
        } else {
            bounced += weight * shade_reflected(scene, here, wi, false);
        }
    }
    return from_lights(scene, here, px, light_samples) + from_delta_lobe(scene, here) + bounced / float(count);
}

// The light arriving along the camera's ray. Through surfaces whose lobes
// are all delta (glass), every lobe is taken, each with its probability: the
// BSDF sampled at both ends of u.x yields each lobe, and the product of a
// delta lobe's probability and its weight is value |cos| (contract 2). A
// reflection is shown where the ray arrives from outside, shaded as a
// reflection; the transmitted part continues, up to max_delta surfaces.
inline float3 radiance(Scene scene, CameraRay camera, PixelSamples px) {
    float3 color = float3(0.0f);
    float3 weight = float3(1.0f);
    uint medium = serenity::contracts::no_medium;  // air, at the camera
    metal::raytracing::ray along(camera.origin, camera.direction, 0.0f, unbounded);
    float3 from = camera.origin;  // where the ray truly left from, for a medium's stretch
    for (uint delta = 0; delta < max_delta; ++delta) {
        const Reached at = reach(scene, along);
        if (!at.found) {
            return color + weight * gradient_sky(scene.sky(), along.direction);
        }
        const float3 point = to_float3(at.surface.position);
        weight *= transmittance(scene.media(), medium, metal::distance(from, point));
        const bool entering = (at.surface.flags & serenity::contracts::arrived_from_outside) != 0u;
        const ShapeLight glowing = light_at(scene.lights(), at.surface.primitive);
        if (glowing.is_light) {
            color += weight * light_emitted(scene.lights(), glowing.light, point, -along.direction);
        }
        const Shading here{at.surface, resolve_bsdf(scene.materials(), scene.textures(), at.surface),
                           -along.direction, medium};
        const uint lobes = bsdf_lobes(here.bsdf);
        if (lobes == 0u) {
            return color;  // scatters nothing: a light's own surface
        }
        if (serenity::contracts::aims_at_lights(lobes)) {
            return color + weight * shade_scattering(scene, here, px);
        }

        // Every lobe delta: each at its end of u.x.
        const float3 shading = to_float3(here.bsdf.normal);
        const serenity::contracts::BsdfSample first = bsdf_sample(here.bsdf, here.wo, float3(0.0f, any_direction));
        const serenity::contracts::BsdfSample last =
            bsdf_sample(here.bsdf, here.wo, float3(last_below_one, any_direction));
        const bool two_lobes = first.pdf > 0.0f && last.pdf > 0.0f && first.lobe != last.lobe;
        const bool first_reflects = (first.lobe & serenity::contracts::lobe_reflection) != 0u;
        if (two_lobes && entering) {
            const serenity::contracts::BsdfSample reflected = first_reflects ? first : last;
            const float3 r = to_float3(reflected.direction);
            color += weight * to_float3(reflected.value) * metal::abs(metal::dot(r, shading)) *
                     shade_reflected(scene, here, r, true);
        }
        // What goes on: the lobe that is not the reflection, or the one lobe.
        const serenity::contracts::BsdfSample passed =
            two_lobes ? (first_reflects ? last : first) : (first.pdf > 0.0f ? first : last);
        if (passed.pdf <= 0.0f) {
            return color;
        }
        const float3 d = to_float3(passed.direction);
        weight *= to_float3(passed.value) * metal::abs(metal::dot(d, shading));
        if ((passed.lobe & serenity::contracts::lobe_transmission) != 0u) {
            medium = entering ? at.surface.interior : serenity::contracts::no_medium;
        }
        from = point;
        along = leave(at.surface, d);
    }
    return color;
}

}  // namespace direct
}  // namespace shaders
}  // namespace serenity
