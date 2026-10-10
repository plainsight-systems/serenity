// Runs the sphere light's shader half (metal/lights/sphere_light.metal.h)
// for tests/gpu/light_test.cpp. The light is its data and its shape's
// transform, as the emitter reads it (emitter.metal.h). Bindings: the output
// at 0, then the light's data, the query and the light's transform
// (probes.h).

#include <metal_stdlib>

#include "core/contracts/emitter.h"
#include "core/lights/sphere_light.h"
#include "metal/device/layout.metal.h"
#include "metal/math/hash.metal.h"
#include "metal/lights/sphere_light.metal.h"
#include "probes.h"

using namespace serenity::shaders;
using serenity::tests::LightDraw;
using serenity::tests::LightFar;
using serenity::tests::LightQuery;
using serenity::tests::LightSampleDraw;

// The direction drawn from sample i of a query.grid x query.grid grid over
// the unit square, and the distance along it to the light's surface.
kernel void light_cone(device LightDraw* out [[buffer(0)]],
                       constant serenity::lights::SphereLightData& data [[buffer(1)]],
                       constant LightQuery& query [[buffer(2)]],
                       constant serenity::contracts::Transform& placed [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    const SphereLight light = sphere_light(data, placed, 1.0f);
    const uint n = query.grid;
    if (i >= n * n) {
        return;
    }
    const LightView view = view_light(light, to_float3(query.point));
    const float2 u = (float2(i % n, i / n) + 0.5f) / float(n);
    const float3 d = direction_to_light(view, u);
    out[i] = LightDraw{to_packed(d), distance_to_light(light, view, d)};
}

// For a light seen from far off: the emitter's sample's pdf, and
// light_pdf() asked of the direction to the light's middle and of one well
// outside it.
kernel void light_far(device LightFar* out [[buffer(0)]],
                      constant serenity::lights::SphereLightData& data [[buffer(1)]],
                      constant LightQuery& query [[buffer(2)]],
                      constant serenity::contracts::Transform& placed [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    const SphereLight light = sphere_light(data, placed, 1.0f);
    if (i != 0) {
        return;
    }
    const float3 point = to_float3(query.point);
    const serenity::contracts::LightSample s = sphere_sample_light(light, point, float2(0.25f, 0.5f));
    const float3 middle = metal::normalize(light.center - point);
    const float3 aside = metal::normalize(middle + float3(0.01f, 0.0f, 0.0f));
    out[0] = LightFar{s.pdf, sphere_light_pdf(light, point, middle), sphere_light_pdf(light, point, aside),
                      s.distance};
}

// Draw i of query.grid draws of the emitter's sample() from query.point,
// from numbers hashed from i, as a path draws them; and pdf() asked of the
// drawn direction.
kernel void light_samples(device LightSampleDraw* out [[buffer(0)]],
                          constant serenity::lights::SphereLightData& data [[buffer(1)]],
                          constant LightQuery& query [[buffer(2)]],
                          constant serenity::contracts::Transform& placed [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= query.grid) {
        return;
    }
    const SphereLight light = sphere_light(data, placed, 1.0f);
    const float3 point = to_float3(query.point);
    const uint h = pcg_hash(i ^ 0x5bd1e995u);
    const serenity::contracts::LightSample s =
        sphere_sample_light(light, point, float2(unit_float(h), unit_float(pcg_hash(h))));
    out[i] = LightSampleDraw{s.direction, s.pdf, s.radiance, s.distance,
                             sphere_light_pdf(light, point, to_float3(s.direction)), {0u, 0u, 0u}};
}
