// Runs the sphere light's shader half (metal/lights/sphere_light.metal.h) on
// a grid of samples, for tests/gpu/light_test.cpp: thread i writes the
// direction drawn from sample i of an n x n grid over the unit square, and
// the distance along it to the light's surface. The light is its data and
// its shape's transform, as the emitter reads it (emitter.metal.h).

#include <metal_stdlib>

#include "core/contracts/emitter.h"
#include "core/lights/sphere_light.h"
#include "metal/lights/sphere_light.metal.h"

using namespace serenity::shaders;

kernel void light_cone(device float4* out [[buffer(0)]],
                       constant serenity::lights::SphereLightData& data [[buffer(1)]],
                       constant float4& point_and_n [[buffer(2)]],
                       constant serenity::contracts::Transform& placed [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    const SphereLight light = sphere_light(data, placed, 1.0f);
    const uint n = uint(point_and_n.w);
    if (i >= n * n) {
        return;
    }
    const LightView view = view_light(light, point_and_n.xyz);
    const float2 u = (float2(i % n, i / n) + 0.5f) / float(n);
    const float3 d = direction_to_light(view, u);
    out[i] = float4(d, distance_to_light(light, view, d));
}

// For a light seen from far off: the emitter's sample's pdf, and
// light_pdf() asked of the direction to the light's middle and of one well
// outside it.
kernel void light_far(device float4* out [[buffer(0)]],
                      constant serenity::lights::SphereLightData& data [[buffer(1)]],
                      constant float4& point [[buffer(2)]],
                      constant serenity::contracts::Transform& placed [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    const SphereLight light = sphere_light(data, placed, 1.0f);
    if (i != 0) {
        return;
    }
    const serenity::contracts::LightSample s = sphere_sample_light(light, point.xyz, float2(0.25f, 0.5f));
    const float3 middle = metal::normalize(light.center - point.xyz);
    const float3 aside = metal::normalize(middle + float3(0.01f, 0.0f, 0.0f));
    out[0] = float4(s.pdf, sphere_light_pdf(light, point.xyz, middle), sphere_light_pdf(light, point.xyz, aside),
                    s.distance);
}
