// Runs the sphere light's shader half (metal/lights/sphere_light.metal.h) on
// a grid of samples, for tests/gpu/light_test.cpp: thread i writes the
// direction drawn from sample i of an n x n grid over the unit square, and
// the distance along it to the light's surface.

#include <metal_stdlib>

#include "core/lights/sphere_light.h"
#include "metal/lights/sphere_light.metal.h"

using namespace serenity::shaders;

kernel void light_cone(device float4* out [[buffer(0)]],
                       constant serenity::lights::SphereLightData& light [[buffer(1)]],
                       constant float4& point_and_n [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    const uint n = uint(point_and_n.w);
    if (i >= n * n) {
        return;
    }
    const LightView view = view_light(light, point_and_n.xyz);
    const float2 u = (float2(i % n, i / n) + 0.5f) / float(n);
    const float3 d = direction_to_light(view, u);
    out[i] = float4(d, distance_to_light(light, view, d));
}
