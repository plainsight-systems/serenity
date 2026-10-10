// Runs the camera's rays (metal/camera/thin_lens.metal.h) for
// tests/gpu/camera_test.cpp: thread i makes the ray through image point
// queries[i].xy from lens point queries[i].zw, of a 1600 x 900 image, and
// writes its origin and direction.

#include <metal_stdlib>

#include "metal/camera/thin_lens.metal.h"

using namespace serenity::shaders;

kernel void camera_probe(device float4* out [[buffer(0)]],
                         device const float4* queries [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant serenity::contracts::CameraData& camera [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < count) {
        const CameraRay ray = camera_ray(camera, queries[i].xy, queries[i].zw, uint2(1600u, 900u));
        out[2 * i] = float4(ray.origin, 0.0f);
        out[2 * i + 1] = float4(ray.direction, 0.0f);
    }
}
