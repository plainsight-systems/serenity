// Runs the camera's rays (metal/camera/thin_lens.metal.h) for
// tests/gpu/camera_test.cpp: thread i makes the ray through image point
// queries[i]'s pixel from its lens point, for an image of image.width x
// image.height. Bindings: the output at 0, then the queries, their count,
// the framed camera and the image (probes.h).

#include <metal_stdlib>

#include "metal/camera/thin_lens.metal.h"
#include "metal/device/layout.metal.h"
#include "probes.h"

using namespace serenity::shaders;
using serenity::tests::CameraImage;
using serenity::tests::CameraProbeRay;
using serenity::tests::CameraQuery;

kernel void camera_probe(device CameraProbeRay* out [[buffer(0)]],
                         device const CameraQuery* queries [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant serenity::contracts::CameraData& camera [[buffer(3)]],
                         constant CameraImage& image [[buffer(4)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < count) {
        const CameraQuery q = queries[i];
        const CameraRay ray = camera_ray(camera, float2(q.pixel_x, q.pixel_y), float2(q.lens_u, q.lens_v),
                                         uint2(image.width, image.height));
        out[i] = CameraProbeRay{to_packed(ray.origin), to_packed(ray.direction)};
    }
}
