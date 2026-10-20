// Runs the camera's rays (cuda/camera/thin_lens.cuh) for
// tests/gpu/camera_test.cpp: thread i makes the ray through image point
// queries[i]'s pixel from its lens point, for an image of image.width x
// image.height. Bindings: the output at 0, then the queries, their count,
// the framed camera and the image (camera_probe.h).

#include "cuda/camera/thin_lens.cuh"
#include "cuda/device/layout.cuh"
#include "cuda/math/vector.cuh"
#include "gpu/kernels/camera_probe.h"

namespace serenity {
namespace shaders {

using serenity::tests::CameraImage;
using serenity::tests::CameraProbeRay;
using serenity::tests::CameraQuery;

extern "C" __global__ void camera_probe(
    CameraProbeRay* out, const CameraQuery* queries, const uint* count_at,
    const serenity::contracts::CameraData* camera_at,
    const CameraImage* image_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint& count = *count_at;
  const serenity::contracts::CameraData& camera = *camera_at;
  const CameraImage& image = *image_at;
  if (i < count) {
    const CameraQuery q = queries[i];
    const CameraRay ray = camera_ray(camera, make_float2(q.pixel_x, q.pixel_y),
                                     make_float2(q.lens_u, q.lens_v),
                                     make_uint2(image.width, image.height));
    out[i] = CameraProbeRay{to_packed(ray.origin), to_packed(ray.direction)};
  }
}

}  // namespace shaders
}  // namespace serenity
