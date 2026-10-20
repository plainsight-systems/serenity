// Runs the procedural textures' shader halves for tests/gpu/texture_test.cpp:
// thread i evaluates point i. The noise (cuda/textures/noise.cuh) at
// points[i].point with seed points[i].seed; the wood (metal/textures/
// metal/kernels/wood.metal.h) and the swirl (cuda/textures/swirl.cuh) at
// points[i].point, given their data. Bindings: the output at 0, then the
// points, their count, and the texture's data (texture_probe.h).

#include "cuda/device/layout.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/textures/noise.cuh"
#include "cuda/textures/swirl.cuh"
#include "cuda/textures/wood.cuh"
#include "gpu/kernels/texture_probe.h"

namespace serenity {
namespace shaders {

using serenity::tests::TexturePoint;

extern "C" __global__ void noise_probe(float* out, const TexturePoint* points,
                                       const uint* count_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint& count = *count_at;
  if (i < count) {
    out[i] = gradient_noise(to_float3(points[i].point), points[i].seed);
  }
}

extern "C" __global__ void wood_probe(
    serenity::contracts::Float3* out, const TexturePoint* points,
    const uint* count_at, const serenity::textures::WoodData* wood_data_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint& count = *count_at;
  const serenity::textures::WoodData& wood_data = *wood_data_at;
  if (i < count) {
    out[i] = to_packed(wood(wood_data, to_float3(points[i].point)));
  }
}

// The swirl at a point in its shape's own coordinates.
extern "C" __global__ void swirl_probe(
    serenity::contracts::Float3* out, const TexturePoint* points,
    const uint* count_at, const serenity::textures::SwirlData* swirl_data_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint& count = *count_at;
  const serenity::textures::SwirlData& swirl_data = *swirl_data_at;
  if (i < count) {
    out[i] = to_packed(swirl(swirl_data, to_float3(points[i].point)));
  }
}

}  // namespace shaders
}  // namespace serenity
