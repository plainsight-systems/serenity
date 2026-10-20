// Runs Film's accumulation (cuda/film/accumulate.cuh) for
// tests/gpu/film_test.cpp: thread i folds samples[i] into pixels[i], starting
// over where starts[i] is 1, and every thread reports its failure to the
// shared counter, as the path pass does. Bindings: the output at 0, then the
// pixels, the samples, the starts, their count and the counter (film_probe.h).

#include "cuda/film/accumulate.cuh"
#include "cuda/math/vector.cuh"
#include "gpu/kernels/film_probe.h"

namespace serenity {
namespace shaders {

using serenity::tests::Float4;

extern "C" __global__ void film_fold(Float4* out, const Float4* pixels,
                                     const Float4* samples, const uint* starts,
                                     const uint* count_at, uint* counter) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint& count = *count_at;
  bool failed = false;
  if (i < count) {
    const float3 sample = make_float3(samples[i].x, samples[i].y, samples[i].z);
    failed = !finite(sample);
    const Float4 p = pixels[i];
    const float4 held =
        accumulate(make_float4(p.x, p.y, p.z, p.w), sample, starts[i] != 0u);
    out[i] = Float4{held.x, held.y, held.z, held.w};
  }
  count_non_finite(failed, counter);
}

}  // namespace shaders
}  // namespace serenity
