// Runs the numbers a path draws (cuda/sampler/sampler.cuh) for
// tests/gpu/sampler_test.cpp. Bindings: the output at 0, the inputs after
// it in the order the kernel lists them (sampler_probe.h).

#include "cuda/math/vector.cuh"
#include "cuda/sampler/sampler.cuh"
#include "gpu/kernels/sampler_probe.h"

namespace serenity {
namespace shaders {

using serenity::tests::PathNumbersDraw;
using serenity::tests::PathNumbersQuery;

// The first path_numbers_drawn numbers of the path through pixel
// (queries[i].x, queries[i].y) in frame queries[i].frame. Inputs: the
// queries, their count.
extern "C" __global__ void path_numbers_probe(PathNumbersDraw* out,
                                              const PathNumbersQuery* queries,
                                              const uint* count_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint& count = *count_at;
  if (i >= count) {
    return;
  }
  PathNumbers numbers =
      path_numbers(make_uint2(queries[i].x, queries[i].y), queries[i].frame);
  PathNumbersDraw drawn;
  for (uint k = 0; k < serenity::tests::path_numbers_drawn; ++k) {
    drawn.u[k] = next_number(numbers);
  }
  out[i] = drawn;
}

}  // namespace shaders
}  // namespace serenity
