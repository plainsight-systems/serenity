// Runs the shapes' surface interaction (cuda/shapes/shapes.cuh) for
// tests/gpu/shape_test.cpp: thread i fills contract 1 at queries[i].point on
// shape queries[i].shape, a ray arriving from +z. Bindings: the output at 0,
// then the queries, their count, and the shapes' records, transforms and
// boxes (shape_probe.h).

#include "cuda/device/layout.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/shapes/shapes.cuh"
#include "gpu/kernels/shape_probe.h"

namespace serenity {
namespace shaders {

using serenity::tests::ShapeQuery;

extern "C" __global__ void interaction_probe(
    serenity::contracts::SurfaceInteraction* out, const ShapeQuery* queries,
    const uint* count_at, const serenity::shapes::ShapeRecord* records,
    const serenity::contracts::Transform* transforms,
    const serenity::shapes::BoxData* boxes) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint& count = *count_at;
  if (i < count) {
    out[i] = surface_interaction(Shapes{records, transforms, boxes},
                                 queries[i].shape, to_float3(queries[i].point),
                                 make_float3(0.0f, 0.0f, -1.0f));
  }
}

}  // namespace shaders
}  // namespace serenity
