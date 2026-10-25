// Runs manifold next event estimation's seed (metal/integrator/
// mnee.metal.h, seed_segment()) over the scene's structure, for
// tests/gpu/mnee_test.cpp: thread i seeds segment i and writes the seed:
// its crossings of caster spheres, sorted, whether another shape lies on
// the segment, whether the solve may take it. The CUDA backend's program of
// the same name (tests/cuda/kernels/traced_probes.cu) seeds the same
// segments its own way, so the one test checks both. Bindings: the output
// at 0, then the segments, their count, the scene's block, the structures'
// block, the shapes' records, transforms and boxes, and the table of the
// seed's intersection function (seed_probe.h, support/traced_probe.h); the
// seed reads the records and boxes through the scene's block, so the
// kernel declares neither.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/transform.h"
#include "core/scene/scene_block.h"
#include "core/shapes/box.h"
#include "core/shapes/primitive.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/device/layout.metal.h"
#include "metal/integrator/mnee.metal.h"
#include "metal/materials/resolve.metal.h"
#include "seed_probe.h"

using namespace serenity::shaders;
using serenity::tests::SeedFound;
using serenity::tests::SeedSegment;

namespace {

// What seed_segment() reads of a scene: the structures, the scene's block
// and the shapes' transforms (scene/scene_block.metal.h's SceneView, without
// what the seed does not read).
struct SeedScene {
  SceneStructure structure;
  constant serenity::gpu::SceneBlock* block;
  constant serenity::contracts::Transform* transforms;
};

}  // namespace

kernel void seed_probe(
    device SeedFound* out [[buffer(0)]],
    device const SeedSegment* segments [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant serenity::gpu::SceneBlock& block [[buffer(3)]],
    SceneStructure structure [[buffer(4)]],
    constant serenity::contracts::Transform* transforms [[buffer(6)]],
    metal::raytracing::intersection_function_table<> seed_functions
    [[buffer(8)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= count) {
    return;
  }
  const SeedScene scene{structure, &block, transforms};
  const SeedSegment s = segments[i];
  const mnee::Seed seed =
      mnee::seed_segment(scene, seed_functions, to_float3(s.origin),
                         to_float3(s.direction), s.distance);
  SeedFound found{};
  found.count = seed.count;
  found.blocked = seed.blocked ? 1u : 0u;
  found.usable = seed.usable ? 1u : 0u;
  for (uint c = 0u; c < mnee::max_interfaces; ++c) {
    found.t[c] = c < seed.count ? seed.t[c] : 0.0f;
    found.shape[c] = c < seed.count ? seed.shape[c] : 0u;
  }
  out[i] = found;
}
