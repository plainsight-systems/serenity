// Axis: Integrator (manifold next event estimation's seed, as the M3 runs
// it).
//
// The intersection function Metal runs for each box of the occluders' box
// structure a seed's segment enters (mnee.metal.h, seed_segment(); the
// structure, acceleration/structure_block.h, seed_boxes): the box's shape,
// its per-primitive data, handed to seed_candidate() with the seed the
// traversal carries as its payload. Where the seed stops, the function
// accepts the box, at the middle of the segment's interval (any distance
// within it would do: the traversal accepts its first and ends, and the
// hit is never read); otherwise it accepts none and Metal goes on. The
// CUDA backend's seed program, the same (cuda/integrator/mnee_seed.cuh).
//
// The library's one intersection function: its own file, so the passes'
// library and the tests' probe library each compile it once; linked into
// the pipeline of every kernel that seeds (Library's PipelineOptions,
// passes/restir_pt_initial/restir_pt_initial.cpp), whose table of it the
// kernel is handed.

#include <metal_raytracing>
#include <metal_stdlib>

#include "metal/integrator/mnee.metal.h"

namespace serenity {
namespace shaders {
namespace mnee {

// What the function tells Metal of a box: whether it is accepted, and at
// what distance.
struct SeedBoxResult {
  bool accept [[accept_intersection]];
  float distance [[distance]];
};

}  // namespace mnee
}  // namespace shaders
}  // namespace serenity

[[intersection(bounding_box)]] serenity::shaders::mnee::SeedBoxResult
mnee_seed_box(const device void* shape [[primitive_data]],
              float min_distance [[min_distance]],
              float max_distance [[max_distance]],
              ray_data serenity::shaders::mnee::SeedQuery& query [[payload]]) {
  if (serenity::shaders::mnee::seed_candidate(
          query, *static_cast<const device uint*>(shape))) {
    return serenity::shaders::mnee::SeedBoxResult{
        true, 0.5f * (min_distance + max_distance)};
  }
  return serenity::shaders::mnee::SeedBoxResult{false, 0.0f};
}
