// The MNEE pass's kernels (mnee.h): the work list's length as an indirect
// grid, and the requests it had no room for counted; each request's walk
// begun and taken a round; a round list's walks taken another round; a
// list's length as the next dispatch's grid; and each converged walk's
// light, its chain's visibility traced, added to ReSTIR PT's light buffer.
//
// A round (docs/research/2026-10-21-tracing-divergence.md, section 4; the
// CUDA backend's, cuda/passes/mnee/mnee_kernels.cu): each walk takes up to
// mnee_round_iterations iterations (contract 16); one that converges is
// appended to the converged list, one still going to the other round
// list, one that fails is dropped, so the next round's SIMD groups hold
// walks still going, not lanes waiting on one (GPU.4). Appended a SIMD
// group's at once (append.metal.h, GDSA.10): every thread of a group
// reaches both reserves, a thread past its list's count with nothing to
// append.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/mnee_request.h"
#include "core/contracts/transform.h"
#include "metal/device/append.metal.h"
#include "metal/device/indirect_grid.h"
#include "metal/film/accumulate.metal.h"
#include "metal/integrator/mnee.metal.h"
#include "metal/integrator/path.metal.h"
#include "metal/passes/bindings.h"
#include "metal/scene/scene_block.metal.h"

using namespace metal;
using namespace metal::raytracing;
using namespace serenity::shaders;
namespace binding = serenity::bindings::mnee;

// The solve's indirect grid sits in contract 16's header, at
// mnee_grid_offset.
static_assert(serenity::contracts::mnee_grid_offset +
                      sizeof(serenity::gpu::IndirectGrid) <=
                  serenity::contracts::mnee_header_bytes,
              "the solve's grid fits in the list's header");
// The last round takes a walk to the walk's last iteration, so the list it
// writes stays empty and nothing reads it (contract 16).
static_assert(serenity::contracts::mnee_rounds *
                      serenity::contracts::mnee_round_iterations ==
                  mnee::max_iterations,
              "the rounds take a walk to its last iteration");

namespace {

// Where a round leaves the walks: the round list it writes and the
// converged list, each with its count.
struct Lists {
  device serenity::contracts::MneeWalk* going;
  device atomic_uint* going_count;
  device serenity::contracts::MneeWalk* converged;
  device atomic_uint* converged_count;
};

// Walk `w` for request `request` after its round, `walked` how it stands
// (failed for a thread with no walk), appended where it goes. Every thread
// of the SIMD group calls it.
void leave(Lists lists, uint request, thread const mnee::Walk& w,
           mnee::Walked walked) {
  const bool converged = walked == mnee::Walked::converged;
  const bool going = walked == mnee::Walked::going;
  const uint at_converged =
      simd_reserve(lists.converged_count, converged ? 1u : 0u);
  const uint at_going = simd_reserve(lists.going_count, going ? 1u : 0u);
  if (converged) {
    lists.converged[at_converged] = mnee::kept_walk(request, w);
  } else if (going) {
    lists.going[at_going] = mnee::kept_walk(request, w);
  }
}

}  // namespace

// One thread: the list's length as the walks' first grid; the requests that
// found no room, by which the count passes the capacity (contract 16),
// added to the frame's count of lost requests; and the count set back to 0
// for the next frame's list, which a barrier orders after the start that
// reads the grid.
kernel void mnee_grid(device atomic_uint* header [[buffer(binding::header)]],
                      device serenity::gpu::IndirectGrid* grid
                      [[buffer(binding::grid)]],
                      device atomic_uint* lost [[buffer(binding::lost)]]) {
  namespace c = serenity::contracts;
  const uint written =
      atomic_load_explicit(header + c::mnee_count_word, memory_order_relaxed);
  const uint capacity = atomic_load_explicit(header + c::mnee_capacity_word,
                                             memory_order_relaxed);
  atomic_store_explicit(header + c::mnee_count_word, 0u, memory_order_relaxed);
  const uint count = min(written, capacity);  // past it, none was written
  if (written > capacity) {
    atomic_fetch_add_explicit(lost, written - capacity, memory_order_relaxed);
  }
  *grid = indirect_grid(count, c::mnee_solve_group);
}

// One thread: a walk list's length (the round's before, or the converged
// list's) as the next dispatch's grid, and its count set back to 0 for the
// next list written there, which a barrier orders after the dispatch that
// reads the grid.
kernel void mnee_walk_grid(device atomic_uint* count
                           [[buffer(binding::walk_count)]],
                           device serenity::gpu::IndirectGrid* grid
                           [[buffer(binding::grid)]]) {
  const uint walks = atomic_load_explicit(count, memory_order_relaxed);
  atomic_store_explicit(count, 0u, memory_order_relaxed);
  *grid = indirect_grid(walks, serenity::contracts::mnee_solve_group);
}

// Each request's chain built, its walk begun and taken a round. A request
// whose weight is not finite is counted (metal/film/failure_count.h) and
// left out, as the solve has always left out a light not finite; one whose
// light faces away gives nothing.
kernel void mnee_walk_start(
    serenity::shaders::SceneStructure structure [[buffer(binding::structure)]],
    constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
    constant float* sphere_glows [[buffer(binding::glows)]],
    constant serenity::contracts::Transform* transforms
    [[buffer(binding::transforms)]],
    device atomic_uint* non_finite [[buffer(binding::non_finite)]],
    const device serenity::contracts::MneeRequest* requests
    [[buffer(binding::requests)]],
    const device serenity::gpu::IndirectGrid* grid [[buffer(binding::grid)]],
    device serenity::contracts::MneeWalk* going [[buffer(binding::walks_out)]],
    device atomic_uint* going_count [[buffer(binding::walk_count_out)]],
    device serenity::contracts::MneeWalk* converged
    [[buffer(binding::converged)]],
    device atomic_uint* converged_count [[buffer(binding::converged_count)]],
    uint n [[thread_position_in_grid]]) {
  mnee::Walk w{};
  mnee::Walked walked = mnee::Walked::failed;
  bool failed = false;
  if (n < grid->count) {
    const path::Scene scene{structure, &block, transforms, sphere_glows};
    const mnee::Requested q = mnee::requested(scene, requests[n]);
    failed = !finite(q.weight);
    if (!failed) {
      const mnee::Chain chain = mnee::chain_start(
          scene, q.here, q.origin, q.light_center, q.sample, q.seed);
      if (chain.faces) {
        w = mnee::begun_walk(chain.count, chain.seeded);
        mnee::Blocks blocks[mnee::max_interfaces];
        walked = mnee::walk(chain.count, w, chain.casters, q.origin,
                            chain.facing, chain.y, blocks,
                            serenity::contracts::mnee_round_iterations);
      }
    }
  }
  count_non_finite(failed, non_finite);
  leave(Lists{going, going_count, converged, converged_count}, n, w, walked);
}

// The walks of a round list taken for another round, each one's chain
// rebuilt from its request, those still going into the other list.
kernel void mnee_walk_round(
    serenity::shaders::SceneStructure structure [[buffer(binding::structure)]],
    constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
    constant float* sphere_glows [[buffer(binding::glows)]],
    constant serenity::contracts::Transform* transforms
    [[buffer(binding::transforms)]],
    const device serenity::contracts::MneeRequest* requests
    [[buffer(binding::requests)]],
    const device serenity::gpu::IndirectGrid* grid [[buffer(binding::grid)]],
    const device serenity::contracts::MneeWalk* walks
    [[buffer(binding::walks_in)]],
    device serenity::contracts::MneeWalk* going [[buffer(binding::walks_out)]],
    device atomic_uint* going_count [[buffer(binding::walk_count_out)]],
    device serenity::contracts::MneeWalk* converged
    [[buffer(binding::converged)]],
    device atomic_uint* converged_count [[buffer(binding::converged_count)]],
    uint n [[thread_position_in_grid]]) {
  mnee::Walk w{};
  mnee::Walked walked = mnee::Walked::failed;
  uint request = 0u;
  if (n < grid->count) {
    const serenity::contracts::MneeWalk held = walks[n];
    request = held.request;
    w = mnee::resumed_walk(held);
    const path::Scene scene{structure, &block, transforms, sphere_glows};
    const mnee::Requested q = mnee::requested(scene, requests[request]);
    const mnee::Chain chain = mnee::chain_start(
        scene, q.here, q.origin, q.light_center, q.sample, q.seed);
    mnee::Blocks blocks[mnee::max_interfaces];
    walked =
        mnee::walk(chain.count, w, chain.casters, q.origin, chain.facing,
                   chain.y, blocks, serenity::contracts::mnee_round_iterations);
  }
  leave(Lists{going, going_count, converged, converged_count}, request, w,
        walked);
}

// Each converged walk's light, its request's chain rebuilt and its
// derivatives at the walk's points computed again (the same numbers the
// walk's last iteration had), added to its pixel's: three atomic adds,
// since the requests of one path's vertices share a pixel. A light not
// finite is counted and left out, as the shade leaves out a sample.
kernel void mnee_finish(
    serenity::shaders::SceneStructure structure [[buffer(binding::structure)]],
    constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
    constant float* sphere_glows [[buffer(binding::glows)]],
    constant serenity::contracts::Transform* transforms
    [[buffer(binding::transforms)]],
    device atomic_uint* non_finite [[buffer(binding::non_finite)]],
    const device serenity::contracts::MneeRequest* requests
    [[buffer(binding::requests)]],
    const device serenity::gpu::IndirectGrid* grid [[buffer(binding::grid)]],
    const device serenity::contracts::MneeWalk* walks
    [[buffer(binding::walks_in)]],
    device atomic_float* light [[buffer(binding::light)]],
    uint n [[thread_position_in_grid]]) {
  bool failed = false;
  if (n < grid->count) {
    const serenity::contracts::MneeWalk walked = walks[n];
    const serenity::contracts::MneeRequest request = requests[walked.request];
    const path::Scene scene{structure, &block, transforms, sphere_glows};
    const mnee::Requested q = mnee::requested(scene, request);
    const mnee::Chain chain = mnee::chain_start(
        scene, q.here, q.origin, q.light_center, q.sample, q.seed);
    float3 p[mnee::max_interfaces];
#pragma unroll
    for (uint i = 0u; i < mnee::max_interfaces; ++i) {
      p[i] = to_float3(walked.p[i]);
    }
    mnee::Blocks blocks[mnee::max_interfaces];
    (void)mnee::constraint_blocks(chain.count, p, chain.casters, q.origin,
                                  chain.y, blocks);
    const float3 found =
        q.weight *
        mnee::chain_light(scene, q.here, q.origin, q.sample, chain, p, blocks);
    failed = !finite(found);
    if (!failed && any(found != 0.0f)) {
      for (uint c = 0u; c < 3u; ++c) {
        atomic_fetch_add_explicit(light + request.pixel * 4u + c, found[c],
                                  memory_order_relaxed);
      }
    }
  }
  count_non_finite(failed, non_finite);
}
