// The MNEE pass's kernels (mnee.h) that trace nothing.
//
//   mnee_walk_start   a thread a request: its chain rebuilt from it
//                     (integrator/mnee.cuh, requested(), chain_start()), its
//                     walk begun and taken for a round
//   mnee_walk_round   a thread a walk still going: its chain rebuilt, its
//                     walk taken for another round
//   mnee_settle       after the solve: the requests that found no room, by
//                     which the count passes the capacity (contract 16),
//                     added to the slot's count of lost requests, and every
//                     count set back to 0 for the next frame's lists.
//                     Metal's mnee_grid (metal/passes/mnee/mnee.metal)
//                     without its grid, which a launch here does not read
//                     (mnee.cu).
//
// A round (docs/research/2026-10-21-tracing-divergence.md, section 4):
// each walk takes up to mnee_round_iterations iterations (contract 16,
// core/contracts/mnee_request.h); one that converges is appended to the
// converged list, one still going to the other round list, one that fails is
// dropped, so the next round's warps hold walks still going, not lanes waiting
// on one. Appended by warp (device/append.cuh). Each launched at the list's
// capacity, a thread past its count returning: cheaper than a fixed set of
// threads striding over it, for the whole walks (the hardware design's
// Phase 2) and for rounds of them (measured 2026-10-21: 8.87 against 9.19
// ms a frame at 1920 x 1080; a round's time is its walks', not its
// launch's).

#include "core/contracts/mnee_request.h"
#include "cuda/device/append.cuh"
#include "cuda/film/accumulate.cuh"
#include "cuda/frame/frame_launch.h"
#include "cuda/integrator/mnee.cuh"
#include "cuda/integrator/path.cuh"
#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {
namespace {

using serenity::contracts::MneeWalk;
using serenity::cuda::FrameLaunch;

__device__ inline path::Scene scene_of(const FrameLaunch& launch) {
  return path::Scene{&launch.structures, launch.scene, launch.transforms,
                     launch.glows};
}

// Walk `w` for request `request` written at the end of the list whose
// count is `*count`.
__device__ inline void append_walk(MneeWalk* list, uint* count, uint request,
                                   const mnee::Walk& w) {
  MneeWalk& out = list[warp_reserve(count, 1u)];
  out.request = request;
  out.iteration = w.iteration;
  out.beta = w.beta;
  out.resolve = w.resolve ? 1u : 0u;
#pragma unroll
  for (uint i = 0u; i < mnee::max_interfaces; ++i) {
    out.p[i] = to_packed(w.p[i]);
    out.dx[i][0] = w.dx[i].x;
    out.dx[i][1] = w.dx[i].y;
  }
}

__device__ inline mnee::Walk loaded_walk(const MneeWalk& in) {
  mnee::Walk w{};
#pragma unroll
  for (uint i = 0u; i < mnee::max_interfaces; ++i) {
    w.p[i] = to_float3(in.p[i]);
    w.dx[i] = make_float2(in.dx[i][0], in.dx[i][1]);
  }
  w.beta = in.beta;
  w.iteration = in.iteration;
  w.resolve = in.resolve != 0u;
  return w;
}

// One round of walk `w` for request `request`, then where it goes: the
// converged list, the round list `next` (its count `*next_count`), or
// nowhere.
__device__ inline void walk_round(const FrameLaunch& launch, uint request,
                                  mnee::Walk w, MneeWalk* next,
                                  uint* next_count) {
  const path::Scene scene = scene_of(launch);
  const mnee::Requested q =
      mnee::requested(scene, launch.mnee_requests[request]);
  const mnee::Chain chain = mnee::chain_start(scene, q.here, q.origin,
                                              q.light_center, q.sample, q.seed);
  mnee::Blocks blocks[mnee::max_interfaces];
  const mnee::Walked walked =
      mnee::walk(chain.count, w, chain.casters, q.origin, chain.facing, chain.y,
                 blocks, serenity::contracts::mnee_round_iterations);
  if (walked == mnee::Walked::converged) {
    append_walk(
        launch.mnee_converged,
        &launch.mnee_walk_header[serenity::contracts::mnee_converged_word],
        request, w);
  } else if (walked == mnee::Walked::going) {
    append_walk(next, next_count, request, w);
  }
}

}  // namespace
}  // namespace shaders
}  // namespace serenity

// The requests' walks begun (above), into round list 0. A request whose
// weight is not finite is counted (film/failure_count.h) and left out, as
// the solve has always left out a light not finite; one whose light faces
// away gives nothing.
extern "C" __global__ void mnee_walk_start(
    const serenity::cuda::FrameLaunch* launch) {
  namespace s = serenity::shaders;
  namespace c = serenity::contracts;
  const unsigned n = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned count = min(launch->mnee_header[c::mnee_count_word],
                             launch->mnee_header[c::mnee_capacity_word]);
  if (n >= count) {
    return;
  }
  const s::path::Scene scene = s::scene_of(*launch);
  const s::mnee::Requested q =
      s::mnee::requested(scene, launch->mnee_requests[n]);
  if (!s::finite(q.weight)) {
    s::count_non_finite(true, launch->non_finite);
    return;
  }
  const s::mnee::Chain chain = s::mnee::chain_start(
      scene, q.here, q.origin, q.light_center, q.sample, q.seed);
  if (!chain.faces) {
    return;
  }
  s::walk_round(*launch, n, s::mnee::begun_walk(chain.count, chain.seeded),
                launch->mnee_walks_a, &launch->mnee_walk_header[0]);
}

// The walks of round list `from` (0 or 1) taken for another round, those
// still going into the other list.
extern "C" __global__ void mnee_walk_round(
    const serenity::cuda::FrameLaunch* launch, unsigned from) {
  namespace s = serenity::shaders;
  const unsigned n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= launch->mnee_walk_header[from]) {
    return;
  }
  const serenity::contracts::MneeWalk* const in =
      from == 0u ? launch->mnee_walks_a : launch->mnee_walks_b;
  serenity::contracts::MneeWalk* const out =
      from == 0u ? launch->mnee_walks_b : launch->mnee_walks_a;
  const serenity::contracts::MneeWalk walked = in[n];
  s::walk_round(*launch, walked.request, s::loaded_walk(walked), out,
                &launch->mnee_walk_header[1u - from]);
}

extern "C" __global__ void mnee_settle(
    const serenity::cuda::FrameLaunch* launch) {
  namespace c = serenity::contracts;
  unsigned int* const header = launch->mnee_header;
  const unsigned int written = header[c::mnee_count_word];
  const unsigned int capacity = header[c::mnee_capacity_word];
  header[c::mnee_count_word] = 0u;
  for (unsigned i = 0u; i < serenity::contracts::mnee_walk_header_words; ++i) {
    launch->mnee_walk_header[i] = 0u;
  }
  if (written > capacity) {
    atomicAdd(launch->mnee_lost, written - capacity);
  }
}
