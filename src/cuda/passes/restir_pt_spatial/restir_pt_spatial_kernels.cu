// ReSTIR PT's paired spatial reuse's kernel (restir_pt_spatial.h) that
// traces nothing: step 11, the merge, Metal's restir_pt_spatial
// (metal/passes/restir_pt_spatial/restir_pt_spatial.metal), taking the
// frame's launch parameters by address (frame/frame_launch.h).

#include "core/contracts/frame_constants.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "cuda/frame/frame_launch.h"
#include "cuda/math/srgb.cuh"
#include "cuda/passes/restir_pt/restir_pt.cuh"
#include "cuda/passes/restir_pt_spatial/partner.cuh"

namespace serenity {
namespace shaders {

// Step 11: the merge, one thread a traced pixel; its estimate added to the
// light buffer (its own pixel's alone, so in place); with temporal reuse in
// the graph, the path it keeps written as the frame's final reservoir.
__device__ inline void spatial_merge(
    const serenity::cuda::FrameLaunch& launch) {
  const serenity::contracts::FrameConstants& frame = launch.frame;
  const serenity::contracts::PathReuseSettings& settings = launch.path_settings;
  const restir_pt::Traced traced = restir_pt::traced(frame, settings);
  const uint2 pixel =
      traced.pixel(make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                              blockIdx.y * blockDim.y + threadIdx.y));
  if (!traced.inside(pixel)) {
    return;
  }
  constexpr uint slots = serenity::contracts::max_path_neighbors;
  const uint i = traced.slot(pixel);
  const serenity::contracts::PathReservoir own = launch.path_reservoirs[i];
  // k: the partners similar enough to have been tried (reuse/path_merge.cuh).
  uint k = 0u;
  for (uint t = 0u; t < settings.neighbors; ++t) {
    k += launch.shifts[i * slots + t].w != restir_pt::not_paired ? 1u : 0u;
  }
  const float own_target = luminance(to_float3(own.integrand));
  const float own_count = float(serenity::contracts::path_count(own.flags));
  // The path kept for the next frame (reuse/path_merge.cuh): picked by the
  // scalar weights, k + 1 times theirs, which the shared factor leaves
  // the pick's odds alone; only the graph with temporal reuse reads it.
  const bool keeps = settings.confidence_cap > 0.0f;
  PathNumbers numbers = purpose_numbers(path_numbers(pixel, frame.frame_index),
                                        restir_pt::spatial_pick_first);
  float weight_sum = 0.0f;
  float count_sum = own_count;
  serenity::contracts::PathReservoir kept = own;
  float canonical = 1.0f;
  float3 sum = make_float3(0.0f);
  for (uint t = 0u; t < settings.neighbors && k > 0u; ++t) {
    // this pixel's path at its partner
    const float4 outward = launch.shifts[i * slots + t];
    if (outward.w == restir_pt::not_paired) {
      continue;
    }
    const spatial::Partner partner =
        spatial::partner_of(launch.pairing, pixel, t, traced);
    const uint j = traced.slot(partner.pixel);
    const serenity::contracts::PathReservoir theirs = launch.path_reservoirs[j];
    const float their_count =
        float(serenity::contracts::path_count(theirs.flags));
    count_sum += their_count;
    canonical += path_merge::canonical_share(
        own_target, own_count, luminance(xyz(outward)) * outward.w, their_count,
        k);
    const float4 inward =
        launch.shifts[j * slots + t];  // the partner's path here
    if (inward.w > 0.0f) {
      const float m = path_merge::neighbor_weight(
          luminance(to_float3(theirs.integrand)), their_count,
          luminance(xyz(inward)) * inward.w, own_count, k);
      sum += m * xyz(inward) * inward.w * theirs.weight;
      if (keeps &&
          stream_pick(weight_sum,
                      m * luminance(xyz(inward)) * inward.w * theirs.weight,
                      next_number(numbers))) {
        kept = path_merge::shifted_reservoir(theirs, xyz(inward), inward.w);
      }
    }
  }
  sum += canonical * restir_pt::estimate(own);
  launch.light[i] += make_float4(sum / float(k + 1u), 0.0f);
  if (!keeps) {
    return;
  }
  // The canonical path last, its own odds the same in any order.
  if (stream_pick(weight_sum, canonical * own_target * own.weight,
                  next_number(numbers))) {
    kept = own;
  }
  const float target = luminance(to_float3(kept.integrand));
  kept.weight = target > 0.0f && weight_sum > 0.0f
                    ? weight_sum / (float(k + 1u) * target)
                    : 0.0f;
  kept.flags = path_merge::with_count(kept.flags, count_sum);
  launch.path_finals[i] = kept;
}

}  // namespace shaders
}  // namespace serenity

extern "C" __global__ void restir_pt_spatial(
    const serenity::cuda::FrameLaunch* launch) {
  serenity::shaders::spatial_merge(*launch);
}
