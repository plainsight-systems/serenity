#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <array>

#include "core/contracts/path_reuse_settings.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "metal/frame/frame_resources.h"
#include "metal/passes/mnee/mnee.h"

namespace serenity::metal {

// The intersection functions a kernel that seeds MNEE traces with, linked
// into its pipeline in this order, its table's entries
// (integrator/mnee_seed.metal; Library's PipelineOptions).
inline constexpr std::array<const char*, 1> seed_intersection_functions{
    "mnee_seed_box"};

// Axis: Pass (ReSTIR PT's initial sampling).
//
// Steps 1 to 4 of metal/passes/restir_pt/restir_pt.metal.h: the frame's
// power pyramid, by one threadgroup (contracts/light_tile.h; light_selection/
// power_light.metal.h); a barrier (GPU.8: the pyramid's write and the draws
// that read it); the frame's light tiles, one thread an entry, each drawn
// from it by power; a barrier (the tiles' write and the candidates that
// read them); then one thread a traced pixel (every
// pixel, or with the checkerboard the frame's field: frame_resources.h,
// traced; contract 20), the path tree along the camera's ray resampled to
// one path; then the tree's MNEE requests solved, their light added to the
// light buffer (metal/passes/mnee/mnee.h).
//
// The path tree's MNEE seeds trace the occluders' boxes with the seed's
// intersection function: linked into the tree's pipeline, its table made
// once and kept resident, bound beside the kernel's arrays
// (integrator/mnee.metal.h, seed_segment()).
//
// Reads: the scene, structure, camera, ReSTIR PT's settings and, with
// temporal reuse, last frame's transforms. Writes: the power pyramid, which
// the reuse passes read too (this frame's, and the next frame's temporal
// reuse as last frame's), the light tiles, and the
// path reservoirs, the primary hits (the G-buffer), the light buffer and,
// with temporal reuse, the previous positions, a pixel's each, which the
// passes after it read (renderer.h records the barrier); MNEE's work list,
// which its pass solves.
//
// Cost: the pyramid, every light placed and its power taken once, and a
// few thousand additions (616 lights: 1365 floats); the tiles, 131072
// lights drawn by a descent of the pyramid's levels (5 for 616 lights, 4
// floats read at each, from 5.3 KB the cache holds), placed and packed,
// 4 MB; the path tree,
// one path's rays (a ray and a shadow ray per vertex, as the path
// tracer's), and NEE's candidates, 32 at the primary hit and fewer after
// (metal/integrator/nee.metal.h), each a 32-byte tile entry read; 128
// bytes written a pixel, 140 with temporal reuse. Measured beside the path
// pass (docs/research/2026-10-11-restir-pt-phase2.md); the pyramid and the
// draws in docs/research/2026-10-18-production-sweep.md. Three dispatches,
// and MNEE's two.
class RestirPtInitialPass {
 public:
  // Its pipelines specialized by the run's `settings`
  // (restir_pt/reuse_constants.h).
  // Its seed's table kept resident on `submission`'s queue.
  RestirPtInitialPass(const Library& library, Submission& submission,
                      const contracts::PathReuseSettings& settings);

  // Records the pass. Throws MetalError if `resources` lacks any array or
  // structure it reads or writes above.
  void record(MTL4::ComputeCommandEncoder* encoder,
              const FrameResources& resources) const;

 private:
  NS::SharedPtr<MTL::ComputePipelineState> power_;
  NS::SharedPtr<MTL::ComputePipelineState> tiles_;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
  NS::SharedPtr<MTL::IntersectionFunctionTable> seed_functions_;
  Resident seed_functions_resident_;
  MneePass mnee_;
};

}  // namespace serenity::metal
