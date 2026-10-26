#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (manifold next event estimation's solve).
//
// The second half of MNEE (metal/integrator/mnee.metal.h): the work list
// a loop wrote (contract 16, core/contracts/mnee_request.h) solved, a
// thread a request and then a walk, and each request's light added to its
// pixel. Recorded
// by ReSTIR PT's initial pass, whose path tree wrote the list, right after
// its own dispatch: the light is added to the frame's light buffer, which
// the shade folds in (metal/passes/restir_pt/restir_pt.metal.h).
//
// The Newton walks in rounds, as on the RTX 3090 Ti (docs/research/
// 2026-10-21-tracing-divergence.md, section 4; cuda/passes/mnee/mnee.h):
// there a fifth of the walks failed and spent 57% of the iterations, and a
// group of 32 lanes waited on its longest. Taken mnee_round_iterations at a
// time (contract 16), the walks still going gathered into full SIMD groups
// between rounds (GPU.4, GDSA.13), each walk takes the same iterations as
// one taken whole. The dispatches, each after a barrier on the one before
// (Metal 4 tracks no hazards; GPU.8), every list's grid written on the GPU
// (metal/device/indirect_grid.h, groups of contract 16's mnee_solve_group)
// in the header's grid, which each grid kernel writes over:
//
//   mnee_grid        one thread: the request list's count as the grid, the
//                    requests that found the list full added to the
//                    frame's count of lost requests, the count set back to
//                    0 for the next frame;
//   mnee_walk_start  a thread a request: its chain built, its walk begun
//                    and taken a round, into round list 0 or the converged
//                    list;
//   7 times          mnee_walk_grid, one thread: the round list the round
//                    before wrote as the grid, its count set back to 0;
//                    mnee_walk_round, a thread a walk: taken another round,
//                    into the other list or the converged one;
//   mnee_walk_grid   the converged list's count as the grid;
//   mnee_finish      a thread a converged walk: its contribution and
//                    visibility (mnee.metal.h, steps 3 and 4), times the
//                    request's weight, added to its pixel's float4 by three
//                    atomic float adds (a path's requests share a pixel).
//
// The lists' bindings are moved between dispatches: Metal 4 takes the
// argument table's bindings at each dispatch. CUDA launches each list at
// its capacity and zeroes a count by a memset on its stream; a compute
// encoder has no memset, so the grid kernels set the counts back.
//
// Determinism (GDSA.2): not guaranteed where a pixel takes three or more
// requests. Two float adds commute exactly; from three on, the sum's
// rounding follows the order the finishes arrive in, which the GPU's
// scheduling decides: the demo graph differs run to run in a few pixels by
// a level or two (found 2026-10-24), kept as it is, since a fold in a
// fixed order would cost frame time. A walk's own steps are the same
// whichever list place a round gives it.
//
// Why a list (contract 16): solved inside the loop, every SIMD group waited
// on its lanes that solved (a tenth of the paths), the path tracer's pass
// 4.1 times as long when it had MNEE; solved here, lanes are requests
// (GDSA.13: divergent work grouped by key, as Cycles' intersect_mnee kernel
// groups it), and between rounds walks still going (above).
//
// Cost per frame: the requests' 128 bytes written and read once (on the
// marbles a tenth of the paths write one: some 28 MB at 1920 x 1080), and
// read again by each round of their walks; each walk's 96 bytes written and
// read once a round it is still going; their walks and n + 1 rays each; 18
// dispatches. Measured in docs/research/2026-10-12-mnee.md.
class MneePass {
 public:
  explicit MneePass(const Library& library);

  // Records the dispatches. Throws MetalError if `resources` has no scene,
  // work list, walk lists, counters or light buffer.
  void record(MTL4::ComputeCommandEncoder* encoder,
              const FrameResources& resources) const;

 private:
  NS::SharedPtr<MTL::ComputePipelineState> grid_;
  NS::SharedPtr<MTL::ComputePipelineState> walk_grid_;
  NS::SharedPtr<MTL::ComputePipelineState> start_;
  NS::SharedPtr<MTL::ComputePipelineState> round_;
  NS::SharedPtr<MTL::ComputePipelineState> finish_;
};

}  // namespace serenity::metal
