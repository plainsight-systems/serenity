#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (manifold next event estimation's solve).
//
// The second half of MNEE (metal/integrator/mnee.metal.h): the work list
// a loop wrote (contract 16, core/contracts/mnee_request.h) solved, one
// thread a request, and each request's light added to its pixel. Recorded
// by ReSTIR PT's initial pass, whose path tree wrote the list, right after
// its own dispatch: the light is added to the frame's light buffer, which
// the shade folds in (metal/passes/restir_pt/restir_pt.metal.h).
//
// Two dispatches, each after a barrier on the one before (Metal 4 tracks
// no hazards; GPU.8):
//
//   mnee_grid   one thread: the list's count as the solve's indirect grid
//               (metal/device/indirect_grid.h, groups of contract 16's
//               mnee_solve_group), the requests that found the list full
//               added to the frame's count of lost requests, and the count
//               set back to 0 for the next frame;
//   mnee_solve  a thread a request: the walk, its contribution, its
//               visibility (mnee.metal.h, steps 2 to 4), times the
//               request's weight, added to its pixel's float4 by three
//               atomic float adds (a path's requests share a pixel).
//
// Determinism (GDSA.2): not guaranteed where a pixel takes three or more
// requests. Two float adds commute exactly; from three on, the sum's
// rounding follows the order the solves arrive in, which the GPU's
// scheduling decides. Measured reproducible all the same: every graph's
// headless output, the MNEE graphs among them, was the same to the bit run
// after run on 2026-10-16 (docs/research/
// 2026-10-16-codebase-guidelines-pass.md, area C's P1), so a byte diff of
// two builds is a fair check here; a fold in a fixed order would make it
// so by construction, at a cost not measured.
//
// Why a list (contract 16): solved inside the loop, every SIMD group waited
// on its lanes that solved (a tenth of the paths), the path tracer's pass
// 4.1 times as long when it had MNEE; solved here, lanes are requests (GDSA.13: divergent work grouped
// by key, as Cycles' intersect_mnee kernel groups it).
//
// Cost per frame: the requests' 128 bytes written and read once (on the
// marbles a tenth of the paths write one: some 28 MB at 1920 x 1080); their
// solves and n + 1 rays each. Measured in docs/research/2026-10-12-mnee.md.
class MneePass {
public:
    explicit MneePass(const Library& library);

    // Records the two dispatches. Throws MetalError if `resources` has no
    // scene, work list, counters or light buffer.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> grid_;
    NS::SharedPtr<MTL::ComputePipelineState> solve_;
};

}  // namespace serenity::metal
