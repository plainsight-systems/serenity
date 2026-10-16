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
// by the pass whose loop wrote the list, right after its own dispatch: the
// path pass, where the light is folded into the accumulated image; ReSTIR
// PT's initial pass, where it is added to the frame's light buffer, which
// the shade folds in (metal/passes/restir_pt/restir_pt.metal.h).
//
// Three dispatches, each after a barrier on the one before (Metal 4 tracks
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
//               atomic float adds (a path's requests share a pixel);
//   mnee_fold   the path pass's only, a thread a pixel: the pixel's MNEE
//               light over the accumulated image's count added to its
//               mean, as if the path's sample had held it, and set back to
//               0.
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
// on its lanes that solved (a tenth of the paths), the path pass 4.1 times
// as long; solved here, lanes are requests (GDSA.13: divergent work grouped
// by key, as Cycles' intersect_mnee kernel groups it).
//
// Cost per frame: the requests' 128 bytes written and read once (on the
// marbles a tenth of the paths write one: some 28 MB at 1920 x 1080); their solves and
// n + 1 rays each; the fold's 16 bytes a pixel read, and the image read and
// written where a pixel has MNEE light. Measured in docs/research/
// 2026-10-12-mnee.md.
class MneePass {
public:
    explicit MneePass(const Library& library);

    // Where the requests' light goes: the path pass's MNEE light, folded
    // into the accumulated image; or ReSTIR PT's light buffer, as it is.
    enum class Into { accumulated_image, light_buffer };

    // Records the three dispatches (two for the light buffer). Throws
    // MetalError if `resources` has no scene, work list, counters or the
    // light it adds to, or, folding, no accumulated image.
    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources, Into into) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> grid_;
    NS::SharedPtr<MTL::ComputePipelineState> solve_;
    NS::SharedPtr<MTL::ComputePipelineState> fold_;
};

}  // namespace serenity::metal
