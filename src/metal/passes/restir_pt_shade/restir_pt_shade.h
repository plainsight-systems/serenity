#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/library.h"
#include "metal/frame/frame_resources.h"

namespace serenity::metal {

// Axis: Pass (ReSTIR PT's shade).
//
// Step 9 of metal/passes/restir_pt/restir_pt.metal.h: one thread a pixel,
// the light buffer (and, with no spatial pass, the initial reservoir's own
// estimate) folded into the accumulated image, which is the frame's
// radiance, as the path pass's is.
//
// Reads: the settings, the path reservoirs and the light buffer. Writes: the
// accumulated image and the count of samples not finite. 80 bytes read a
// pixel, the accumulated image read and written. One dispatch.
class RestirPtShadePass {
public:
    explicit RestirPtShadePass(const Library& library);

    void record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const;

private:
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
};

}  // namespace serenity::metal
