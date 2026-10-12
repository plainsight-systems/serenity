#include "metal/passes/restir_pt_duplication/restir_pt_duplication.h"

#include <string>

#include "core/contracts/path_reuse_settings.h"

#include "metal/device/error.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

namespace {

// metal/reuse/duplication.metal.h's group.
constexpr NS::UInteger group = contracts::duplication_group;

}  // namespace

RestirPtDuplicationPass::RestirPtDuplicationPass(const Library& library)
    : pipeline_(library.compute_pipeline("restir_pt_duplication")) {
    if (pipeline_->maxTotalThreadsPerThreadgroup() < group * group) {
        throw MetalError("RestirPtDuplicationPass: the pipeline takes " +
                         std::to_string(pipeline_->maxTotalThreadsPerThreadgroup()) + " threads a group, under its " +
                         std::to_string(group * group));
    }
}

void RestirPtDuplicationPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.path_finals == 0 || resources.duplication == 0) {
        throw MetalError("RestirPtDuplicationPass: the frame lacks its final path reservoirs or duplication map");
    }
    namespace binding = bindings::restir_pt;
    MTL4::ArgumentTable* arguments = resources.arguments;
    arguments->setAddress(resources.constants, binding::constants);
    arguments->setAddress(resources.path_finals, binding::finals);
    arguments->setAddress(resources.duplication, binding::duplication);
    encoder->setComputePipelineState(pipeline_.get());
    encoder->dispatchThreadgroups(
        MTL::Size{(resources.size.width + group - 1) / group, (resources.size.height + group - 1) / group, 1},
        MTL::Size{group, group, 1});
}

}  // namespace serenity::metal
