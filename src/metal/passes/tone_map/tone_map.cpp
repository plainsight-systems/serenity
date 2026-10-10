#include "metal/passes/tone_map/tone_map.h"

#include <cstring>
#include <optional>
#include <string>
#include <type_traits>

#include "core/frame/schedule.h"
#include "metal/device/error.h"

namespace serenity::metal {

namespace {

// One thread per texel of `size`, in rows of the execution width (GPU.2).
void dispatch(MTL4::ComputeCommandEncoder* encoder, const MTL::ComputePipelineState* pipeline,
              NS::UInteger width, NS::UInteger height) {
    const NS::UInteger simd = pipeline->threadExecutionWidth();
    const NS::UInteger rows = pipeline->maxTotalThreadsPerThreadgroup() / simd;
    encoder->dispatchThreads(MTL::Size(width, height, 1), MTL::Size(simd, rows, 1));
}

// Each dispatch reads what the one before wrote (tone_map.h).
void barrier(MTL4::ComputeCommandEncoder* encoder) {
    encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
}

}  // namespace

ToneMapPass::ToneMapPass(const Device& device, const Library& library, Submission& submission,
                         const passes::ToneMap& settings)
    : down_first_(library.compute_pipeline("tone_map_down_first")),
      down_(library.compute_pipeline("tone_map_down")),
      up_(library.compute_pipeline("tone_map_up")),
      finish_(library.compute_pipeline("tone_map_finish")) {
    if (const std::optional<std::string> reason = frame::invalid(settings)) {
        throw Error("ToneMapPass: " + *reason);
    }
    if (!device.handle()->supports32BitFloatFiltering()) {
        throw Error("ToneMapPass: the device cannot filter the 32-bit radiance image (supports32BitFloatFiltering)");
    }
    settings_ = NS::TransferPtr(device.handle()->newBuffer(sizeof(passes::ToneMap), MTL::ResourceStorageModeShared));
    if (!settings_) {
        throw Error("ToneMapPass: the device made no buffer for the settings");
    }
    static_assert(std::is_trivially_copyable_v<passes::ToneMap>, "the settings are copied as their bytes");
    std::memcpy(settings_->contents(), &settings, sizeof(settings));
    resident_ = submission.keep_resident(settings_.get());
}

void ToneMapPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.radiance == nullptr) {
        throw Error("ToneMapPass: the frame has no radiance image");
    }
    for (MTL::Texture* level : resources.bloom) {
        if (level == nullptr) {
            throw Error("ToneMapPass: the frame has no bloom pyramid");
        }
    }
    MTL4::ArgumentTable* arguments = resources.arguments;
    const auto& bloom = resources.bloom;

    // Steps 1 and 2 for B_0. Bindings match tone_map.metal.
    arguments->setAddress(settings_->gpuAddress(), 0);
    arguments->setTexture(resources.radiance->gpuResourceID(), 0);
    arguments->setTexture(bloom[0]->gpuResourceID(), 1);
    encoder->setComputePipelineState(down_first_.get());
    dispatch(encoder, down_first_.get(), bloom[0]->width(), bloom[0]->height());

    // Step 2 for B_1 .. B_5.
    encoder->setComputePipelineState(down_.get());
    for (std::size_t k = 1; k < bloom.size(); ++k) {
        barrier(encoder);
        arguments->setTexture(bloom[k - 1]->gpuResourceID(), 0);
        arguments->setTexture(bloom[k]->gpuResourceID(), 1);
        dispatch(encoder, down_.get(), bloom[k]->width(), bloom[k]->height());
    }

    // Step 3 for B_4 .. B_0.
    encoder->setComputePipelineState(up_.get());
    for (std::size_t k = bloom.size() - 1; k-- > 0;) {
        barrier(encoder);
        arguments->setTexture(bloom[k + 1]->gpuResourceID(), 0);
        arguments->setTexture(bloom[k]->gpuResourceID(), 1);
        dispatch(encoder, up_.get(), bloom[k]->width(), bloom[k]->height());
    }

    // Steps 1 and 4 to 6, into the target.
    barrier(encoder);
    arguments->setTexture(resources.radiance->gpuResourceID(), 0);
    arguments->setTexture(bloom[0]->gpuResourceID(), 1);
    arguments->setTexture(resources.target->gpuResourceID(), 2);
    encoder->setComputePipelineState(finish_.get());
    dispatch(encoder, finish_.get(), resources.size.width, resources.size.height);
}

}  // namespace serenity::metal
