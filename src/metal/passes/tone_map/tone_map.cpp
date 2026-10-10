#include "metal/passes/tone_map/tone_map.h"

#include <cstring>
#include <optional>
#include <string>
#include <type_traits>

#include "core/frame/schedule.h"
#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

namespace {

// A texture's size, as the dispatch over it takes it.
frame::Extent extent(const MTL::Texture* texture) {
    return {static_cast<std::uint32_t>(texture->width()), static_cast<std::uint32_t>(texture->height())};
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
        throw MetalError("ToneMapPass: " + *reason);
    }
    if (!device.handle()->supports32BitFloatFiltering()) {
        throw MetalError(
            "ToneMapPass: the device cannot filter the 32-bit radiance image (supports32BitFloatFiltering)");
    }
    settings_ = NS::TransferPtr(device.handle()->newBuffer(sizeof(passes::ToneMap), MTL::ResourceStorageModeShared));
    if (!settings_) {
        throw MetalError("ToneMapPass: the device made no buffer for the settings");
    }
    static_assert(std::is_trivially_copyable_v<passes::ToneMap>, "the settings are copied as their bytes");
    std::memcpy(settings_->contents(), &settings, sizeof(settings));
    resident_ = submission.keep_resident(settings_.get());
}

void ToneMapPass::record(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources) const {
    if (resources.radiance == nullptr) {
        throw MetalError("ToneMapPass: the frame has no radiance image");
    }
    for (const MTL::Texture* level : resources.bloom) {
        if (level == nullptr) {
            throw MetalError("ToneMapPass: the frame has no bloom pyramid");
        }
    }
    // Bindings: passes/bindings.h, as tone_map.metal declares them.
    namespace binding = bindings::tone_map;
    MTL4::ArgumentTable* arguments = resources.arguments;
    const auto& bloom = resources.bloom;

    // Steps 1 and 2 for B_0.
    arguments->setAddress(settings_->gpuAddress(), binding::settings);
    arguments->setTexture(resources.radiance->gpuResourceID(), binding::input);
    arguments->setTexture(bloom[0]->gpuResourceID(), binding::level);
    encoder->setComputePipelineState(down_first_.get());
    dispatch_per_pixel(encoder, down_first_.get(), extent(bloom[0]));

    // Step 2 for B_1 .. B_5, each from the level before.
    encoder->setComputePipelineState(down_.get());
    for (std::size_t k = 1; k < bloom.size(); ++k) {
        barrier(encoder);
        arguments->setTexture(bloom[k - 1]->gpuResourceID(), binding::input);
        arguments->setTexture(bloom[k]->gpuResourceID(), binding::level);
        dispatch_per_pixel(encoder, down_.get(), extent(bloom[k]));
    }

    // Step 3 for B_4 .. B_0, each from the level below it: `below` counts
    // down from the last level to B_1.
    encoder->setComputePipelineState(up_.get());
    for (std::size_t below = bloom.size() - 1; below > 0; --below) {
        const std::size_t k = below - 1;
        barrier(encoder);
        arguments->setTexture(bloom[below]->gpuResourceID(), binding::input);
        arguments->setTexture(bloom[k]->gpuResourceID(), binding::level);
        dispatch_per_pixel(encoder, up_.get(), extent(bloom[k]));
    }

    // Steps 1 and 4 to 6, into the target.
    barrier(encoder);
    arguments->setTexture(resources.radiance->gpuResourceID(), binding::input);
    arguments->setTexture(bloom[0]->gpuResourceID(), binding::bloom);
    arguments->setTexture(resources.target->gpuResourceID(), binding::target);
    encoder->setComputePipelineState(finish_.get());
    dispatch_per_pixel(encoder, finish_.get(), resources.size);
}

}  // namespace serenity::metal
