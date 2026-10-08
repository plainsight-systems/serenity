#include "metal/frame/renderer.h"

#include <cstring>

#include "core/contracts/frame_constants.h"
#include "metal/device/error.h"
#include "serenity/metallib/shaders.h"

namespace serenity::metal {

namespace {

// Bytes between the ring's slots: more than a FrameConstants needs, so a slot
// can grow, and aligned for any buffer binding Metal takes.
constexpr std::size_t slot_stride = 256;
static_assert(sizeof(contracts::FrameConstants) <= slot_stride);

// Every pass binds at most this many buffers and textures through the table.
constexpr NS::UInteger max_buffers = 4;
constexpr NS::UInteger max_textures = 4;

std::string describe(const NS::Error* error) {
    if (error == nullptr || error->localizedDescription() == nullptr) {
        return "Metal gave no description";
    }
    return error->localizedDescription()->utf8String();
}

}  // namespace

Renderer::Renderer(const Device& device, Submission& submission, const frame::Schedule& schedule)
    : library_(device, serenity::metallib::shaders) {
    if (schedule.passes.empty()) {
        throw Error("Renderer: the schedule has no passes");
    }
    auto drained = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

    constants_ = NS::TransferPtr(
        device.handle()->newBuffer(slot_stride * frames_in_flight, MTL::ResourceStorageModeShared));
    if (!constants_) {
        throw Error("Renderer: the device made no buffer for the frame constants");
    }
    submission.make_resident(constants_.get());

    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(max_buffers);
    descriptor->setMaxTextureBindCount(max_textures);
    for (auto& table : arguments_) {
        NS::Error* error = nullptr;
        table = NS::TransferPtr(device.handle()->newArgumentTable(descriptor.get(), &error));
        if (!table) {
            throw Error("Renderer: the device made no argument table: " + describe(error));
        }
    }

    passes_.reserve(schedule.passes.size());
    for (frame::PassKind kind : schedule.passes) {
        // No default: a kind this backend does not implement fails the build
        // (core/frame/schedule.h).
        switch (kind) {
        case frame::PassKind::test_pattern:
            passes_.emplace_back(std::in_place_type<TestPatternPass>, device, library_);
            break;
        }
    }
}

void Renderer::record(const FrameSlot& frame, const frame::FrameInputs& inputs, MTL::Texture* target,
                      frame::Extent size) {
    if (frame.commands == nullptr || target == nullptr || size.width == 0 || size.height == 0) {
        throw Error("Renderer::record: no command buffer, no target, or an empty image");
    }

    const contracts::FrameConstants constants{
        static_cast<float>(inputs.time.count()),
        static_cast<std::uint32_t>(inputs.index),
        size.width,
        size.height,
    };
    const std::size_t offset = std::size_t{frame.slot} * slot_stride;
    std::memcpy(static_cast<std::byte*>(constants_->contents()) + offset, &constants, sizeof(constants));
    const MTL::GPUAddress address = constants_->gpuAddress() + offset;

    MTL4::ArgumentTable* arguments = arguments_[frame.slot].get();
    MTL4::ComputeCommandEncoder* encoder = frame.commands->computeCommandEncoder();
    encoder->setArgumentTable(arguments);
    for (const Pass& pass : passes_) {
        std::visit([&](const auto& p) { p.record(encoder, arguments, address, target, size); }, pass);
    }
    encoder->endEncoding();
}

bool render_to_window(Submission& submission, Presenter& presenter, Renderer& renderer,
                      const frame::FrameInputs& inputs) {
    auto drained = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    CA::MetalDrawable* drawable = presenter.acquire();
    if (drawable == nullptr) {
        return false;
    }
    const FrameSlot frame = submission.begin();
    renderer.record(frame, inputs, drawable->texture(), presenter.size());
    submission.present(drawable);
    return true;
}

std::uint64_t render_to_offscreen(Submission& submission, Offscreen& target, Renderer& renderer,
                                  const frame::FrameInputs& inputs) {
    const FrameSlot frame = submission.begin();
    renderer.record(frame, inputs, target.texture(), target.size());
    submission.commit();
    return frame.sequence;
}

}  // namespace serenity::metal
