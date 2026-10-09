#include "metal/frame/renderer.h"

#include <cstring>

#include "core/camera/pinhole.h"
#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "metal/device/error.h"
#include "serenity/metallib/shaders.h"

namespace serenity::metal {

namespace {

// Bytes between the ring's slots, and where in a slot each contract is
// (renderer.h): each aligned for any buffer binding Metal takes, with room
// to grow.
constexpr std::size_t slot_stride = 256;
constexpr std::size_t constants_offset = 0;
constexpr std::size_t camera_offset = 128;
static_assert(constants_offset + sizeof(contracts::FrameConstants) <= camera_offset);
static_assert(camera_offset + sizeof(contracts::CameraData) <= slot_stride);

// Every pass binds at most this many buffers and textures through the table:
// the preview binds the constants, the camera, the structure and the scene's
// thirteen arrays (preview.metal).
constexpr NS::UInteger max_buffers = 16;
constexpr NS::UInteger max_textures = 4;

std::string describe(const NS::Error* error) {
    if (error == nullptr || error->localizedDescription() == nullptr) {
        return "Metal gave no description";
    }
    return error->localizedDescription()->utf8String();
}

}  // namespace

Renderer::Renderer(const Device& device, Submission& submission, const frame::Schedule& schedule,
                   const scene::SceneDescription* scene)
    : library_(device, serenity::metallib::shaders) {
    if (schedule.passes.empty()) {
        throw Error("Renderer: the schedule has no passes");
    }
    for (frame::PassKind kind : schedule.passes) {
        needs_scene_ = needs_scene_ || frame::needs_scene(kind);
    }
    if (needs_scene_ && scene == nullptr) {
        throw Error("Renderer: the frame graph's passes read a scene, and none was given (--scene)");
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
        case frame::PassKind::preview:
            passes_.emplace_back(std::in_place_type<PreviewPass>, device, library_);
            break;
        }
    }

    // Only a graph that reads the scene has it put on the GPU.
    if (needs_scene_) {
        scene_ = std::make_unique<SceneBuffers>(device, submission, *scene);
        acceleration_ = std::make_unique<PrimitiveAcceleration>(device, submission, shapes::bounds(scene->shapes));
    }
}

void Renderer::record(const FrameSlot& frame, const frame::FrameInputs& inputs, MTL::Texture* target,
                      frame::Extent size) {
    if (frame.commands == nullptr || target == nullptr || size.width == 0 || size.height == 0) {
        throw Error("Renderer::record: no command buffer, no target, or an empty image");
    }

    if (needs_scene_ && !inputs.camera) {
        throw Error("Renderer::record: the frame graph reads a scene, and the frame has no camera");
    }

    const contracts::FrameConstants constants{
        static_cast<float>(inputs.time.count()),
        static_cast<std::uint32_t>(inputs.index),
        size.width,
        size.height,
    };
    const std::size_t slot = std::size_t{frame.slot} * slot_stride;
    auto* ring = static_cast<std::byte*>(constants_->contents());
    std::memcpy(ring + slot + constants_offset, &constants, sizeof(constants));

    FrameResources resources;
    resources.arguments = arguments_[frame.slot].get();
    resources.constants = constants_->gpuAddress() + slot + constants_offset;
    resources.target = target;
    resources.size = size;
    if (inputs.camera) {
        const contracts::CameraData camera = camera::shader_form(*inputs.camera, size);
        std::memcpy(ring + slot + camera_offset, &camera, sizeof(camera));
        resources.camera = constants_->gpuAddress() + slot + camera_offset;
    }
    if (scene_) {
        resources.scene = &scene_->addresses();
        resources.acceleration = acceleration_.get();
    }

    MTL4::ComputeCommandEncoder* encoder = frame.commands->computeCommandEncoder();
    encoder->setArgumentTable(resources.arguments);
    for (const Pass& pass : passes_) {
        std::visit([&](const auto& p) { p.record(encoder, resources); }, pass);
    }
    encoder->endEncoding();
}

std::optional<WindowFrame> render_to_window(Submission& submission, Presenter& presenter, Renderer& renderer,
                                            const frame::FrameInputs& inputs) {
    auto drained = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    CA::MetalDrawable* drawable = presenter.acquire();
    if (drawable == nullptr) {
        return std::nullopt;
    }
    const FrameSlot frame = submission.begin();
    renderer.record(frame, inputs, drawable->texture(), presenter.size());
    submission.present(drawable);
    return WindowFrame{frame.sequence, frame.settled};
}

std::uint64_t render_to_offscreen(Submission& submission, Offscreen& target, Renderer& renderer,
                                  const frame::FrameInputs& inputs) {
    const FrameSlot frame = submission.begin();
    renderer.record(frame, inputs, target.texture(), target.size());
    submission.commit();
    return frame.sequence;
}

}  // namespace serenity::metal
