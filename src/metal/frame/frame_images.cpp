#include "metal/frame/frame_images.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <string>

#include "core/contracts/reservoir.h"
#include "core/contracts/surface_record.h"

#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

namespace {

// Half of `size` on each axis, rounded up, at least 1 (core/passes/tone_map.h,
// step 2).
constexpr frame::Extent half(frame::Extent size) noexcept {
    return {std::max(1u, (size.width + 1) / 2), std::max(1u, (size.height + 1) / 2)};
}

}  // namespace

FrameImages::FrameImages(const Device& device, Submission& submission, Radiance radiance, Bloom bloom)
    : device_(NS::RetainPtr(device.handle())), submission_(submission), radiance_kind_(radiance), bloom_(bloom) {}

MTL::Texture* FrameImages::bloom(std::uint32_t level) const {
    if (level >= pyramid_.size()) {
        throw MetalError("FrameImages: no bloom level " + std::to_string(level));
    }
    return pyramid_[level].texture.get();
}

void FrameImages::prepare(frame::Extent size) {
    check_texture_size(size, "FrameImages");
    if (size == size_) {
        return;
    }
    // Frames in flight may still read the old images, whole or, after a
    // failed prepare(), in part. What the drain settles goes untimed: a
    // resize's frame or two.
    size_ = {};
    bool settled_in_flight = false;
    const auto release = [&](Image& image) {
        if (!image.texture) {
            return;
        }
        if (!settled_in_flight) {
            // Settled here, so a failure in a frame in flight is reported.
            submission_.drain();
            settled_in_flight = true;
        }
        image.resident.reset();
        image.texture.reset();
    };
    release(radiance_);
    for (auto& level : pyramid_) {
        release(level);
    }
    const auto pool = scoped_pool();
    const auto make = [&](Image& image, MTL::PixelFormat format, frame::Extent at, const char* what) {
        image.texture = make_private_texture(device_.get(), format, at, what);
        image.resident = submission_.keep_resident(image.texture.get());
    };
    if (radiance_kind_ == Radiance::image) {
        make(radiance_, MTL::PixelFormatRGBA32Float, size, "radiance image");
    }
    if (bloom_ == Bloom::pyramid) {
        frame::Extent level = size;
        for (auto& image : pyramid_) {
            level = half(level);
            make(image, MTL::PixelFormatRGBA16Float, level, "bloom level");
        }
    }
    size_ = size;
}

ReuseImages::ReuseImages(const Device& device, Submission& submission, Temporal temporal, Working working)
    : device_(NS::RetainPtr(device.handle())), submission_(submission) {
    // Refused, not made without them (frame_images.h).
    if (temporal != Temporal::none) {
        throw MetalError("ReuseImages: last frame's records and reservoirs are restir_temporal's, which this backend "
                         "does not implement yet");
    }
    if (working != Working::none) {
        throw MetalError("ReuseImages: the working reservoirs are the merges', restir_temporal's and "
                         "restir_spatial's, which this backend does not implement yet");
    }
}

void ReuseImages::make(Array& array, std::size_t bytes, const char* what) {
    array.buffer = NS::TransferPtr(device_->newBuffer(bytes, MTL::ResourceStorageModePrivate));
    if (!array.buffer) {
        throw MetalError(std::string("ReuseImages: the device made no buffer for the ") + what);
    }
    array.resident = submission_.keep_resident(array.buffer.get());
}

std::uint64_t ReuseImages::prepare(const frame::FrameInputs& /*inputs: the temporal merge's history's*/,
                                   frame::Extent size) {
    check_texture_size(size, "ReSTIR DI's arrays");
    if (size == size_) {
        return 0;
    }
    // Frames in flight may still read the old arrays: settled first, as
    // FrameImages settles its images, and untimed, a resize's frame or two.
    size_ = {};
    if (records_.buffer || path_states_.buffer || final_reservoirs_.buffer) {
        submission_.drain();
    }
    for (Array* array : {&records_, &path_states_, &final_reservoirs_}) {
        array->resident.reset();
        array->buffer.reset();
    }
    const auto pool = scoped_pool();
    const std::size_t pixels = std::size_t{size.width} * size.height;
    make(records_, pixels * sizeof(contracts::SurfaceRecord), "surface records");
    make(path_states_, pixels * sizeof(contracts::PathState), "path states");
    make(final_reservoirs_, pixels * sizeof(contracts::Reservoir), "final reservoirs");
    size_ = size;
    return 0;
}

ReuseImages::Addresses ReuseImages::addresses(std::uint64_t /*index: the ping-pongs' parity, with them*/) const noexcept {
    const auto address = [](const Array& array) { return array.buffer ? array.buffer->gpuAddress() : 0; };
    Addresses out;
    out.records = address(records_);
    out.path_states = address(path_states_);
    out.final_reservoirs = address(final_reservoirs_);
    return out;
}

}  // namespace serenity::metal
