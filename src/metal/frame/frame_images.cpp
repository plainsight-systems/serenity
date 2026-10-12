#include "metal/frame/frame_images.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <initializer_list>
#include <string>

#include "core/contracts/float3.h"
#include "core/contracts/light_tile.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/primary_hit.h"
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
    : device_(NS::RetainPtr(device.handle())), submission_(submission), temporal_(temporal), working_kind_(working) {}

void ReuseImages::make(Array& array, std::size_t bytes, const char* what) {
    array.buffer = NS::TransferPtr(device_->newBuffer(bytes, MTL::ResourceStorageModePrivate));
    if (!array.buffer) {
        throw MetalError(std::string("ReuseImages: the device made no buffer for the ") + what);
    }
    array.resident = submission_.keep_resident(array.buffer.get());
}

std::uint64_t ReuseImages::prepare(const frame::FrameInputs& inputs, frame::Extent size) {
    check_texture_size(size, "ReSTIR DI's arrays");
    const bool temporal = temporal_ == Temporal::merged;
    std::uint64_t held = 0;
    if (temporal) {
        try {
            held = history_.join(inputs, size).held;
        } catch (const frame::HistoryError& refused) {
            throw MetalError(std::string("ReSTIR DI's reuse history: ") + refused.what());
        }
    }
    if (size == size_) {
        return held;
    }
    // A new size starts the history over (the rule refuses one that does
    // not), so nothing of the old arrays is read again. Frames in flight
    // may still read them: settled first, as FrameImages settles its
    // images, and untimed, a resize's frame or two.
    size_ = {};
    const auto all = {&records_[0], &records_[1], &path_states_, &previous_positions_, &working_,
                      &final_reservoirs_[0], &final_reservoirs_[1]};
    if (std::ranges::any_of(all, [](const Array* array) { return static_cast<bool>(array->buffer); })) {
        submission_.drain();
    }
    for (Array* array : all) {
        array->resident.reset();
        array->buffer.reset();
    }
    const auto pool = scoped_pool();
    const std::size_t pixels = std::size_t{size.width} * size.height;
    const std::size_t halves = temporal ? 2 : 1;
    for (std::size_t half = 0; half < halves; ++half) {
        make(records_[half], pixels * sizeof(contracts::SurfaceRecord), "surface records");
        make(final_reservoirs_[half], pixels * sizeof(contracts::Reservoir), "final reservoirs");
    }
    make(path_states_, pixels * sizeof(contracts::PathState), "path states");
    if (temporal) {
        make(previous_positions_, pixels * sizeof(contracts::Float3), "previous positions");
    }
    if (working_kind_ == Working::needed) {
        make(working_, pixels * sizeof(contracts::Reservoir), "working reservoirs");
    }
    size_ = size;
    return held;
}

ReuseImages::Addresses ReuseImages::addresses(std::uint64_t index) const noexcept {
    const auto address = [](const Array& array) { return array.buffer ? array.buffer->gpuAddress() : 0; };
    // This frame's half by its parity, last frame's the other; one half
    // without a temporal merge.
    const std::size_t now = temporal_ == Temporal::merged ? index % 2 : 0;
    const std::size_t then = 1 - now;
    Addresses out;
    out.records = address(records_[now]);
    out.path_states = address(path_states_);
    out.previous_positions = address(previous_positions_);
    out.working = address(working_);
    out.final_reservoirs = address(final_reservoirs_[now]);
    if (temporal_ == Temporal::merged) {
        out.last_records = address(records_[then]);
        out.last_final = address(final_reservoirs_[then]);
    }
    return out;
}

PathReuseImages::PathReuseImages(const Device& device, Submission& submission, Spatial spatial, Temporal temporal)
    : device_(NS::RetainPtr(device.handle())), submission_(submission), spatial_(spatial), temporal_(temporal) {
    // The count starts at 0, and the grid pass keeps setting it back.
    const auto pool = scoped_pool();
    counter_.buffer = NS::TransferPtr(device_->newBuffer(grid_offset * 2, MTL::ResourceStorageModeShared));
    if (!counter_.buffer) {
        throw MetalError("PathReuseImages: the device made no buffer for the replay count");
    }
    std::memset(counter_.buffer->contents(), 0, grid_offset * 2);
    counter_.resident = submission_.keep_resident(counter_.buffer.get());
    make(light_tiles_, std::size_t{contracts::light_tiles} * contracts::light_tile_size * sizeof(contracts::LightTileEntry),
         "light tiles");
}

void PathReuseImages::make(Array& array, std::size_t bytes, const char* what) {
    array.buffer = NS::TransferPtr(device_->newBuffer(bytes, MTL::ResourceStorageModePrivate));
    if (!array.buffer) {
        throw MetalError(std::string("PathReuseImages: the device made no buffer for the ") + what);
    }
    array.resident = submission_.keep_resident(array.buffer.get());
}

std::uint64_t PathReuseImages::prepare(const frame::FrameInputs& inputs, frame::Extent size) {
    check_texture_size(size, "ReSTIR PT's arrays");
    const bool temporal = temporal_ == Temporal::merged;
    std::uint64_t held = 0;
    if (temporal) {
        try {
            held = history_.join(inputs, size).held;
        } catch (const frame::HistoryError& refused) {
            throw MetalError(std::string("ReSTIR PT's reuse history: ") + refused.what());
        }
    }
    if (size == size_) {
        return held;
    }
    // A new size starts the history over (the rule refuses one that does
    // not), so nothing of the old arrays is read again; frames in flight
    // may still read them, so settled first.
    size_ = {};
    const auto all = {&reservoirs_, &primaries_[0], &primaries_[1], &light_,           &shifts_,
                      &replays_,    &finals_[0],    &finals_[1],    &previous_positions_, &previous_pixels_,
                      &duplication_};
    if (std::ranges::any_of(all, [](const Array* array) { return static_cast<bool>(array->buffer); })) {
        submission_.drain();
    }
    for (Array* array : all) {
        array->resident.reset();
        array->buffer.reset();
    }
    const auto pool = scoped_pool();
    const std::size_t pixels = std::size_t{size.width} * size.height;
    const std::size_t halves = temporal ? 2 : 1;
    make(reservoirs_, pixels * sizeof(contracts::PathReservoir), "path reservoirs");
    for (std::size_t half = 0; half < halves; ++half) {
        make(primaries_[half], pixels * sizeof(contracts::PrimaryHit), "primary hits");
    }
    make(light_, pixels * 4 * sizeof(float), "light buffer");
    if (spatial_ == Spatial::merged || temporal) {
        make(shifts_, pixels * contracts::max_path_neighbors * 4 * sizeof(float), "shifted paths");
        make(replays_, pixels * contracts::max_path_neighbors * sizeof(std::uint32_t), "replaying shifts");
    }
    if (temporal) {
        for (Array& half : finals_) {
            make(half, pixels * sizeof(contracts::PathReservoir), "final path reservoirs");
        }
        make(previous_positions_, pixels * sizeof(contracts::Float3), "previous positions");
        make(previous_pixels_, pixels * sizeof(std::uint32_t), "previous pixels");
        make(duplication_, pixels * sizeof(float), "duplication map");
    }
    size_ = size;
    return held;
}

PathReuseImages::Addresses PathReuseImages::addresses(std::uint64_t index) const noexcept {
    const auto address = [](const Array& array) { return array.buffer ? array.buffer->gpuAddress() : 0; };
    const bool temporal = temporal_ == Temporal::merged;
    const std::size_t now = temporal ? index % 2 : 0;
    const std::size_t then = 1 - now;
    Addresses out;
    out.reservoirs = address(reservoirs_);
    out.primaries = address(primaries_[now]);
    out.light = address(light_);
    out.shifts = address(shifts_);
    out.replays = address(replays_);
    out.counter = address(counter_);
    out.grid = out.counter + grid_offset;
    out.light_tiles = address(light_tiles_);
    if (temporal) {
        out.last_primaries = address(primaries_[then]);
        out.finals = address(finals_[now]);
        out.last_finals = address(finals_[then]);
        out.previous_positions = address(previous_positions_);
        out.previous_pixels = address(previous_pixels_);
        out.duplication = address(duplication_);
    }
    return out;
}

}  // namespace serenity::metal
