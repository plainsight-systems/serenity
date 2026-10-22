#include "metal/frame/frame_images.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>

#include "core/passes/bloom.h"
#include "core/passes/restir_pt_arrays.h"
#include "metal/device/error.h"
#include "metal/device/indirect_grid.h"
#include "metal/device/resident.h"
#include "metal/device/support.h"

namespace serenity::metal {

FrameImages::FrameImages(const Device& device, Submission& submission,
                         Radiance radiance, Bloom bloom)
    : device_(NS::RetainPtr(device.handle())),
      submission_(submission),
      radiance_kind_(radiance),
      bloom_(bloom) {}

MTL::Texture* FrameImages::bloom(std::uint32_t level) const {
  if (level >= pyramid_.size()) {
    throw MetalError("FrameImages: no bloom level " + std::to_string(level));
  }
  return pyramid_.at(level).texture.get();
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
  const auto release = [&](ResidentTexture& image) {
    if (!image) {
      return;
    }
    if (!settled_in_flight) {
      // Settled here, so a failure in a frame in flight is reported.
      submission_.drain();
      settled_in_flight = true;
    }
    image.reset();
  };
  release(radiance_);
  for (auto& level : pyramid_) {
    release(level);
  }
  const auto pool = scoped_pool();
  const auto make = [&](ResidentTexture& image, MTL::PixelFormat format,
                        frame::Extent at, std::string_view what) {
    image = make_resident_texture(device_.get(), submission_, format, at, what);
  };
  if (radiance_kind_ == Radiance::image) {
    make(radiance_, MTL::PixelFormatRGBA32Float, size, "radiance image");
  }
  if (bloom_ == Bloom::pyramid) {
    frame::Extent level = size;
    for (auto& image : pyramid_) {
      level = passes::bloom_level(level);
      make(image, MTL::PixelFormatRGBA16Float, level, "bloom level");
    }
  }
  size_ = size;
}

static_assert(PathReuseImages::grid_offset >= sizeof(std::uint32_t) &&
              PathReuseImages::grid_offset + sizeof(gpu::IndirectGrid) <=
                  PathReuseImages::counter_bytes);
static_assert(PathReuseImages::grid_offset % alignof(gpu::IndirectGrid) == 0 &&
              PathReuseImages::grid_offset % constant_offset_alignment == 0);

PathReuseImages::PathReuseImages(const Device& device, Submission& submission,
                                 Spatial spatial, Temporal temporal, Grid grid,
                                 std::uint32_t lights)
    : device_(NS::RetainPtr(device.handle())),
      submission_(submission),
      spatial_(spatial),
      temporal_(temporal),
      grid_(grid) {
  // The count starts at 0, and the grid pass keeps setting it back.
  const auto pool = scoped_pool();
  counter_ = make_resident_buffer(device_.get(), submission_, counter_bytes,
                                  MTL::ResourceStorageModeShared,
                                  "ReSTIR PT's replay count", Zeroed::yes);
  make(light_tiles_, passes::light_tiles_bytes(), "light tiles");
  for (std::size_t h = 0; h < (temporal == Temporal::merged ? 2u : 1u); ++h) {
    make(light_power_.at(h), passes::light_power_bytes(lights),
         "power pyramid");
  }
}

void PathReuseImages::make(Array& array, std::size_t bytes,
                           std::string_view what) {
  array = make_resident_buffer(device_.get(), submission_, bytes,
                               MTL::ResourceStorageModePrivate,
                               "ReSTIR PT's " + std::string(what));
}

std::uint64_t PathReuseImages::prepare(const frame::FrameInputs& inputs,
                                       frame::Extent size) {
  check_texture_size(size, "ReSTIR PT's arrays");
  const bool temporal = temporal_ == Temporal::merged;
  // The history joined on a copy, kept only once the arrays that hold
  // the frame exist: a throw below leaves it claiming nothing they never
  // held (E.4), as Accumulation's (accumulation.cpp).
  frame::ReuseHistory joined = history_;
  std::uint64_t held = 0;
  if (temporal) {
    try {
      held = joined.join(inputs, size).held;
    } catch (const frame::HistoryError& refused) {
      throw MetalError(std::string("ReSTIR PT's reuse history: ") +
                       refused.what());
    }
  }
  if (size == size_) {
    history_ = joined;
    return held;
  }
  // A new size starts the history over (the rule refuses one that does
  // not), so nothing of the old arrays is read again; frames in flight
  // may still read them, so settled first.
  size_ = {};
  const auto all = {&reservoirs_,      &primaries_[0], &primaries_[1],
                    &light_,           &shifts_,       &replays_,
                    &finals_[0],       &finals_[1],    &previous_positions_,
                    &previous_pixels_, &duplication_};
  if (std::ranges::any_of(
          all, [](const Array* array) { return static_cast<bool>(*array); })) {
    submission_.drain();
  }
  for (Array* array : all) {
    array->reset();
  }
  const auto pool = scoped_pool();
  const passes::PathReuseArrays sizes = passes::path_reuse_arrays(
      size, spatial_ == Spatial::merged, temporal, grid_ == Grid::checkerboard);
  make(reservoirs_, sizes.reservoirs, "path reservoirs");
  for (std::size_t h = 0; h < sizes.copies; ++h) {
    make(primaries_.at(h), sizes.primaries, "primary hits");
  }
  make(light_, sizes.light, "light buffer");
  if (sizes.shifts != 0) {
    make(shifts_, sizes.shifts, "shifted paths");
    make(replays_, sizes.replays, "replaying shifts");
  }
  if (sizes.finals != 0) {
    for (Array& copy : finals_) {
      make(copy, sizes.finals, "final path reservoirs");
    }
    make(previous_positions_, sizes.previous_positions, "previous positions");
    make(previous_pixels_, sizes.previous_pixels, "previous pixels");
    make(duplication_, sizes.duplication, "duplication map");
  }
  size_ = size;
  history_ = joined;
  return held;
}

PathReuseImages::Addresses PathReuseImages::addresses(
    std::uint64_t index) const noexcept {
  const auto address = [](const Array& array) { return array.address(); };
  const bool temporal = temporal_ == Temporal::merged;
  const std::size_t now = passes::this_frame_copy(index, temporal);
  const std::size_t then = 1 - now;
  Addresses out;
  out.reservoirs = address(reservoirs_);
  out.primaries = address(primaries_.at(now));
  out.light = address(light_);
  out.shifts = address(shifts_);
  out.replays = address(replays_);
  out.counter = address(counter_);
  out.grid = out.counter + grid_offset;
  out.light_tiles = address(light_tiles_);
  out.light_power = address(light_power_.at(now));
  if (temporal) {
    out.last_primaries = address(primaries_.at(then));
    out.last_light_power = address(light_power_.at(then));
    out.finals = address(finals_.at(now));
    out.last_finals = address(finals_.at(then));
    out.previous_positions = address(previous_positions_);
    out.previous_pixels = address(previous_pixels_);
    out.duplication = address(duplication_);
  }
  return out;
}

}  // namespace serenity::metal
