#include "cuda/scene/scene_buffers.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/scene/scene_arrays.h"
#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

// Where each array starts: 256-byte aligned (the driver's own alignment for
// an allocation, more than any layout needs), after a zeroed block of that
// size for the empty ones.
constexpr std::size_t array_alignment = 256;

std::size_t aligned(std::size_t bytes) {
  return (bytes + array_alignment - 1) / array_alignment * array_alignment;
}

}  // namespace

SceneBuffers::SceneBuffers(Submission& submission,
                           const scene::SceneDescription& scene) {
  const std::array<std::span<const std::byte>, scene::scene_array_count>
      arrays = scene::scene_arrays(scene);
  std::array<std::size_t, scene::scene_array_count> offsets{};
  std::size_t total = array_alignment;  // the zeroed block
  for (std::size_t i = 0; i < arrays.size(); ++i) {
    offsets.at(i) = arrays.at(i).empty() ? 0 : total;
    total += aligned(arrays.at(i).size());
  }
  // The arrays laid out on the host as they will be on the device, padding
  // and the zeroed block zero, and copied in one call.
  std::vector<std::byte> laid_out(total, std::byte{0});
  for (std::size_t i = 0; i < arrays.size(); ++i) {
    std::ranges::copy(
        arrays.at(i),
        laid_out.begin() + static_cast<std::ptrdiff_t>(offsets.at(i)));
  }
  arrays_ = DeviceBuffer(total);
  // Before any frame (the constructor's contract).
  submission.finish();
  upload_now(arrays_.address(), laid_out, "SceneBuffers");

  std::array<std::uint64_t, scene::scene_array_count> addresses{};
  for (std::size_t i = 0; i < addresses.size(); ++i) {
    addresses.at(i) = arrays_.address() + offsets.at(i);
  }
  block_ = scene::scene_block(addresses);
  block_buffer_ = DeviceBuffer(sizeof(block_));
  upload_now(block_buffer_.address(), std::as_bytes(std::span(&block_, 1)),
             "SceneBuffers");
}

}  // namespace serenity::cuda
