#include "metal/scene/scene_buffers.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "core/scene/scene_arrays.h"
#include "metal/device/resident.h"

namespace serenity::metal {

SceneBuffers::SceneBuffers(const Device& device, Submission& submission,
                           const scene::SceneDescription& scene)
    : arrays_(device, submission, scene::scene_arrays(scene)) {
  std::array<std::uint64_t, scene::scene_array_count> addresses{};
  for (std::size_t i = 0; i < addresses.size(); ++i) {
    addresses.at(i) = arrays_.address(i);
  }
  block_ = scene::scene_block(addresses);
  // The block itself, written once (core/scene/scene_block.h); shared storage,
  // which Apple silicon's unified memory lets the GPU read where the CPU wrote.
  block_buffer_ =
      make_resident_buffer(device.handle(), submission, sizeof(block_),
                           MTL::ResourceStorageModeShared, "the scene's block");
  std::memcpy(block_buffer_.buffer->contents(), &block_, sizeof(block_));
}

}  // namespace serenity::metal
