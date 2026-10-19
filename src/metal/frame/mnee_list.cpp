#include "metal/frame/mnee_list.h"

#include <cstdint>
#include <cstring>
#include <string>

#include "core/contracts/mnee_request.h"
#include "metal/device/error.h"
#include "metal/device/indirect_grid.h"
#include "metal/device/support.h"

namespace serenity::metal {

namespace {

// The header's bytes: the count and capacity, then the grid at
// mnee_grid_offset (contract 16).
constexpr std::size_t header_bytes = contracts::mnee_header_bytes;
static_assert(contracts::mnee_grid_offset + sizeof(gpu::IndirectGrid) <=
              header_bytes);
static_assert(contracts::mnee_grid_offset % alignof(gpu::IndirectGrid) == 0 &&
              contracts::mnee_grid_offset % constant_offset_alignment == 0);

}  // namespace

MneeList::MneeList(const Device& device, Submission& submission)
    : device_(NS::RetainPtr(device.handle())), submission_(submission) {}

void MneeList::make(ResidentBuffer& array, std::size_t bytes,
                    MTL::ResourceOptions storage, std::string_view what) {
  array = make_resident_buffer(device_.get(), submission_, bytes, storage,
                               "MNEE's " + std::string(what));
}

void MneeList::prepare(frame::Extent size) {
  check_texture_size(size, "MNEE's work list");
  if (size == size_) {
    return;
  }
  size_ = {};
  if (requests_) {
    submission_.drain();  // frames in flight may still read the old arrays
  }
  for (ResidentBuffer* array : {&requests_, &header_}) {
    array->reset();
  }
  const auto pool = scoped_pool();
  const std::size_t pixels = std::size_t{size.width} * size.height;
  const std::size_t capacity = pixels;  // one request a pixel (contract 16)
  make(requests_, capacity * sizeof(contracts::MneeRequest),
       MTL::ResourceStorageModePrivate, "requests");
  make(header_, header_bytes, MTL::ResourceStorageModeShared, "header");
  std::memset(header_.buffer->contents(), 0, header_bytes);
  const auto words = static_cast<std::uint32_t*>(header_.buffer->contents());
  words[contracts::mnee_capacity_word] = static_cast<std::uint32_t>(capacity);
  size_ = size;
}

MneeList::Addresses MneeList::addresses() const noexcept {
  Addresses out;
  out.requests = requests_.address();
  out.header = header_.address();
  out.grid = out.header != 0 ? out.header + contracts::mnee_grid_offset : 0;
  return out;
}

}  // namespace serenity::metal
