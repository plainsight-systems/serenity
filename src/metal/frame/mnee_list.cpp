#include "metal/frame/mnee_list.h"

#include <cstdint>
#include <cstring>
#include <string>

#include "core/contracts/mnee_request.h"
#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

namespace {

// The header's bytes: the count and capacity, then the grid at
// mnee_grid_offset (contract 16).
constexpr std::size_t header_bytes = contracts::mnee_grid_offset * 2;

}  // namespace

MneeList::MneeList(const Device& device, Submission& submission, Light light)
    : device_(NS::RetainPtr(device.handle())), submission_(submission), light_kind_(light) {}

void MneeList::make(Array& array, std::size_t bytes, MTL::ResourceOptions storage, const char* what) {
    array.buffer = NS::TransferPtr(device_->newBuffer(bytes, storage));
    if (!array.buffer) {
        throw MetalError(std::string("MneeList: the device made no buffer for the ") + what);
    }
    array.resident = submission_.keep_resident(array.buffer.get());
}

void MneeList::prepare(frame::Extent size) {
    check_texture_size(size, "MNEE's work list");
    if (size == size_) {
        return;
    }
    size_ = {};
    if (requests_.buffer) {
        submission_.drain();  // frames in flight may still read the old arrays
    }
    for (Array* array : {&requests_, &header_, &light_}) {
        array->resident.reset();
        array->buffer.reset();
    }
    const auto pool = scoped_pool();
    const std::size_t pixels = std::size_t{size.width} * size.height;
    const std::size_t capacity = pixels;  // one request a pixel (contract 16)
    make(requests_, capacity * sizeof(contracts::MneeRequest), MTL::ResourceStorageModePrivate, "requests");
    make(header_, header_bytes, MTL::ResourceStorageModeShared, "header");
    std::memset(header_.buffer->contents(), 0, header_bytes);
    const auto words = static_cast<std::uint32_t*>(header_.buffer->contents());
    words[1] = static_cast<std::uint32_t>(capacity);
    if (light_kind_ == Light::own) {
        make(light_, pixels * 4 * sizeof(float), MTL::ResourceStorageModeShared, "MNEE light");
        std::memset(light_.buffer->contents(), 0, pixels * 4 * sizeof(float));
    }
    size_ = size;
}

MneeList::Addresses MneeList::addresses() const noexcept {
    const auto address = [](const Array& array) { return array.buffer ? array.buffer->gpuAddress() : 0; };
    Addresses out;
    out.requests = address(requests_);
    out.header = address(header_);
    out.grid = out.header != 0 ? out.header + contracts::mnee_grid_offset : 0;
    out.light = address(light_);
    return out;
}

}  // namespace serenity::metal
