#pragma once

#include <cstddef>
#include <string_view>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/extent.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: GPU backend.
//
// A GPU allocation with its residency, released together, the residency
// first (submission.h, Lifetime): what every image, array and counter the
// backend makes for its frames holds, written once here (ES.3, SF.3) where
// each owner wrote its own struct, its make(), its address-or-0 and its
// release. Move-only, as Resident is; an empty one holds nothing and has
// address 0.

struct ResidentBuffer {
    NS::SharedPtr<MTL::Buffer> buffer;
    Resident resident;

    explicit operator bool() const noexcept { return static_cast<bool>(buffer); }
    // Its address, as a shader binds it; 0 when empty.
    MTL::GPUAddress address() const noexcept { return buffer ? buffer->gpuAddress() : 0; }
    // Empties it: the residency, then the buffer.
    void reset() noexcept {
        resident.reset();
        buffer.reset();
    }
};

struct ResidentTexture {
    NS::SharedPtr<MTL::Texture> texture;
    Resident resident;

    explicit operator bool() const noexcept { return static_cast<bool>(texture); }
    void reset() noexcept {
        resident.reset();
        texture.reset();
    }
};

// Whether a shared buffer starts as zeros or as whatever it holds (I.4).
enum class Zeroed { no, yes };

// A buffer of `bytes` in `storage`, made resident through `submission`;
// a shared one zeroed when `zeroed` says. Throws MetalError, naming `what`,
// if the device makes none. Called inside an autorelease pool.
ResidentBuffer make_resident_buffer(MTL::Device* device, Submission& submission, std::size_t bytes,
                                    MTL::ResourceOptions storage, std::string_view what,
                                    Zeroed zeroed = Zeroed::no);

// A private image (support.h, make_private_texture), made resident through
// `submission`. Throws as make_private_texture() does.
ResidentTexture make_resident_texture(MTL::Device* device, Submission& submission, MTL::PixelFormat format,
                                      frame::Extent size, std::string_view what);

}  // namespace serenity::metal
