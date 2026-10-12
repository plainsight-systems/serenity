#pragma once

#include <cstddef>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/extent.h"
#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Frame (MNEE's work list's storage).
//
// The arrays manifold next event estimation's two halves share within a
// frame (contract 16, core/contracts/mnee_request.h; metal/passes/mnee/
// mnee.h), a graph with mnee only:
//
//   requests  one a pixel (contract 16), 128 bytes: 988 MB at 3456 x 2234,
//             265 MB at 1920 x 1080; private to the GPU;
//   header    the count and capacity, then the solve's indirect grid, in
//             one small shared buffer the CPU writes once (the grid kernel
//             sets the count back to 0);
//   light     for the path pass, each pixel's MNEE light before the fold,
//             a float4 a pixel, 16 bytes; shared, zeroed when made, and
//             set back to 0 by the fold where it added any.
//
// Nothing is carried from frame to frame but zeros. Made at the frame's
// size, remade when it changes, settled first (frames in flight may still
// read the old arrays). Throws MetalError if the device cannot make a
// buffer.
class MneeList {
public:
    enum class Light { none, own };

    MneeList(const Device& device, Submission& submission, Light light);

    MneeList(const MneeList&) = delete;
    MneeList& operator=(const MneeList&) = delete;
    MneeList(MneeList&&) = delete;
    MneeList& operator=(MneeList&&) = delete;
    ~MneeList() = default;

    // Ready for a frame at `size`.
    void prepare(frame::Extent size);

    struct Addresses {
        MTL::GPUAddress requests = 0;
        MTL::GPUAddress header = 0;
        MTL::GPUAddress grid = 0;
        MTL::GPUAddress light = 0;  // 0 without Light::own
    };
    Addresses addresses() const noexcept;

private:
    struct Array {
        NS::SharedPtr<MTL::Buffer> buffer;
        Resident resident;
    };

    void make(Array& array, std::size_t bytes, MTL::ResourceOptions storage, const char* what);

    NS::SharedPtr<MTL::Device> device_;
    Submission& submission_;
    Light light_kind_;
    frame::Extent size_;
    Array requests_;
    Array header_;
    Array light_;
};

}  // namespace serenity::metal
