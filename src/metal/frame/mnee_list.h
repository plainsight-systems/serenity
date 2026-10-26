#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <array>
#include <cstddef>
#include <string_view>

#include "core/contracts/mnee_request.h"
#include "core/frame/extent.h"
#include "metal/device/device.h"
#include "metal/device/resident.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Frame (MNEE's work list's storage).
//
// The arrays manifold next event estimation's two halves share within a
// frame (contract 16, core/contracts/mnee_request.h; metal/passes/mnee/
// mnee.h), a ReSTIR PT graph's:
//
//   requests  one a pixel (contract 16), 128 bytes: 988 MB at 3456 x 2234,
//             265 MB at 1920 x 1080; private to the GPU;
//   header    the count and capacity, then the pass's indirect grid, in
//             one small shared buffer the CPU writes once (the grid kernel
//             sets the count back to 0);
//   walks     the walks between rounds (contract 16's MneeWalk, 96 bytes):
//             two round lists and the converged list, each the requests'
//             capacity (a round appends no more walks than it reads), 2.2
//             GB together at 3456 x 2234 beside the requests' 988 MB, 597
//             MB at 1920 x 1080; private to the GPU;
//   walk header  their three counts, in one small shared buffer the CPU
//             zeroes once (each grid kernel sets the count it reads back
//             to 0).
//
// Nothing is carried from frame to frame but zeros. Made at the frame's
// size, remade when it changes, settled first (frames in flight may still
// read the old arrays). Throws MetalError if the device cannot make a
// buffer.
class MneeList {
 public:
  MneeList(const Device& device, Submission& submission);

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
    std::array<MTL::GPUAddress, contracts::mnee_walk_lists> walks{};
    MTL::GPUAddress converged = 0;
    MTL::GPUAddress walk_header = 0;  // contract 16's counts, a word each
  };
  Addresses addresses() const noexcept;

 private:
  void make(ResidentBuffer& array, std::size_t bytes,
            MTL::ResourceOptions storage, std::string_view what);

  NS::SharedPtr<MTL::Device> device_;
  Submission& submission_;
  frame::Extent size_;
  ResidentBuffer requests_;
  ResidentBuffer header_;
  std::array<ResidentBuffer, contracts::mnee_walk_lists> walks_;
  ResidentBuffer converged_;
  ResidentBuffer walk_header_;
};

}  // namespace serenity::metal
