#pragma once

#include <cuda.h>

#include "core/frame/extent.h"
#include "cuda/device/buffer.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: Frame graph (MNEE's work list's storage).
//
// The arrays manifold next event estimation's two halves share within a
// frame (contract 16, core/contracts/mnee_request.h; cuda/passes/mnee/
// mnee.h), Metal's MneeList (metal/frame/mnee_list.h, which gives their
// sizes): the requests, one a pixel; and the header, the count, which the
// trees append to and the settling sets back to 0 after the solve, and the
// capacity, written once with each make. In device memory, the header
// written by the host before any frame that uses it. Nothing is carried
// from frame to frame but zeros. Made at the frame's size, remade when it
// changes, the frames in flight finished first (they may still read the
// old arrays). Throws CudaError if the memory cannot be made or written.
class MneeList {
 public:
  explicit MneeList(Submission& submission);

  MneeList(const MneeList&) = delete;
  MneeList& operator=(const MneeList&) = delete;
  MneeList(MneeList&&) = delete;
  MneeList& operator=(MneeList&&) = delete;
  ~MneeList() = default;

  void prepare(frame::Extent size);

  CUdeviceptr requests() const noexcept { return requests_.address(); }
  CUdeviceptr header() const noexcept { return header_.address(); }

 private:
  Submission& submission_;
  frame::Extent size_;
  DeviceBuffer requests_;
  DeviceBuffer header_;
};

}  // namespace serenity::cuda
