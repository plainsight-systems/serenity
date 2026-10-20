#pragma once

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <type_traits>

#include "cuda/device/buffer.h"
#include "cuda/device/error.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: GPU backend.
//
// An array of bytes the CPU rewrites while frames are in flight, on a GPU
// whose memory is not the CPU's: Metal's FrameArray (metal/device/
// frame_array.h) for the RTX 3090 Ti (docs/research/2026-10-19-cuda-
// backend.md, "Phase 3's design"). The CPU stages each frame's bytes in
// pinned host memory, one copy a frame slot, and upload() copies the
// frame's copy on the frame's stream into the one copy on the device, which
// the frame's kernels and programs read. It knows nothing of what the bytes
// are; whoever hands them over says (cuda/scene/shape_transforms.h, a
// pass's launch parameters).
//
// Why one device copy and a staging copy a slot. The stream orders frame
// n + 1's upload after frame n's work, so no frame's work reads the device
// copy while the next frame's bytes arrive in it, and one device copy is
// enough. The copy from the host runs when the stream reaches it, after the
// call returns, so the bytes it copies must not change until it has: a
// staging copy a slot, written only once the slot's last frame has
// completed (GPU.7: reusing one staging buffer every frame is a hidden
// fence; MEM.4). Pinned, so the copy is the copy engine's and runs in the
// stream's order rather than being staged by the driver first.
//
// When the CPU may write a copy: bytes(slot) is slot `slot`'s, to write only
// between Submission::begin() returning the slot and the frame's upload();
// the frame that last used the slot has then completed (begin() waited for
// it), and its upload with it. The device copy and every staging copy
// start as the bytes given, uploaded before the constructor returns, so an
// array no frame rewrites needs no upload() at all.
//
// One device allocation, and the staging copies in one pinned allocation
// at a stride of `alignment` bytes (GPU.9): made once, nothing allocated a
// frame (MEM.9). Per frame: one copy of the array's bytes, on the stream.
//
// Throws CudaError if `initial` is empty or the driver cannot make or fill
// the memory; bytes(), view() and upload() throw CudaError for a slot that
// is not one (I.6).
class FrameUpload {
 public:
  FrameUpload(Submission& submission, std::span<const std::byte> initial);

  FrameUpload(const FrameUpload&) = delete;
  FrameUpload& operator=(const FrameUpload&) = delete;
  FrameUpload(FrameUpload&&) = delete;
  FrameUpload& operator=(FrameUpload&&) = delete;
  ~FrameUpload() = default;

  // Frame slot `slot`'s staging copy, for the CPU to write; see above. Not
  // const: it hands out the array to write (Con.2).
  std::span<std::byte> bytes(std::uint32_t slot);

  // The same copy as the array of T it was made from: the one place its
  // bytes are reinterpreted. T must be a type whose bytes are its value
  // (SL.con.4, COPY.6: the copy was made by memcpy), and the copy whole
  // T's, else CudaError (I.6); each copy starts `alignment`-aligned, more
  // than any T here needs.
  template <typename T>
  std::span<T> view(std::uint32_t slot) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "a FrameUpload holds bytes copied from Ts");
    static_assert(alignof(T) <= alignment);
    const std::span<std::byte> copy = bytes(slot);
    if (copy.size() % sizeof(T) != 0) {
      throw CudaError("FrameUpload::view: a copy of " +
                      std::to_string(copy.size()) + " bytes is not whole " +
                      std::to_string(sizeof(T)) + "-byte values");
    }
    // Pinned memory is untyped (cuMemHostAlloc): viewing it as the Ts
    // copied into it is the one cast this interop needs.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return {reinterpret_cast<T*>(copy.data()), copy.size() / sizeof(T)};
  }

  // Copies `frame`'s staging copy to the device copy on its stream.
  void upload(const FrameSlot& frame);

  // The device copy, as a kernel or program reads it.
  CUdeviceptr address() const noexcept { return device_.address(); }

  // Bytes between staging copies: a multiple of every alignment a shared
  // layout has, and of the 256 bytes the driver aligns its allocations to.
  static constexpr std::size_t alignment = 256;

 private:
  std::size_t size_;    // bytes in each copy
  std::size_t stride_;  // bytes between staging copies
  PinnedBuffer staging_;
  DeviceBuffer device_;
};

}  // namespace serenity::cuda
