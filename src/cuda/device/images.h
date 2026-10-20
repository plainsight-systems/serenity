#pragma once

#include <cuda.h>

#include <cstddef>
#include <span>

#include "core/frame/extent.h"
#include "cuda/device/device.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: GPU backend.
//
// The images a frame's passes write and read on CUDA (docs/research/
// 2026-10-19-cuda-backend.md, "Phase 3's design"), Metal's textures for
// this machine: what each pixel holds is its owner's (cuda/frame/
// accumulation.h, frame_images.h); how it is laid out, written and sampled
// is here. Every one samples as Metal's one sampler does: bilinear, clamped
// to the edge, normalized coordinates at texel centers. CUDA filters with
// weights held to 8 fractional bits (the CUDA Programming Guide's
// texture-fetching appendix), the hardware's, as Metal's filter is Apple's.
//
// Each is made once at its size, outside the frame loop (MEM.9), and
// released with its owner (R.1), after the frames that used it
// (submission.h, Lifetime). Rows from the top, row-major, so a warp's
// threads along a row touch neighbouring addresses (GPU.2).

// An image of four 32-bit floats a pixel in pitched linear memory: the
// accumulated image (contract 6), which the path program writes through
// its address and pitch, and the tone map reads filtered through its
// texture object, as Metal reads the RGBA32Float image through its sampler.
// Pitched by the driver (cuMemAllocPitch), and its pitch a multiple of the
// device's alignment for a texture over linear memory, checked here.
//
// Throws CudaError if the size is empty, the memory or the texture cannot
// be made, or the pitch is not one a texture can read.
class PitchedImage {
 public:
  PitchedImage(const Device& device, Submission& submission,
               frame::Extent size);

  PitchedImage(const PitchedImage&) = delete;
  PitchedImage& operator=(const PitchedImage&) = delete;
  PitchedImage(PitchedImage&&) = delete;
  PitchedImage& operator=(PitchedImage&&) = delete;
  ~PitchedImage();

  frame::Extent size() const noexcept { return size_; }
  // Where row 0 starts, and the bytes from one row's start to the next's.
  CUdeviceptr address() const noexcept { return address_; }
  std::size_t pitch() const noexcept { return pitch_; }
  // The image, read filtered (above).
  CUtexObject texture() const noexcept { return texture_; }

  // Waits for every committed frame (Submission::finish) and copies the
  // image into `out`, four floats a pixel, rows from the top, tightly
  // packed: exactly width x height x 4 floats, else CudaError.
  void read(std::span<float> out) const;

 private:
  Submission& submission_;
  frame::Extent size_;
  CUdeviceptr address_ = 0;
  std::size_t pitch_ = 0;
  CUtexObject texture_ = 0;
};

// An image of four 16-bit floats a pixel in a CUDA array: a level of the
// tone map's bloom pyramid (core/passes/bloom.h), which its kernels write
// through its surface object and read through its texture object, filtered,
// or through the surface, exact, as Metal reads and writes its levels'
// RGBA16Float textures.
//
// Throws CudaError if the size is empty or the array, the surface or the
// texture cannot be made.
class LevelImage {
 public:
  explicit LevelImage(frame::Extent size);

  LevelImage(const LevelImage&) = delete;
  LevelImage& operator=(const LevelImage&) = delete;
  LevelImage(LevelImage&&) = delete;
  LevelImage& operator=(LevelImage&&) = delete;
  ~LevelImage();

  frame::Extent size() const noexcept { return size_; }
  CUsurfObject surface() const noexcept { return surface_; }
  CUtexObject texture() const noexcept { return texture_; }

 private:
  frame::Extent size_;
  CUarray array_ = nullptr;
  CUsurfObject surface_ = 0;
  CUtexObject texture_ = 0;
};

}  // namespace serenity::cuda
