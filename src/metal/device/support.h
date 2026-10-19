#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstddef>
#include <string>
#include <string_view>

#include "core/frame/extent.h"

namespace serenity::metal {

// Axis: the Metal API surface.
//
// The few operations every part of the backend repeats, named once (ES.3,
// F.10): how Metal's errors are described, how an autorelease pool is
// scoped, how buffer offsets are aligned, how a private image is made, how
// a pass dispatches a thread per pixel, and the barrier between one
// dispatch and the next. Each was written out where it was
// used, four or five times over; a change to one (a texture's usage, the
// dispatch's shape) is now made here. And the half float, the storage of
// the RGBA16Float images and the denoiser's weights, as the host writes and
// reads it.

// A half float: _Float16, a compiler extension (P.2), until the
// toolchain's C++ has std::float16_t (C++23).
using Half = _Float16;
static_assert(sizeof(Half) == 2, "a half float is two bytes, as Metal's half");

// Metal's description of `error`, or a fixed text when it gave none. Called
// inside the pool that owns `error`, so the text is copied out before the
// pool drains.
std::string describe(const NS::Error* error);

// An autorelease pool, drained when the returned pointer is released: for a
// scope that receives autoreleased objects from Metal (R.1).
[[nodiscard]] NS::SharedPtr<NS::AutoreleasePool> scoped_pool();

// What Metal asks of a constant buffer's offset on every GPU family this
// renderer runs on (Apple7 and later, device.h): 4 bytes, the "Minimum
// constant buffer offset alignment" of Apple's Metal Feature Set Tables
// (May 21, 2026 edition, page 7, "Resources": 4 B for Apple2 to Apple10).
// Every address bound as `constant` lies on it, which each placement
// asserts beside itself.
inline constexpr std::size_t constant_offset_alignment = 4;

// Where the backend starts every array and copy it places in a buffer: 256
// bytes, a choice, not Metal's requirement (above), which it exceeds, as it
// exceeds every shared layout's alignment; the scene block measured fast so
// (scene_block.h). Each placement inside a 256-byte start (a ring slot's
// camera, a dispatch's parameters, a grid beside its count) asserts the
// alignment it needs, not this one.
inline constexpr std::size_t buffer_alignment = 256;
static_assert(buffer_alignment % constant_offset_alignment == 0);

// `bytes` rounded up to a multiple of buffer_alignment.
constexpr std::size_t align_up(std::size_t bytes) noexcept {
  return (bytes + buffer_alignment - 1) / buffer_alignment * buffer_alignment;
}

// A 2D image of `format` and `size` in private storage, read and written by
// shaders: the GPU's alone. Throws MetalError naming `what` if `size` is not
// one Metal makes (check_texture_size, device.h) or the device makes none. Not
// made resident: the caller keeps it so (submission.h, keep_resident).
NS::SharedPtr<MTL::Texture> make_private_texture(MTL::Device* device,
                                                 MTL::PixelFormat format,
                                                 frame::Extent size,
                                                 std::string_view what);

// Dispatches `pipeline`, already set on `encoder`, over `size`, one thread
// per pixel, in threadgroups one execution width wide and as many rows as
// the pipeline allows, so adjacent threads take adjacent pixels of a row
// (GPU.2).
void dispatch_per_pixel(MTL4::ComputeCommandEncoder* encoder,
                        const MTL::ComputePipelineState* pipeline,
                        frame::Extent size);

// A barrier from the encoder's dispatches before to its dispatches after,
// their device memory made visible: what a dispatch that reads an earlier
// one's writes records between them, Metal 4 tracking no hazards (GPU.8).
inline void barrier_after_dispatch(MTL4::ComputeCommandEncoder* encoder) {
  encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch,
                                     MTL4::VisibilityOptionDevice);
}

}  // namespace serenity::metal
