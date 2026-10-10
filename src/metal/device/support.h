#pragma once

#include <cstddef>
#include <string>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/extent.h"

namespace serenity::metal {

// Axis: the Metal API surface.
//
// The few operations every part of the backend repeats, named once (ES.3,
// F.10): how Metal's errors are described, how an autorelease pool is
// scoped, how buffer offsets are aligned, how a private image is made, and
// how a pass dispatches a thread per pixel. Each was written out where it was
// used, four or five times over; a change to one (a texture's usage, the
// dispatch's shape) is now made here.

// Metal's description of `error`, or a fixed text when it gave none. Called
// inside the pool that owns `error`, so the text is copied out before the
// pool drains.
std::string describe(const NS::Error* error);

// An autorelease pool, drained when the returned pointer is released: for a
// scope that receives autoreleased objects from Metal (R.1).
[[nodiscard]] NS::SharedPtr<NS::AutoreleasePool> scoped_pool();

// What Metal asks of a constant buffer's offset, and more than any shared
// layout's alignment: every array and copy the backend places in a buffer
// starts on it.
inline constexpr std::size_t buffer_alignment = 256;

// `bytes` rounded up to a multiple of buffer_alignment.
constexpr std::size_t align_up(std::size_t bytes) noexcept {
    return (bytes + buffer_alignment - 1) / buffer_alignment * buffer_alignment;
}

// A 2D image of `format` and `size` in private storage, read and written by
// shaders: the GPU's alone. Throws Error naming `what` if `size` is not one
// Metal makes (check_texture_size, device.h) or the device makes none. Not
// made resident: the caller keeps it so (submission.h, keep_resident).
NS::SharedPtr<MTL::Texture> make_private_texture(MTL::Device* device, MTL::PixelFormat format, frame::Extent size,
                                                 const char* what);

// Dispatches `pipeline`, already set on `encoder`, over `size`, one thread
// per pixel, in threadgroups one execution width wide and as many rows as
// the pipeline allows, so adjacent threads take adjacent pixels of a row
// (GPU.2).
void dispatch_per_pixel(MTL4::ComputeCommandEncoder* encoder, const MTL::ComputePipelineState* pipeline,
                        frame::Extent size);

}  // namespace serenity::metal
