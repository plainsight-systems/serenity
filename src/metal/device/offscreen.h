#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/extent.h"
#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: GPU backend.
//
// An image a frame renders into instead of a drawable: what the headless
// renderer and the GPU tests use. Its format is RGBA8Unorm where the
// drawable's is BGRA8Unorm (presenter.h): a shader writes colors, not bytes,
// and the format alone decides the byte order, so the window and headless
// images are the same image (principle 1), and this one reads back in the
// order a PNG wants.
//
// The texture is in shared storage, which Apple silicon's unified memory
// allows, so reading it back is a copy out of memory the CPU can already see,
// with no staging buffer. It is made resident once, at construction, until
// it is destroyed, which waits for the GPU first (submission.h, Lifetime).
//
// Readback is the one place the CPU waits for a frame to finish (GPU.1): the
// caller waits with Submission::wait_until_complete(i) and then calls
// read_rgba(). It belongs to the headless renderer and the tests, never to
// the window's loop.
//
// read_rgba() copies the image out as R, G, B, A, rows from the top, the
// order output/png.h takes. `out` must hold exactly width x height x 4
// bytes; any other size throws MetalError. Construction throws MetalError for a size
// Metal makes no image of (device.h, max_texture_side), or if the device
// cannot make the texture.
//
// Cost of a readback: width x height x 4 bytes, copied once. At 3456 x 2234
// that is 30.9 MB. Not on any budgeted path.
class Offscreen {
public:
    Offscreen(const Device& device, Submission& submission, frame::Extent size);

    Offscreen(const Offscreen&) = delete;
    Offscreen& operator=(const Offscreen&) = delete;
    Offscreen(Offscreen&&) = delete;
    Offscreen& operator=(Offscreen&&) = delete;
    ~Offscreen() = default;

    MTL::Texture* texture() const noexcept { return texture_.get(); }
    frame::Extent size() const noexcept { return size_; }

    // The bytes read_rgba() fills: width x height x 4. The one place that
    // size is computed (ES.3).
    std::size_t rgba_size() const;

    void read_rgba(std::span<std::uint8_t> out) const;

private:
    NS::SharedPtr<MTL::Texture> texture_;
    Resident resident_;  // the texture's residency, released after the GPU is done with it
    frame::Extent size_;
};

}  // namespace serenity::metal
