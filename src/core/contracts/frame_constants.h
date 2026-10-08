#pragma once

// Contract 7: frame constants. Owned by the Frame graph; read by every pass.
//
// The values every shader in a frame may read: when the frame is, which frame
// it is, and the size of the image being written. One definition for both
// sides: C++ writes these bytes and the Metal shaders read them, so this
// header compiles as C++20 and as the Metal shading language
// (file-mapping.md, "How a layout crosses into a shader"). It shares a layout,
// never math.
//
// Layout rules, for this and every contract a shader reads:
//   - fixed-width fields only, 4 bytes each here, so no field is padded and
//     the order is the alignment order as well (CACHE.5);
//   - no pointers, no bool (its size is not fixed across the two languages),
//     nothing that only the host has;
//   - the size is asserted on both sides, so a field added on one side alone
//     fails to compile on the other.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

namespace serenity {
namespace contracts {

struct FrameConstants {
    // frame::FrameInputs::time, narrowed to float.
    float time_seconds;
    // The low 32 bits of frame::FrameInputs::index: wraps after 2^32 frames,
    // over two years at 60 frames a second.
    uint32_t frame_index;
    // The image being written, in pixels.
    uint32_t width;
    uint32_t height;
};

static_assert(sizeof(FrameConstants) == 16, "FrameConstants must be the same 16 bytes on the host and in shaders");

}  // namespace contracts
}  // namespace serenity
