#pragma once

// Contract 7: frame constants. Owned by the Frame graph; read by every pass.
//
// The values every shader in a frame may read: when the frame is, which frame
// it is, the size of the image being written, how many frames the
// accumulated image already holds, and where in its pixel the frame's
// camera rays pass, where the frame takes the camera's jitter
// (core/sampling/jitter.h): one offset for every pixel, which the camera's
// ray generation reads (metal/camera/thin_lens.metal.h, pixel_ray), in
// place of each pixel's own random point. Which frames take it is the
// frame graph's (core/frame/schedule.h, jitters): ReSTIR PT's light and its
// features, written into the light's constants and never the output
// size's, whose full-size albedo stays unjittered. The camera is a contract of
// its own (contracts/camera.h), owned by Camera, so a change to the camera
// changes nothing here. One definition for both
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
//     fails to compile on the other, and on the host that the type is
//     trivially copyable, since the host writes it as bytes (SL.con.4);
//   - the differences between the two languages, the standard headers and
//     how a namespace-scope constant is declared (SERENITY_CONSTANT), are
//     written once, in contracts/shared_layout.h, which every shared header
//     includes (P.11).

#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

// The camera's jitter's phases (core/sampling/jitter.h): 8 x the square of
// the denoiser's upscale, DLSS's rule, asserted there.
SERENITY_CONSTANT uint32_t jitter_phases = 32u;

struct FrameConstants {
  // frame::FrameInputs::time, narrowed to float.
  float time_seconds;
  // The low 32 bits of frame::FrameInputs::index: wraps after 2^32 frames,
  // over two years at 60 frames a second.
  uint32_t frame_index;
  // The image being written, in pixels.
  uint32_t width;
  uint32_t height;
  // frame::FrameInputs::index - accumulated_since: the frames already in
  // the accumulated image, which this one joins; 0 starts it over. At most
  // 2^24 - 1 (frame::max_accumulated_frames, core/frame/frame_inputs.h).
  uint32_t accumulated_frames;
  // The camera's jitter, in pixels of this image, in [-1/2, 1/2): every
  // camera ray passes through its pixel's center plus it, where
  // `jittered` is 1; read by nothing where it is 0, each pixel drawing
  // its own point from its numbers as before.
  float jitter_x;
  float jitter_y;
  uint32_t jittered;
};

static_assert(
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): Metal's layout
    sizeof(FrameConstants) == 32,
    "FrameConstants must be the same 32 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<FrameConstants>,
              "FrameConstants is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
