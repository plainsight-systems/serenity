#pragma once

#include <cstdint>
#include <span>

#include <Metal/Metal.hpp>

#include "metal/device/device.h"
#include "metal/device/frame_array.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Scene content (on the GPU: how bright each light is).
//
// Each sphere light's glow as a frame lights it (core/animation/glow.h): a
// factor on its radiance, one float per sphere light, in the order of the
// scene's sphere lights. The second array of the scene that changes from
// frame to frame, beside the shapes' transforms (shape_transforms.h). This
// file says which array it is, and nothing more: how it is allocated, made
// resident and copied per frame in flight, and when the CPU may write it, is
// the GPU backend's (metal/device/frame_array.h).
//
// A sphere light's emitter reads its factor where it reads its radiance
// (metal/lights/emitter.metal.h), so a light sampled, a light seen and a
// light reached through glass are all as bright as this frame says.
//
// A scene with no glowing light keeps one copy, every factor 1; a scene
// where some light glows, one per frame in flight, every one starting at 1,
// the core then rewriting the glowing lights' factors in the copy of the
// frame being recorded (animation::animate). A scene with no sphere lights
// keeps one factor, unread, since a buffer must have bytes.
//
// Last frame's glows, for a graph with ReSTIR's temporal merge, as last
// frame's transforms are kept (shape_transforms.h): a second LightGlows,
// lit at the previous frame's time each frame, so the merge's test of last
// frame's surface sees each firefly as bright as last frame did: one asleep
// then gave that surface nothing, whatever it gives now
// (metal/reuse/unbiased.metal.h). A scene with no glowing light needs none:
// its one copy is last frame's too.
//
// Throws MetalError if the backend cannot make the array.
class LightGlows {
public:
    // `lights` factors of 1: one copy when `glowing` is false, one per frame
    // in flight when it is true.
    LightGlows(const Device& device, Submission& submission, std::uint32_t lights, bool glowing);

    LightGlows(const LightGlows&) = delete;
    LightGlows& operator=(const LightGlows&) = delete;
    LightGlows(LightGlows&&) = delete;
    LightGlows& operator=(LightGlows&&) = delete;
    ~LightGlows() = default;

    // Frame slot `slot`'s factors, to light the frame's lights in, under
    // FrameArray's rule for writing (frame_array.h).
    std::span<float> glows(std::uint32_t slot);

    // Frame slot `slot`'s factors, as a shader binds them.
    MTL::GPUAddress address(std::uint32_t slot) const { return array_.address(slot); }

private:
    std::uint32_t lights_;
    FrameArray array_;
};

}  // namespace serenity::metal
