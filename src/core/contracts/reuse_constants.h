#pragma once

// The reuse constants. Owned by Sample reuse; written by the Frame graph
// each frame, read by ReSTIR DI's passes.
//
// What a ReSTIR pass reads beyond the frame's constants (contract 7) and
// camera (contract 8): the previous frame's camera, framed for the same
// image, which the temporal merge reprojects through and the surfaces pass
// projects last frame's position through; whether there is a previous frame
// to merge; and the estimator's settings (core/passes/restir_di.h). Its own
// buffer, bound only by ReSTIR's passes, so contract 7, which every pass
// binds, is unchanged, and so is every graph without them
// (docs/research/2026-10-10-restir-di-layout.md, "Additive").
//
//   previous_camera  the camera of the frame before, in this history
//                    (core/frame/history.h, the reuse history), framed for
//                    this frame's size, which the history keeps the same.
//                    Unread when `previous` is 0.
//   settings         the graph's, the same every frame, checked on the CPU
//                    before they are written (I.5; core/frame/schedule.h).
//   previous         1 when the frame before is this frame's history, so
//                    its surface records and final reservoirs are there to
//                    merge (metal/frame/frame_images.h); 0 when this frame
//                    starts the history over, and the temporal merge merges
//                    nothing. A word, not a bool (no bool in a shared
//                    layout: contracts/frame_constants.h).
//
// The previous frame's time reaches no shader: what it changes, where the
// shapes were and how bright the lights were, the CPU computes at it, into
// last frame's transforms and glows (metal/scene/shape_transforms.h,
// light_glows.h), and its structure is kept
// (metal/acceleration/scene_acceleration.h).
//
// Written by the CPU once a frame into a ring of one copy per frame in
// flight (metal/device/frame_array.h): 112 bytes a frame.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/shared_layout.h"
#include "core/contracts/camera.h"
#include "core/passes/restir_di.h"

namespace serenity {
namespace contracts {

struct ReuseConstants {
    CameraData previous_camera;
    passes::RestirDi settings;
    uint32_t previous;  // 1: the frame before is this one's history; 0: it starts over
    uint32_t padding[3];
};

static_assert(sizeof(ReuseConstants) == 112, "ReuseConstants must be the same 112 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<ReuseConstants>, "ReuseConstants is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
