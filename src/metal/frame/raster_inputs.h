#pragma once

#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"

namespace serenity::metal {

// Axis: Frame (how Metal traces a frame whose features it rasterizes).
//
// The frame `inputs` as Metal's passes trace it: the core's rule
// (frame::traced_inputs(): through a pinhole in a graph that feeds the
// network), and also through a pinhole in any graph with the features pass,
// since Metal rasterizes the features' first surface (passes/features/
// features.h; Apple's hybrid design, docs/research/
// 2026-10-23-apple-silicon-design.md, section 6, change 2) and a rasterizer
// draws a pinhole's image; the light the same rays (contract 17). The frame's
// camera and the frame before's with their lens taken off (a radius of 0,
// the rest of each camera as it was). Metal's own rule, not the core's: the
// CUDA backend traces its features, and keeps the scene's lens in a
// features graph that does not feed the network. Pure (F.8).
frame::FrameInputs raster_inputs(const frame::Schedule& schedule,
                                 const frame::FrameInputs& inputs);

}  // namespace serenity::metal
