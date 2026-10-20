#pragma once

#include <span>

#include "cuda/device/frame_upload.h"
#include "cuda/device/optix_context.h"
#include "cuda/device/pipeline.h"
#include "cuda/device/submission.h"
#include "cuda/passes/path/path_launch.h"

namespace serenity::cuda {

// Axis: Pass (path tracing).
//
// The pass that runs the path tracer without reuse, Metal's PathPass
// (metal/passes/path/path.h, which says what it computes per pixel and what
// it costs): next event estimation weighed against BSDF sampling by MIS
// (cuda/integrator/path.cuh), once per pixel per frame, folded into the
// accumulated image (cuda/film/accumulate.cuh), so a still image converges
// toward the scene's full light transport.
//
// On CUDA it is an OptiX launch (cuda/device/pipeline.h): its ray
// generation program, one thread a pixel, is the pass's body (path.cu),
// and the shapes' intersection program its hit group. Its launch
// parameters (path_launch.h) are the renderer's for the frame, staged in
// the frame's slot and uploaded on the frame's stream before the launch
// (cuda/device/frame_upload.h): 168 bytes a frame.
//
// Ordering: the pass reads the image the previous frame wrote; the one
// stream runs it after that frame's work, and every pass of a frame in
// order, so it needs no barrier, unlike Metal's encoder.
//
// Made once, outside the frame loop (MEM.9). Throws CudaError if OptiX
// refuses its programs or the memory cannot be made.
class PathPass {
 public:
  // `structures`, the hit records' data, one a structure in record order
  // (cuda/acceleration/scene_acceleration.h, hit_records).
  PathPass(const OptixContext& context, Submission& submission,
           std::span<const Pipeline::HitData> structures);

  // Launches the pass for `frame`, with `launch`, over its constants' width
  // and height. Throws CudaError if the launch is refused.
  void record(const FrameSlot& frame, const PathLaunch& launch);

 private:
  Pipeline pipeline_;
  FrameUpload launch_;
};

}  // namespace serenity::cuda
