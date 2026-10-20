#pragma once

// Axis: Frame graph (device half: what every tracing module holds once).
//
// Included once by each pass's programs' `.cu` (one module a pass,
// cuda/device/pipeline.h), before its ray generation programs: the frame's
// launch parameters as OptiX's launch-parameter variable, `launch`
// (frame_launch.h), and the backend's two intersection programs, the
// shapes' (acceleration/intersection.cuh) and MNEE's seed's
// (integrator/mnee_seed.cuh), which every tracing pipeline holds a hit group
// each of (traced_program.h).

#include "cuda/acceleration/intersection.cuh"
#include "cuda/frame/frame_launch.h"
#include "cuda/integrator/mnee_seed.cuh"

extern "C" {
__constant__ serenity::cuda::FrameLaunch launch;
}
