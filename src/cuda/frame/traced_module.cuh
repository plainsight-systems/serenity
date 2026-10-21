#pragma once

// Axis: Frame graph (device half: what every tracing module holds once).
//
// Included once by each pass's programs' `.cu` (one module a pass,
// cuda/device/pipeline.h), before its ray generation programs: the frame's
// launch parameters as OptiX's launch-parameter variable, `launch`
// (frame_launch.h), and the backend's one intersection program, MNEE's
// seed's (integrator/mnee_seed.cuh); the shapes are triangles, whose hits
// are OptiX's own. Every tracing pipeline holds a hit group of each
// (traced_program.h).

#include "cuda/frame/frame_launch.h"
#include "cuda/integrator/mnee_seed.cuh"

extern "C" {
__constant__ serenity::cuda::FrameLaunch launch;
}
