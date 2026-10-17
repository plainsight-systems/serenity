// ReSTIR DI's spatial merge (restir_spatial.h): step 5 of
// metal/passes/restir_di/restir_di.metal.h, one thread a pixel (GDSA.8). The
// reservoirs of k neighbors within the radius, similar in geometry and
// material, merged into the pixel's own (metal/reuse/spatial.metal.h), asked
// of this frame's lights, camera and, in raytraced mode, structure; read
// from the working array, written to this frame's final one, never in place.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/reservoir.h"
#include "core/contracts/reuse_constants.h"
#include "core/contracts/surface_record.h"
#include "core/contracts/transform.h"
#include "metal/passes/bindings.h"
#include "metal/passes/restir_di/restir_di.metal.h"

using namespace metal;
using namespace metal::raytracing;
using namespace serenity::shaders;
namespace binding = serenity::bindings::restir_spatial;

kernel void restir_spatial(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                           constant serenity::contracts::CameraData& camera [[buffer(binding::camera)]],
                           serenity::shaders::SceneStructure structure [[buffer(binding::structure)]],
                           constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
                           constant float* sphere_glows [[buffer(binding::glows)]],
                           constant serenity::contracts::Transform* transforms [[buffer(binding::transforms)]],
                           constant serenity::contracts::ReuseConstants& reuse [[buffer(binding::reuse)]],
                           const device serenity::contracts::SurfaceRecord* records [[buffer(binding::records)]],
                           const device serenity::contracts::Reservoir* input [[buffer(binding::input)]],
                           device serenity::contracts::Reservoir* output [[buffer(binding::output)]],
                           uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= frame.width || pixel.y >= frame.height) {
        return;
    }
    const uint i = pixel.y * frame.width + pixel.x;
    const uint2 size = uint2(frame.width, frame.height);
    const restir::FrameView now{restir::Scene{structure, &block, transforms, sphere_glows}, &camera, size};
    output[i] = restir::spatial(now, Merging{records[i], input[i]}, pixel, NeighborArrays{records, input, size},
                                frame.frame_index, reuse.settings);
}
