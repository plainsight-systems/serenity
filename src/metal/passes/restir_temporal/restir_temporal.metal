// ReSTIR DI's temporal merge (restir_temporal.h): step 4 of
// metal/passes/restir_di/restir_di.metal.h, one thread a pixel (GDSA.8). Last
// frame's final reservoir, where the pixel's surface was, merged into its
// initial reservoir (metal/reuse/temporal.metal.h), asked of this frame's
// lights and last frame's (`now`, `then`), and written to `output`, the
// array the renderer binds by the graph.
//
// Bound here: what basic mode reads. The renderer refuses a raytraced
// graph with a temporal merge until last frame's structure is kept
// (metal/frame/renderer.h), so `then` is given this frame's, which basic
// mode never traces.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/float3.h"
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
namespace binding = serenity::bindings::restir_temporal;

kernel void restir_temporal(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                            constant serenity::contracts::CameraData& camera [[buffer(binding::camera)]],
                            serenity::shaders::SceneStructure structure [[buffer(binding::structure)]],
                            constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
                            constant float* sphere_glows [[buffer(binding::glows)]],
                            constant serenity::contracts::Transform* transforms [[buffer(binding::transforms)]],
                            constant serenity::contracts::ReuseConstants& reuse [[buffer(binding::reuse)]],
                            const device serenity::contracts::SurfaceRecord* records [[buffer(binding::records)]],
                            const device serenity::contracts::Float3* previous_positions
                            [[buffer(binding::previous_positions)]],
                            const device serenity::contracts::SurfaceRecord* last_records
                            [[buffer(binding::last_records)]],
                            const device serenity::contracts::Reservoir* last_finals [[buffer(binding::last_final)]],
                            const device serenity::contracts::Reservoir* input [[buffer(binding::input)]],
                            device serenity::contracts::Reservoir* output [[buffer(binding::output)]],
                            constant float* last_glows [[buffer(binding::last_glows)]],
                            constant serenity::contracts::Transform* last_transforms
                            [[buffer(binding::last_transforms)]],
                            uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= frame.width || pixel.y >= frame.height) {
        return;
    }
    const uint i = pixel.y * frame.width + pixel.x;
    const uint2 size = uint2(frame.width, frame.height);
    const restir::FrameView now{restir::Scene{structure, &block, transforms, sphere_glows}, &camera, size};
    const restir::FrameView then{restir::Scene{structure, &block, last_transforms, last_glows},
                                 &reuse.previous_camera, size};
    const Merging here{records[i], input[i]};
    output[i] = restir::temporal(now, then, here, to_float3(previous_positions[i]),
                                 LastArrays{last_records, last_finals, size}, reuse.previous != 0u, pixel,
                                 frame.frame_index, reuse.settings);
}
