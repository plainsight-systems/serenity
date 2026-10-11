// ReSTIR DI's initial pass (restir_initial.h): steps 1 to 3 of
// metal/passes/restir_di/restir_di.metal.h, one thread a pixel. The camera
// ray through a point in the pixel drawn anew each frame, as the path pass
// draws it (passes/path/path.metal), so the camera's segment draws what the
// path tracer's would; its record and path state written; then M
// candidates at the record, unpacked, and visibility reuse; the reservoir
// written to `output`, the array the renderer binds by the graph.
//
// Bound here: what a graph without a temporal merge reads and writes. The
// previous positions and last frame's transforms (bindings.h) are the
// temporal merge's, bound with it; a graph with one is refused by the
// renderer until it is implemented (metal/frame/renderer.h).

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
namespace binding = serenity::bindings::restir_initial;

kernel void restir_initial(constant serenity::contracts::FrameConstants& frame [[buffer(binding::constants)]],
                           constant serenity::contracts::CameraData& camera [[buffer(binding::camera)]],
                           primitive_acceleration_structure structure [[buffer(binding::structure)]],
                           constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
                           constant float* sphere_glows [[buffer(binding::glows)]],
                           constant serenity::contracts::Transform* transforms [[buffer(binding::transforms)]],
                           constant serenity::contracts::ReuseConstants& reuse [[buffer(binding::reuse)]],
                           device serenity::contracts::SurfaceRecord* records [[buffer(binding::records)]],
                           device serenity::contracts::PathState* path_states [[buffer(binding::path_states)]],
                           device serenity::contracts::Reservoir* output [[buffer(binding::output)]],
                           uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= frame.width || pixel.y >= frame.height) {
        return;
    }
    const uint i = pixel.y * frame.width + pixel.x;
    const restir::Scene scene{structure, &block, transforms, sphere_glows};

    // Step 1, the path's own numbers from 0, as the path pass draws them.
    const PathNumbers stream = path_numbers(pixel, frame.frame_index);
    PathNumbers numbers = stream;
    const float2 within = next_numbers2(numbers);  // the point in the pixel
    const float2 lens = has_lens(camera) ? next_numbers2(numbers) : float2(0.0f);
    const CameraRay ray = camera_ray(camera, float2(pixel) + within, lens, uint2(frame.width, frame.height));
    const restir::Surfaces surfaces = restir::first_surface(scene, ray, nullptr, numbers);
    records[i] = surfaces.record;
    path_states[i] = surfaces.state;  // written before step 2: not alive across step 3's query

    // Steps 2 and 3, at the record as every later pass reads it.
    if (packed_kind(surfaces.record.bsdf) == serenity::contracts::BsdfKind::none ||
        scene.selection().count == 0u) {
        output[i] = serenity::contracts::Reservoir{};  // empty: no sample, W = 0, M = 0
        return;
    }
    PathNumbers candidates_numbers = purpose_numbers(stream, restir::candidates_numbers);
    output[i] = restir::candidates(scene, unpack_record(surfaces.record), reuse.settings.candidates,
                                   candidates_numbers);
}
