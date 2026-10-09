// The path pass (path.h): one path per pixel per frame, through a point in
// the pixel drawn anew each frame, folded into the accumulated image, whose
// mean is the frame's radiance.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "core/lights/gradient_sky.h"
#include "core/lights/light.h"
#include "core/lights/sphere_light.h"
#include "metal/camera/pinhole.metal.h"
#include "metal/film/accumulate.metal.h"
#include "metal/integrator/path.metal.h"

using namespace metal;
using namespace metal::raytracing;
using namespace serenity::shaders;

kernel void path_trace(constant serenity::contracts::FrameConstants& frame [[buffer(0)]],
                       constant serenity::contracts::CameraData& camera [[buffer(1)]],
                       primitive_acceleration_structure structure [[buffer(2)]],
                       constant serenity::lights::GradientSkyData& sky [[buffer(3)]],
                       device const serenity::textures::TextureRecord* texture_records [[buffer(4)]],
                       device const serenity::textures::CheckerData* checkers [[buffer(5)]],
                       device const serenity::materials::MaterialRecord* materials [[buffer(6)]],
                       device const serenity::materials::RoughData* rough [[buffer(7)]],
                       device const serenity::materials::DielectricData* dielectrics [[buffer(8)]],
                       device const serenity::materials::ConductorData* conductors [[buffer(9)]],
                       device const float* sphere_glows [[buffer(10)]],
                       device const serenity::shapes::ShapeRecord* shape_records [[buffer(11)]],
                       device const serenity::contracts::Transform* transforms [[buffer(12)]],
                       device const serenity::shapes::BoxData* boxes [[buffer(13)]],
                       device const serenity::lights::LightRecord* light_records [[buffer(14)]],
                       device const uint* shape_lights [[buffer(15)]],
                       device const serenity::lights::SphereLightData* sphere_lights [[buffer(16)]],
                       constant serenity::lights::LightCounts& light_counts [[buffer(17)]],
                       device atomic_uint* non_finite [[buffer(18)]],
                       texture2d<float, access::read_write> accumulated [[texture(0)]],
                       texture2d<float, access::write> radiance [[texture(1)]],
                       uint2 pixel [[thread_position_in_grid]]) {
    // Every thread reaches count_non_finite() below, which sums over its
    // SIMD group: none returns early.
    const bool inside = pixel.x < frame.width && pixel.y < frame.height;
    bool failed = false;
    if (inside) {
        path::Scene scene;
        scene.structure = structure;
        scene.shapes = Shapes{shape_records, transforms, boxes};
        scene.materials = Materials{materials, rough, dielectrics, conductors};
        scene.textures = Textures{texture_records, checkers};
        scene.selection = UniformLight{light_records, light_counts.lights};
        scene.lights = Lights{light_records, shape_lights, sphere_lights, transforms, sphere_glows};
        scene.sky = sky;

        PathNumbers numbers = path_numbers(pixel, frame.frame_index);
        const float2 within = next_numbers2(numbers);  // the point in the pixel
        const float3 direction = pinhole_direction(camera, float2(pixel) + within, frame.width, frame.height);
        const float3 sample = path::radiance(scene, to_float3(camera.origin), direction, numbers);

        failed = !finite(sample);
        const float4 held = accumulate(accumulated.read(pixel), sample, frame.accumulated_frames == 0u);
        accumulated.write(held, pixel);
        radiance.write(float4(held.rgb, 1.0f), pixel);
    }
    count_non_finite(failed, non_finite);
}
