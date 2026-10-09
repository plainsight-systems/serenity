// The preview (preview.h): the pass that launches the direct-light
// integrator (integrator/direct.metal.h) for each pixel, at four positions
// within it, and writes the mean to the radiance image.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "core/lights/gradient_sky.h"
#include "core/lights/light.h"
#include "core/lights/sphere_light.h"
#include "metal/camera/pinhole.metal.h"
#include "metal/integrator/direct.metal.h"

using namespace metal;
using namespace metal::raytracing;
using namespace serenity::shaders;

namespace {

// The pixel's sample positions: a rotated grid, the four points that
// separate both rows and columns of near-horizontal and near-vertical edges.
constant constexpr uint pixel_samples = 4;
constant constexpr float2 positions[pixel_samples] = {
    float2(0.375f, 0.125f), float2(0.875f, 0.375f), float2(0.625f, 0.875f), float2(0.125f, 0.625f)};


}  // namespace

kernel void preview(constant serenity::contracts::FrameConstants& frame [[buffer(0)]],
                    constant serenity::contracts::CameraData& camera [[buffer(1)]],
                    primitive_acceleration_structure structure [[buffer(2)]],
                    constant serenity::lights::GradientSkyData& sky [[buffer(3)]],
                    device const serenity::textures::TextureRecord* texture_records [[buffer(4)]],
                    device const serenity::textures::CheckerData* checkers [[buffer(5)]],
                    device const serenity::textures::WoodData* woods [[buffer(19)]],
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
                    texture2d<float, access::write> radiance [[texture(0)]],
                    uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= frame.width || pixel.y >= frame.height) {
        return;
    }
    direct::Scene scene;
    scene.structure = structure;
    scene.shapes = Shapes{shape_records, transforms, boxes};
    scene.textures = Textures{texture_records, checkers, woods};
    scene.materials = Materials{materials, rough, dielectrics, conductors};
    scene.selection = EveryLight{light_records, light_counts.lights};
    scene.lights = Lights{light_records, shape_lights, sphere_lights, transforms, sphere_glows};
    scene.sky = sky;

    float3 color = float3(0.0f);
    for (uint i = 0; i < pixel_samples; ++i) {
        const float3 direction = pinhole_direction(camera, float2(pixel) + positions[i], frame.width, frame.height);
        color += direct::radiance(scene, to_float3(camera.origin), direction, direct::Pixel{pixel, i});
    }
    radiance.write(float4(color / float(pixel_samples), 1.0f), pixel);
}
