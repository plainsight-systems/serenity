// The features' raster pass (features.h): the meshes drawn into the
// visibility buffer (visibility.h), each pixel's nearest triangle by
// reversed depth, its shape and its index in its mesh's per-primitive
// data. A still structure's vertices are in the world, its fragments'
// shape its triangle's (acceleration/structure_block.h, MeshTriangle); an
// instanced mesh's are placed by each of its shapes' instance transforms,
// the shape carried to its fragments. Projected by the pinhole camera's
// own projection, its sample at a pixel's center the camera ray through it
// plus the frame's jitter where the frame takes it (camera/
// thin_lens.metal.h, camera_clip()). Bindings: passes/bindings.h, raster.

#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/float3.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/transform.h"
#include "core/scene/scene_block.h"
#include "core/shapes/box.h"
#include "core/shapes/primitive.h"
#include "metal/acceleration/structure_block.h"
#include "metal/camera/thin_lens.metal.h"
#include "metal/device/layout.metal.h"
#include "metal/math/transform.metal.h"
#include "metal/passes/bindings.h"
#include "metal/passes/features/visibility.h"
#include "metal/shapes/shapes.metal.h"

using namespace serenity::shaders;
namespace binding = serenity::bindings::raster;
namespace contracts = serenity::contracts;

namespace serenity {
namespace shaders {

// A vertex as the rasterizer takes it: its clip position and, for an
// instanced mesh, its raster entry (its mesh and shape), the same at every
// vertex of an instance.
struct RasterVertex {
  float4 position [[position]];
  uint shape [[flat]];
};

// `world` in clip space for the frame `frame`: its sample at a pixel's
// center the camera ray through it plus the frame's jitter where the frame
// takes it, else none (contract 17).
inline float4 raster_position(constant contracts::CameraData& camera,
                              constant contracts::FrameConstants& frame,
                              float3 world) {
  return camera_clip(camera, world, uint2(frame.width, frame.height),
                     frame.jittered != 0u
                         ? float2(frame.jitter_x, frame.jitter_y)
                         : float2(0.0f));
}

}  // namespace shaders
}  // namespace serenity

vertex RasterVertex raster_still(uint vertex_index [[vertex_id]],
                                 constant contracts::CameraData& camera
                                 [[buffer(binding::camera)]],
                                 constant contracts::FrameConstants& frame
                                 [[buffer(binding::constants)]],
                                 const device contracts::Float3* vertices
                                 [[buffer(binding::vertices)]]) {
  return RasterVertex{
      raster_position(camera, frame, to_float3(vertices[vertex_index])), 0u};
}

vertex RasterVertex raster_instanced(
    uint vertex_index [[vertex_id]], uint instance [[instance_id]],
    constant contracts::CameraData& camera [[buffer(binding::camera)]],
    constant contracts::FrameConstants& frame [[buffer(binding::constants)]],
    const device contracts::Float3* vertices [[buffer(binding::vertices)]],
    const device uint* instance_shapes [[buffer(binding::instance_shapes)]],
    constant serenity::gpu::SceneBlock& block [[buffer(binding::scene)]],
    constant contracts::Transform* transforms [[buffer(binding::transforms)]]) {
  // the instance id counts from the draw's base instance, its mesh's first
  // entry (acceleration/scene_acceleration.h, raster()): the mesh above
  // the shape (structure_block.h, instance_mesh_shift), carried to the
  // fragments with it
  const uint entry = instance_shapes[instance];
  const uint shape = entry & serenity::gpu::instance_low_mask;
  const contracts::Transform placed =
      instance_placement(Shapes{block.shapes, transforms, block.boxes}, shape);
  return RasterVertex{
      raster_position(
          camera, frame,
          transform_to_world(placed, to_float3(vertices[vertex_index]))),
      entry};
}

fragment uint2 raster_still_fragment(
    RasterVertex in [[stage_in]], uint primitive [[primitive_id]],
    const device serenity::gpu::MeshTriangle* triangles
    [[buffer(binding::triangles)]]) {
  return uint2(triangles[primitive].shape, primitive);
}

fragment uint2 raster_instanced_fragment(RasterVertex in [[stage_in]],
                                         uint primitive [[primitive_id]]) {
  // the shape, and the triangle with the mesh drawn above it
  return uint2(in.shape & serenity::gpu::instance_low_mask,
               (in.shape & ~serenity::gpu::instance_low_mask) | primitive);
}
