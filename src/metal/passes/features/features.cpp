#include "metal/passes/features/features.h"

#include <cstdint>

#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"
#include "metal/passes/features/visibility.h"
#include "metal/passes/scene_binding.h"

namespace serenity::metal {

FeaturesPass::FeaturesPass(const Device& device, const Library& library)
    : pipeline_(library.compute_pipeline(
          "features", {.group_threads = apple_group_threads})),
      still_(library.render_pipeline("raster_still", "raster_still_fragment",
                                     MTL::PixelFormatRG32Uint,
                                     MTL::PixelFormatDepth32Float)),
      instanced_(library.render_pipeline(
          "raster_instanced", "raster_instanced_fragment",
          MTL::PixelFormatRG32Uint, MTL::PixelFormatDepth32Float)) {
  const auto pool = scoped_pool();
  // Reversed depth (camera_clip(): near / view depth): the nearest is the
  // greatest, the clear 0 the farthest.
  auto depth = NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
  depth->setDepthCompareFunction(MTL::CompareFunctionGreater);
  depth->setDepthWriteEnabled(true);
  nearest_ =
      NS::TransferPtr(device.handle()->newDepthStencilState(depth.get()));
  auto table = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
  table->setMaxBufferBindCount(bindings::raster::transforms + 1);
  NS::Error* error = nullptr;
  raster_arguments_ =
      NS::TransferPtr(device.handle()->newArgumentTable(table.get(), &error));
  if (!nearest_ || !raster_arguments_) {
    throw MetalError(
        "FeaturesPass: the device made no depth test or argument "
        "table: " +
        describe(error));
  }
}

void FeaturesPass::rasterize(MTL4::CommandBuffer* commands,
                             const FrameResources& resources,
                             bool after_readers) const {
  if (commands == nullptr || resources.camera == 0 ||
      resources.constants == 0 || resources.scene == 0 ||
      resources.transforms == 0 || resources.instance_shapes == 0 ||
      resources.visibility == nullptr || resources.raster_depth == nullptr) {
    throw MetalError(
        "FeaturesPass::rasterize: no command buffer, or the frame lacks its "
        "camera, constants, scene, meshes or raster attachments");
  }
  const auto pool = scoped_pool();
  auto pass = NS::TransferPtr(MTL4::RenderPassDescriptor::alloc()->init());
  MTL::RenderPassColorAttachmentDescriptor* color =
      pass->colorAttachments()->object(0);
  color->setTexture(resources.visibility);
  color->setLoadAction(MTL::LoadActionClear);
  // where nothing is drawn, the sky (visibility.h)
  color->setClearColor(MTL::ClearColor(
      static_cast<double>(bindings::visibility::sky_shape), 0.0, 0.0, 0.0));
  color->setStoreAction(MTL::StoreActionStore);
  MTL::RenderPassDepthAttachmentDescriptor* depth = pass->depthAttachment();
  depth->setTexture(resources.raster_depth);
  depth->setLoadAction(MTL::LoadActionClear);
  depth->setClearDepth(0.0);
  depth->setStoreAction(MTL::StoreActionDontCare);
  MTL4::RenderCommandEncoder* encoder =
      commands->renderCommandEncoder(pass.get());
  if (encoder == nullptr) {
    throw MetalError("FeaturesPass::rasterize: no render encoder");
  }
  if (after_readers) {
    encoder->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageFragment,
                                     MTL4::VisibilityOptionDevice);
  }
  encoder->setViewport(
      MTL::Viewport{0.0, 0.0, static_cast<double>(resources.size.width),
                    static_cast<double>(resources.size.height), 0.0, 1.0});
  encoder->setDepthStencilState(nearest_.get());
  encoder->setCullMode(MTL::CullModeNone);
  // Bindings: passes/bindings.h, raster, as raster.metal declares them.
  namespace binding = bindings::raster;
  MTL4::ArgumentTable* arguments = raster_arguments_.get();
  arguments->setAddress(resources.camera, binding::camera);
  arguments->setAddress(resources.constants, binding::constants);
  arguments->setAddress(resources.instance_shapes, binding::instance_shapes);
  arguments->setAddress(resources.scene, binding::scene);
  arguments->setAddress(resources.transforms, binding::transforms);
  encoder->setArgumentTable(arguments,
                            MTL::RenderStageVertex | MTL::RenderStageFragment);
  for (const SceneAcceleration::Draw& draw : resources.draws) {
    if (!draw.in_world && draw.shape_count == 0) {
      continue;  // a mesh no shape takes
    }
    arguments->setAddress(draw.vertices, binding::vertices);
    arguments->setAddress(draw.triangles, binding::triangles);
    encoder->setRenderPipelineState(draw.in_world ? still_.get()
                                                  : instanced_.get());
    // Each structure's triangles as built, so a fragment's primitive id is
    // its triangle's index (visibility.h).
    encoder->drawIndexedPrimitives(
        MTL::PrimitiveTypeTriangle, draw.index_count, MTL::IndexTypeUInt32,
        draw.indices, NS::UInteger{draw.index_count} * sizeof(std::uint32_t),
        draw.in_world ? 1u : draw.shape_count, 0,
        draw.in_world ? 0u : draw.first_shape);
  }
  encoder->endEncoding();
}

void FeaturesPass::record(MTL4::ComputeCommandEncoder* encoder,
                          const FrameResources& resources) const {
  if (resources.feature_constants == 0 ||
      resources.feature_last_transforms == 0 || resources.albedo == nullptr ||
      resources.normal == nullptr || resources.depth == nullptr ||
      resources.motion == nullptr || resources.visibility == nullptr ||
      resources.mesh_block == 0) {
    throw MetalError(
        "FeaturesPass: the frame lacks one of its feature constants, last "
        "frame's transforms, feature images, visibility buffer or meshes");
  }
  // Bindings: passes/bindings.h, as features.metal declares them.
  namespace binding = bindings::features;
  MTL4::ArgumentTable* arguments = resources.arguments;
  bind_scene(resources, SceneCamera::bound, "FeaturesPass");
  arguments->setAddress(resources.feature_constants,
                        binding::feature_constants);
  arguments->setAddress(resources.feature_last_transforms,
                        binding::last_transforms);
  arguments->setTexture(resources.albedo->gpuResourceID(), binding::albedo);
  arguments->setTexture(resources.normal->gpuResourceID(), binding::normal);
  arguments->setTexture(resources.depth->gpuResourceID(), binding::depth);
  arguments->setTexture(resources.motion->gpuResourceID(), binding::motion);
  arguments->setTexture(resources.visibility->gpuResourceID(),
                        binding::visibility);
  arguments->setAddress(resources.mesh_block, binding::mesh_block);
  encoder->setComputePipelineState(pipeline_.get());
  dispatch_per_pixel(encoder, pipeline_.get(), resources.size);
}

}  // namespace serenity::metal
