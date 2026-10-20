#pragma once

// Axis: Scene content (device half of core/scene/scene_block.h).
//
// The scene as an estimator reads it, Metal's SceneView (metal/scene/
// scene_block.metal.h, which says why each view is built where it is used):
// the scene's block (scene_block.h), the frame's transforms and glows, and
// the structures, one a ray mask (acceleration/trace.cuh); and from them
// each family's view, the shapes, materials, textures, lights and media
// (shapes/shapes.cuh, materials/resolve.cuh, textures/textures.cuh,
// lights/emitter.cuh, media/media.cuh). A new kind's array joins its
// family's view here, as it joins the block, so no pass changes for it.
// `Selection` is how the estimator reading it chooses among the lights
// (light_selection/): UniformLight for the path tracer. ReSTIR PT's
// selection by power, and the power pyramid it reads, arrive with it
// (Phase 4).
//
// The block and the arrays it names are read from global memory, through
// const pointers (cuda/scene/scene_buffers.h); Metal reads them from its
// constant address space.

#include "core/contracts/transform.h"
#include "core/lights/gradient_sky.h"
#include "core/scene/scene_block.h"
#include "cuda/acceleration/trace.cuh"
#include "cuda/lights/emitter.cuh"
#include "cuda/materials/resolve.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/media/media.cuh"
#include "cuda/shapes/shapes.cuh"
#include "cuda/textures/textures.cuh"

namespace serenity {
namespace shaders {

template <typename Selection>
struct SceneView {
  // the structures by mask (acceleration/structure_block.h)
  SceneStructure structure;
  const serenity::gpu::SceneBlock* block;
  const serenity::contracts::Transform*
      transforms;      // as this frame places the shapes
  const float* glows;  // as bright as this frame lights them

  __device__ Shapes shapes() const {
    return Shapes{block->shapes, transforms, block->boxes};
  }

  __device__ Materials materials() const {
    return Materials{block->materials, block->roughs, block->dielectrics,
                     block->conductors, block->coateds};
  }

  __device__ Textures textures() const {
    return Textures{block->textures, block->checkers, block->woods,
                    block->swirls};
  }

  __device__ Lights lights() const {
    return Lights{block->light_records, block->shape_lights,
                  block->sphere_lights, transforms, glows};
  }

  __device__ Media media() const {
    return Media{block->media, block->absorbings};
  }

  __device__ serenity::lights::GradientSkyData sky() const {
    return *block->environment;
  }

  __device__ Selection selection() const {
    return Selection{block->light_records, block->light_counts->lights};
  }
};

}  // namespace shaders
}  // namespace serenity
