#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstdint>
#include <string>
#include <string_view>

#include "metal/device/error.h"
#include "metal/frame/frame_resources.h"
#include "metal/passes/bindings.h"

namespace serenity::metal {

// Axis: Pass (every pass that reads the scene).
//
// The scene as each such pass binds it, at bindings::scene_view's indices
// 0 to 5, which every pass's own namespace is asserted to repeat
// (bindings.h): the frame's constants and camera, the structures (one a
// ray mask, metal/acceleration/structure_block.h), the
// scene's block whole, and what changes per frame beside it, the sphere
// lights' glows and the shapes' transforms (core/scene/scene_block.h).
// Written once (ES.3, P.5) where eleven passes each wrote it and its test.

// Whether a pass reads the camera, and so binds it (I.4).
enum class SceneCamera : std::uint8_t { bound, none };

// Binds the frame's scene into `resources.arguments`, and the camera when
// `camera` says. Throws MetalError, naming `pass`, if the frame lacks its
// scene, structures, transforms or glows, or, bound, its camera.
inline void bind_scene(const FrameResources& resources, SceneCamera camera,
                       std::string_view pass) {
  const bool with_camera = camera == SceneCamera::bound;
  if (resources.scene == 0 || resources.acceleration == 0 ||
      resources.transforms == 0 || resources.glows == 0 ||
      (with_camera && resources.camera == 0)) {
    throw MetalError(
        std::string(pass) +
        ": the frame lacks one of its scene, structures, transforms, glows" +
        (with_camera ? " or camera" : ""));
  }
  namespace binding = bindings::scene_view;
  MTL4::ArgumentTable* arguments = resources.arguments;
  arguments->setAddress(resources.constants, binding::constants);
  if (with_camera) {
    arguments->setAddress(resources.camera, binding::camera);
  }
  arguments->setAddress(resources.acceleration,
                        binding::structure);  // the structures by mask
  arguments->setAddress(resources.scene, binding::scene);
  arguments->setAddress(resources.glows,
                        binding::glows);  // the sphere lights' glows this frame
  arguments->setAddress(
      resources.transforms,
      binding::transforms);  // as this frame places the shapes
}

}  // namespace serenity::metal
