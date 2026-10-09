#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/contracts/camera.h"
#include "core/lights/gradient_sky.h"
#include "core/lights/light.h"
#include "core/lights/sphere_light.h"
#include "core/materials/conductor.h"
#include "core/materials/dielectric.h"
#include "core/materials/emissive.h"
#include "core/materials/material.h"
#include "core/materials/rough.h"
#include "core/scene/animate.h"
#include "core/shapes/shapes.h"
#include "core/textures/checker.h"
#include "core/textures/texture.h"

namespace serenity::scene {

// Axis: Scene content (reading it).
//
// What is rendered, as data: the camera, the environment, the textures,
// the materials, the shapes that wear them, the lights, which are the
// spheres that wear an emissive material, and how the shapes that move
// move. Read from a scene file in
// scenes/, kept apart from frame graph files (core/frame/graph_file.h), so
// one scene runs under any graph.
//
// The description is the scene as the GPU will hold it, built on the CPU: an
// array per kind in its shared layout (shapes/, materials/, textures/,
// lights/), and the records that say which kind each shape, material and
// texture is. Names in the file become indices here, so nothing downstream
// looks anything up by name.
//
// The format (TOML, read with toml++, as graph files are):
//
//   [camera]
//   position = [0, 1.2, 4]          # x, y, z; y is up
//   look_at = [0, 0.8, 0]
//   up = [0, 1, 0]                  # optional; this is the default
//   vertical_fov_degrees = 40
//
//   [environment]
//   kind = "gradient"
//   zenith = [0.02, 0.03, 0.08]     # linear RGB
//   horizon = [0.15, 0.17, 0.25]
//
//   [textures.floor_checks]         # a name, used by materials
//   kind = "checker"
//   size = 0.5
//   a = [0.9, 0.9, 0.9]
//   b = [0.1, 0.1, 0.1]
//
//   [materials.glass]               # a name, used by shapes
//   kind = "dielectric"
//   ior = 1.5
//
//   [materials.floor]
//   kind = "rough"
//   texture = "floor_checks"        # or: color = [r, g, b]
//
//   [materials.brass]
//   kind = "conductor"
//   f0 = [0.91, 0.78, 0.42]         # reflectance at normal incidence
//   roughness = 0.35                # in (0, 1]
//
//   [materials.glow]
//   kind = "emissive"
//   radiance = [40, 36, 12]         # linear RGB, may exceed 1
//
//   [[shapes]]
//   kind = "sphere"
//   center = [0, 0.8, 0]
//   radius = 0.8
//   material = "glass"
//
//   [[shapes]]
//   kind = "box"
//   min = [-6, -0.1, -6]
//   max = [6, 0, 6]
//   material = "floor"
//
//   [[shapes]]
//   kind = "sphere"
//   center = [1, 1.4, -0.5]         # a moving shape's anchor
//   radius = 0.03
//   material = "glow"
//   motion = { kind = "wander", reach = 0.25, speed = 0.3, seed = 7 }
//
// A motion (core/animation/motion.h) is optional, and only a sphere has
// one. A wander (core/animation/wander.h) drifts about the sphere's center:
// at most `reach` along each axis, at a root-mean-square `speed` in meters
// a second, on a path drawn from `seed`, an integer from 0. The shape must
// touch no shape that does not move, wherever its motion takes it: its
// motion's extent, grown by its radius on every side, may not meet any
// still shape, by that shape kind's exact test (shapes/shapes.h).
// Moving shapes are not checked against each other: fireflies may pass
// through one another, which renders as what it is.
//
// Every key is checked, as in graph files: a missing or unknown key, a
// value of the wrong type or out of range (a radius or size not greater
// than 0, an ior not greater than 1, an f0 outside [0, 1], a roughness
// outside (0, 1], a negative radiance, a box whose min is not below its max,
// a camera that cannot be framed, a reach or speed not greater than 0, a
// seed below 0), an unknown kind, a name used and never defined, an
// emissive material or a motion on anything but a sphere, a moving shape
// that could touch a still one, or no shapes at all is an Error naming the
// file and the line (E.2, E.14). Nothing has a silent default except `up`.
//
// Not performance-sensitive: read once, at start-up.

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

struct SceneDescription {
    // The camera as the file places it (contracts/camera.h); valid
    // (camera/pinhole.h). Each frame is seen through it until the camera
    // moves.
    contracts::Camera camera;
    lights::GradientSkyData environment;

    std::vector<textures::TextureRecord> textures;
    std::vector<textures::CheckerData> checkers;

    std::vector<materials::MaterialRecord> materials;
    std::vector<materials::RoughData> rough;
    std::vector<materials::DielectricData> dielectrics;
    std::vector<materials::ConductorData> conductors;
    std::vector<materials::EmissiveData> emissives;

    // In file order: shape i is shapes.records[i] and shapes.transforms[i],
    // each at rest, and instance i of the acceleration structure
    // (shapes/primitive.h). A sphere is the unit sphere placed at its center,
    // scaled by its radius; a box, a box about its origin, its half extent
    // the file's max - min halved, placed at its middle.
    shapes::Shapes shapes;

    // One record per light, and one array per light kind: a sphere light
    // for each sphere that wears an emissive material, in shape order.
    std::vector<lights::LightRecord> lights;
    std::vector<lights::SphereLightData> sphere_lights;
    // One per shape, in primitive order: its light record's index, or
    // lights::no_light (core/lights/light.h).
    std::vector<std::uint32_t> shape_lights;
    lights::LightCounts light_counts{};

    // The shapes that move, and their motions (animate.h). Empty for a
    // still scene. The transforms above place each moving shape at its
    // anchor, where nothing renders it: a frame places it at its time first.
    SceneAnimation animation;
};

// Reads the scene file at `path`.
SceneDescription load(const std::filesystem::path& path);

// Reads a scene from `text`, naming it `source` in errors. For tests, and
// for load() itself.
SceneDescription parse(std::string_view text, std::string_view source);

}  // namespace serenity::scene
