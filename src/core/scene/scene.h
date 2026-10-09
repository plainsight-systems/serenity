#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/animation/animate.h"
#include "core/contracts/camera.h"
#include "core/lights/gradient_sky.h"
#include "core/lights/light.h"
#include "core/lights/sphere_light.h"
#include "core/materials/conductor.h"
#include "core/materials/dielectric.h"
#include "core/materials/emissive.h"
#include "core/materials/material.h"
#include "core/materials/rough.h"
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
//   name = "marble"                 # optional: what a flight may circle
//   center = [2, 0.4, 0]
//   radius = 0.4
//   material = "glass"
//
//   [[shapes]]
//   kind = "sphere"
//   center = [1, 1.4, -0.5]         # a moving shape's anchor, or start
//   radius = 0.03
//   material = "glow"
//   motion = { kind = "wander", reach = 0.25, speed = 0.3, seed = 7 }
//   glow = { kind = "rhythm", period = 5.5, flash = 0.35, dim = 0.05, seed = 3 }
//
//   [[shapes]]
//   kind = "sphere"
//   center = [-1, 1.2, 1]
//   radius = 0.03
//   material = "glow"
//   motion = { kind = "flight", min = [-3, 0.1, -3], max = [3, 2.5, 3],
//              targets = ["marble"], speed = 0.5, clearance = 0.08,
//              circle = 3, swoop = 1, drift = 1, seed = 11 }
//   glow = { kind = "flight", flash = 0.35, dim = 0.05 }
//
// A name is optional and unique among the shapes: it is how a flight names
// what it circles.
//
// A motion (core/animation/motion.h) is optional, and only a sphere has
// one. Each kind is made clear of the still shapes, by that shape kind's
// exact tests (shapes/shapes.h), and refused otherwise (motion.h):
//
//   - wander (core/animation/wander.h) drifts about the sphere's center: at
//     most `reach` along each axis, at a root-mean-square `speed` in meters
//     a second, on a path drawn from `seed`, an integer from 0. Its reach,
//     grown by the sphere's radius, may meet no still shape;
//   - flight (core/animation/flight.h) flies free within the box `min` to
//     `max`, starting at the sphere's center: circling the `targets`, still
//     spheres named in this file (none, and `circle` must be 0), swooping
//     and drifting, in the proportions `circle`, `swoop` and `drift` (each 0
//     or more, not all 0), at a cruising `speed`, its surface at least
//     `clearance` meters from every still surface, on a loop drawn from
//     `seed`. Its every stretch is checked clear at load.
//
// Everything a motion can reach, grown by the sphere's radius, must lie
// within float's range on every axis, computed in double, so no frame places
// it where a float cannot hold. Moving shapes are not checked against each
// other: fireflies may pass through one another, which renders as what it
// is.
//
// A glow (core/animation/glow.h) is optional, and only a sphere that is a
// light has one; without it, the light shines at its radiance always. Its
// brightness is a factor on the radiance, from `dim` between flashes (in
// [0, 1)) up to 1 at a flash's peak, a flash lasting `flash` seconds:
//
//   - rhythm flashes every `period` seconds, each moved by up to a fifth of
//     it, drawn from `seed`; `flash` at most half the period;
//   - flight flashes when the light's flight says: on each swoop's climb,
//     now and then while circling or drifting (its flight's flash schedule,
//     copied into a glow of the schedule kind, core/animation/flashes.h).
//     The sphere's motion must be a flight; `flash` under a second.
//
// Every key is checked, as in graph files: a missing or unknown key, a
// value of the wrong type or out of range (a radius or size not greater
// than 0, an ior not greater than 1, an f0 outside [0, 1], a roughness
// outside (0, 1], a negative radiance, a box whose min is not below its max,
// a camera that cannot be framed, a reach or speed not greater than 0, a
// seed below 0, a motion that could carry its shape out of float's range, a
// flight's box whose min is not below its max, a negative clearance or
// weight, weights all 0, a circle weight with no targets, a period or flash
// out of range, a dim outside [0, 1)), an unknown kind, a name used twice, a
// name used and never defined, a target that is not a still sphere, an
// emissive material or a motion on anything but a sphere, a glow on
// anything but a light, a flight glow on a light that does not fly, a motion
// that cannot be made clear of the still shapes, or no shapes at all is an
// Error naming the file and the line (E.2, E.14). Nothing has a silent
// default except `up`.
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
    // each at rest, and primitive i of the acceleration structure
    // (shapes/primitive.h). A sphere is the unit sphere placed at its center,
    // scaled by its radius; a box, its corners as the file gives them,
    // placed by the identity transform.
    shapes::Shapes shapes;

    // One record per light, and one array per light kind: a sphere light
    // for each sphere that wears an emissive material, in shape order.
    std::vector<lights::LightRecord> lights;
    std::vector<lights::SphereLightData> sphere_lights;
    // One per shape, in primitive order: its light record's index, or
    // lights::no_light (core/lights/light.h).
    std::vector<std::uint32_t> shape_lights;
    lights::LightCounts light_counts{};

    // Which shapes move and which lights glow, and how: a mover per moving
    // shape, its target the shape's index, and a glower per glowing light,
    // its target the light's index among the sphere lights
    // (core/animation/animate.h). Empty for a still scene. The transforms
    // above place each moving shape at its anchor or start, where nothing
    // renders it: a frame places it at its time first.
    animation::Animation animation;
};

// Whether anything in `scene` changes with time, moving or glowing: what
// the history plans ask (core/frame/history.h).
inline bool changes(const SceneDescription& scene) {
    return animation::changes(scene.animation);
}

// Reads the scene file at `path`.
SceneDescription load(const std::filesystem::path& path);

// Reads a scene from `text`, naming it `source` in errors. For tests, and
// for load() itself.
SceneDescription parse(std::string_view text, std::string_view source);

}  // namespace serenity::scene
