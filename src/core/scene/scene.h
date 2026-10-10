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
#include "core/textures/wood.h"

namespace serenity::scene {

// Axis: Scene content (reading it).
//
// What is rendered, as data: the camera, the environment, the textures,
// the materials, the shapes that wear them, the lights, which are the
// spheres that wear an emissive material, and how the shapes that move
// move; and swarms, many fireflies from one entry (swarm.h). Read from a scene file in
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
//   lens = { radius = 0.012, focus = 4.0 }  # optional (contracts/camera.h):
//                                   # a thin lens, its radius and the distance
//                                   # in meters to the plane in focus;
//                                   # without one, a pinhole, all in focus
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
//   [textures.walnut]
//   kind = "wood"                   # a plank tabletop (core/textures/wood.h)
//   light = [0.13, 0.065, 0.03]     # earlywood, linear RGB in [0, 1]
//   dark = [0.045, 0.02, 0.008]     # latewood
//   ring = 0.004                    # meters between growth rings
//   board = 0.16                    # meters across a board; boards run along x
//   seed = 3
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
//   [[swarms]]                      # many fireflies (swarm.h)
//   count = 512                     # 1 to 4096
//   radius = 0.008                  # each firefly's
//   material = "glow"               # emissive
//   min = [-1.6, 0.8, -1.2]         # its flights' volume, where they start
//   max = [1.6, 1.7, 0.9]
//   targets = ["marble"]
//   speed = 0.35
//   clearance = 0.03
//   circle = 2
//   swoop = 1
//   drift = 2
//   flash = 0.35                    # its flight glows'
//   dim = 0.25
//   seed = 1
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
// The world is the cube within world_extent of the origin on every axis:
// every shape must lie inside it, a sphere's center grown by its radius and
// a box's corners, and so must everything a motion can reach, grown by the
// sphere's radius, computed in double. So no frame places a shape where a
// float cannot hold it, and every point a ray can hit is bounded, which is
// what keeps the textures' arithmetic finite: the wood's divisions and its
// noise's lattice coordinates (core/textures/wood.h), the checker's squares.
// A thousand kilometers: a scene of a table and its ground is a few hundred
// meters across. Moving shapes are not checked against each other:
// fireflies may pass through one another, which renders as what it is.
//
// A swarm (swarm.h) is `count` fireflies, each a sphere of `radius` wearing
// `material`, which must be emissive, with a flight motion of the swarm's
// numbers and a flight glow of its `flash` and `dim`, each with its own seed
// and start, drawn from the swarm's seed: what a firefly written as a
// [[shapes]] entry with a flight and a flight glow is, checked by the same
// rules. Its shapes follow the file's [[shapes]], swarm by swarm.
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
// Every key is checked, as in graph files: a missing or unknown key, a value
// of the wrong type or out of range (a radius or size not greater than 0, an
// ior not greater than 1, an f0 outside [0, 1], a roughness outside (0, 1], a
// negative radiance, a box whose min is not below its max, a lens radius
// below 0 or focus not above 0, a camera that cannot be framed, a reach or speed not greater than 0, a seed below 0, a
// shape outside the world or a motion that could carry it out, a flight's box
// whose min is not below its max, a negative clearance or weight, weights all
// 0, a circle weight with no targets, a period or flash out of range, a dim
// outside [0, 1), a wood color outside [0, 1], a ring under wood_least_ring or
// a board outside wood_least_board to wood_most_board (core/textures/wood.h),
// a wood's seed past 2^32 - 1, a swarm's count outside 1 to 4096), an unknown
// kind, a name used twice, a name used and never defined, a target that is not
// a still sphere, an emissive material or a motion on anything but a sphere, a
// swarm whose material is not emissive, a glow on anything but a light, a
// flight glow on a light that does not fly, a motion that cannot be made clear
// of the still shapes, a swarm whose fireflies cannot start clear of them, or
// no shapes at all is an Error naming the file and the line (E.2, E.14).
// Nothing has a silent default except `up`.
//
// Read once, at start-up. Every flight in the scene, written or a swarm's,
// is made once every shape is read, all together, in parallel
// (core/animation/flight.h, make_flights): for the marbles' 512 fireflies
// the load's largest cost; the scene loads in 104 ms on the M3 Max.

// How far the world reaches from the origin on every axis, in meters (above).
inline constexpr double world_extent = 1.0e6;

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

struct SceneDescription {
    // The camera as the file places it (contracts/camera.h); valid
    // (camera/thin_lens.h). Each frame is seen through it until the camera
    // moves.
    contracts::Camera camera;
    lights::GradientSkyData environment;

    std::vector<textures::TextureRecord> textures;
    std::vector<textures::CheckerData> checkers;
    std::vector<textures::WoodData> woods;

    std::vector<materials::MaterialRecord> materials;
    std::vector<materials::RoughData> rough;
    std::vector<materials::DielectricData> dielectrics;
    std::vector<materials::ConductorData> conductors;
    std::vector<materials::EmissiveData> emissives;

    // In file order, the [[shapes]] and then each swarm's fireflies
    // (swarm.h): shape i is shapes.records[i] and shapes.transforms[i],
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
