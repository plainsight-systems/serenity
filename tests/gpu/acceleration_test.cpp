// The acceleration structure (metal/acceleration/scene_acceleration.h), on
// the GPU, probed ray by ray through the tracing loop (kernels/
// trace_probe.metal): a hit, tested in a shape's own space, is at the
// world's distance, within the backend's gap where it traces a mesh
// (support/traced_probe.h, hit_gap()), and names the shape, at any scale; and a
// moved shape is hit where it moved to and not where it was, through both frame
// slots; and contract 19's masks (core/contracts/ray_mask.h): the occluders
// never find a light and find each surface as everything would, moving or not.
//
// Its own dispatch, not the probe runner's (support/traced_probe.h): it
// binds the structures' block, and records the structures' update in
// the same encoder as the trace, as a frame does.

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// the backend's harness, by the include path (SF.12)
#include <traced_probe.h>

#include "core/scene/scene.h"
#include "gpu/kernels/trace_probe.h"
#include "support/vector.h"

namespace contracts = serenity::contracts;
namespace scene = serenity::scene;
namespace tests = serenity::tests;
using tests::TraceHit;
using tests::TraceRay;

namespace {

// Traces `rays` through the structure for the next frame's slot, in one
// command buffer; if `moved` is given, first writes them as the slot's
// transforms and records the structure's update, as a frame does
// (support/traced_probe.h).
std::vector<TraceHit> probe(
    tests::TracedScene& traced, const std::vector<TraceRay>& rays,
    const std::optional<std::vector<contracts::Transform>>& moved) {
  const auto count = static_cast<std::uint32_t>(rays.size());
  return tests::run_traced<TraceHit>(
      traced, "trace_probe", count,
      {tests::Binding::of(rays), tests::Binding::of(count)}, moved);
}

// A ray from `origin` along the unit vector toward `toward`, carrying
// `mask`; its shadow-ray twin looks as far as `reach`.
TraceRay ray(tests::Vec3 origin, tests::Vec3 toward,
             contracts::RayMask mask = contracts::RayMask::everything,
             float reach = 1.0e30f) {
  const tests::Vec3 d = tests::normalized(toward);
  return TraceRay{tests::float3(origin), tests::float3(d), mask, reach};
}

// A tiny firefly and a sphere wider than one, and a box: every scale the
// structure carries rays into object space at.
constexpr const char* scales = R"(
[camera]
position = [0, 0, 10]
look_at = [0, 0, 0]
vertical_fov_degrees = 40
[environment]
kind = "gradient"
zenith = [0, 0, 0]
horizon = [0, 0, 0]
[materials.glow]
kind = "emissive"
radiance = [1, 1, 1]
[materials.matte]
kind = "rough"
color = [0.5, 0.5, 0.5]
[[shapes]]
kind = "sphere"
center = [0, 0, -5]
radius = 0.03
material = "glow"
motion = { kind = "wander", reach = 0.01, speed = 0.01, seed = 1 }
[[shapes]]
kind = "sphere"
center = [20, 0, 0]
radius = 7.5
material = "matte"
[[shapes]]
kind = "box"
min = [-1, 9, -2]
max = [1, 11, 2]
material = "matte"
)";

// The firefly, its radius and its center's distance from the origin.
constexpr double firefly_radius = 0.03;
constexpr double firefly_distance = 5.0;

}  // namespace

TEST_CASE(
    "a hit, tested in a shape's own space, is at the world's distance, and "
    "names its shape") {
  const scene::SceneDescription description = scene::parse(scales, "scales");
  // Still: the shapes where the file puts them.
  tests::TracedScene traced(description, {});

  const std::vector<TraceRay> rays = {
      ray({0, 0, 0}, {0, 0, -1}),  // the firefly, 5 away: t = 4.97
      ray({0, 0, 0},
          {1, 0, 0}),  // the wide sphere, radius 7.5, 20 away: t = 12.5
      ray({0, 0, 0}, {0, 1, 0}),      // the box, its bottom face at y = 9
      ray({0, 0, 0}, {0, -1, 0}),     // nothing
      ray({0, 0.02, 0}, {0, 0, -1}),  // the firefly off its middle: sqrt(0.03^2
                                      // - 0.02^2) short of 5
  };
  const std::vector<TraceHit> hits = probe(traced, rays, std::nullopt);
  // Within `relative` of `expected`, and the backend's gap for `shape`
  // along a ray meeting its surface at `cosine` (traced_probe.h).
  const auto near = [&](float t, double expected, double relative,
                        std::uint32_t shape, double cosine) {
    return std::abs(t - expected) <=
           relative * expected + traced.hit_gap(shape) / cosine;
  };
  CHECK(hits[0].found == 1u);
  CHECK(near(hits[0].t, firefly_distance - firefly_radius, 1e-6, 0u, 1.0));
  CHECK(hits[0].shape == 0u);
  CHECK(hits[1].found == 1u);
  CHECK(near(hits[1].t, 12.5, 1e-6, 1u, 1.0));
  CHECK(hits[1].shape == 1u);
  CHECK(hits[2].found == 1u);
  CHECK(near(hits[2].t, 9.0, 1e-6, 2u, 1.0));
  CHECK(hits[2].shape == 2u);
  CHECK(hits[3].found == 0u);
  CHECK(hits[4].found == 1u);
  // Off the firefly's middle by 0.02 of its 0.03: the ray meets its
  // surface at a cosine of sqrt(1 - (2/3)^2).
  const double off_cosine = std::sqrt(1.0 - (0.02 / 0.03) * (0.02 / 0.03));
  CHECK(near(hits[4].t,
             firefly_distance -
                 std::sqrt(firefly_radius * firefly_radius - 0.02 * 0.02),
             1e-5, 0u, off_cosine));
}

TEST_CASE("a moved shape is hit where it moved to, and not where it was") {
  const scene::SceneDescription description = scene::parse(scales, "scales");
  const std::vector<std::uint32_t> moving = {0};
  tests::TracedScene traced(description, moving);

  // Each slot in turn, so both structures are built and traced.
  for (int round = 0; round < 4; ++round) {
    std::vector<contracts::Transform> moved = description.shapes.transforms;
    const float x = 3.0f + static_cast<float>(round);
    moved[0] = contracts::moved_to(moved[0], {x, 0.0f, -5.0f});
    const std::vector<TraceRay> rays = {
        ray({x, 0, 0}, {0, 0, -1}),  // where it is now
        ray({0, 0, 0}, {0, 0, -1}),  // where it was at rest
    };
    const std::vector<TraceHit> hits = probe(traced, rays, moved);
    INFO("round " << round);
    CHECK(hits[0].found == 1u);
    CHECK(hits[0].t == doctest::Approx(firefly_distance - firefly_radius)
                           .scale(0)
                           .epsilon(1e-6));
    CHECK(hits[0].shape == 0u);
    CHECK(hits[1].found == 0u);
  }
}

namespace {

// Lights and surfaces interleaved in shape order, so the occluders'
// structure names its shapes through its table, not by position: a light
// (0) with a surface behind it along -z (1), a second light (2) with a box
// beyond it along +x (3). With `moving`, the first light and the surface
// behind it each wander.
std::string masks_scene(bool moving) {
  const std::string wander = moving ? "\nmotion = { kind = \"wander\", reach = "
                                      "0.01, speed = 0.01, seed = 1 }"
                                    : "";
  return R"(
[camera]
position = [0, 0, 10]
look_at = [0, 0, 0]
vertical_fov_degrees = 40
[environment]
kind = "gradient"
zenith = [0, 0, 0]
horizon = [0, 0, 0]
[materials.glow]
kind = "emissive"
radiance = [1, 1, 1]
[materials.matte]
kind = "rough"
color = [0.5, 0.5, 0.5]
[[shapes]]
kind = "sphere"
center = [0, 0, -5]
radius = 0.03
material = "glow")" +
         wander + R"(
[[shapes]]
kind = "sphere"
center = [0, 0, -8]
radius = 1
material = "matte")" +
         wander + R"(
[[shapes]]
kind = "sphere"
center = [3, 0, 0]
radius = 0.1
material = "glow"
[[shapes]]
kind = "box"
min = [6, -1, -1]
max = [7, 1, 1]
material = "matte"
)";
}

}  // namespace

// Contract 19's guarantee (core/contracts/ray_mask.h): a ray carrying the
// occluders never stops at, is never blocked by, and never reports a
// light, and finds the surfaces beyond as their own shapes at their own
// distances; a ray carrying everything still finds the lights.
TEST_CASE(
    "a ray carrying the occluders skips every light; one carrying everything "
    "still finds them") {
  const scene::SceneDescription description =
      scene::parse(masks_scene(false), "masks");
  REQUIRE(scene::light_shapes(description) == std::vector<std::uint32_t>{0, 2});
  tests::TracedScene traced(description, {});
  using contracts::RayMask;
  const std::vector<TraceRay> rays = {
      ray({0, 0, 0}, {0, 0, -1}, RayMask::everything,
          6.0f),  // the light at 4.97, within reach
      ray({0, 0, 0}, {0, 0, -1}, RayMask::occluders,
          6.0f),  // through it, to the surface at 7
      ray({0, 0, 0}, {0, 0, -1}, RayMask::occluders,
          7.5f),  // reaching past the surface
      ray({0, 0, 0}, {1, 0, 0}, RayMask::everything,
          3.0f),  // the second light at 2.9
      ray({0, 0, 0}, {1, 0, 0}, RayMask::occluders,
          3.0f),  // through it, to the box at 6
      ray({0, 0, 0}, {0, 1, 0}, RayMask::occluders),  // nothing
  };
  const std::vector<TraceHit> hits = probe(traced, rays, std::nullopt);
  CHECK(hits[0].found == 1u);
  CHECK(hits[0].shape == 0u);
  CHECK(hits[0].t == doctest::Approx(5.0 - 0.03).scale(0).epsilon(1e-6));
  CHECK(hits[0].occluded == 1u);
  CHECK(hits[1].found == 1u);
  CHECK(hits[1].shape == 1u);
  CHECK(hits[1].t == doctest::Approx(7.0).scale(0).epsilon(1e-6));
  CHECK(hits[1].occluded == 0u);  // the light in its way does not block it
  CHECK(hits[2].occluded == 1u);
  CHECK(hits[3].found == 1u);
  CHECK(hits[3].shape == 2u);
  CHECK(hits[3].t == doctest::Approx(2.9).scale(0).epsilon(1e-6));
  CHECK(hits[3].occluded == 1u);
  CHECK(hits[4].found == 1u);
  CHECK(hits[4].shape == 3u);
  CHECK(hits[4].t == doctest::Approx(6.0).scale(0).epsilon(1e-6));
  CHECK(hits[4].occluded == 0u);
  CHECK(hits[5].found == 0u);
  CHECK(hits[5].occluded == 0u);
}

// The occluders' structure keeps up with its own moving shapes, slot by
// slot, and never holds a moving light: a moved surface is found where it
// moved to by both masks, a moved light by everything alone.
TEST_CASE(
    "a moved surface is found by both masks where it moved to, a moved light "
    "by everything alone") {
  const scene::SceneDescription description =
      scene::parse(masks_scene(true), "masks_moving");
  const std::vector<std::uint32_t> moving = {0, 1};
  tests::TracedScene traced(description, moving);
  using contracts::RayMask;
  for (int round = 0; round < 4; ++round) {
    std::vector<contracts::Transform> moved = description.shapes.transforms;
    const float y = 20.0f + static_cast<float>(round);
    moved[0] = contracts::moved_to(moved[0], {0.0f, y, -5.0f});   // the light
    moved[1] = contracts::moved_to(moved[1], {0.0f, -y, -8.0f});  // the surface
    const std::vector<TraceRay> rays = {
        ray({0, y, 0}, {0, 0, -1},
            RayMask::everything),  // the light, where it is now
        ray({0, y, 0}, {0, 0, -1}, RayMask::occluders),  // not by the occluders
        ray({0, -y, 0}, {0, 0, -1},
            RayMask::occluders),  // the surface, where it is now
        ray({0, -y, 0}, {0, 0, -1}, RayMask::everything),
        ray({0, 0, 0}, {0, 0, -1},
            RayMask::occluders),  // where both were at rest: nothing
    };
    const std::vector<TraceHit> hits = probe(traced, rays, moved);
    INFO("round " << round);
    CHECK(hits[0].found == 1u);
    CHECK(hits[0].shape == 0u);
    CHECK(hits[1].found == 0u);
    CHECK(hits[2].found == 1u);
    CHECK(hits[2].shape == 1u);
    CHECK(hits[2].t == doctest::Approx(7.0).scale(0).epsilon(1e-6));
    CHECK(hits[3].found == 1u);
    CHECK(hits[3].shape == 1u);
    CHECK(hits[4].found == 0u);
  }
}

// The two edges of structure_block.h: with no light the occluders are
// everything, named through the identity; with no surface they find
// nothing, and no structure is made for them.
TEST_CASE(
    "the occluders are every shape in a scene of surfaces, and none in a scene "
    "of lights") {
  constexpr const char* head = R"(
[camera]
position = [0, 0, 10]
look_at = [0, 0, 0]
vertical_fov_degrees = 40
[environment]
kind = "gradient"
zenith = [0, 0, 0]
horizon = [0, 0, 0]
[materials.glow]
kind = "emissive"
radiance = [1, 1, 1]
[materials.matte]
kind = "rough"
color = [0.5, 0.5, 0.5]
)";
  for (const std::string material : {"matte", "glow"}) {
    std::string text(head);
    for (const std::string_view center : {"[0, 0, -5]", "[4, 0, 0]"}) {
      text += "[[shapes]]\nkind = \"sphere\"\ncenter = ";
      text += center;
      text += "\nradius = 1\nmaterial = \"";
      text += material;
      text += "\"\n";
    }
    const scene::SceneDescription description = scene::parse(text, material);
    tests::TracedScene traced(description, {});
    using contracts::RayMask;
    const std::vector<TraceRay> rays = {
        ray({0, 0, 0}, {0, 0, -1}, RayMask::occluders),
        ray({0, 0, 0}, {1, 0, 0}, RayMask::occluders),
        ray({0, 0, 0}, {1, 0, 0}, RayMask::everything),
    };
    const std::vector<TraceHit> hits = probe(traced, rays, std::nullopt);
    const bool surfaces = material == "matte";
    INFO(material);
    CHECK(hits[0].found == (surfaces ? 1u : 0u));
    CHECK(hits[0].occluded == (surfaces ? 1u : 0u));
    CHECK(hits[1].found == (surfaces ? 1u : 0u));
    if (surfaces) {
      CHECK(hits[0].shape == 0u);
      CHECK(hits[1].shape == 1u);
      CHECK(hits[1].t == doctest::Approx(3.0).scale(0).epsilon(1e-6));
    }
    CHECK(hits[2].found == 1u);
    CHECK(hits[2].shape == 1u);
  }
}
