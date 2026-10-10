// The shapes' surface interaction on the GPU (tests/gpu/kernels/
// shape_probe.metal), held to contract 1: the point in the shape's own
// coordinates is where the hit is on its geometry before its transform, and
// the interior is the shape's medium (contract 12).

#include <cstdint>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/medium.h"
#include "core/contracts/surface_interaction.h"
#include "core/scene/scene.h"
#include "gpu/kernels/probes.h"
#include "gpu/support/probe_runner.h"

using namespace serenity;
using tests::Binding;
using tests::ShapeQuery;

namespace {

// interaction_probe over `queries`, the scene's shape arrays bound beside.
std::vector<contracts::SurfaceInteraction> interactions(const scene::SceneDescription& description,
                                                        const std::vector<ShapeQuery>& queries) {
    tests::ProbeRunner gpu;
    const auto count = static_cast<std::uint32_t>(queries.size());
    const shapes::Shapes& shapes = description.shapes;
    return gpu.run<contracts::SurfaceInteraction>(
        "interaction_probe", queries.size(),
        {Binding::of(queries), Binding::of(count), Binding::of(shapes.records), Binding::of(shapes.transforms),
         Binding::of(shapes.boxes)});
}

}  // namespace

TEST_CASE("a surface interaction carries the point on its shape's own geometry, and the medium inside") {
    const scene::SceneDescription description = scene::parse(R"(
[camera]
position = [0, 0, 5]
look_at = [0, 0, 0]
vertical_fov_degrees = 40
[environment]
kind = "gradient"
zenith = [0, 0, 0]
horizon = [0, 0, 0]
[materials.glass]
kind = "dielectric"
ior = 1.5
[materials.matte]
kind = "rough"
color = [0.5, 0.5, 0.5]
[media.tint]
kind = "absorbing"
tint = [0.5, 0.5, 0.5]
tint_distance = 1
[[shapes]]
kind = "sphere"
center = [1, 2, 3]
radius = 0.5
material = "glass"
interior = "tint"
[[shapes]]
kind = "box"
min = [-2, -1, -1]
max = [-1, 0, 0]
material = "matte"
)",
                                                             "s");
    // On the sphere, 0.5 from (1, 2, 3) along (0.6, 0, 0.8): on the unit
    // sphere at (0.6, 0, 0.8). On the box's top face, its own corners being
    // the world's.
    const std::vector<ShapeQuery> queries = {{{1.3f, 2.0f, 3.4f}, 0u}, {{-1.5f, 0.0f, -0.5f}, 1u}};
    const auto out = interactions(description, queries);
    CHECK(out[0].object_position.x == doctest::Approx(0.6f).scale(0).epsilon(1e-5));
    CHECK(out[0].object_position.y == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(out[0].object_position.z == doctest::Approx(0.8f).scale(0).epsilon(1e-5));
    CHECK(out[0].interior == 0u);
    CHECK(out[0].geometric_normal.x == doctest::Approx(0.6f).scale(0).epsilon(1e-5));
    CHECK(out[1].object_position.x == doctest::Approx(-1.5f).scale(0).epsilon(1e-6));
    CHECK(out[1].object_position.y == doctest::Approx(0.0f).epsilon(1e-6));
    CHECK(out[1].object_position.z == doctest::Approx(-0.5f).scale(0).epsilon(1e-6));
    CHECK(out[1].interior == contracts::no_medium);
}
