// The Metal backend's rasterized features (metal/passes/features/
// features.h; docs/research/2026-10-23-apple-silicon-design.md, section 6,
// change 2): the features' primary hit, which Metal rasterizes through a
// pinhole and each pixel's center (its own rule, metal/frame/
// raster_inputs.h), is the camera ray's traced one: its depth the traced
// hit's, but at the edges of triangles, where the two may round to
// neighbors. Metal's alone: the CUDA backend traces its features as the
// light passes trace their rays (contract 17).

#include <doctest/doctest.h>
#include <frame_rig.h>
#include <traced_probe.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/camera/thin_lens.h"
#include "core/frame/schedule.h"
#include "core/scene/scene.h"
#include "gpu/kernels/trace_probe.h"
#include "gpu/support/rendering.h"
#include "support/scene_text.h"
#include "support/schedules.h"

namespace camera = serenity::camera;
namespace contracts = serenity::contracts;
namespace frame = serenity::frame;
namespace scene = serenity::scene;
namespace tests = serenity::tests;

TEST_CASE(
    "the features' primary hit is the camera ray's traced hit, but at "
    "triangles' edges") {
  // A still scene: the brass ball (a sphere's mesh), the ground 400 m
  // across (a box's, its corners far from the ball's points), two
  // fireflies; the features alone, through a pinhole, each pixel's ray
  // through its center (contract 17).
  const scene::SceneDescription description =
      scene::parse(tests::still_fireflies(), "still fireflies");
  constexpr frame::Extent drawn{96, 72};
  const frame::Schedule alone{{frame::PassKind::features}, std::nullopt};
  tests::FrameRig rig;
  const auto target = rig.target(drawn);
  const auto renderer = rig.renderer(alone, &description);
  rig.render(*target, *renderer,
             tests::frame_at(0, 0, 0.0, description.camera));
  const auto features = renderer->read_features();
  REQUIRE(features.depth.extent == drawn);

  // The same rays traced (kernels/trace_probe.metal): from the camera
  // through each pixel's center, every shape.
  const contracts::CameraData camera =
      camera::shader_form(description.camera, drawn);
  const auto vec = [](contracts::Float3 v) {
    return std::array<double, 3>{v.x, v.y, v.z};
  };
  const auto forward = vec(camera.forward);
  const auto right = vec(camera.right);
  const auto up = vec(camera.up);
  std::vector<tests::TraceRay> rays;
  std::vector<double> along;  // each ray's view depth a unit of t
  for (std::uint32_t y = 0; y < drawn.height; ++y) {
    for (std::uint32_t x = 0; x < drawn.width; ++x) {
      const double sx = 2.0 * (x + 0.5) / drawn.width - 1.0;
      const double sy = 1.0 - 2.0 * (y + 0.5) / drawn.height;
      std::array<double, 3> d{};
      for (std::size_t c = 0; c < 3; ++c) {
        d.at(c) = forward.at(c) + sx * right.at(c) + sy * up.at(c);
      }
      const double length = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      rays.push_back(
          tests::TraceRay{camera.origin,
                          contracts::Float3{static_cast<float>(d[0] / length),
                                            static_cast<float>(d[1] / length),
                                            static_cast<float>(d[2] / length)},
                          contracts::RayMask::everything, 1.0e30f});
      along.push_back(1.0 / length);  // dot(forward, d) is 1
    }
  }
  tests::TracedScene traced(description, {});
  const auto count = static_cast<std::uint32_t>(rays.size());
  const std::vector<tests::TraceHit> hits = tests::run_traced<tests::TraceHit>(
      traced, "trace_probe", rays.size(),
      {tests::Binding::of(rays), tests::Binding::of(count)});
  std::size_t off = 0;
  std::size_t surfaces = 0;
  for (std::size_t p = 0; p < hits.size(); ++p) {
    const double expected =
        hits[p].found != 0u ? static_cast<double>(hits[p].t) * along[p] : 0.0;
    const double found = features.depth.rgb[p * 3];
    surfaces += hits[p].found != 0u ? 1u : 0u;
    if (std::abs(found - expected) > 1e-4 * std::max(1.0, expected)) {
      ++off;
    }
  }
  INFO(off << " of " << hits.size() << " pixels off the traced depth");
  // Most of the image is surface, so the check sees the meshes.
  CHECK(surfaces > hits.size() / 2);
  CHECK(off <= hits.size() / 100);
}
