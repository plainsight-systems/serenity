// The camera's rays on the GPU (tests/gpu/kernels/camera_probe.metal),
// held to contracts/camera.h: through a lens, every ray through a point of
// the image, wherever on the lens it starts, meets the same point of the
// plane of focus, and starts on the lens; without one, the pinhole's ray,
// exactly.

#include <doctest/doctest.h>
#include <probe_runner.h>  // the backend's runner, by the include path (SF.12)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/animation/draw.h"
#include "core/camera/thin_lens.h"
#include "gpu/kernels/camera_probe.h"
#include "support/references.h"
#include "support/vector.h"

namespace animation = serenity::animation;
namespace camera = serenity::camera;
namespace contracts = serenity::contracts;
namespace frame = serenity::frame;
namespace tests = serenity::tests;
using tests::Binding;
using tests::CameraProbeRay;
using tests::CameraQuery;
using tests::Vec3;

namespace {

// The image the rays are made for.
constexpr frame::Extent image{1600, 900};

std::vector<CameraProbeRay> rays(const contracts::CameraData& camera,
                                 const std::vector<CameraQuery>& queries) {
  tests::ProbeRunner gpu;
  const auto count = static_cast<std::uint32_t>(queries.size());
  return gpu.run<CameraProbeRay>(
      "camera_probe", queries.size(),
      {Binding::of(queries), Binding::of(count), Binding::of(camera),
       Binding::of(tests::CameraImage{image.width, image.height})});
}

double uniform(std::uint64_t i, std::uint64_t which) {
  return animation::draw(0xCA3u, i, which);
}

// A point of the image and of the lens, drawn from i.
CameraQuery query(std::uint64_t pixel, std::uint64_t lens) {
  return {static_cast<float>(image.width * uniform(pixel, 0)),
          static_cast<float>(image.height * uniform(pixel, 1)),
          static_cast<float>(uniform(lens, 2)),
          static_cast<float>(uniform(lens, 3))};
}

contracts::Camera framed_camera(float radius) {
  contracts::Camera c{
      {0.0f, 0.93f, 1.45f}, {0.0f, 0.87f, 0.0f}, {0.0f, 1.0f, 0.0f}, 38.0f};
  c.lens_radius = radius;
  c.focus_distance = 1.4f;
  return c;
}

}  // namespace

TEST_CASE(
    "through a lens, every ray through an image point meets one point of the "
    "plane of focus") {
  constexpr float lens_radius = 0.012f;
  constexpr std::uint64_t points = 64;
  // lens points through each image point
  constexpr std::uint64_t per_point = 16;
  const contracts::CameraData data =
      camera::shader_form(framed_camera(lens_radius), image);
  std::vector<CameraQuery> queries;
  queries.reserve(points * per_point);
  for (std::uint64_t p = 0; p < points; ++p) {
    for (std::uint64_t l = 0; l < per_point; ++l) {
      queries.push_back(query(p, p * per_point + l));
    }
  }
  const std::vector<CameraProbeRay> traced = rays(data, queries);
  const Vec3 origin = tests::vec(data.origin);
  const Vec3 forward = tests::vec(data.forward);
  double widest = 0.0;
  double farthest_start = 0.0;
  double farthest_off_lens = 0.0;
  for (std::size_t p = 0; p < points; ++p) {
    Vec3 first{};
    for (std::size_t l = 0; l < per_point; ++l) {
      const CameraProbeRay& r = traced[p * per_point + l];
      const Vec3 start = tests::vec(r.origin);
      const Vec3 direction = tests::vec(r.direction);
      // Where it meets the plane square to forward, focus_distance ahead.
      const double along = dot(direction, forward);
      REQUIRE(along > 0.0);
      const Vec3 offset = start - origin;
      const Vec3 hit =
          start +
          ((data.focus_distance - dot(offset, forward)) / along) * direction;
      if (l == 0) {
        first = hit;
      }
      widest = std::max(widest, tests::length(hit - first));
      // It starts on the lens: in the plane through the origin square
      // to forward, within the radius.
      farthest_off_lens =
          std::max(farthest_off_lens, std::abs(dot(offset, forward)));
      farthest_start = std::max(farthest_start, tests::length(offset));
    }
  }
  INFO("spread at the plane of focus " << widest << " m; farthest start "
                                       << farthest_start << " m");
  CHECK(widest < 1e-5);
  CHECK(farthest_off_lens < 1e-6);
  CHECK(farthest_start <= lens_radius * (1.0 + 1e-5));
  CHECK(farthest_start > 0.010);  // the lens is used, to its edge
}

TEST_CASE(
    "without a lens, the ray is the pinhole's, whatever lens point is given") {
  const contracts::CameraData data =
      camera::shader_form(framed_camera(0.0f), image);
  std::vector<CameraQuery> queries;
  queries.reserve(256);
  for (std::uint64_t i = 0; i < 256; ++i) {
    queries.push_back(query(i, i));
  }
  const std::vector<CameraProbeRay> traced = rays(data, queries);
  for (std::size_t i = 0; i < queries.size(); ++i) {
    const Vec3 d = tests::camera_direction(data, queries[i].pixel_x,
                                           queries[i].pixel_y, image);
    CHECK(traced[i].origin.x == data.origin.x);
    CHECK(traced[i].origin.y ==
          doctest::Approx(data.origin.y).scale(0).epsilon(1e-7));
    CHECK(traced[i].origin.z == data.origin.z);
    // A unit vector's components, each to 1e-5 of the vector's length.
    CHECK(traced[i].direction.x ==
          doctest::Approx(d.x).scale(1.0).epsilon(1e-5));
    CHECK(traced[i].direction.y ==
          doctest::Approx(d.y).scale(1.0).epsilon(1e-5));
    CHECK(traced[i].direction.z ==
          doctest::Approx(d.z).scale(1.0).epsilon(1e-5));
  }
}

namespace {

// Where the pinhole ray through `point` crosses the image, in pixels:
// contracts/camera.h's formula solved for (px, py), as the shaders'
// camera_pixel() solves it.
std::pair<double, double> image_point(const contracts::CameraData& data,
                                      Vec3 point) {
  const Vec3 v = point - tests::vec(data.origin);
  const Vec3 d = (1.0 / dot(v, tests::vec(data.forward))) * v;
  const Vec3 right = tests::vec(data.right);
  const Vec3 up = tests::vec(data.up);
  const double sx = dot(d, right) / dot(right, right);
  const double sy = dot(d, up) / dot(up, up);
  return {(sx + 1.0) * 0.5 * image.width, (1.0 - sy) * 0.5 * image.height};
}

}  // namespace

TEST_CASE(
    "the depth of field's circle of confusion is the blur the lens rays "
    "trace") {
  // A point at view depth z, through the lens: each lens point's ray
  // through it lands on the image where that ray meets the plane of
  // focus; the lens's edge draws the blur's circle, whose radius the
  // depth of field takes as circle_of_confusion_scale |z - F| / z
  // (core/camera/thin_lens.h).
  constexpr float lens_radius = 0.012f;
  const contracts::Camera camera = framed_camera(lens_radius);
  const contracts::CameraData data = camera::shader_form(camera, image);
  const double scale = camera::circle_of_confusion_scale(camera, image);
  const Vec3 origin = tests::vec(data.origin);
  const Vec3 forward = tests::vec(data.forward);
  // Lens points on its edge: concentric_disk maps the square's edge to the
  // disk's (metal/math/warp.metal.h).
  constexpr std::uint32_t edge = 16;
  for (const double depth : {0.5, 0.9, 2.5, 8.0}) {
    const Vec3 at = tests::camera_direction(data, 517.0f, 311.0f, image);
    const Vec3 point = origin + (depth / dot(at, forward)) * at;
    const auto [cx, cy] = image_point(data, point);
    // Each edge point's start on the lens, from the GPU's rays.
    std::vector<CameraQuery> starts;
    starts.reserve(edge);
    for (std::uint32_t i = 0; i < edge; ++i) {
      starts.push_back({517.0f, 311.0f, 1.0f,
                        static_cast<float>(i) / static_cast<float>(edge)});
    }
    const std::vector<CameraProbeRay> from = rays(data, starts);
    std::vector<CameraQuery> through;
    double widest = 0.0;
    for (std::uint32_t i = 0; i < edge; ++i) {
      const Vec3 start = tests::vec(from[i].origin);
      const Vec3 toward = point - start;
      const Vec3 focus =
          start + ((data.focus_distance - dot(start - origin, forward)) /
                   dot(toward, forward)) *
                      toward;
      const auto [x, y] = image_point(data, focus);
      widest = std::max(widest, std::hypot(x - cx, y - cy));
      through.push_back({static_cast<float>(x), static_cast<float>(y), 1.0f,
                         static_cast<float>(i) / static_cast<float>(edge)});
    }
    // The GPU's ray from that lens point through that image point passes
    // through the point: the blur is the traced lens's.
    const std::vector<CameraProbeRay> traced = rays(data, through);
    for (std::uint32_t i = 0; i < edge; ++i) {
      const Vec3 d = tests::vec(traced[i].direction);
      const Vec3 off = point - tests::vec(traced[i].origin);
      CHECK(tests::length(off - dot(off, d) * d) < 1e-5 * depth);
    }
    const double expected =
        scale * std::abs(depth - static_cast<double>(data.focus_distance)) /
        depth;
    INFO("depth " << depth << ": the lens's blur " << widest
                  << " pixels, the circle of confusion " << expected);
    CHECK(widest == doctest::Approx(expected).epsilon(1e-3));
  }
}
