// A ray leaving a surface (metal/integrator/surface.metal.h, leave()) does
// not hit that surface again, on the GPU, through the tracing loop and the
// shapes' own tests (kernels/leave_probe.metal): the surface point found as
// the path tracer finds it, a ray's origin + t d put back on its shape
// (shapes/shapes.metal.h), and rays leaving it on both sides of its
// geometric normal, head on to grazing. Spheres and boxes, the scenes'
// shapes, at the scales and places the scenes put them and past them: a
// firefly 3 mm across resting on the marbles' table, a marble, a sphere and
// a box a meter across at the origin and 10 km off it, the table's top,
// the ground 400 m across, each hit from near and from hundreds of meters.
//
// Every shape here is convex, so a ray leaving it outward hits it nowhere,
// and a ray leaving it inward hits it once, on its far side: a self-hit is
// an outward ray that hits the shape at all, or an inward ray that hits it
// nearer than half the way across (for a sphere, half its chord 2 R |cos|;
// for a box, a hundredth of its least half size, every query's point being
// in the middle half of its face). An inward ray that hits no shape, or
// another, started outside the shape: a leak, the offset too large for it.
//
// Its dispatch is support/traced_probe.h's, as acceleration_test.cpp's: it
// binds the structures' block, which the probe runner does not.

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "core/animation/draw.h"
#include "core/scene/scene.h"
#include "gpu/kernels/leave_probe.h"
#include "gpu/support/traced_probe.h"
#include "metal/acceleration/scene_acceleration.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "metal/scene/scene_buffers.h"
#include "metal/scene/shape_transforms.h"
#include "serenity/metallib/probes.h"
#include "support/scene_text.h"
#include "support/vector.h"

namespace animation = serenity::animation;
namespace metal = serenity::metal;
namespace metallib = serenity::metallib;
namespace scene = serenity::scene;
namespace tests = serenity::tests;
using tests::LeaveQuery;
using tests::LeaveResult;
using tests::Vec3;

namespace {

// Runs leave_probe over `queries` in `description`'s one still scene
// (support/traced_probe.h).
std::vector<LeaveResult> leave(const scene::SceneDescription& description,
                               const std::vector<LeaveQuery>& queries) {
  metal::Device device;
  metal::Submission submission(device);
  const metal::SceneBuffers buffers(device, submission, description);
  metal::ShapeTransforms transforms(device, submission,
                                    description.shapes.transforms,
                                    metal::FrameArray::Copies::one);
  const std::vector<std::uint32_t> lights = scene::light_shapes(description);
  metal::SceneAcceleration acceleration(device, submission, description.shapes,
                                        {.moving = {}, .lights = lights});
  const metal::Library library(device, metallib::probes);
  const auto count = static_cast<std::uint32_t>(queries.size());
  return tests::run_traced<LeaveResult>(
      {device, submission, library, buffers, transforms, acceleration},
      "leave_probe", std::size_t{count} * tests::leave_directions,
      {tests::Binding::of(queries), tests::Binding::of(count)});
}

// Numbers for placing the queries, the n-th of a seed drawn as the
// animations draw theirs (core/animation/draw.h), so every run asks the
// same rays (principle 2).
class Numbers {
 public:
  explicit Numbers(std::uint64_t seed) : seed_(seed) {}

  double unit() { return animation::draw(seed_, next_++); }

  // A unit vector, uniform over the sphere.
  Vec3 direction() {
    const double z = 2.0 * unit() - 1.0;
    const double phi = 2.0 * std::numbers::pi * unit();
    const double r = std::sqrt(1.0 - z * z);
    return {r * std::cos(phi), r * std::sin(phi), z};
  }

  // A unit vector at least 0.2 in cosine from the surface's tangent plane
  // on `n`'s side: where a ray to the surface comes from.
  Vec3 above(Vec3 n) {
    for (;;) {
      const Vec3 v = direction();
      const double c = dot(v, n);
      if (std::abs(c) >= 0.2) {
        return c > 0.0 ? v : -v;
      }
    }
  }

 private:
  std::uint64_t seed_;
  std::uint64_t next_ = 0;
};

LeaveQuery query(Vec3 origin, Vec3 toward, std::uint32_t shape) {
  const Vec3 d = tests::normalized(toward - origin);
  return LeaveQuery{tests::float3(origin), shape, tests::float3(d), 0u};
}

// One shape, alone in a scene, and the distances its queries come from.
struct Case {
  std::string name;
  bool sphere;
  Vec3 center;  // a sphere's
  double radius;
  Vec3 min;  // a box's
  Vec3 max;
  std::vector<double> distances;
};

using tests::exact;

std::string triple(Vec3 v) {
  return "[" + exact(v.x) + ", " + exact(v.y) + ", " + exact(v.z) + "]";
}

std::string scene_of(const Case& c) {
  std::string text = R"([camera]
position = [0, 0, 10]
look_at = [0, 0, 0]
vertical_fov_degrees = 40
[environment]
kind = "gradient"
zenith = [0, 0, 0]
horizon = [0, 0, 0]
[materials.glass]
kind = "dielectric"
ior = 1.5
[[shapes]]
)";
  if (c.sphere) {
    text += "kind = \"sphere\"\ncenter = " + triple(c.center) +
            "\nradius = " + exact(c.radius) + "\n";
  } else {
    text += "kind = \"box\"\nmin = " + triple(c.min) +
            "\nmax = " + triple(c.max) + "\n";
  }
  return text + "material = \"glass\"\n";
}

// The queries for `c`: per distance, `per_distance` rays from that far off
// the surface, each aimed at a point on it from outside, 0.2 or more in
// cosine from its tangent plane. A box's points are in the middle half of
// a face, each face in turn.
std::vector<LeaveQuery> queries_for(const Case& c, std::uint32_t per_distance) {
  Numbers numbers(0x5eed0ff5e7ull);
  std::vector<LeaveQuery> queries;
  for (const double distance : c.distances) {
    for (std::uint32_t k = 0; k < per_distance; ++k) {
      if (c.sphere) {
        const Vec3 u = numbers.direction();
        const Vec3 target = c.center + c.radius * u;
        queries.push_back(
            query(target + distance * numbers.above(u), target, 0u));
      } else {
        const int axis = static_cast<int>(k % 3u);
        const double side = (k / 3u) % 2u == 0u ? 1.0 : -1.0;
        const auto along = [](Vec3 v, int a) {
          return a == 0 ? v.x : (a == 1 ? v.y : v.z);
        };
        std::array<double, 3> coords{};
        for (int a = 0; a < 3; ++a) {
          const double lo = along(c.min, a);
          const double hi = along(c.max, a);
          coords.at(static_cast<std::size_t>(a)) =
              a == axis ? (side > 0.0 ? hi : lo)
                        : lo + (hi - lo) * (0.25 + 0.5 * numbers.unit());
        }
        const Vec3 target{coords[0], coords[1], coords[2]};
        const Vec3 n{axis == 0 ? side : 0.0, axis == 1 ? side : 0.0,
                     axis == 2 ? side : 0.0};
        queries.push_back(
            query(target + distance * numbers.above(n), target, 0u));
      }
    }
  }
  return queries;
}

// What the rays leaving `c`'s surface did.
struct Tally {
  std::size_t found = 0;     // queries that found the shape, nearest
  std::size_t outward = 0;   // rays leaving outward
  std::size_t inward = 0;    // rays leaving inward
  std::size_t self_out = 0;  // outward rays that hit the shape again
  std::size_t self_in =
      0;  // inward rays that hit it nearer than half way across
  std::size_t leaked =
      0;  // inward rays that did not hit it: they started outside it

  std::size_t failures() const { return self_out + self_in + leaked; }
};

Tally tally(const Case& c, const std::vector<LeaveResult>& results) {
  double least_half = 0.0;
  if (!c.sphere) {
    least_half = 0.5 * std::min({c.max.x - c.min.x, c.max.y - c.min.y,
                                 c.max.z - c.min.z});
  }
  Tally t;
  for (std::size_t i = 0; i < results.size(); ++i) {
    const LeaveResult& r = results[i];
    if (i % tests::leave_directions == 0u) {
      t.found += r.found;
    }
    if (r.found == 0u) {
      continue;
    }
    const bool again = r.hit != 0u && r.hit_shape == 0u;
    if (r.outward != 0u) {
      ++t.outward;
      t.self_out += again ? 1u : 0u;
    } else {
      ++t.inward;
      const double half_way =
          c.sphere ? c.radius * std::abs(static_cast<double>(r.cosine))
                   : 0.01 * least_half;
      if (!again) {
        ++t.leaked;
      } else if (static_cast<double>(r.t) < half_way) {
        ++t.self_in;
      }
    }
  }
  return t;
}

Case sphere(std::string name, Vec3 center, double radius,
            std::vector<double> distances) {
  return Case{std::move(name),     true, center, radius, {}, {},
              std::move(distances)};
}

Case box(std::string name, Vec3 min, Vec3 max, std::vector<double> distances) {
  return Case{std::move(name), false, {}, 0.0, min, max, std::move(distances)};
}

Tally run(const Case& c) {
  constexpr std::uint32_t per_distance = 96;
  const scene::SceneDescription description = scene::parse(scene_of(c), c.name);
  const std::vector<LeaveQuery> queries = queries_for(c, per_distance);
  const Tally t = tally(c, leave(description, queries));
  MESSAGE(c.name << ": " << t.found << " of " << queries.size()
                 << " points found; " << t.outward << " rays out, "
                 << t.self_out << " hit it again; " << t.inward << " in, "
                 << t.self_in << " hit it short, " << t.leaked << " leaked");
  INFO(c.name);
  CHECK(t.found == queries.size());
  return t;
}

}  // namespace

TEST_CASE(
    "a ray leaving a surface does not hit it again, on either side, at the "
    "scenes' scales and past them") {
  const std::vector<Case> cases{
      // The marbles (scenes/marbles.toml): a firefly resting on the table,
      // hit from 1 cm, from the camera's 0.4 m and from 100 m; a marble.
      sphere("firefly, 3 mm across, on the table", {0.03, 0.7515, -0.1}, 0.0015,
             {0.01, 0.4, 100.0}),
      sphere("marble, 16 mm across", {-0.0204, 0.758, -0.167}, 0.008,
             {0.4, 200.0}),
      box("the table's top", {-1.4, 0.72, -1.6}, {1.4, 0.75, 1.0},
          {0.4, 100.0}),
      box("the ground, 400 m across", {-200.0, -0.1, -200.0},
          {200.0, 0.0, 200.0}, {1.0, 300.0}),
      // A meter across at the origin, where offset_ray's fixed step rules,
      // and 10 km off it, where its step in ulps is 25 cm.
      sphere("sphere 2 m across at the origin", {0.0, 0.0, 0.0}, 1.0,
             {3.0, 300.0}),
      box("box 1 m across at the origin", {-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5},
          {2.0, 500.0}),
      sphere("sphere 2 m across 10 km off", {1.0e4, -3.0e3, 7.0e3}, 1.0,
             {3.0, 1000.0}),
      box("box 1 m across 10 km off", {-1.0e4, 3.0e3, 7.0e3},
          {-1.0e4 + 1.0, 3.0e3 + 1.0, 7.0e3 + 1.0}, {2.0, 1000.0}),
      // Small, far from the origin: a centimeter 170 m off.
      sphere("sphere 2 cm across 170 m off", {100.0, 100.0, 100.0}, 0.01,
             {0.1, 300.0}),
      box("box 1 cm across 170 m off", {100.0, 100.0, 100.0},
          {100.01, 100.01, 100.01}, {0.1, 300.0}),
      // Large, its surface through the origin's band.
      sphere("sphere 20 m across, its top at the origin", {0.0, -10.0, 0.0},
             10.0, {1.0, 300.0}),
  };
  for (const Case& c : cases) {
    const Tally t = run(c);
    INFO(c.name);
    CHECK(t.outward > 0u);
    CHECK(t.inward > 0u);
    CHECK(t.failures() == 0u);
  }
  // And the check can fail, past what the offset promises: a sphere 2 mm
  // across 170 m off, where the step is 256 ulps of 100, 2 mm, its
  // diameter: rays leaving it inward start near its far side and hit it
  // short.
  const Tally small = run(sphere("sphere 2 mm across 170 m off",
                                 {100.0, 100.0, 100.0}, 0.001, {0.1, 300.0}));
  CHECK(small.self_in > 0u);
}
