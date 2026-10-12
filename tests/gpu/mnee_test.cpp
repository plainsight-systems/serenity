// Manifold next event estimation (metal/integrator/mnee.metal.h), on the
// GPU, against what it must give, worked out here:
//
//   - the solve: from the straight segment's two crossings of a glass
//     sphere resting on a floor, Newton's method converges to the path
//     through the sphere that refracts by Snell's law at both, found here in
//     double precision by a search in the plane the path must lie in (the
//     sphere's center and both ends); and the generalized geometry term it
//     gives is the solid angle at the receiver per area at the light, found
//     here by moving the light's point and solving again;
//   - the estimator: the path tracer with MNEE, and ReSTIR PT with it,
//     converge to the path tracer's mean without it on a caustic, a glass
//     sphere on a diffuse floor under a small light, to within MNEE's bias,
//     measured here and stated; and the caustic is counted once, so a build
//     that counted it both ways (MNEE's and the bounce's) fails.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
#include "gpu/kernels/mnee_probe.h"
#include "gpu/support/probe_runner.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"
#include "support/scene_text.h"
#include "support/vector.h"

using namespace serenity;
using tests::Binding;
using tests::Vec3;

namespace {

Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// A glass sphere resting on the floor y = 0, the scene's and the probe's.
constexpr double sphere_radius = 0.5;
constexpr Vec3 sphere_center{0.0, 0.5, 0.0};
constexpr double ior = 1.5;
// A light point above and to one side, on a light whose center is 0.1 above
// it: its outward normal there points down.
constexpr Vec3 light_point{0.6, 3.0, -0.4};
constexpr Vec3 light_normal{0.0, -1.0, 0.0};

// The nearer root along unit `d` from `o` of the sphere, past `least`, if
// any.
std::optional<double> root(Vec3 o, Vec3 d, double least) {
    const Vec3 oc = o - sphere_center;
    const double b = dot(oc, d);
    const double disc = b * b - (dot(oc, oc) - sphere_radius * sphere_radius);
    if (disc < 0.0) {
        return std::nullopt;
    }
    const double s = std::sqrt(disc);
    if (-b - s > least) {
        return -b - s;
    }
    if (-b + s > least) {
        return -b + s;
    }
    return std::nullopt;
}

// Unit `d` refracted at a surface of unit normal `n` facing it, from index
// 1 into `eta` times it; none under total internal reflection.
std::optional<Vec3> refract(Vec3 d, Vec3 n, double eta) {
    const double cos_i = -dot(d, n);
    const double k = 1.0 - (1.0 / (eta * eta)) * (1.0 - cos_i * cos_i);
    if (k < 0.0) {
        return std::nullopt;
    }
    return (1.0 / eta) * d + (cos_i / eta - std::sqrt(k)) * n;
}

// The path from `x0` through the sphere to `y`, in double: the direction
// leaving x0 at angle `a` from the straight one, in the plane of x0, y and
// the center, traced through both crossings; and how far its exit misses
// `y`, signed, in that plane.
struct Traced {
    bool through;
    Vec3 first;
    Vec3 second;
    double miss;
};

Traced trace_at(Vec3 x0, Vec3 y, double a) {
    const Vec3 u = normalized(y - x0);
    const Vec3 normal = normalized(cross(u, sphere_center - x0));
    const Vec3 v = cross(normal, u);
    const Vec3 d = std::cos(a) * u + std::sin(a) * v;
    const std::optional<double> t1 = root(x0, d, 1e-9);
    if (!t1) {
        return {false, {}, {}, 0.0};
    }
    const Vec3 p1 = x0 + *t1 * d;
    const std::optional<Vec3> inside = refract(d, normalized(p1 - sphere_center), ior);
    if (!inside) {
        return {false, {}, {}, 0.0};
    }
    const std::optional<double> t2 = root(p1, *inside, 1e-9);
    if (!t2) {
        return {false, {}, {}, 0.0};
    }
    const Vec3 p2 = p1 + *t2 * *inside;
    const std::optional<Vec3> out = refract(*inside, -normalized(p2 - sphere_center), 1.0 / ior);
    if (!out) {
        return {false, {}, {}, 0.0};
    }
    const Vec3 toward = normalized(y - p2);
    return {true, p1, p2, std::atan2(dot(cross(*out, toward), normal), dot(*out, toward))};
}

// The refraction path from x0 to y nearest the straight one: the sign
// change of the miss nearest a = 0, then bisection.
std::optional<Traced> refraction_path(Vec3 x0, Vec3 y) {
    constexpr double step = 1e-4;
    for (int k = 0; k < 20000; ++k) {
        for (const double side : {1.0, -1.0}) {
            const double a = side * step * k;
            const double b = side * step * (k + 1);
            Traced ta = trace_at(x0, y, a);
            const Traced tb = trace_at(x0, y, b);
            if (!ta.through || !tb.through || (ta.miss > 0.0) == (tb.miss > 0.0) || std::abs(ta.miss - tb.miss) > 1.0) {
                continue;
            }
            double lo = a;
            double hi = b;
            for (int i = 0; i < 200; ++i) {
                const double mid = 0.5 * (lo + hi);
                const Traced tm = trace_at(x0, y, mid);
                if ((tm.miss > 0.0) == (ta.miss > 0.0)) {
                    lo = mid;
                    ta = tm;
                } else {
                    hi = mid;
                }
            }
            return trace_at(x0, y, 0.5 * (lo + hi));
        }
    }
    return std::nullopt;
}

contracts::Float3 f3(Vec3 v) {
    return {static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)};
}

// Floor points in the sphere's shadow of the light point, whose straight
// segment to it crosses the sphere: four in the caustic's bright core, one
// at its rim, where G is a hundredth of the core's. Over a grid of the
// shadow, the solve from the straight seed converges at the core and at
// some of the rim; where it does not, MNEE finds no light, part of its
// bias, which the mean tests below measure.
const std::array<Vec3, 5> receivers = {Vec3{-0.12, 0.0, 0.08}, Vec3{-0.12, 0.0, -0.02}, Vec3{-0.02, 0.0, 0.08},
                                       Vec3{-0.22, 0.0, 0.08}, Vec3{-0.52, 0.0, 0.28}};

}  // namespace

TEST_CASE("MNEE's solve: through a glass sphere on a floor, the refraction path and its geometry term") {
    std::vector<tests::MneeQuery> queries;
    for (const Vec3& x0 : receivers) {
        queries.push_back(tests::MneeQuery{f3(x0), static_cast<float>(ior), f3(light_point),
                                           static_cast<float>(sphere_radius), f3(sphere_center), 0u,
                                           f3(light_normal), 0u, {0.0f, 1.0f, 0.0f}, 0u});
    }
    tests::ProbeRunner gpu;
    const std::vector<tests::MneeChain> chains =
        gpu.run<tests::MneeChain>("mnee_solve", queries.size(), {Binding::of(queries)});
    REQUIRE(chains.size() == receivers.size());
    for (std::size_t i = 0; i < receivers.size(); ++i) {
        const Vec3 x0 = receivers[i];
        const std::optional<Traced> path = refraction_path(x0, light_point);
        REQUIRE(path);
        INFO("receiver " << i << ": (" << x0.x << ", " << x0.z << ")");
        REQUIRE(chains[i].found == 1u);
        // The solve stops where every constraint, a sine, is under 10^-3:
        // its points are that close in angle, a few 10^-4 of the radius.
        const double off = std::max(length(tests::vec(chains[i].first) - path->first),
                                    length(tests::vec(chains[i].second) - path->second));
        CHECK(off < 2e-3 * sphere_radius);

        // G: the solid angle at x0 per area at the light, from the
        // directions toward the first crossing as the light's point moves
        // by h along two tangents of its surface.
        constexpr double h = 1e-4;
        const Vec3 w = normalized(path->first - x0);
        const std::optional<Traced> along_x = refraction_path(x0, light_point + h * Vec3{1.0, 0.0, 0.0});
        const std::optional<Traced> along_z = refraction_path(x0, light_point + h * Vec3{0.0, 0.0, 1.0});
        REQUIRE(along_x);
        REQUIRE(along_z);
        const Vec3 wx = normalized(along_x->first - x0) - w;
        const Vec3 wz = normalized(along_z->first - x0) - w;
        const double expected = length(cross(wx, wz)) / (h * h);
        INFO("G expected " << expected << ", found " << chains[i].geometry);
        CHECK(static_cast<double>(chains[i].geometry) == doctest::Approx(expected).epsilon(0.02));
    }
}

namespace {

// The caustic: the sphere above, its floor 0.8 diffuse, a light of radius
// 0.1 whose lowest point is light_point, under a black sky, seen from above
// and in front, the shadow and its caustic in the middle of the image.
std::string caustic_scene() {
    return tests::camera_text("[0, 2.4, 1.6]", "[-0.1, 0.1, 0.1]", 40) + tests::sky("[0, 0, 0]", "[0, 0, 0]") +
           "[materials.floor]\nkind = \"rough\"\ncolor = [0.8, 0.8, 0.8]\n"
           "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
           "[materials.glow]\nkind = \"emissive\"\nradiance = [50, 50, 50]\n"
           "[[shapes]]\nkind = \"box\"\nmin = [-50, -1, -50]\nmax = [50, 0, 50]\nmaterial = \"floor\"\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0.5, 0]\nradius = 0.5\nmaterial = \"glass\"\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0.6, 3.3, -0.4]\nradius = 0.3\nmaterial = \"glow\"\n";
}

constexpr frame::Extent caustic_size{96, 96};

// The accumulated image of `frames` frames of `graph` over the caustic.
std::vector<float> render_caustic(const std::string& graph, std::uint64_t frames) {
    const scene::SceneDescription description = scene::parse(caustic_scene(), "caustic");
    metal::Device device;
    metal::Submission submission(device);
    metal::Offscreen target(device, submission, caustic_size);
    metal::Renderer renderer(device, submission, frame::parse_schedule(graph, "graph"), &description);
    std::uint64_t sequence = 0;
    for (std::uint64_t i = 0; i < frames; ++i) {
        sequence = metal::render_to_offscreen(submission, target, renderer,
                                              tests::frame_at(i, 0, 0.0, description.camera));
    }
    (void)submission.wait_until_complete(sequence);
    CHECK(renderer.non_finite_samples() == 0);
    std::vector<float> rgba(std::size_t{caustic_size.width} * caustic_size.height * 4);
    renderer.read_accumulated(rgba);
    return rgba;
}

// The red channel's mean over the square of half-width `half` about
// (x, y).
double mean_red(const std::vector<float>& rgba, std::uint32_t x, std::uint32_t y, std::uint32_t half) {
    double sum = 0.0;
    std::uint32_t n = 0;
    for (std::uint32_t j = y - half; j <= y + half; ++j) {
        for (std::uint32_t i = x - half; i <= x + half; ++i) {
            sum += rgba[(std::size_t{j} * caustic_size.width + i) * 4];
            ++n;
        }
    }
    return sum / n;
}

constexpr const char* path_graph = "passes = [\"path\", \"display\"]\n";
constexpr const char* path_mnee_graph = "passes = [\"path\", \"display\"]\nmnee = true\n";
constexpr const char* restir_pt_mnee_graph =
    "passes = [\"restir_pt_initial\", \"restir_pt_temporal\", \"restir_pt_spatial\", \"restir_pt_shade\", "
    "\"restir_pt_duplication\", \"display\"]\nmnee = true\n"
    "[restir_pt]\nnee_candidates = 32\nmax_surfaces = 9\nfootprint = 0.02\nroughness = 0.2\nneighbors = 3\n"
    "sigma = 16.0\nconfidence_cap = 20.0\ncap_floor = 1.0\ncap_exponent = 0.1\n";

// Where the caustic lies in the image (the sphere's shadow on the floor,
// and the floor seen through the sphere above it) and the whole image,
// each with the ratio to the path tracer's mean MNEE must give: (x, y,
// half-width, least ratio, most ratio).
//
// MNEE's bias here, measured against the path tracer at 16384 frames, the
// path tracer with MNEE at 8192 (docs/research/2026-10-12-mnee.md): over
// the shadow, -6.5% (-13% over its middle third); over the whole image,
// -0.25%. It is the light MNEE does not find: where Newton's method from
// the straight seed does not reach the refraction path (most of the
// caustic's rim, the solve's test above), and light the sphere bends
// outside the straight segment's shadow; the path tracer's bounces count
// none of it, so none comes back. The bounds hold that bias and the noise
// of a caustic found by rare bounces and of MNEE's own near its folds,
// some 5% over the shadow; counting the caustic both ways, MNEE's and the
// bounce's, puts the shadow 14% (path tracer) and 21% (ReSTIR PT) over.
struct Region {
    const char* name;
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t half;
    double least;
    double most;
};
constexpr std::array<Region, 2> regions = {Region{"shadow", 47, 60, 14, 0.85, 1.03},
                                           Region{"whole", 48, 48, 47, 0.99, 1.005}};

void check_caustic(const char* graph, std::uint64_t frames, const std::vector<float>& reference) {
    const std::vector<float> image = render_caustic(graph, frames);
    for (const Region& r : regions) {
        const double ratio = mean_red(image, r.x, r.y, r.half) / mean_red(reference, r.x, r.y, r.half);
        INFO(std::string(r.name) << ": with MNEE over the path tracer " << ratio);
        CHECK(ratio >= r.least);
        CHECK(ratio <= r.most);
    }
}

}  // namespace

TEST_CASE("MNEE: the path tracer with it converges to the caustic the path tracer finds, counting it once") {
    const std::vector<float> reference = render_caustic(path_graph, 8192);
    check_caustic(path_mnee_graph, 2048, reference);
}

TEST_CASE("MNEE: ReSTIR PT with it converges to the caustic the path tracer finds, counting it once") {
    const std::vector<float> reference = render_caustic(path_graph, 8192);
    check_caustic(restir_pt_mnee_graph, 1024, reference);
}
