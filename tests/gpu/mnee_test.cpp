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
//     that counted it both ways (MNEE's and the bounce's) fails. The path
//     tracer's reference is rendered once for both;
//   - a request the full work list has no room for is counted lost.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/bsdf.h"
#include "core/contracts/emitter.h"
#include "core/contracts/medium.h"
#include "core/contracts/mnee_request.h"
#include "core/scene/scene.h"
#include "core/shapes/primitive.h"
#include "gpu/kernels/mnee_probe.h"
#include "gpu/support/probe_runner.h"
#include "gpu/support/rendering.h"
#include "metal/frame/renderer.h"
#include "support/scene_text.h"
#include "support/schedules.h"
#include "support/vector.h"

using namespace serenity;
using tests::Binding;
using tests::Vec3;

namespace {

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
        queries.push_back(tests::MneeQuery{tests::float3(x0), static_cast<float>(ior), tests::float3(light_point),
                                           static_cast<float>(sphere_radius), tests::float3(sphere_center), 0u,
                                           tests::float3(light_normal), 0u, {0.0f, 1.0f, 0.0f}, 0u});
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

TEST_CASE("MNEE's chain: a bounce reads its shape's kind only on a chain at a refraction, never at no_primitive") {
    // A replayed shift's first surface is the destination's primary hit,
    // whose primitive is no_primitive (shift.metal.h, unpack_primary_hit());
    // reading its record there indexes 64 GiB past the records. Until
    // 2026-10-16 every replayed shift did, its kind unused
    // (docs/research/2026-10-16-codebase-guidelines-pass.md, C1). Neither
    // Metal's shader validation (tests/gpu/CMakeLists.txt) nor the GPU
    // stops on that read on this machine: through an argument table's
    // address it reads 0, unseen. So this checks chain_after()'s every
    // case, the no_primitive ones among them, under validation; that the
    // record is read only behind the refraction's test is the source's,
    // and the compiled kernels' (that note).
    namespace c = contracts;
    const std::vector<shapes::ShapeRecord> records = {
        shapes::ShapeRecord{shapes::ShapeKind::sphere, 0u, 0u, c::no_medium},
        shapes::ShapeRecord{shapes::ShapeKind::box, 0u, 0u, c::no_medium}};
    constexpr std::uint32_t refraction = c::lobe_transmission | c::lobe_delta;
    constexpr std::uint32_t reflection = c::lobe_reflection | c::lobe_delta;
    constexpr std::uint32_t off = 0u;
    constexpr std::uint32_t most = c::mnee_most_crossings;
    struct Case {
        const char* name;
        tests::ChainQuery query;
        std::uint32_t expected;
    };
    const std::array<Case, 8> cases = {
        Case{"a refraction at no_primitive, off the chain", {off, refraction, c::no_primitive, 0u}, off},
        Case{"a diffuse bounce at no_primitive", {off, c::lobe_diffuse, c::no_primitive, 0u}, 1u},
        Case{"a reflection at no_primitive, off the chain", {off, reflection, c::no_primitive, 0u}, off},
        Case{"a reflection on the chain", {1u, reflection, 0u, 0u}, off},
        Case{"a refraction at a sphere, on the chain", {1u, refraction, 0u, 0u}, 2u},
        Case{"a refraction at a box, on the chain", {1u, refraction, 1u, 0u}, off},
        Case{"the last refraction the chain holds", {most, refraction, 0u, 0u}, most + 1u},
        Case{"a refraction past the chain's most", {most + 1u, refraction, 0u, 0u}, off}};
    std::vector<tests::ChainQuery> queries;
    for (const Case& k : cases) {
        queries.push_back(k.query);
    }
    tests::ProbeRunner gpu;
    const std::vector<std::uint32_t> chains =
        gpu.run<std::uint32_t>("mnee_chain_after", queries.size(), {Binding::of(queries), Binding::of(records)});
    REQUIRE(chains.size() == cases.size());
    for (std::size_t i = 0; i < cases.size(); ++i) {
        INFO(cases[i].name);
        CHECK(chains[i] == cases[i].expected);
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
std::vector<float> render_caustic(const frame::Schedule& graph, std::uint64_t frames) {
    tests::AccumulatingRig rig(caustic_scene(), graph, caustic_size);
    return rig.render_accumulated(frames);
}

// The red channel's mean over the square of half-width `half` about
// (x, y).
double mean_red(const std::vector<float>& rgba, std::uint32_t x, std::uint32_t y, std::uint32_t half) {
    return tests::patch_mean(rgba, caustic_size, {x, y, half});
}

// The path tracer, with MNEE and without; ReSTIR PT as graphs/restir_pt.toml
// runs it, with MNEE, its settings the written ones (support/schedules.h).
frame::Schedule path_graph(bool mnee) {
    frame::Schedule graph = tests::path_graph();
    graph.mnee = mnee;
    return graph;
}
const frame::Schedule restir_pt_mnee_graph = tests::restir_pt_graph(tests::restir_pt_passes, true);

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

void check_caustic(const frame::Schedule& graph, std::uint64_t frames, const std::vector<float>& reference) {
    const std::vector<float> image = render_caustic(graph, frames);
    for (const Region& r : regions) {
        const double ratio = mean_red(image, r.x, r.y, r.half) / mean_red(reference, r.x, r.y, r.half);
        INFO(std::string(r.name) << ": with MNEE over the path tracer " << ratio);
        CHECK(ratio >= r.least);
        CHECK(ratio <= r.most);
    }
}

}  // namespace

namespace {

// A light inside a glass sphere, inside a closed white cavity, the camera
// between them: every light sample from the wall crosses the glass, so
// every bounce off the wall writes MNEE a request, and a path, bouncing
// among walls that keep 95% of the light, writes several. The work list
// holds one a pixel (contract 16), so it fills.
std::string light_in_glass_in_a_cavity() {
    return tests::camera_text("[0, 0, 1.5]", "[0, 0, 0]", 60) + tests::sky("[0, 0, 0]", "[0, 0, 0]") +
           "[materials.wall]\nkind = \"rough\"\ncolor = [0.95, 0.95, 0.95]\n"
           "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
           "[materials.glow]\nkind = \"emissive\"\nradiance = [5, 5, 5]\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 2\nmaterial = \"wall\"\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 0.6\nmaterial = \"glass\"\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 0.2\nmaterial = \"glow\"\n";
}

}  // namespace

TEST_CASE("MNEE: a request the full work list has no room for is counted lost, apart from samples not finite") {
    // Until 2026-10-16 a lost request made its path's sample a NaN, under
    // fast math, which need not keep one (docs/research/
    // 2026-10-16-codebase-guidelines-pass.md, C14). Now the MNEE pass's grid
    // kernel counts it; the path's sample keeps its other light, finite.
    for (const auto& [name, graph] : {std::pair{"the path tracer with MNEE", path_graph(true)},
                                      std::pair{"ReSTIR PT with MNEE", restir_pt_mnee_graph}}) {
        INFO(name);
        // Two frames; no sample left out for not being finite (the rig's
        // render() checks it).
        tests::AccumulatingRig rig(light_in_glass_in_a_cavity(), graph, {32, 32});
        const std::vector<float> rgba = rig.render_accumulated(2);
        MESSAGE("lost requests: " << rig.renderer().mnee_lost_requests());
        CHECK(rig.renderer().mnee_lost_requests() > 0);
        CHECK(std::ranges::all_of(rgba, [](float v) { return std::isfinite(v); }));
    }
}

TEST_CASE("MNEE: the path tracer and ReSTIR PT with it converge to the caustic the path tracer finds, counting "
          "it once") {
    // One reference for both, the costliest render here (P.9).
    const std::vector<float> reference = render_caustic(path_graph(false), 8192);
    {
        INFO("the path tracer with MNEE");
        check_caustic(path_graph(true), 2048, reference);
    }
    {
        INFO("ReSTIR PT with MNEE");
        check_caustic(restir_pt_mnee_graph, 1024, reference);
    }
}
