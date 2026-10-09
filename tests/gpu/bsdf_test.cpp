// Contract 2 (core/contracts/bsdf.h), its shader side run on the GPU
// (tests/gpu/kernels/bsdf_probe.metal), checked against what the contract
// says:
//
//   - consistency: pdf() and evaluate() asked of a sampled direction give
//     what sample() returned with it;
//   - sampling follows pdf(): a million samples, binned over the sphere,
//     fall into each bin as often as pdf() integrated over it predicts, and
//     pdf() integrates to the fraction of samples that are not refused;
//   - the weight value |cos| / pdf: exactly the albedo for Lambert, never
//     above 1 for a metal of f0 = 1, and its mean the directional albedo,
//     the integral of evaluate() times the cosine;
//   - glass: reflection with Fresnel's probability, weights 1 and 1 / eta^2,
//     total internal reflection past the critical angle, nothing for
//     evaluate() and pdf();
//   - resolve(): each material kind to its Bsdf, a texture read at the point.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <numbers>
#include <span>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/bsdf.h"
#include "core/contracts/surface_interaction.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "kernels/bsdf_probe.h"
#include "serenity/metallib/smoke.h"

using namespace serenity;
using contracts::Bsdf;
using contracts::BsdfKind;

namespace {

// Runs one kernel of the test library over `threads` threads, buffer i of
// `inputs` bound at i + 1 and the output at 0, and returns the output.
class Gpu {
public:
    template <typename Out>
    std::vector<Out> run(const char* kernel, std::size_t threads, std::initializer_list<std::span<const std::byte>> inputs) {
        auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        MTL::Device* mtl = device_.handle();
        auto pipeline = library_.compute_pipeline(kernel);
        std::vector<NS::SharedPtr<MTL::Buffer>> buffers;
        buffers.push_back(NS::TransferPtr(mtl->newBuffer(threads * sizeof(Out), MTL::ResourceStorageModeShared)));
        for (std::span<const std::byte> input : inputs) {
            auto buffer = NS::TransferPtr(mtl->newBuffer(std::max<std::size_t>(input.size(), 16),
                                                         MTL::ResourceStorageModeShared));
            std::memcpy(buffer->contents(), input.data(), input.size());
            buffers.push_back(buffer);
        }
        auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
        descriptor->setMaxBufferBindCount(buffers.size());
        NS::Error* error = nullptr;
        auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
        REQUIRE(table);
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            REQUIRE(buffers[i]);
            submission_.make_resident(buffers[i].get());
            table->setAddress(buffers[i]->gpuAddress(), i);
        }
        const auto frame = submission_.begin();
        MTL4::ComputeCommandEncoder* encoder = frame.commands->computeCommandEncoder();
        encoder->setArgumentTable(table.get());
        encoder->setComputePipelineState(pipeline.get());
        encoder->dispatchThreads(MTL::Size(threads, 1, 1), MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
        encoder->endEncoding();
        submission_.commit();
        (void)submission_.wait_until_complete(frame.sequence);
        std::vector<Out> out(threads);
        std::memcpy(out.data(), buffers[0]->contents(), threads * sizeof(Out));
        return out;
    }

private:
    metal::Device device_;
    metal::Submission submission_{device_};
    metal::Library library_{device_, metallib::smoke};
};

template <typename T>
std::span<const std::byte> bytes(const T& value) {
    return std::as_bytes(std::span(&value, 1));
}

template <typename T>
std::span<const std::byte> bytes(const std::vector<T>& values) {
    return std::as_bytes(std::span(values));
}

struct V3 {
    double x, y, z;
};
double dot(V3 a, V3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
V3 unit(double x, double y, double z) {
    const double l = std::sqrt(x * x + y * y + z * z);
    return {x / l, y / l, z / l};
}

using tests::Probe;

Bsdf make(BsdfKind kind, V3 normal, std::array<float, 3> color, float alpha, float ior) {
    Bsdf b{};
    b.normal = {float(normal.x), float(normal.y), float(normal.z)};
    b.kind = kind;
    b.color = {color[0], color[1], color[2]};
    b.alpha = alpha;
    b.ior = ior;
    return b;
}

constexpr std::uint32_t sample_count = 1u << 20;
constexpr std::uint32_t grid = 512;  // bsdf_density's cells: grid x grid

std::vector<Probe> samples(Gpu& gpu, const Bsdf& bsdf, V3 wo) {
    const float wo_count[4] = {float(wo.x), float(wo.y), float(wo.z), float(sample_count)};
    return gpu.run<Probe>("bsdf_samples", sample_count, {bytes(bsdf), bytes(wo_count)});
}

std::vector<std::array<float, 4>> density(Gpu& gpu, const Bsdf& bsdf, V3 wo) {
    const float wo_res[4] = {float(wo.x), float(wo.y), float(wo.z), float(grid)};
    return gpu.run<std::array<float, 4>>("bsdf_density", std::size_t{grid} * grid, {bytes(bsdf), bytes(wo_res)});
}

// The cell of bsdf_density's grid a direction falls in, coarsened by
// `coarse` in each direction.
std::size_t coarse_cell(contracts::Float3 d, std::uint32_t coarse) {
    const double z = std::clamp<double>(d.z, -1.0, 1.0);
    const double phi = std::atan2(d.y, d.x);
    const auto row = std::min<std::uint32_t>(grid - 1, std::uint32_t((1.0 - z) * 0.5 * grid));
    const auto column =
        std::min<std::uint32_t>(grid - 1, std::uint32_t((phi < 0 ? phi + 2 * std::numbers::pi : phi) /
                                                         (2 * std::numbers::pi) * grid));
    return std::size_t(row / coarse) * (grid / coarse) + column / coarse;
}

// The checks every non-delta kind passes: consistency, the pdf's integral,
// and the histogram of samples against it.
void check_sampling_follows_pdf(Gpu& gpu, const Bsdf& bsdf, V3 wo, std::uint32_t lobe) {
    const auto drawn = samples(gpu, bsdf, wo);
    const auto cells = density(gpu, bsdf, wo);
    const double cell_solid_angle = 4.0 * std::numbers::pi / (double(grid) * grid);

    std::uint32_t valid = 0;
    std::uint32_t inconsistent = 0;
    for (const Probe& p : drawn) {
        if (p.pdf <= 0.0f) {
            continue;
        }
        ++valid;
        CHECK(std::uint32_t(p.lobe) == lobe);
        const bool same_pdf = std::abs(p.evaluated_pdf - p.pdf) <= 1e-3f * p.pdf;
        const auto close = [](float a, float b) { return std::abs(a - b) <= 1e-3f * std::max(b, 1e-6f); };
        const bool same_value =
            close(p.evaluated.x, p.value.x) && close(p.evaluated.y, p.value.y) && close(p.evaluated.z, p.value.z);
        inconsistent += (same_pdf && same_value) ? 0u : 1u;
    }
    CHECK(inconsistent == 0);
    REQUIRE(valid > sample_count / 2);

    // The pdf integrates over the sphere to the fraction of samples not
    // refused.
    double integral = 0.0;
    for (const auto& cell : cells) {
        integral += cell[3] * cell_solid_angle;
    }
    INFO("pdf integrates to " << integral << ", valid fraction " << double(valid) / sample_count);
    CHECK(std::abs(integral - double(valid) / sample_count) < 0.01);

    // Samples per coarse bin against the pdf's integral over it.
    constexpr std::uint32_t coarse = 32;  // 16 x 16 bins
    std::vector<double> expected((grid / coarse) * (grid / coarse), 0.0);
    std::vector<double> counted(expected.size(), 0.0);
    for (std::uint32_t i = 0; i < grid * grid; ++i) {
        expected[std::size_t(i / grid / coarse) * (grid / coarse) + (i % grid) / coarse] +=
            cells[i][3] * cell_solid_angle * sample_count;
    }
    for (const Probe& p : drawn) {
        if (p.pdf > 0.0f) {
            counted[coarse_cell(p.direction, coarse)] += 1.0;
        }
    }
    // Pearson's chi-square over the bins the pdf expects at least 20 samples
    // in, the test pbrt and Mitsuba apply to their samplers. Sampling that
    // follows the pdf gives chi^2 / dof near 1, within a few sqrt(2 / dof);
    // a sampler that departs from its pdf by a few percent over a region
    // holding thousands of samples gives many times that.
    double chi2 = 0.0;
    std::uint32_t bins = 0;
    for (std::size_t b = 0; b < expected.size(); ++b) {
        if (expected[b] >= 20.0) {
            chi2 += (counted[b] - expected[b]) * (counted[b] - expected[b]) / expected[b];
            ++bins;
        }
    }
    const double dof = bins - 1.0;
    INFO("chi^2 / dof = " << chi2 / dof << " over " << bins << " bins");
    REQUIRE(bins > 10);
    CHECK(chi2 / dof < 1.0 + 5.0 * std::sqrt(2.0 / dof));
}

}  // namespace

TEST_CASE("Lambert: sampling follows the pdf, and every sample's weight is the albedo") {
    Gpu gpu;
    const V3 n = unit(0.3, 0.2, 0.9);
    const Bsdf bsdf = make(BsdfKind::lambert, n, {0.5f, 0.7f, 0.9f}, 0.0f, 0.0f);
    const V3 wo = unit(-0.4, 0.1, 0.8);
    check_sampling_follows_pdf(gpu, bsdf, wo, contracts::lobe_reflection | contracts::lobe_diffuse);

    std::uint32_t wrong_weight = 0;
    for (const Probe& p : samples(gpu, bsdf, wo)) {
        const double cos = std::abs(dot({p.direction.x, p.direction.y, p.direction.z}, n));
        const double weight = p.value.x * cos / p.pdf;
        wrong_weight += std::abs(weight - 0.5) < 1e-3 ? 0u : 1u;
    }
    CHECK(wrong_weight == 0);

    // The directional albedo, the integral of f |cos|, is the albedo.
    double albedo = 0.0;
    const auto cells = density(gpu, bsdf, wo);
    for (std::uint32_t i = 0; i < grid * grid; ++i) {
        const double z = 1.0 - 2.0 * ((i / grid) + 0.5) / grid;
        const double phi = 2 * std::numbers::pi * ((i % grid) + 0.5) / grid;
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        albedo += cells[i][0] * std::abs(dot({r * std::cos(phi), r * std::sin(phi), z}, n)) * 4.0 *
                  std::numbers::pi / (double(grid) * grid);
    }
    CHECK(albedo == doctest::Approx(0.5).epsilon(0.005));

    // wo on the far side of the normal: it reflects there instead. Checked
    // against the normal directly, not only against evaluate() and pdf(),
    // which would agree with sample() on the wrong side too.
    const V3 behind{-wo.x, -wo.y, -wo.z};
    check_sampling_follows_pdf(gpu, bsdf, behind, contracts::lobe_reflection | contracts::lobe_diffuse);
    std::uint32_t wrong_side = 0;
    std::uint32_t wrong_back_weight = 0;
    for (const Probe& p : samples(gpu, bsdf, behind)) {
        const double cos = dot({p.direction.x, p.direction.y, p.direction.z}, n);
        wrong_side += (p.pdf > 0.0f && cos < 0.0) ? 0u : 1u;
        wrong_back_weight += std::abs(p.value.x * -cos / p.pdf - 0.5) < 1e-3 ? 0u : 1u;
    }
    CHECK(wrong_side == 0);
    CHECK(wrong_back_weight == 0);
}

TEST_CASE("conductor: sampling follows the pdf, and a metal of f0 = 1 returns no more than it receives") {
    Gpu gpu;
    // The normal along x, so a sharp lobe lies where the density grid's
    // cells are finest, at its equator, not its poles.
    const V3 n = unit(1.0, 0.0, 0.0);
    for (float alpha : {0.25f, 0.06f}) {
        for (V3 wo : {unit(1.0, 0.1, 0.0), unit(0.6, 0.0, 0.8), unit(0.3, 0.1, 0.95)}) {
            INFO("alpha " << alpha << ", wo (" << wo.x << ", " << wo.y << ", " << wo.z << ")");
            const Bsdf bsdf = make(BsdfKind::conductor, n, {1.0f, 1.0f, 1.0f}, alpha, 0.0f);
            check_sampling_follows_pdf(gpu, bsdf, wo, contracts::lobe_reflection | contracts::lobe_glossy);

            double total = 0.0;
            double largest = 0.0;
            for (const Probe& p : samples(gpu, bsdf, wo)) {
                if (p.pdf > 0.0f) {
                    const double weight = p.value.x * std::abs(p.direction.x) / p.pdf;
                    total += weight;
                    largest = std::max(largest, weight);
                }
            }
            CHECK(largest <= 1.0 + 1e-3);
            // The mean weight over all draws, refused ones counting 0, is the
            // directional albedo.
            const auto cells = density(gpu, bsdf, wo);
            double albedo = 0.0;
            for (std::uint32_t i = 0; i < grid * grid; ++i) {
                const double z = 1.0 - 2.0 * ((i / grid) + 0.5) / grid;
                const double phi = 2 * std::numbers::pi * ((i % grid) + 0.5) / grid;
                const double x = std::sqrt(std::max(0.0, 1.0 - z * z)) * std::cos(phi);  // n . wi
                albedo += cells[i][0] * std::abs(x) * 4.0 * std::numbers::pi / (double(grid) * grid);
            }
            CHECK(total / sample_count == doctest::Approx(albedo).epsilon(0.01));
        }
    }
}

TEST_CASE("dielectric: Fresnel chooses reflection, the weights are 1 and 1 / eta^2, and past the critical angle all reflects") {
    Gpu gpu;
    const V3 n = unit(0.0, 0.0, 1.0);
    const Bsdf bsdf = make(BsdfKind::dielectric, n, {1.0f, 1.0f, 1.0f}, 0.0f, 1.5f);

    // Straight on from outside: F = ((1.5 - 1) / (1.5 + 1))^2 = 0.04.
    std::uint32_t reflected = 0;
    for (const Probe& p : samples(gpu, bsdf, n)) {
        REQUIRE(p.pdf > 0.0f);
        CHECK(p.evaluated.x == 0.0f);
        CHECK(p.evaluated_pdf == 0.0f);
        const double weight = p.value.x * std::abs(p.direction.z) / p.pdf;
        if (std::uint32_t(p.lobe) == (contracts::lobe_reflection | contracts::lobe_delta)) {
            ++reflected;
            CHECK(p.direction.z == doctest::Approx(1.0f));
            CHECK(weight == doctest::Approx(1.0));
        } else {
            CHECK(std::uint32_t(p.lobe) == (contracts::lobe_transmission | contracts::lobe_delta));
            CHECK(p.direction.z == doctest::Approx(-1.0f));  // straight through, into the glass
            CHECK(weight == doctest::Approx(1.0 / 2.25));
        }
    }
    CHECK(double(reflected) / sample_count == doctest::Approx(0.04).epsilon(0.03));

    // From inside, 60 degrees from the normal: past the critical angle,
    // asin(1 / 1.5) = 41.8 degrees, so every sample reflects, weight 1.
    const V3 inside = unit(std::sin(std::numbers::pi / 3), 0.0, -std::cos(std::numbers::pi / 3));
    for (const Probe& p : samples(gpu, bsdf, inside)) {
        CHECK(std::uint32_t(p.lobe) == (contracts::lobe_reflection | contracts::lobe_delta));
        CHECK(p.value.x * std::abs(p.direction.z) / p.pdf == doctest::Approx(1.0));
        CHECK(p.direction.z < 0.0f);  // stays inside
    }
}

TEST_CASE("lobes: each kind's, from the shader, and an estimator aims at lights only where one is not delta") {
    using namespace contracts;
    const V3 n = unit(0.0, 0.0, 1.0);
    const std::vector<Bsdf> bsdfs = {
        make(BsdfKind::none, n, {1, 1, 1}, 0.0f, 0.0f),
        make(BsdfKind::lambert, n, {0.5f, 0.5f, 0.5f}, 0.0f, 0.0f),
        make(BsdfKind::conductor, n, {1, 1, 1}, 0.25f, 0.0f),
        make(BsdfKind::dielectric, n, {1, 1, 1}, 0.0f, 1.5f),
    };
    const std::uint32_t count = std::uint32_t(bsdfs.size());
    Gpu gpu;
    const auto lobes = gpu.run<std::uint32_t>("bsdf_lobe_bits", count, {bytes(bsdfs), bytes(count)});
    CHECK(lobes[0] == 0u);
    CHECK(lobes[1] == (lobe_reflection | lobe_diffuse));
    CHECK(lobes[2] == (lobe_reflection | lobe_glossy));
    CHECK(lobes[3] == (lobe_reflection | lobe_transmission | lobe_delta));

    CHECK_FALSE(aims_at_lights(lobes[0]));
    CHECK(aims_at_lights(lobes[1]));
    CHECK(aims_at_lights(lobes[2]));
    CHECK_FALSE(aims_at_lights(lobes[3]));
    CHECK(aims_at_lights(lobe_reflection | lobe_delta | lobe_glossy));  // a mirror over a glossy coat
}

TEST_CASE("resolve: each material kind to its Bsdf, a texture read at the point") {
    const scene::SceneDescription scene = scene::parse(R"(
[camera]
position = [0, 1, 3]
look_at = [0, 0, 0]
vertical_fov_degrees = 40
[environment]
kind = "gradient"
zenith = [0, 0, 0]
horizon = [0, 0, 0]
[textures.checks]
kind = "checker"
size = 1
a = [0.9, 0.8, 0.7]
b = [0.1, 0.2, 0.3]
[materials.a_checks]
kind = "rough"
texture = "checks"
[materials.b_brass]
kind = "conductor"
f0 = [0.91, 0.78, 0.42]
roughness = 0.5
[materials.c_glass]
kind = "dielectric"
ior = 1.5
[materials.d_glow]
kind = "emissive"
radiance = [5, 5, 5]
[[shapes]]
kind = "sphere"
center = [0, 0, 0]
radius = 1
material = "a_checks"
)",
                                                       "resolve test");
    const auto at = [](float x, float z, std::uint32_t material) {
        contracts::SurfaceInteraction s{};
        s.position = {x, 0.0f, z};
        s.material = material;
        s.geometric_normal = {0.0f, 1.0f, 0.0f};
        s.shading_normal = {0.0f, 1.0f, 0.0f};
        s.flags = contracts::arrived_from_outside;
        return s;
    };
    // Materials in name order: a_checks 0, b_brass 1, c_glass 2, d_glow 3.
    const std::vector<contracts::SurfaceInteraction> surfaces = {at(0.5f, 0.5f, 0), at(1.5f, 0.5f, 0),
                                                                 at(0.0f, 0.0f, 1), at(0.0f, 0.0f, 2),
                                                                 at(0.0f, 0.0f, 3)};
    const std::uint32_t count = std::uint32_t(surfaces.size());
    Gpu gpu;
    const auto resolved = gpu.run<Bsdf>(
        "bsdf_resolve", count,
        {bytes(surfaces), bytes(scene.materials), bytes(scene.rough), bytes(scene.dielectrics),
         bytes(scene.conductors), bytes(scene.textures), bytes(scene.checkers), bytes(count)});

    CHECK(resolved[0].kind == BsdfKind::lambert);
    CHECK(resolved[0].color.x == doctest::Approx(0.9f));  // floor(0.5) + floor(0.5) = 0: even, a
    CHECK(resolved[1].color.x == doctest::Approx(0.1f));  // floor(1.5) + floor(0.5) = 1: odd, b
    CHECK(resolved[0].normal.y == 1.0f);
    CHECK(resolved[2].kind == BsdfKind::conductor);
    CHECK(resolved[2].color.z == doctest::Approx(0.42f));
    CHECK(resolved[2].alpha == doctest::Approx(0.25f));
    CHECK(resolved[3].kind == BsdfKind::dielectric);
    CHECK(resolved[3].ior == doctest::Approx(1.5f));
    CHECK(resolved[4].kind == BsdfKind::none);
}
