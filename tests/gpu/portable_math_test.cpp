// core/portable_math.h's contract, measured: the same source gives the same
// bits on this machine's CPU and GPU when the shader is built with the
// project's pinned flags (a NaN excepted, which the GPU returns as its one
// canonical NaN), and does not when it is built with Metal's fast
// math. The second half is what makes the first mean something: it shows the
// comparison can see a difference, and that the flags are what remove it.
//
// Inputs, one triple per GPU thread:
//   - special values: zeros, infinities, NaN, the smallest and largest
//     normals and subnormals, and the triple that separates fused from
//     unfused a * b + c (tests/portable_math_test.cpp);
//   - random bit patterns: every class of float, in proportion to its share
//     of the encodings;
//   - random normal floats with exponents in [-20, 20], where most values a
//     renderer computes lie.
// All drawn from std::mt19937's raw output, which the standard fixes
// exactly, so the inputs are the same on every platform; a standard
// distribution would not be.

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "kernels/portable_math_probe.h"
#include "metal/device.h"
#include "metal/library.h"
#include "serenity/metallib/portable_math_probe.h"
#include "serenity/metallib/portable_math_probe_fast.h"

namespace {

using serenity::probe::operation_count;

constexpr const char* operation_names[] = {"add", "sub", "mul", "div", "sqrt", "fma", "mad"};
static_assert(std::size(operation_names) == operation_count);

constexpr std::uint32_t random_triples = 1u << 20;

std::vector<float> make_inputs() {
    constexpr float inf = std::numeric_limits<float>::infinity();
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    constexpr float min_normal = std::numeric_limits<float>::min();
    constexpr float max_normal = std::numeric_limits<float>::max();
    constexpr float min_subnormal = std::numeric_limits<float>::denorm_min();
    const float max_subnormal = std::bit_cast<float>(0x007fffffu);

    std::vector<float> in = {
        0.0f, 0.0f, 0.0f,
        -0.0f, 0.0f, -0.0f,
        1.0f, -1.0f, 1.0f,
        inf, 1.0f, -inf,
        nan, 2.0f, 3.0f,
        min_normal, 0.5f, min_subnormal,
        max_normal, 2.0f, -max_normal,
        min_subnormal, 0.5f, max_subnormal,
        max_subnormal, 1.0f, min_subnormal,
        1.0f + 0x1p-12f, 1.0f + 0x1p-12f, -(1.0f + 0x1p-11f),
    };

    std::mt19937 bits(20261008u);
    for (std::uint32_t i = 0; i < random_triples / 2; ++i) {
        for (int k = 0; k < 3; ++k) {
            in.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(bits())));
        }
    }
    for (std::uint32_t i = 0; i < random_triples / 2; ++i) {
        for (int k = 0; k < 3; ++k) {
            const std::uint32_t r = static_cast<std::uint32_t>(bits());
            const std::uint32_t sign = r & 0x80000000u;
            const std::uint32_t exponent = 127u - 20u + (r >> 23) % 41u;  // 2^-20 .. 2^20
            const std::uint32_t mantissa = static_cast<std::uint32_t>(bits()) & 0x007fffffu;
            in.push_back(std::bit_cast<float>(sign | (exponent << 23) | mantissa));
        }
    }
    return in;
}

// Runs the probe kernel of `metallib` over `in`, one thread per triple.
std::vector<float> run_on_gpu(const serenity::metal::Device& device,
                              std::span<const unsigned char> metallib,
                              const std::vector<float>& in) {
    serenity::metal::Library library(device, metallib);
    auto pipeline = library.compute_pipeline("portable_math_probe");

    const std::uint32_t count = static_cast<std::uint32_t>(in.size() / 3);
    const std::size_t out_size = std::size_t{count} * operation_count;

    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    MTL::Device* mtl = device.handle();
    auto in_buffer = NS::TransferPtr(
        mtl->newBuffer(in.data(), in.size() * sizeof(float), MTL::ResourceStorageModeShared));
    auto out_buffer =
        NS::TransferPtr(mtl->newBuffer(out_size * sizeof(float), MTL::ResourceStorageModeShared));
    auto queue = NS::TransferPtr(mtl->newCommandQueue());
    REQUIRE(in_buffer);
    REQUIRE(out_buffer);
    REQUIRE(queue);

    MTL::CommandBuffer* commands = queue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commands->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline.get());
    encoder->setBuffer(in_buffer.get(), 0, 0);
    encoder->setBuffer(out_buffer.get(), 0, 1);
    encoder->setBytes(&count, sizeof(count), 2);
    const NS::UInteger width = pipeline->threadExecutionWidth();
    encoder->dispatchThreads(MTL::Size(count, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commands->commit();
    commands->waitUntilCompleted();

    if (commands->status() != MTL::CommandBufferStatusCompleted) {
        const NS::Error* error = commands->error();
        FAIL("the probe's command buffer did not complete: "
             << (error != nullptr && error->localizedDescription() != nullptr
                     ? error->localizedDescription()->utf8String()
                     : "no description"));
    }

    const float* out = static_cast<const float*>(out_buffer->contents());
    return std::vector<float>(out, out + out_size);
}

// The one NaN the GPU returns: positive, quiet, no payload
// (core/portable_math.h).
constexpr std::uint32_t gpu_nan = 0x7fc00000u;

// Equal under the contract: the same bits, or, where the CPU's result is a
// NaN, the GPU's canonical NaN. The CPU keeps the sign and payload of a NaN
// input and the GPU does not, so a NaN is compared by class and by the GPU's
// one encoding, never by the CPU's bits.
bool same(float cpu, float gpu) {
    if (std::isnan(cpu)) {
        return std::bit_cast<std::uint32_t>(gpu) == gpu_nan;
    }
    return std::bit_cast<std::uint32_t>(cpu) == std::bit_cast<std::uint32_t>(gpu);
}

// Whether a value is outside the contract: Apple's GPU flushes subnormal
// floats to zero, as inputs and as results, whatever the math mode
// (docs/research/2026-10-08-metal-math-modes.md), and the CPU does not.
bool subnormal(float x) {
    return std::fpclassify(x) == FP_SUBNORMAL;
}

// Whether the contract covers `op` on (a, b, c): no input, no rounded
// intermediate and no result is subnormal. The intermediates are each
// operation's, as written in kernels/portable_math_probe.h.
bool in_contract(int op, float a, float b, float c) {
    using namespace serenity::probe;
    if (subnormal(a) || subnormal(b) || subnormal(apply(op, a, b, c))) {
        return false;
    }
    switch (op) {
    case op_sqrt: return !subnormal(a * a) && !subnormal(b * b) && !subnormal(a * a + b * b);
    case op_fma: return !subnormal(c);
    case op_mad: return !subnormal(c) && !subnormal(a * b);
    default: return true;
    }
}

struct Mismatches {
    // Inside the contract, and outside it.
    std::uint32_t inside[operation_count] = {};
    std::uint32_t nan_results[operation_count] = {};  // inside, where the CPU gave a NaN
    std::uint32_t outside[operation_count] = {};
    std::uint32_t outside_total[operation_count] = {};
    std::string first[operation_count];
};

Mismatches compare(const std::vector<float>& in, const std::vector<float>& gpu) {
    Mismatches m;
    const std::size_t triples = in.size() / 3;
    for (std::size_t i = 0; i < triples; ++i) {
        const float a = in[3 * i];
        const float b = in[3 * i + 1];
        const float c = in[3 * i + 2];
        for (int op = 0; op < operation_count; ++op) {
            const float cpu = serenity::probe::apply(op, a, b, c);
            const float g = gpu[i * operation_count + op];
            const bool covered = in_contract(op, a, b, c);
            if (!covered) {
                ++m.outside_total[op];
            } else if (std::isnan(cpu)) {
                ++m.nan_results[op];
            }
            if (same(cpu, g)) {
                continue;
            }
            if (!covered) {
                ++m.outside[op];
                continue;
            }
            if (m.inside[op]++ == 0) {
                std::ostringstream s;
                s << std::hexfloat << "a=" << a << " b=" << b << " c=" << c << ": cpu " << cpu
                  << ", gpu " << g;
                m.first[op] = s.str();
            }
        }
    }
    return m;
}

}  // namespace

TEST_CASE("with the pinned flags, every probe operation gives the CPU's bits, NaN as the GPU's one NaN") {
    serenity::metal::Device device;
    const std::vector<float> in = make_inputs();
    const std::vector<float> gpu = run_on_gpu(device, serenity::metallib::portable_math_probe, in);
    const Mismatches m = compare(in, gpu);

    for (int op = 0; op < operation_count; ++op) {
        const std::string name = operation_names[op];
        MESSAGE(name << ": " << m.outside[op] << " of " << m.outside_total[op]
                     << " triples outside the contract differ");
        INFO(name << ": first mismatch " << m.first[op]);
        CHECK(m.inside[op] == 0);
        // The NaN rule was exercised, not vacuous: the inputs hold NaNs
        // with signs and payloads, and every operation propagates them.
        CHECK(m.nan_results[op] > 0);
    }

    // The GPU did not fuse a * b + c on its own: the separating triple
    // (index 9) gives the unfused 0, and fma() the fused 2^-24.
    CHECK(gpu[9 * operation_count + serenity::probe::op_mad] == 0.0f);
    CHECK(gpu[9 * operation_count + serenity::probe::op_fma] == 0x1p-24f);

    // The reason for the contract's domain, held as a fact about this GPU: a
    // subnormal result is flushed to zero (triple 5: the smallest normal
    // times 0.5). If this ever fails, the flush has gone and the domain can
    // widen.
    CHECK(gpu[5 * operation_count + serenity::probe::op_mul] == 0.0f);
}

TEST_CASE("with Metal's fast math, the same source does not give the CPU's bits") {
    serenity::metal::Device device;
    const std::vector<float> in = make_inputs();
    const std::vector<float> gpu =
        run_on_gpu(device, serenity::metallib::portable_math_probe_fast, in);
    const Mismatches m = compare(in, gpu);

    for (int op = 0; op < operation_count; ++op) {
        MESSAGE("fast math, " << std::string(operation_names[op]) << ": " << m.inside[op]
                              << " of " << in.size() / 3 - m.outside_total[op]
                              << " triples inside the contract differ");
    }
    // Inside the contract's domain, so the flush to zero explains none of
    // it: division and sqrt are approximated under fast math.
    CHECK(m.inside[serenity::probe::op_div] > 0);
    CHECK(m.inside[serenity::probe::op_sqrt] > 0);
}
