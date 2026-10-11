#pragma once

// The records restir_probe.metal and tests/gpu/restir_test.cpp share (Sample
// reuse: ReSTIR DI's pieces): one definition for both sides, fixed-width
// fields, counts and indices as integers, each size asserted on both sides
// and trivial copyability on the host (core/contracts/shared_layout.h gives
// the rules; SF.3, I.4). Included by that kernel and that test alone, so a
// change to one probe's records touches no other probe (SF.11;
// docs/architecture/change-axes.md). The kernel binds its output at buffer 0
// and its inputs after it in the order its comment lists them
// (tests/gpu/support/probe_runner.h).

#include "core/contracts/shared_layout.h"

#include "core/contracts/float3.h"
#include "core/contracts/reservoir.h"
#include "core/contracts/reuse_settings.h"

namespace serenity {
namespace tests {

// A unit vector's equal-area octahedral code, and its decoding (unit_trip).
struct UnitTrip {
    contracts::Float3 decoded;
    uint32_t code;
};

static_assert(sizeof(UnitTrip) == 16, "UnitTrip must be the same 16 bytes on the host and in shaders");

// How unit_codes draws its directions: uniformly over the sphere, or by the
// cosine about +z (a distribution that is not uniform, to show the
// equal-area test can fail).
SERENITY_CONSTANT uint32_t draw_uniform = 0u;
SERENITY_CONSTANT uint32_t draw_cosine = 1u;

struct UnitDraws {
    uint32_t count;
    uint32_t distribution;  // draw_uniform or draw_cosine
};

static_assert(sizeof(UnitDraws) == 8, "UnitDraws must be the same 8 bytes on the host and in shaders");

// Where a light point is drawn from, and the u it is drawn with
// (light_points).
struct LightPointQuery {
    contracts::Float3 point;
    float u0;
    float u1;
    uint32_t padding[3];
};

static_assert(sizeof(LightPointQuery) == 32, "LightPointQuery must be the same 32 bytes on the host and in shaders");

// What contract 3's sample_light() draws from a point with u, and what
// contract 4's draw_light_point() then evaluate_light_point() give the
// same point for the same u (light_points).
struct LightPointResult {
    contracts::Float3 sample_direction;
    float sample_distance;
    contracts::Float3 sample_radiance;
    float sample_pdf;
    contracts::Float3 direction;  // evaluate_light_point()'s
    float distance;
    contracts::Float3 radiance;
    float cos_light;
    float area_scale;
    float pdf;           // light_pdf() asked of the evaluated direction
    float area_density;  // light_area_density(pdf, the evaluation)
    uint32_t light;      // the identity's light word
};

static_assert(sizeof(LightPointResult) == 80, "LightPointResult must be the same 80 bytes on the host and in shaders");

// A surface the target is evaluated at, and the identity it is asked of
// (target_at): the surface as a record unpacks it, and the light point.
struct TargetQuery {
    contracts::Float3 position;
    uint32_t light_point;  // contract 4's point word, of light record 0
    contracts::Float3 wo;
    float selection;       // the light's selection probability, for candidate_weight()
};

static_assert(sizeof(TargetQuery) == 32, "TargetQuery must be the same 32 bytes on the host and in shaders");

// target(), candidate_weight() at the cone's density, light_area_density()
// of that density, and direct_light() with W = 1 (target_at).
struct TargetResult {
    float target;
    float weight;
    float area_density;
    float cos_light;
    contracts::Float3 direct;
    float distance;
};

static_assert(sizeof(TargetResult) == 32, "TargetResult must be the same 32 bytes on the host and in shaders");

// Where a light candidate is drawn from, seen from where, with which
// numbers, and how many candidates of each strategy weigh it
// (candidate_mis).
struct MisQuery {
    contracts::Float3 position;
    float u0;
    contracts::Float3 wo;
    float u1;
    uint32_t light_candidates;  // n_l
    uint32_t bsdf_candidates;   // n_b
    uint32_t padding[2];
};

static_assert(sizeof(MisQuery) == 48, "MisQuery must be the same 48 bytes on the host and in shaders");

// A light point y drawn by the light strategy (draw_candidate()) and the
// same point as the BSDF strategy weighs it once a ray struck it
// (weigh_struck()), each with its weight, the luminance of f |cos| L_e
// toward it (the weight's numerator, target.metal.h), and the two
// strategies' densities toward it per unit solid angle, light_pdf() and
// bsdf_pdf() asked here, apart from the code under test (candidate_mis).
struct MisResult {
    float light_scattered;
    float light_weight;
    float light_density;       // p_l at the light candidate's direction
    float light_bsdf_density;  // p_b at the light candidate's direction
    float bsdf_scattered;
    float bsdf_weight;
    float bsdf_light_density;  // p_l at the BSDF candidate's direction
    float bsdf_density;        // p_b at the BSDF candidate's direction
};

static_assert(sizeof(MisResult) == 32, "MisResult must be the same 32 bytes on the host and in shaders");

// One temporal merge (temporal_merge): the pixel's own reservoir and last
// frame's, under `settings`, with a number for each choice. The merge asks
// its targets of a probe's Frame (restir_probe.metal, ProbeFrame), whose
// target for a light point is `scale` x (1 + its light's index), the same
// at every surface, both frames.
struct MergeQuery {
    contracts::Reservoir own;
    contracts::Reservoir last;
    contracts::ReuseSettings settings;
    uint32_t seed;
    float scale;
    uint32_t padding[2];
};

static_assert(sizeof(MergeQuery) == 112, "MergeQuery must be the same 112 bytes on the host and in shaders");

// What find_last_frame() answered (last_frame_found): whether a pixel
// passed, its offset from the reprojected pixel, and its index among last
// frame's 3 x 3.
struct FoundResult {
    uint32_t found;
    int32_t dx;
    int32_t dy;
    uint32_t index;
};

static_assert(sizeof(FoundResult) == 16, "FoundResult must be the same 16 bytes on the host and in shaders");

// The streams of softmax_streams: `streams` reservoirs, each fed the
// `count` scores, weight exp(scale x score), in an order `order` says,
// from numbers keyed by its index and `seed`. zero: every weight 0.
SERENITY_CONSTANT uint32_t order_forward = 0u;
SERENITY_CONSTANT uint32_t order_reversed = 1u;
SERENITY_CONSTANT uint32_t order_rotated = 2u;  // stream i starts at score i mod count
SERENITY_CONSTANT uint32_t max_scores = 16u;

struct SoftmaxQuery {
    uint32_t count;
    uint32_t streams;
    uint32_t order;
    uint32_t seed;
    float scale;
    uint32_t zero;  // 1: every weight 0, an empty stream
    uint32_t padding[2];
    float scores[max_scores];
};

static_assert(sizeof(SoftmaxQuery) == 96, "SoftmaxQuery must be the same 96 bytes on the host and in shaders");

// One stream's finished reservoir and the sum it finished with.
struct SoftmaxDraw {
    uint32_t light;   // the kept sample's light word: light_valid | the score's index; 0 for none
    uint32_t count;   // M
    float target;     // the kept candidate's target, exp(scale x score)
    float weight;     // W
    float weight_sum;
    uint32_t padding[3];
};

static_assert(sizeof(SoftmaxDraw) == 32, "SoftmaxDraw must be the same 32 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<UnitTrip>, "UnitTrip is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<UnitDraws>, "UnitDraws is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<LightPointQuery>, "LightPointQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<LightPointResult>, "LightPointResult is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<TargetQuery>, "TargetQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<TargetResult>, "TargetResult is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<MisQuery>, "MisQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<MisResult>, "MisResult is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<MergeQuery>, "MergeQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<FoundResult>, "FoundResult is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<SoftmaxQuery>, "SoftmaxQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<SoftmaxDraw>, "SoftmaxDraw is copied from the GPU as bytes");
#endif

}  // namespace tests
}  // namespace serenity
