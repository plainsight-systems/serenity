#pragma once

// What the probe kernels (tests/gpu/kernels/*.metal) read and write, and the
// GPU tests that run them (tests/gpu/*_test.cpp) write and read: one
// definition for both sides of every record, fixed-width fields, counts and
// indices as integers, each size asserted on both sides and trivial
// copyability on the host (core/contracts/shared_layout.h gives the rules;
// SF.3, I.4). A probe kernel binds its output at buffer 0 and its inputs
// after it in the order its comment lists them (tests/gpu/support/
// probe_runner.h).

#include "core/contracts/shared_layout.h"

#include "core/contracts/float3.h"

namespace serenity {
namespace tests {

// Four floats, a shading language float4's sixteen bytes.
struct Float4 {
    float x;
    float y;
    float z;
    float w;
};

static_assert(sizeof(Float4) == 16, "Float4 must be the same 16 bytes on the host and in shaders");

// ---- bsdf_probe.metal -------------------------------------------------------

// Which wo a BSDF probe asks about, and how many draws (bsdf_samples) or
// cells a side (bsdf_density).
struct BsdfQuery {
    contracts::Float3 wo;
    uint32_t count;
};

static_assert(sizeof(BsdfQuery) == 16, "BsdfQuery must be the same 16 bytes on the host and in shaders");

// One draw of sample(), and what evaluate() and pdf() then say of its wi.
struct BsdfDraw {
    contracts::Float3 direction;  // the sample's wi
    float pdf;                    // the sample's pdf
    contracts::Float3 value;      // the sample's value
    uint32_t lobe;                // the sample's lobe bits
    contracts::Float3 evaluated;  // evaluate(wo, wi), asked afterwards
    float evaluated_pdf;          // pdf(wo, wi), asked afterwards
};

static_assert(sizeof(BsdfDraw) == 48, "BsdfDraw must be the same 48 bytes on the host and in shaders");

// evaluate() and pdf() at one cell of the density grid.
struct BsdfCell {
    contracts::Float3 evaluated;
    float pdf;
};

static_assert(sizeof(BsdfCell) == 16, "BsdfCell must be the same 16 bytes on the host and in shaders");

// ---- sampler_probe.metal ----------------------------------------------------

// A pixel and a frame whose path's numbers are drawn.
struct PathNumbersQuery {
    uint32_t x;
    uint32_t y;
    uint32_t frame;
    uint32_t padding;
};

static_assert(sizeof(PathNumbersQuery) == 16, "PathNumbersQuery must be the same 16 bytes on the host and in shaders");

// How many of a path's numbers a probe draws.
SERENITY_CONSTANT uint32_t path_numbers_drawn = 8u;

// The first path_numbers_drawn numbers of one path.
struct PathNumbersDraw {
    float u[path_numbers_drawn];
};

static_assert(sizeof(PathNumbersDraw) == 32, "PathNumbersDraw must be the same 32 bytes on the host and in shaders");

// ---- camera_probe.metal -----------------------------------------------------

// A point of the image, in pixels, and a point of the lens, in [0, 1)^2.
struct CameraQuery {
    float pixel_x;
    float pixel_y;
    float lens_u;
    float lens_v;
};

static_assert(sizeof(CameraQuery) == 16, "CameraQuery must be the same 16 bytes on the host and in shaders");

// The image the camera's rays are made for.
struct CameraImage {
    uint32_t width;
    uint32_t height;
};

static_assert(sizeof(CameraImage) == 8, "CameraImage must be the same 8 bytes on the host and in shaders");

// A camera ray.
struct CameraProbeRay {
    contracts::Float3 origin;
    contracts::Float3 direction;
};

static_assert(sizeof(CameraProbeRay) == 24, "CameraProbeRay must be the same 24 bytes on the host and in shaders");

// ---- shape_probe.metal ------------------------------------------------------

// A point on shape `shape`, where a ray arriving from +z meets it.
struct ShapeQuery {
    contracts::Float3 point;
    uint32_t shape;
};

static_assert(sizeof(ShapeQuery) == 16, "ShapeQuery must be the same 16 bytes on the host and in shaders");

// ---- texture_probe.metal ----------------------------------------------------

// A point a texture is evaluated at, and the noise's seed (noise_probe only).
struct TexturePoint {
    contracts::Float3 point;
    uint32_t seed;
};

static_assert(sizeof(TexturePoint) == 16, "TexturePoint must be the same 16 bytes on the host and in shaders");

// ---- trace_probe.metal ------------------------------------------------------

// A ray traced from `origin` along the unit `direction`.
struct TraceRay {
    contracts::Float3 origin;
    contracts::Float3 direction;
};

static_assert(sizeof(TraceRay) == 24, "TraceRay must be the same 24 bytes on the host and in shaders");

// What a ray hit: its distance and its shape, or found 0.
struct TraceHit {
    float t;
    uint32_t shape;
    uint32_t found;
};

static_assert(sizeof(TraceHit) == 12, "TraceHit must be the same 12 bytes on the host and in shaders");

// ---- light_probe.metal ------------------------------------------------------

// The point a light is seen from, and the side of the grid of samples drawn
// toward it (light_cone).
struct LightQuery {
    contracts::Float3 point;
    uint32_t grid;
};

static_assert(sizeof(LightQuery) == 16, "LightQuery must be the same 16 bytes on the host and in shaders");

// A direction drawn toward the light, and the distance along it to the
// light's surface.
struct LightDraw {
    contracts::Float3 direction;
    float distance;
};

static_assert(sizeof(LightDraw) == 16, "LightDraw must be the same 16 bytes on the host and in shaders");

// A draw of the emitter's sample(), and pdf() asked of the direction to the
// light's middle and of one well outside it (light_far).
struct LightFar {
    float sample_pdf;
    float middle_pdf;
    float aside_pdf;
    float sample_distance;
};

static_assert(sizeof(LightFar) == 16, "LightFar must be the same 16 bytes on the host and in shaders");

// One draw of the emitter's sample() from a point, in full (light_samples).
struct LightSampleDraw {
    contracts::Float3 direction;
    float pdf;
    contracts::Float3 radiance;
    float distance;
    float pdf_asked;  // pdf() asked of the drawn direction afterwards
    uint32_t padding[3];
};

static_assert(sizeof(LightSampleDraw) == 48, "LightSampleDraw must be the same 48 bytes on the host and in shaders");

// ---- scene_block_probe.metal ------------------------------------------------

// A 32-bit word of an element of one of the scene block's arrays: field i
// of gpu::SceneBlock, in the order it declares them.
struct BlockQuery {
    uint32_t field;
    uint32_t element;
    uint32_t word;
};

static_assert(sizeof(BlockQuery) == 12, "BlockQuery must be the same 12 bytes on the host and in shaders");

// ---- restir_probe.metal -----------------------------------------------------

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
static_assert(std::is_trivially_copyable_v<SoftmaxQuery>, "SoftmaxQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<SoftmaxDraw>, "SoftmaxDraw is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<BlockQuery>, "BlockQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<Float4>, "Float4 is copied to and from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<BsdfQuery>, "BsdfQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<BsdfDraw>, "BsdfDraw is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<BsdfCell>, "BsdfCell is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<PathNumbersQuery>, "PathNumbersQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<PathNumbersDraw>, "PathNumbersDraw is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<CameraQuery>, "CameraQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<CameraImage>, "CameraImage is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<CameraProbeRay>, "CameraProbeRay is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<ShapeQuery>, "ShapeQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<TexturePoint>, "TexturePoint is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<TraceRay>, "TraceRay is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<TraceHit>, "TraceHit is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<LightQuery>, "LightQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<LightDraw>, "LightDraw is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<LightFar>, "LightFar is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<LightSampleDraw>, "LightSampleDraw is copied from the GPU as bytes");
#endif

}  // namespace tests
}  // namespace serenity
