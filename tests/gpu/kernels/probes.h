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

#if !defined(__METAL_VERSION__)
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
