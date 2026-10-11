#pragma once

// The records light_probe.metal and tests/gpu/light_test.cpp share (Light:
// the emitter's answers): one definition for both sides, fixed-width fields,
// counts and indices as integers, each size asserted on both sides and
// trivial copyability on the host (core/contracts/shared_layout.h gives the
// rules; SF.3, I.4). Included by that kernel and that test alone, so a change
// to one probe's records touches no other probe (SF.11;
// docs/architecture/change-axes.md). The kernel binds its output at buffer 0
// and its inputs after it in the order its comment lists them
// (tests/gpu/support/probe_runner.h).

#include "core/contracts/shared_layout.h"

#include "core/contracts/float3.h"

namespace serenity {
namespace tests {

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

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<LightQuery>, "LightQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<LightDraw>, "LightDraw is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<LightFar>, "LightFar is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<LightSampleDraw>, "LightSampleDraw is copied from the GPU as bytes");
#endif

}  // namespace tests
}  // namespace serenity
