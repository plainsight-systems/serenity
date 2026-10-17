#pragma once

// The records encoding_probe.metal and tests/gpu/encoding_test.cpp share
// (the packed forms ReSTIR PT stores: unit vectors, BSDFs, light points):
// one definition for both sides, fixed-width fields, counts and indices as
// integers, each size asserted on both sides and trivial copyability on the
// host (core/contracts/shared_layout.h gives the rules; SF.3, I.4).
// Included by that kernel and that test alone, so a change to one probe's
// records touches no other probe (SF.11; docs/architecture/change-axes.md).
// The kernel binds its output at buffer 0 and its inputs after it in the
// order its comment lists them (tests/gpu/support/probe_runner.h).

#include "core/contracts/shared_layout.h"

#include "core/contracts/float3.h"

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

// Where a light sample is drawn from, and the u it is drawn with
// (light_points).
struct LightPointQuery {
    contracts::Float3 point;
    float u0;
    float u1;
    uint32_t padding[3];
};

static_assert(sizeof(LightPointQuery) == 32, "LightPointQuery must be the same 32 bytes on the host and in shaders");

// What contract 3's sample_light() draws from a point with u, and what
// contract 4 gives back of the point it reaches, kept as an identity
// (light_point_at()) and evaluated from the same point
// (evaluate_light_point(), light_point_pdf()): the light points' path in
// ReSTIR PT (metal/integrator/nee.metal.h, kept_point).
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
    float pdf;       // light_point_pdf() of the evaluation
    uint32_t light;  // the identity's light word
    uint32_t padding;
};

static_assert(sizeof(LightPointResult) == 80, "LightPointResult must be the same 80 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<UnitTrip> && std::is_trivially_copyable_v<UnitDraws> &&
              std::is_trivially_copyable_v<LightPointQuery> && std::is_trivially_copyable_v<LightPointResult>);
#endif

}  // namespace tests
}  // namespace serenity
