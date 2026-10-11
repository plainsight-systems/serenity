#pragma once

// The records path_probe.metal and tests/gpu/path_test.cpp share (Integrator:
// the path tracer's MIS): one definition for both sides, fixed-width fields,
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

// Where the MIS probe asks (mis_weights): a surface's point and wo, how many
// directions each strategy draws, and the light count each of the path
// tracer's two steps takes its selection probability from: step 5's (the
// light sample's) and step 3's (the struck light's). The same count in the
// path tracer; a different one shows the check failing.
struct MisQuery {
    contracts::Float3 point;
    uint32_t draws;  // directions drawn by each strategy
    contracts::Float3 wo;
    uint32_t light_count;  // N, step 5's
    uint32_t struck_light_count;  // N, step 3's
    uint32_t padding[3];
};

static_assert(sizeof(MisQuery) == 48, "MisQuery must be the same 48 bytes on the host and in shaders");

// One direction, drawn by one strategy, and both strategies' weights and
// densities for it, each asked as the path tracer asks it.
struct MisDraw {
    float light_weight;   // w_l, as step 5 weighs a light sample along it
    float bsdf_weight;    // w_b, as step 3 weighs the light struck along it
    float light_density;  // P p_l: 0 where the light's cone misses it
    float bsdf_density;   // p_b
    uint32_t by_bsdf;     // 0: drawn toward the light (step 5); 1: by the BSDF (step 6)
    uint32_t padding[3];
};

static_assert(sizeof(MisDraw) == 32, "MisDraw must be the same 32 bytes on the host and in shaders");

// A surface point the glossy probe integrates the light at, its wo, the side
// of the grid it integrates over the light's cone with, and N.
struct DirectQuery {
    contracts::Float3 point;
    uint32_t grid;
    contracts::Float3 wo;
    uint32_t light_count;
};

static_assert(sizeof(DirectQuery) == 32, "DirectQuery must be the same 32 bytes on the host and in shaders");

// The light's integral over the BSDF at a point, f |cos| L_e over the cone
// (its red channel), and the share of it step 5's weighted light samples
// carry in expectation; the rest is step 3's.
struct DirectIntegral {
    float direct;
    float light_share;
    uint32_t padding[2];
};

static_assert(sizeof(DirectIntegral) == 16, "DirectIntegral must be the same 16 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<MisQuery>, "MisQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<MisDraw>, "MisDraw is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<DirectQuery>, "DirectQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<DirectIntegral>, "DirectIntegral is copied from the GPU as bytes");
#endif

}  // namespace tests
}  // namespace serenity
