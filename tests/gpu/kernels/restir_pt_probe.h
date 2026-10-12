#pragma once

// The records restir_pt_probe.metal's kernels read and write, shared with
// tests/gpu/restir_pt_test.cpp alone. Layout rules as for every shared
// contract (core/contracts/frame_constants.h).

#include "core/contracts/float3.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace tests {

// One pair (x_{k-1}, x_k) for the reconnection criteria
// (metal/integrator/reconnection.metal.h), the primary ray's x0 and x1
// beside it.
struct CriteriaQuery {
    contracts::Float3 camera;   // x0
    float footprint;            // c
    contracts::Float3 primary;  // x1
    float min_roughness;        // alpha_min
    contracts::Float3 primary_normal;
    uint32_t lobe_before;       // the lobe sampled at x_{k-1}
    contracts::Float3 before;   // x_{k-1}
    float density_before;       // the density it drew its direction to x_k with
    contracts::Float3 before_normal;
    uint32_t lobe_at;           // the lobe sampled at x_k: 0 for NEE's direction
    contracts::Float3 at;       // x_k
    float density_at;           // the density it drew its direction on with
    contracts::Float3 at_normal;
    float alpha_before;         // x_{k-1}'s GGX alpha, for a glossy lobe
};

static_assert(sizeof(CriteriaQuery) == 112, "CriteriaQuery must be the same 112 bytes on the host and in shaders");

struct CriteriaResult {
    uint32_t connectable;
    float threshold;
    float forward;
    float inverse;
};

static_assert(sizeof(CriteriaResult) == 16, "CriteriaResult must be the same 16 bytes on the host and in shaders");

// One frame of one reuse texture's pairing over an image.
struct PairingQuery {
    uint32_t frame;
    uint32_t texture;
    uint32_t width;
    uint32_t height;
};

static_assert(sizeof(PairingQuery) == 16, "PairingQuery must be the same 16 bytes on the host and in shaders");

// A pixel's partner, and the partner's partner.
struct PairingResult {
    int32_t x;
    int32_t y;
    int32_t back_x;
    int32_t back_y;
};

static_assert(sizeof(PairingResult) == 16, "PairingResult must be the same 16 bytes on the host and in shaders");

// A path tree drawn at one pixel and frame, and its path shifted to another
// pixel of the same frame.
struct ShiftQuery {
    uint32_t source_x;
    uint32_t source_y;
    uint32_t destination_x;
    uint32_t destination_y;
    uint32_t frame;
    uint32_t width;
    uint32_t height;
    uint32_t padding;
};

static_assert(sizeof(ShiftQuery) == 32, "ShiftQuery must be the same 32 bytes on the host and in shaders");

struct ShiftResult {
    contracts::Float3 stored;           // the reservoir's F
    uint32_t flags;                     // its flags
    contracts::Float3 shifted;          // F at the destination
    float jacobian;                     // the shift's
    contracts::Float3 source_hit;       // x1
    uint32_t succeeded;                 // whether the shift did
    contracts::Float3 destination_hit;  // y1
    float weight;                       // the reservoir's W
    contracts::Float3 reconnection;     // x_k, decoded
    float source_jacobian;              // the reservoir's cached denominator
};

static_assert(sizeof(ShiftResult) == 80, "ShiftResult must be the same 80 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<CriteriaQuery>);
static_assert(std::is_trivially_copyable_v<CriteriaResult>);
static_assert(std::is_trivially_copyable_v<PairingQuery>);
static_assert(std::is_trivially_copyable_v<PairingResult>);
static_assert(std::is_trivially_copyable_v<ShiftQuery>);
static_assert(std::is_trivially_copyable_v<ShiftResult>);
#endif

}  // namespace tests
}  // namespace serenity
