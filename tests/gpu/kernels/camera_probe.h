#pragma once

// The records camera_probe.metal and tests/gpu/camera_test.cpp share (Camera:
// its rays): one definition for both sides, fixed-width fields, counts and
// indices as integers, each size asserted on both sides and trivial
// copyability on the host (core/contracts/shared_layout.h gives the rules;
// SF.3, I.4). Included by that kernel and that test alone, so a change to one
// probe's records touches no other probe (SF.11;
// docs/architecture/change-axes.md). The kernel binds its output at buffer 0
// and its inputs after it in the order its comment lists them
// (tests/gpu/support/probe_runner.h).

#include "core/contracts/float3.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace tests {

// A point of the image, in pixels, and a point of the lens, in [0, 1)^2.
struct CameraQuery {
  float pixel_x;
  float pixel_y;
  float lens_u;
  float lens_v;
};

static_assert(
    sizeof(CameraQuery) == 16,
    "CameraQuery must be the same 16 bytes on the host and in shaders");

// The image the camera's rays are made for.
struct CameraImage {
  uint32_t width;
  uint32_t height;
};

static_assert(
    sizeof(CameraImage) == 8,
    "CameraImage must be the same 8 bytes on the host and in shaders");

// A camera ray.
struct CameraProbeRay {
  contracts::Float3 origin;
  contracts::Float3 direction;
};

static_assert(
    sizeof(CameraProbeRay) == 24,
    "CameraProbeRay must be the same 24 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<CameraQuery>,
              "CameraQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<CameraImage>,
              "CameraImage is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<CameraProbeRay>,
              "CameraProbeRay is copied from the GPU as bytes");
#endif

}  // namespace tests
}  // namespace serenity
