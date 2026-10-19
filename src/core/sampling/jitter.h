#pragma once

#include <cstdint>

#include "core/contracts/denoise.h"
#include "core/contracts/frame_constants.h"

namespace serenity::sampling {

// Axis: Sampler (the camera's jitter).
//
// Where in its pixel every camera ray of a ReSTIR PT frame passes: one
// offset a frame, the same for every pixel, from a fixed low-discrepancy
// sequence, as temporal upscalers take it (logical-overview.md, "Primary
// visibility"; the denoiser reads it, contract 18). The sequence is
// Halton's in bases 2 and 3, DLSS's, TSR's and FSR's: frame n's offset is
//
//   (h2(k) - 1/2, h3(k) - 1/2),  k = n mod jitter_phases + 1,
//
// h_b the radical inverse in base b, so in [-1/2, 1/2) of a pixel on each
// axis, x to the right and y down the image's rows (the point is the
// pixel's center plus it); k starts at 1, as DLSS's guide and Unreal's TSR
// index it, since h(0) = 0 would put the first frame's samples on every
// pixel's corner. The phase count is the guide's rule, about 8 times the
// square of the upscale ratio: 8 x denoise_scale^2 = 32 phases at 2 x 2
// (contracts::jitter_phases, beside the frame's constants that carry it).
// The first 32 points of Halton(2, 3) put every x in its own 1/32 of the
// pixel and every y in its own 1/27 of the first 27, so 32 frames cover the
// pixel, which a test holds (tests/jitter_test.cpp).
//
// A function of the frame's index alone, as every number the sampler
// gives is of where and when it is used (principle 2), and constexpr
// (Per.11, F.4): computed in double and narrowed once, to the float the
// shaders read (contract 7). The data set's script writes each frame's
// with the same formula (scripts/make_denoiser_dataset.py, camera_jitter),
// both held to one table of values by their tests.

static_assert(contracts::jitter_phases == 8u * contracts::denoise_scale * contracts::denoise_scale,
              "the jitter's phases are DLSS's rule for the network's upscale");

struct CameraJitter {
    float x = 0.0f;  // pixels, to the right
    float y = 0.0f;  // pixels, down
};

// The radical inverse of `k` in `base`: its digits mirrored about the
// point. Precondition: base >= 2.
constexpr double radical_inverse(std::uint64_t k, std::uint64_t base) noexcept {
    double result = 0.0;
    double scale = 1.0 / static_cast<double>(base);
    while (k > 0) {
        result += static_cast<double>(k % base) * scale;
        k /= base;
        scale /= static_cast<double>(base);
    }
    return result;
}

// Frame `index`'s offset (above).
constexpr CameraJitter camera_jitter(std::uint64_t index) noexcept {
    const std::uint64_t k = index % contracts::jitter_phases + 1u;
    return {static_cast<float>(radical_inverse(k, 2) - 0.5), static_cast<float>(radical_inverse(k, 3) - 0.5)};
}

}  // namespace serenity::sampling
