#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/contracts/linear_image.h"
#include "core/frame/extent.h"

namespace serenity::measurement {

// Axis: Measurement (the reference).
//
// The reference every estimator is judged against (logical-overview.md):
// the same light transport, time frozen, with many samples. Not an
// integrator of its own (change-axes.md): its images are the path graph's,
// rendered by the headless renderer; what is here is how they are combined,
// and how much error the reference itself still has.
//
// A reference is made of batches: B independent images of the same scene,
// time, size and graph, each the GPU's mean of N samples (the headless
// renderer, --time, --samples N, --first k for batch k, so no two batches
// share a sample's random numbers, every index under 2^32, which the
// headless renderer holds them to: headless/options.h). Why batches, and
// not one image of B x N samples: the GPU keeps a pixel's running mean in
// float (metal/film/accumulate.metal.h), and a sample's step, (sample -
// mean) / (n + 1), under half the mean's last place is lost; at n = 2^16
// that is any sample within some 0.4% of the mean. At N = 1024 it is under
// 0.01%. Batches are combined here in double, with no such loss, and their
// spread says how far the reference is from its own limit
// (docs/research/2026-10-10-reference.md).
//
// Each pixel's channel keeps a Welford state, (n, mean, M2), in double: one
// pass, its update stable where summing squares is not, and a fixed order,
// the batches' as add() is given them, so the same batches in the same
// order give the same bits (CDSA.23; GDSA.2, run to run). From it:
//
//   the reference      mean, the batches' mean, rounded to float once;
//   its own error      var = M2 / (n - 1), the batches' variance, so the
//                      mean of n of them has variance var / n: the
//                      squared error this reference itself is expected to
//                      have at the pixel, from its noise alone. Summarized
//                      as two floors, each the mean over every pixel and
//                      channel: the MSE floor, var / n, and the relative
//                      MSE floor, var / n / (mean + epsilon)^2, error.h's
//                      epsilon. An error measured against this reference
//                      estimates the image's own error plus the floor
//                      (the reference's noise independent of the image's,
//                      error.h), so a figure within a small multiple of
//                      the floor is as converged as this reference can
//                      tell.
//
// The batches' independence and sameness, which the floor's meaning rests
// on, are the caller's to make, and the Makefile's `reference` target makes
// them; what is checked here is what can be: each batch's extent the
// first's, and every value finite and 0 or more (radiance is not negative),
// refused by std::invalid_argument naming the batch and the pixel before it
// is folded in (I.5), so a refused batch leaves the state as it was. The
// floors need two batches or more (std::logic_error asked of fewer).
//
// Memory: the state is 2 doubles a channel, mean and M2, and one count, n,
// shared, for it is the same at every pixel: 48 bytes a pixel, some 100 MB
// at 1920 x 1080, allocated once, when the first batch is added (MEM.9).
// A batch is read, folded in and dropped: never more than one held.
// Cost: two passes over each batch, the check above and then the fold,
// some 20 floating-point operations a channel in all, on the CPU; at
// 1920 x 1080, tens of milliseconds a batch beside its seconds of
// rendering. One pass over the batches: Welford's, never a second.

class ReferenceBuilder {
public:
    // Folds `batch` in; see above.
    void add(const contracts::LinearImage& batch);

    std::size_t batches() const noexcept { return count_; }

    // The reference: the batches' mean, rounded to float. Throws
    // std::logic_error before the first batch.
    contracts::LinearImage mean() const;

    // The two floors, above. Throw std::logic_error with fewer than two
    // batches.
    double mse_floor() const;
    double relative_mse_floor() const;

private:
    frame::Extent extent_{};
    std::size_t count_ = 0;
    std::vector<double> mean_;  // 3 a pixel, rows from the top
    std::vector<double> m2_;    // the same, the sum of squared deviations
};

}  // namespace serenity::measurement
