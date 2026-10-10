#pragma once

#include "core/contracts/linear_image.h"

namespace serenity::measurement {

// Axis: Measurement (error against the reference).
//
// How far an image is from the reference (reference.h): two numbers, each a
// mean over every pixel and channel of the 3 x P values, v the image's and
// r the reference's:
//
//   MSE            (v - r)^2
//   relative MSE   (v - r)^2 / (r + epsilon)^2,   epsilon = 0.01
//
// The relative MSE is pbrt-v4's MRSE (its Image::MRSE, and imgtool's
// --metric MRSE), its epsilon pbrt's, so a figure here is pbrt's figure for
// the same two images; it weighs an error by the brightness it is an error
// of, so the dark table's noise counts beside the bright fireflies'. MSE is
// Falcor's ErrorMeasurePass's L2, and the measure the ReSTIR GI paper
// reports; both are kept, so either comparison can be made. Both are summed
// in double, in pixel order, row by row from the top, so the same images
// give the same bits (GDSA.2, run to run).
//
// Refused, by std::invalid_argument naming the pixel, before any sum (I.5):
// images of two extents; a value of either that is not finite; a reference
// value below 0. pbrt skips a pixel whose relative error is infinite; here
// none is skipped, for an error measured over fewer pixels than the image
// has understates it, and nothing would say so (E.2).
//
// How to read the numbers. The reference has error of its own, its floor
// (reference.h). Its noise is independent of the image's (their samples
// share no random numbers: the Makefile's targets render them from
// disjoint frame indices), so an error measured against it estimates the
// image's own error plus the floor. The naive estimator's own error falls
// as 1 / samples; on doubling axes its measured error falls by half a
// doubling until it nears the floor, which is how the reference and this
// measure are checked against each other (docs/research/
// 2026-10-10-reference.md).
//
// Not performance-sensitive: one pass of a few operations a value, on the
// CPU, tens of milliseconds at 1920 x 1080.

inline constexpr double relative_epsilon = 0.01;

struct ImageError {
    double mse = 0.0;
    double relative_mse = 0.0;
};

// The two images an error is of, each by name, so they cannot be passed in
// each other's places (I.24): the relative MSE divides by the reference's
// values, and the image's alone may be below 0. A parameter only, held for
// the call it is made for, so its members refer rather than copy some 25 MB
// each.
struct Judged {
    const contracts::LinearImage& image;
    const contracts::LinearImage& reference;
};

// The error of `judged.image` against `judged.reference`; see above.
ImageError error_against(Judged judged);  // two references: cheap to copy (F.16)

}  // namespace serenity::measurement
