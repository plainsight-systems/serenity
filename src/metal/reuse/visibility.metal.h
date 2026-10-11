#pragma once

// Axis: Sample reuse (a visibility test, reused).
//
// Whether a reservoir's own visibility test stands for its pixel's surface
// now, so the shade traces no ray for it (ReSTIR DI's step 6,
// metal/passes/restir_di/restir_di.metal.h; RTXDI's reuseFinalVisibility):
// a test is reused across pixels and frames as the reservoir that carries
// it is (contracts/reservoir.h, tested), which makes the rule this family's,
// beside the merges that move a test (reservoir.metal.h, reused_from). The
// rule, by the settings (contracts/reuse_settings.h):
//
//   - a test here and now (its offset and age 0: the initial pass's ray, or
//     a raytraced merge's, from this record this frame): taken always.
//   - in basic mode with visibility_reuse on, one made within the
//     settings' visibility_age frames and closer than their
//     visibility_distance pixels: taken. RTXDI takes a test only from age 1
//     on, its age 0 meaning none is stored; here every sample is tested
//     when it is drawn, so age 0 is a test, and one with offset 0 is this
//     pixel's own.
//   - otherwise: not taken; the caller traces the ray, and records its
//     result as the test here and now (tested_here()), with W = 0 where it
//     found the sample hidden (unbiased.metal.h, hide()).
//
// What reusing a test costs in bias, stated: a test made up to
// visibility_age frames ago, or up to visibility_distance pixels away, is
// taken for the surface's now. Every test a reservoir with W > 0 carries
// found its sample visible (contracts/reservoir.h), so the error is light
// where the surface's own ray would have found the sample hidden: within
// that many pixels of a shadow's edge, and for that many frames behind an
// occluder or a firefly that moved. A brightening, where basic mode's
// weights darken; both at shadow edges, both what production ships, both
// measured against the reference with the rest of basic mode's error.
//
// Depends on contract 14 and the reuse settings alone. Pure (F.8). Cost: a
// few integer operations and one length.

#include <metal_stdlib>

#include "core/contracts/reservoir.h"
#include "core/contracts/reuse_settings.h"

namespace serenity {
namespace shaders {

// Whether `tested`, a reservoir's test word (contracts/reservoir.h), is
// taken for its pixel's surface now by `settings` (above).
inline bool takes_test(uint tested, serenity::contracts::ReuseSettings settings) {
    // The offset's two signed 8-bit coordinates, their bits taken apart as
    // unsigned (ES.101) and read back as two's complement, and the age.
    const uint mask = serenity::contracts::tested_field_mask;
    const int2 offset =
        int2(int(as_type<char>(uchar((tested >> serenity::contracts::tested_offset_x_shift) & mask))),
             int(as_type<char>(uchar((tested >> serenity::contracts::tested_offset_y_shift) & mask))));
    const uint age = (tested >> serenity::contracts::tested_age_shift) & mask;
    if (age == 0u && offset.x == 0 && offset.y == 0) {
        return true;
    }
    return settings.bias_correction == serenity::contracts::BiasCorrection::basic &&
           settings.visibility_reuse == serenity::contracts::VisibilityReuse::on &&
           age <= settings.visibility_age && metal::length(float2(offset)) < settings.visibility_distance;
}

// `reservoir` with its test made here and now, by the caller's ray from its
// pixel's surface this frame (contracts/reservoir.h, tested 0).
inline serenity::contracts::Reservoir tested_here(serenity::contracts::Reservoir reservoir) {
    reservoir.tested = 0u;
    return reservoir;
}

}  // namespace shaders
}  // namespace serenity
