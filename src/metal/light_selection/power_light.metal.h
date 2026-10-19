#pragma once

// Axis: Light selection (by power: RTXDI's local-light presampling).
//
// ReSTIR PT's selection: a light drawn in proportion to its power at this
// frame's glows, from the frame's power pyramid (contracts/light_tile.h,
// whose layout and reasons are there), as RTXDI draws its light tiles
// (RTXDI_PresampleLocalLights: RTXDI_SamplePdfMipmap's descent of the PDF
// texture's mip chain, then the light at the texel reached), and the
// probability of any one light, its power over the frame's total (the
// sample's RAB_EvaluateLocalLightSourcePdf, a texel over the texels' sum),
// which every density ReSTIR PT weighs a light by asks: its NEE candidates'
// (each drawn from a tile entry, so from this distribution), and the
// light-sampling density MIS weighs a light struck by a bounced ray by,
// and the shifts' and reconnections' alike, so the merges stay unbiased.
// The probability is a light's leaf over the top, the descent's own
// product of ratios in exact arithmetic and to float rounding as computed
// (RTXDI makes the same trade).
//
// A light whose power is 0, a firefly not yet awake, is never drawn (the
// descent skips a child of no weight, even where rounding would land on
// it) and has probability 0: a bounced ray that strikes it counts it in
// full, MIS's other strategy having no chance at it, and it gives no light
// either way. A frame whose lights all have no power draws nothing: every
// probability is 0 and every candidate's density with it, so NEE keeps no
// candidate (its target is computed only for a density above 0).
//
// The uniform selection (uniform_light.metal.h) stays the path tracer's:
// it is the reference, and the baseline its noise is measured by.
//
// The pyramid's build is one threadgroup's (light_power_threads threads,
// contracts/light_tile.h), each level after a device-memory barrier on the
// one below, each sum of four in a fixed order (GDSA.2: the same lights
// build the same bits). At 616 lights it is 1365 floats, 5.3 KB, which the
// candidates' reads keep in cache; the build is some 2,000 additions, a few
// microseconds, once a frame (measured with the frame,
// docs/research/2026-10-18-production-sweep.md).

#include <metal_stdlib>

#include "core/contracts/light_tile.h"
#include "metal/lights/emitter.metal.h"
#include "metal/sampler/sampler.metal.h"

namespace serenity {
namespace shaders {

struct PowerLight {
    const device float* pyramid;  // the frame's (contracts/light_tile.h)
    uint count;                   // light records
    uint levels;                  // contracts::light_power_levels(count)
};

inline PowerLight power_light(const device float* pyramid, uint count) {
    return PowerLight{pyramid, count, serenity::contracts::light_power_levels(count)};
}

// The frame's total power: the pyramid's top.
inline float power_total(PowerLight selection) {
    return selection.pyramid[serenity::contracts::light_power_offset(selection.levels, selection.levels)];
}

// The probability the draw below takes light record `index` with (above):
// its power over the total, 0 where the total is.
inline float power_probability(PowerLight selection, uint index) {
    const float total = power_total(selection);
    return total > 0.0f ? selection.pyramid[index] / total : 0.0f;
}

struct PowerSelected {
    uint index;         // into the light records
    float probability;  // power_probability(index); 0 where nothing has power
};

// A light drawn in proportion to its power: from the top down, at each
// level one of the node's four children by their weights, a number of
// `numbers` a level (RTXDI_SamplePdfMipmap). A child of no weight is never
// taken: the last child of weight is, should rounding pass every one.
inline PowerSelected select_power_index(PowerLight selection, thread PathNumbers& numbers) {
    uint node = 0u;
    for (uint level = selection.levels; level > 0u; --level) {
        const uint first = serenity::contracts::light_power_offset(selection.levels, level - 1u) + 4u * node;
        const float4 weights = float4(selection.pyramid[first], selection.pyramid[first + 1u],
                                      selection.pyramid[first + 2u], selection.pyramid[first + 3u]);
        const float sum = (weights.x + weights.y) + (weights.z + weights.w);
        if (!(sum > 0.0f)) {
            return PowerSelected{0u, 0.0f};
        }
        const float u = next_number(numbers) * sum;
        float reached = 0.0f;
        uint chosen = 4u;
        uint heaviest_last = 0u;
        for (uint c = 0u; c < 4u; ++c) {
            if (weights[c] > 0.0f) {
                heaviest_last = c;
                reached += weights[c];
                if (chosen == 4u && u < reached) {
                    chosen = c;
                }
            }
        }
        node = 4u * node + (chosen == 4u ? heaviest_last : chosen);
    }
    return PowerSelected{node, power_probability(selection, node)};
}

// The pyramid's build, by the one threadgroup of `threads` threads, this
// one `rank` among them (above): the leaves, light record i's power (0 past the
// count), then each level's sums of four. Every thread of the group must
// call it: it holds barriers.
inline void build_power_pyramid(Lights lights, uint count, device float* pyramid, uint rank, uint threads) {
    const uint levels = serenity::contracts::light_power_levels(count);
    const uint leaves = 1u << (2u * levels);
    for (uint i = rank; i < leaves; i += threads) {
        pyramid[i] = i < count ? light_power(place_light(lights, i)) : 0.0f;
    }
    for (uint level = 1u; level <= levels; ++level) {
        metal::threadgroup_barrier(metal::mem_flags::mem_device);
        const uint below = serenity::contracts::light_power_offset(levels, level - 1u);
        const uint here = serenity::contracts::light_power_offset(levels, level);
        const uint nodes = 1u << (2u * (levels - level));
        for (uint n = rank; n < nodes; n += threads) {
            const uint first = below + 4u * n;
            pyramid[here + n] = (pyramid[first] + pyramid[first + 1u]) + (pyramid[first + 2u] + pyramid[first + 3u]);
        }
    }
}

}  // namespace shaders
}  // namespace serenity
