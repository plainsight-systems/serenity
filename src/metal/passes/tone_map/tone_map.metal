// The tone-map pass (tone_map.h): the core's steps 1 to 6
// (core/passes/tone_map.h), with its constants, in four kernels.
//
// Every image is sampled at its levels' texel centers, (i + 0.5) / size,
// through one sampler: bilinear, clamped to the edge, normalized
// coordinates. The radiance image is RGBA32Float, which the hardware filters
// where the device says it can (tone_map.h).

#include <metal_stdlib>

#include "core/passes/tone_map.h"
#include "metal/passes/bindings.h"
#include "metal/math/srgb.metal.h"

using namespace metal;
using namespace serenity::shaders;
using serenity::passes::ToneMap;

namespace {

constexpr sampler bilinear(coord::normalized, address::clamp_to_edge, filter::linear);

// Step 5's span above the roll-off's start: 1 - neutral_start.
constant constexpr float neutral_span = 1.0f - serenity::passes::neutral_start;

// Step 1: a read of L, exposed and clamped to the ceiling, per channel.
float3 expose(float3 radiance, float scale) {
    return min(radiance * scale, float3(serenity::passes::bloom_ceiling));
}

// Step 2's filter about `uv` in a source whose texels are `texel` apart:
// five 2 x 2 boxes, from 13 bilinear reads of `read`: a 3 x 3 grid two
// texels apart, named by compass point from the center, and four reads one
// texel diagonally off it.
template <typename Read>
float3 down13(Read read, float2 uv, float2 texel) {
    const auto s = [&](float dx, float dy) { return read(uv + float2(dx, dy) * texel); };
    const float3 north_west = s(-2, -2);
    const float3 north = s(0, -2);
    const float3 north_east = s(2, -2);
    const float3 west = s(-2, 0);
    const float3 center = s(0, 0);
    const float3 east = s(2, 0);
    const float3 south_west = s(-2, 2);
    const float3 south = s(0, 2);
    const float3 south_east = s(2, 2);
    const float3 inner_north_west = s(-1, -1);
    const float3 inner_north_east = s(1, -1);
    const float3 inner_south_west = s(-1, 1);
    const float3 inner_south_east = s(1, 1);
    const float3 middle = 0.25f * (inner_north_west + inner_north_east + inner_south_west + inner_south_east);
    const float3 corners =
        0.25f * ((north_west + north + west + center) + (north + north_east + center + east) +
                 (west + center + south_west + south) + (center + east + south + south_east));
    return serenity::passes::down_middle * middle + serenity::passes::down_corner * corners;
}

// Step 3's 3 x 3 tent of `level` about `uv`, a texel of `level` apart.
float3 tent(texture2d<float, access::sample> level, float2 uv) {
    const float2 texel = 1.0f / float2(level.get_width(), level.get_height());
    const auto s = [&](float dx, float dy) { return level.sample(bilinear, uv + float2(dx, dy) * texel).rgb; };
    return (s(-1, -1) + 2.0f * s(0, -1) + s(1, -1) + 2.0f * s(-1, 0) + 4.0f * s(0, 0) + 2.0f * s(1, 0) +
            s(-1, 1) + 2.0f * s(0, 1) + s(1, 1)) /
           16.0f;
}

// Step 5: Khronos' PBR Neutral.
float3 neutral(float3 color) {
    const float x = min(color.r, min(color.g, color.b));
    color -= x < 0.08f ? x - 6.25f * x * x : 0.04f;
    const float peak = max(color.r, max(color.g, color.b));
    if (peak < serenity::passes::neutral_start) {
        return color;
    }
    const float rolled =
        1.0f - neutral_span * neutral_span / (peak + neutral_span - serenity::passes::neutral_start);
    color *= rolled / peak;
    const float g = 1.0f - 1.0f / (serenity::passes::neutral_desaturation * (peak - rolled) + 1.0f);
    return mix(color, float3(rolled), g);
}

float2 center(uint2 pixel, uint2 size) {
    return (float2(pixel) + 0.5f) / float2(size);
}

}  // namespace

// Steps 1 and 2 for B_0: the exposed radiance, filtered to B_0's size and
// divided by the level count.
kernel void tone_map_down_first(constant ToneMap& settings [[buffer(serenity::bindings::tone_map::settings)]],
                                texture2d<float, access::sample> radiance
                                [[texture(serenity::bindings::tone_map::input)]],
                                texture2d<float, access::write> level [[texture(serenity::bindings::tone_map::level)]],
                                uint2 pixel [[thread_position_in_grid]]) {
    const uint2 size = uint2(level.get_width(), level.get_height());
    if (pixel.x >= size.x || pixel.y >= size.y) {
        return;
    }
    const float scale = exp2(settings.exposure);
    const float2 texel = 1.0f / float2(radiance.get_width(), radiance.get_height());
    const auto read = [&](float2 uv) { return expose(radiance.sample(bilinear, uv).rgb, scale); };
    const float3 filtered = down13(read, center(pixel, size), texel);
    level.write(float4(filtered / float(serenity::passes::bloom_levels), 1.0f), pixel);
}

// Step 2 for B_1 .. B_5: the level before, filtered to this one's size.
kernel void tone_map_down(texture2d<float, access::sample> source [[texture(serenity::bindings::tone_map::input)]],
                          texture2d<float, access::write> level [[texture(serenity::bindings::tone_map::level)]],
                          uint2 pixel [[thread_position_in_grid]]) {
    const uint2 size = uint2(level.get_width(), level.get_height());
    if (pixel.x >= size.x || pixel.y >= size.y) {
        return;
    }
    const float2 texel = 1.0f / float2(source.get_width(), source.get_height());
    const auto read = [&](float2 uv) { return source.sample(bilinear, uv).rgb; };
    level.write(float4(down13(read, center(pixel, size), texel), 1.0f), pixel);
}

// Step 3 for one level: B_k += tent(B_(k+1)), in place.
kernel void tone_map_up(texture2d<float, access::sample> below [[texture(serenity::bindings::tone_map::input)]],
                        texture2d<float, access::read_write> level [[texture(serenity::bindings::tone_map::level)]],
                        uint2 pixel [[thread_position_in_grid]]) {
    const uint2 size = uint2(level.get_width(), level.get_height());
    if (pixel.x >= size.x || pixel.y >= size.y) {
        return;
    }
    const float3 sum = level.read(pixel).rgb + tent(below, center(pixel, size));
    level.write(float4(sum, 1.0f), pixel);
}

// Steps 1 and 4 to 6, at the frame's size, into the target.
kernel void tone_map_finish(constant ToneMap& settings [[buffer(serenity::bindings::tone_map::settings)]],
                            texture2d<float, access::read> radiance [[texture(serenity::bindings::tone_map::input)]],
                            texture2d<float, access::sample> bloom [[texture(serenity::bindings::tone_map::bloom)]],
                            texture2d<float, access::write> target [[texture(serenity::bindings::tone_map::target)]],
                            uint2 pixel [[thread_position_in_grid]]) {
    const uint2 size = uint2(radiance.get_width(), radiance.get_height());
    if (pixel.x >= size.x || pixel.y >= size.y) {
        return;
    }
    // Step 1.
    const float3 exposed = expose(radiance.read(pixel).rgb, exp2(settings.exposure));
    // Step 4.
    const float3 glare = tent(bloom, center(pixel, size));
    const float3 composite = (1.0f - settings.bloom) * exposed + settings.bloom * glare;
    // Steps 5 and 6.
    target.write(float4(transfer_srgb(saturate(neutral(composite))), 1.0f), pixel);
}
