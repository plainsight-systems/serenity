// The depth of field's six kernels (depth_of_field.h, which states every
// rule this file keeps, Jimenez's steps in order): the presort to half
// resolution, the tiles' largest near radius and its dilation, the
// scatter-as-gather over the 49-tap disc, the median, and the composite at
// the output's size.

#include <metal_stdlib>

#include "core/contracts/depth_of_field.h"
#include "metal/passes/bindings.h"

using namespace metal;
namespace binding = serenity::bindings::depth_of_field;
namespace contracts = serenity::contracts;

namespace {

constant constexpr float pi = 3.14159265358979f;

// The signed circle of confusion of view depth `z`, in output pixels
// (contract 21): negative in front of the plane of focus; the sky, z = 0,
// at infinity; clamped to the largest radius.
float circle_of_confusion(constant contracts::DepthOfFieldConstants& k, float z) {
    const float c = z > 0.0f ? k.coc_scale * (z - k.focus_distance) / z : k.coc_scale;
    return clamp(c, -k.max_radius, k.max_radius);
}

// A pixel of a `size` grid, clamped into it.
uint2 clamped(int2 p, uint2 size) {
    return uint2(clamp(p, int2(0), int2(size) - 1));
}

// Jimenez's spread comparison: how much of a disc of `radius` covers a
// point `distance` from its center, a pixel's soft edge.
float spreads(float radius, float distance) {
    return saturate(radius - distance + 0.5f);
}

// Jimenez's sample alpha, without its pi: the share of a pixel's light a
// disc of `radius` gives each pixel it covers; a radius under half a pixel
// is half a pixel's, the pixel itself.
float spread_weight(float radius) {
    return 1.0f / max(radius * radius, 0.25f);
}

}  // namespace

// 1. The presort: one thread a half-resolution pixel, its 2 x 2 block of
// the output's (clamped at an odd edge): their mean, and the circle of
// confusion of the nearest surface among them, in half pixels; the sky is
// the farthest of all.
kernel void dof_presort(constant contracts::DepthOfFieldConstants& k [[buffer(binding::constants)]],
                        texture2d<float, access::read> radiance [[texture(binding::radiance)]],
                        texture2d<float, access::read> depth [[texture(binding::depth)]],
                        texture2d<float, access::write> presort [[texture(binding::presort)]],
                        uint2 p [[thread_position_in_grid]]) {
    if (p.x >= k.half_width || p.y >= k.half_height) {
        return;
    }
    const uint2 size = uint2(k.width, k.height);
    float3 color = float3(0.0f);
    float nearest = 0.0f;  // 0: the sky, until a surface is seen
    for (uint q = 0; q < 4u; ++q) {
        const uint2 x = clamped(int2(contracts::dof_scale * p + uint2(q & 1u, q >> 1)), size);
        color += radiance.read(x).rgb;
        const float z = depth.read(x).r;
        if (z > 0.0f && (nearest == 0.0f || z < nearest)) {
            nearest = z;
        }
    }
    presort.write(float4(0.25f * color, circle_of_confusion(k, nearest) / float(contracts::dof_scale)), p);
}

// 2. The tiles: one thread a tile, the largest near-field radius among its
// half pixels, in half pixels (0 where none is in front of the focus).
kernel void dof_tiles(constant contracts::DepthOfFieldConstants& k [[buffer(binding::constants)]],
                      device float* tiles [[buffer(binding::tiles)]],
                      texture2d<float, access::read> presort [[texture(binding::presort)]],
                      uint2 t [[thread_position_in_grid]]) {
    if (t.x >= k.tiles_x || t.y >= k.tiles_y) {
        return;
    }
    float largest = 0.0f;
    const uint2 first = t * contracts::dof_tile;
    const uint2 last = min(first + contracts::dof_tile, uint2(k.half_width, k.half_height));
    for (uint y = first.y; y < last.y; ++y) {
        for (uint x = first.x; x < last.x; ++x) {
            largest = max(largest, -presort.read(uint2(x, y)).a);
        }
    }
    tiles[t.y * k.tiles_x + t.x] = largest;
}

// 3. The dilation: one thread a tile, the largest near radius of the tiles
// within k.dilation of it that reach it (depth_of_field.h, step 3).
kernel void dof_dilate(constant contracts::DepthOfFieldConstants& k [[buffer(binding::constants)]],
                       device const float* tiles [[buffer(binding::tiles)]],
                       device float* dilated [[buffer(binding::dilated)]],
                       uint2 t [[thread_position_in_grid]]) {
    if (t.x >= k.tiles_x || t.y >= k.tiles_y) {
        return;
    }
    const int reach = int(k.dilation);
    float largest = 0.0f;
    for (int dy = -reach; dy <= reach; ++dy) {
        for (int dx = -reach; dx <= reach; ++dx) {
            const int2 n = int2(t) + int2(dx, dy);
            if (any(n < int2(0)) || n.x >= int(k.tiles_x) || n.y >= int(k.tiles_y)) {
                continue;
            }
            const float radius = tiles[uint(n.y) * k.tiles_x + uint(n.x)];
            const int away = max(abs(dx), abs(dy));
            if (radius >= float(max(away - 1, 0) * int(contracts::dof_tile))) {
                largest = max(largest, radius);
            }
        }
    }
    dilated[t.y * k.tiles_x + t.x] = largest;
}

// 4. The gather: one thread a half pixel, scatter as gather over the
// 49-tap disc into its far and near fields (depth_of_field.h, step 4).
kernel void dof_gather(constant contracts::DepthOfFieldConstants& k [[buffer(binding::constants)]],
                       device const float* dilated [[buffer(binding::dilated)]],
                       texture2d<float, access::read> presort [[texture(binding::presort)]],
                       texture2d<float, access::write> far [[texture(binding::far)]],
                       texture2d<float, access::write> near [[texture(binding::near)]],
                       uint2 p [[thread_position_in_grid]]) {
    if (p.x >= k.half_width || p.y >= k.half_height) {
        return;
    }
    const uint2 size = uint2(k.half_width, k.half_height);
    const float4 center = presort.read(p);
    const float own = center.a;  // signed, half pixels
    const uint2 tile = p / contracts::dof_tile;
    const float radius = max(abs(own), dilated[tile.y * k.tiles_x + tile.x]);
    if (radius < 0.5f) {
        // No blur reaches this pixel: its own light, nothing in front.
        far.write(float4(center.rgb, 0.0f), p);
        near.write(float4(0.0f), p);
        return;
    }
    // The center: in its own field, by its own spread.
    const float own_weight = spread_weight(abs(own));
    float3 far_sum = float3(0.0f);
    float far_weight = 0.0f;
    float3 near_sum = float3(0.0f);
    float near_weight = 0.0f;
    if (own >= 0.0f) {
        far_sum = own_weight * center.rgb;
        far_weight = own_weight;
    } else {
        near_sum = own_weight * center.rgb;
        near_weight = own_weight;
    }
    // A tap behind the plane of focus reaches the center with at most the
    // center's own radius where it lies behind it (depth_of_field.h).
    const float behind_limit = max(own, 0.0f);
    for (uint ring = 1u; ring <= contracts::dof_rings; ++ring) {
        const uint taps = contracts::dof_ring_taps * ring;
        const float distance = radius * float(ring) / float(contracts::dof_rings);
        for (uint i = 0u; i < taps; ++i) {
            const float angle = 2.0f * pi * (float(i) + 0.5f * float(ring & 1u)) / float(taps);
            const float2 offset = distance * float2(cos(angle), sin(angle));
            const float4 tap = presort.read(clamped(int2(round(float2(p) + offset)), size));
            const float c = tap.a;
            const float weight = spread_weight(abs(c));
            if (c >= 0.0f) {
                const float reach = c > own ? min(c, behind_limit) : c;
                const float w = spreads(reach, distance) * weight;
                far_sum += w * tap.rgb;
                far_weight += w;
            } else {
                const float w = spreads(-c, distance) * weight;
                near_sum += w * tap.rgb;
                near_weight += w;
            }
        }
    }
    // The near field's coverage: each tap stands for pi radius^2 / taps
    // pixels, each spreading 1 / (pi r^2) of its light here (Jimenez's
    // normalization), times his factor.
    const float coverage =
        saturate(contracts::dof_alpha_gain * near_weight * radius * radius / float(contracts::dof_taps));
    const float3 far_color = far_weight > 0.0f ? far_sum / far_weight : center.rgb;
    const float3 near_color = near_weight > 0.0f ? near_sum / near_weight : float3(0.0f);
    far.write(float4(far_color, 0.0f), p);
    near.write(float4(near_color, coverage), p);
}

namespace {

// The median of nine, by the exchange network of Paeth's (Graphics Gems),
// 19 exchanges: elementwise, so on four channels at once.
void exchange(thread float4& a, thread float4& b) {
    const float4 low = min(a, b);
    b = max(a, b);
    a = low;
}

float4 median9(thread float4 (&v)[9]) {
    exchange(v[1], v[2]);
    exchange(v[4], v[5]);
    exchange(v[7], v[8]);
    exchange(v[0], v[1]);
    exchange(v[3], v[4]);
    exchange(v[6], v[7]);
    exchange(v[1], v[2]);
    exchange(v[4], v[5]);
    exchange(v[7], v[8]);
    exchange(v[0], v[3]);
    exchange(v[5], v[8]);
    exchange(v[4], v[7]);
    exchange(v[3], v[6]);
    exchange(v[1], v[4]);
    exchange(v[2], v[5]);
    exchange(v[4], v[7]);
    exchange(v[4], v[2]);
    exchange(v[6], v[4]);
    exchange(v[4], v[2]);
    return v[4];
}

}  // namespace

// 5. The median: one thread a half pixel, the 3 x 3 median of each field,
// channel by channel (clamped at the edge).
kernel void dof_median(constant contracts::DepthOfFieldConstants& k [[buffer(binding::constants)]],
                       texture2d<float, access::read> far [[texture(binding::far)]],
                       texture2d<float, access::read> near [[texture(binding::near)]],
                       texture2d<float, access::write> far_filtered [[texture(binding::far_filtered)]],
                       texture2d<float, access::write> near_filtered [[texture(binding::near_filtered)]],
                       uint2 p [[thread_position_in_grid]]) {
    if (p.x >= k.half_width || p.y >= k.half_height) {
        return;
    }
    const uint2 size = uint2(k.half_width, k.half_height);
    float4 f[9];
    float4 n[9];
    for (uint i = 0; i < 9u; ++i) {
        const uint2 x = clamped(int2(p) + int2(int(i % 3u) - 1, int(i / 3u) - 1), size);
        f[i] = far.read(x);
        n[i] = near.read(x);
    }
    far_filtered.write(median9(f), p);
    near_filtered.write(median9(n), p);
}

// 6. The composite: one thread an output pixel, in place (depth_of_field.h,
// step 6): the network's own light where the pixel's circle is under a
// pixel, the far field where it is over two, the near field over both.
kernel void dof_composite(constant contracts::DepthOfFieldConstants& k [[buffer(binding::constants)]],
                          texture2d<float, access::read_write> radiance [[texture(binding::radiance)]],
                          texture2d<float, access::read> depth [[texture(binding::depth)]],
                          texture2d<float, access::sample> far_filtered [[texture(binding::far_filtered)]],
                          texture2d<float, access::sample> near_filtered [[texture(binding::near_filtered)]],
                          uint2 x [[thread_position_in_grid]]) {
    if (x.x >= k.width || x.y >= k.height) {
        return;
    }
    constexpr sampler bilinear(coord::normalized, address::clamp_to_edge, filter::linear);
    // The output pixel's center in the half grid's normalized coordinates.
    const float2 at = (float2(x) + 0.5f) / (float(contracts::dof_scale) * float2(k.half_width, k.half_height));
    const float4 sharp = radiance.read(x);
    const float c = abs(circle_of_confusion(k, depth.read(x).r));
    const float3 far_color = far_filtered.sample(bilinear, at).rgb;
    const float4 near = near_filtered.sample(bilinear, at);
    const float3 base = mix(sharp.rgb, far_color, smoothstep(1.0f, 2.0f, c));
    radiance.write(float4(mix(base, near.rgb, near.a), sharp.a), x);
}
