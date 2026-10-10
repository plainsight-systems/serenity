// The test pattern (test_pattern.h): red across, green down, blue pulsing
// once every four seconds, from the frame's constants alone.

#include <metal_stdlib>

#include "core/contracts/frame_constants.h"

using namespace metal;

namespace {

constant constexpr float pulse_seconds = 4.0f;  // blue's period

}  // namespace

kernel void test_pattern(constant serenity::contracts::FrameConstants& frame [[buffer(0)]],
                         texture2d<float, access::write> target [[texture(0)]],
                         uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= frame.width || pixel.y >= frame.height) {
        return;
    }
    const float red = frame.width > 1u ? float(pixel.x) / float(frame.width - 1u) : 0.0f;
    const float green = frame.height > 1u ? float(pixel.y) / float(frame.height - 1u) : 0.0f;
    const float blue = 0.5f + 0.5f * sin(2.0f * M_PI_F * frame.time_seconds / pulse_seconds);
    target.write(float4(red, green, blue, 1.0f), pixel);
}
