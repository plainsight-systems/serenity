// The display pass (display.h): the radiance image, shown as it is.

#include <metal_stdlib>

#include "core/contracts/frame_constants.h"
#include "metal/passes/display/display.metal.h"

using namespace metal;
using namespace serenity::shaders;

kernel void display(constant serenity::contracts::FrameConstants& frame [[buffer(0)]],
                    texture2d<float, access::read> radiance [[texture(0)]],
                    texture2d<float, access::write> target [[texture(1)]],
                    uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= frame.width || pixel.y >= frame.height) {
        return;
    }
    target.write(float4(display_color(radiance.read(pixel).rgb), 1.0f), pixel);
}
