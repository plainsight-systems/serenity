#include "metal/device/device.h"

#include "metal/device/error.h"

namespace serenity::metal {

namespace {

std::string to_string(const NS::String* s) {
    return s != nullptr ? std::string(s->utf8String()) : std::string();
}

// The highest Apple family supported, searched from the newest this SDK
// names down to Apple7, the first Apple silicon Mac.
int highest_apple_family(MTL::Device* device) {
    constexpr MTL::GPUFamily families[] = {
        MTL::GPUFamilyApple10, MTL::GPUFamilyApple9, MTL::GPUFamilyApple8, MTL::GPUFamilyApple7};
    constexpr int numbers[] = {10, 9, 8, 7};
    for (int i = 0; i < 4; ++i) {
        if (device->supportsFamily(families[i])) {
            return numbers[i];
        }
    }
    return 0;
}

}  // namespace

void check_texture_size(frame::Extent size, const char* what) {
    if (size.width == 0 || size.height == 0 || size.width > max_texture_side || size.height > max_texture_side) {
        throw Error(std::string(what) + ": " + std::to_string(size.width) + " x " + std::to_string(size.height) +
                    " is not an image Metal makes (each side from 1 to " + std::to_string(max_texture_side) + ")");
    }
}

Device::Device() : device_(NS::TransferPtr(MTL::CreateSystemDefaultDevice())) {
    if (!device_) {
        throw Error("no Metal device: MTL::CreateSystemDefaultDevice returned none");
    }
    MTL::Device* device = device_.get();

    info_.name = to_string(device->name());
    MTL::Architecture* architecture = device->architecture();
    info_.architecture = architecture != nullptr ? to_string(architecture->name()) : std::string();
    info_.apple_family = highest_apple_family(device);
    info_.unified_memory = device->hasUnifiedMemory();
    info_.recommended_max_working_set_size = device->recommendedMaxWorkingSetSize();

    if (!device->supportsRaytracing()) {
        throw Error("Metal device '" + info_.name + "' does not support ray tracing");
    }
}

}  // namespace serenity::metal
