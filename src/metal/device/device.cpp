#include "metal/device/device.h"

#include <array>
#include <string>
#include <utility>

#include "metal/device/error.h"

namespace serenity::metal {

namespace {

std::string to_string(const NS::String* s) {
    return s != nullptr ? std::string{s->utf8String()} : std::string{};
}

// The highest Apple family supported, searched from the newest this SDK
// names down to Apple7, the first Apple silicon Mac: each family with its
// number (C.1).
int highest_apple_family(MTL::Device* device) {
    constexpr std::array<std::pair<MTL::GPUFamily, int>, 4> families{{
        {MTL::GPUFamilyApple10, 10},
        {MTL::GPUFamilyApple9, 9},
        {MTL::GPUFamilyApple8, 8},
        {MTL::GPUFamilyApple7, 7},
    }};
    for (const auto& [family, number] : families) {
        if (device->supportsFamily(family)) {
            return number;
        }
    }
    return 0;
}

NS::SharedPtr<MTL::Device> system_default() {
    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device) {
        throw Error("no Metal device: MTL::CreateSystemDefaultDevice returned none");
    }
    return device;
}

DeviceInfo describe_device(MTL::Device* device) {
    const MTL::Architecture* architecture = device->architecture();
    return DeviceInfo{
        .name = to_string(device->name()),
        .architecture = architecture != nullptr ? to_string(architecture->name()) : std::string{},
        .apple_family = highest_apple_family(device),
        .unified_memory = device->hasUnifiedMemory(),
        .recommended_max_working_set_size = device->recommendedMaxWorkingSetSize(),
    };
}

}  // namespace

void check_texture_size(frame::Extent size, const char* what) {
    if (size.width == 0 || size.height == 0 || size.width > max_texture_side || size.height > max_texture_side) {
        throw Error(std::string{what} + ": " + std::to_string(size.width) + " x " + std::to_string(size.height) +
                    " is not an image Metal makes (each side from 1 to " + std::to_string(max_texture_side) + ")");
    }
}

Device::Device() : device_(system_default()), info_(describe_device(device_.get())) {
    if (!device_->supportsRaytracing()) {
        throw Error("Metal device '" + info_.name + "' does not support ray tracing");
    }
}

}  // namespace serenity::metal
