#pragma once

#include <cstdint>
#include <string>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

namespace serenity::metal {

// Axis: the Metal API surface.
//
// What a device is, as every measurement records it: a figure means nothing
// without the machine it ran on.
struct DeviceInfo {
    std::string name;          // "Apple M3 Max"
    std::string architecture;  // Metal's name for the GPU architecture
    // The highest Apple GPU family the device supports, 7 to 10, or 0 if it
    // supports none of them. Apple9 (M3 and later) is the first with ray
    // tracing in hardware; earlier families run the same API in software.
    int apple_family = 0;
    bool unified_memory = false;
    std::uint64_t recommended_max_working_set_size = 0;  // bytes
};

// The system's default Metal device, required to support ray tracing.
//
// Construction establishes the invariant or throws Error naming why it could
// not (E.5): there is no Metal device, or the device does not support ray
// tracing. The renderer has no path without ray tracing, so a device that
// lacks it is refused here, at start-up, rather than discovered at the first
// acceleration structure. Hardware acceleration is not required: it is
// recorded in info().apple_family, and a figure taken below Apple9 says so.
//
// Owns one retained reference to the device (R.1); copying and moving are
// deleted so that everything made from it refers to one object with one
// lifetime. Not performance-sensitive: constructed once at start-up.
class Device {
public:
    Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;
    ~Device() = default;

    // Valid for the Device's lifetime. Not retained for the caller: retain
    // it (NS::RetainPtr) to keep it longer.
    MTL::Device* handle() const { return device_.get(); }

    const DeviceInfo& info() const { return info_; }

private:
    NS::SharedPtr<MTL::Device> device_;
    DeviceInfo info_;
};

}  // namespace serenity::metal
