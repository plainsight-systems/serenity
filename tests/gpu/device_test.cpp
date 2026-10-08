// The Metal device: acquired, supporting ray tracing, and named, so a green
// run shows which GPU it ran on.
//
// Not tested here: refusal of a device without ray tracing. Every Apple
// silicon Mac supports the API (Apple7 onward, in software before Apple9), so
// no machine available to these tests can show it; the refusal is the one
// branch in metal/device.cpp that says so.

#include <string>

#include <doctest/doctest.h>

#include "metal/device/device.h"

TEST_CASE("a Metal device with ray tracing is acquired") {
    serenity::metal::Device device;
    const auto& info = device.info();

    MESSAGE("device: " << info.name << ", architecture " << info.architecture
                       << ", Apple" << info.apple_family << ", unified memory "
                       << std::string(info.unified_memory ? "yes" : "no") << ", working set "
                       << (info.recommended_max_working_set_size >> 20) << " MiB");

    CHECK(device.handle() != nullptr);
    CHECK(device.handle()->supportsRaytracing());
    CHECK_FALSE(info.name.empty());
    CHECK(info.apple_family >= 7);
}
