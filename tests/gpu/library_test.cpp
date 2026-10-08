// metal::Library: a compiled-in library loads, its kernel becomes a
// pipeline, and every failure throws metal::Error naming its cause.

#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "metal/device.h"
#include "metal/error.h"
#include "metal/library.h"
#include "serenity/metallib/portable_math_probe.h"

using serenity::metal::Device;
using serenity::metal::Error;
using serenity::metal::Library;

TEST_CASE("a compiled-in library loads and builds a pipeline for its kernel") {
    Device device;
    Library library(device, serenity::metallib::portable_math_probe);
    auto pipeline = library.compute_pipeline("portable_math_probe");
    CHECK(pipeline);
    CHECK(pipeline->maxTotalThreadsPerThreadgroup() > 0);
}

TEST_CASE("a function the library lacks is an error that names it") {
    Device device;
    Library library(device, serenity::metallib::portable_math_probe);
    try {
        (void)library.compute_pipeline("no_such_kernel");
        FAIL("expected metal::Error");
    } catch (const Error& error) {
        CHECK(std::string(error.what()).find("no_such_kernel") != std::string::npos);
    }
}

TEST_CASE("a null function name is an error") {
    Device device;
    Library library(device, serenity::metallib::portable_math_probe);
    CHECK_THROWS_AS((void)library.compute_pipeline(nullptr), Error);
}

TEST_CASE("bytes that are not a library are an error") {
    Device device;
    const std::vector<unsigned char> garbage(4096, 0x5a);
    CHECK_THROWS_AS(Library(device, garbage), Error);
    CHECK_THROWS_AS(Library(device, std::span<const unsigned char>()), Error);
}
