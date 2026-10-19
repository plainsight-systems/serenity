// metal::Library: Metal's own refusals, a function it lacks and bytes that
// are no library, each turned into metal::MetalError naming its cause. A
// compiled-in library's kernels running is every probe test's and every
// pass's (argument_table_test.cpp runs the smallest, smoke, alone).

#include "metal/device/library.h"

#include <doctest/doctest.h>

#include <cstddef>
#include <vector>

#include "metal/device/device.h"
#include "metal/device/error.h"
#include "serenity/metallib/probes.h"

using serenity::metal::Device;
using serenity::metal::Library;
using serenity::metal::MetalError;

TEST_CASE("a function the library lacks is an error that names it") {
  Device device;
  Library library(device, serenity::metallib::probes);
  CHECK_THROWS_WITH_AS((void)library.compute_pipeline("no_such_kernel"),
                       doctest::Contains("no_such_kernel"), MetalError);
}

TEST_CASE("bytes that are not a library are an error") {
  Device device;
  const std::vector<std::byte> garbage(4096, std::byte{0x5a});
  CHECK_THROWS_AS(Library(device, garbage), MetalError);
  CHECK_THROWS_AS(Library(device, std::span<const std::byte>()), MetalError);
}
