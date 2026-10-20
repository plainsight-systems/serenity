// The swapchain's choices (cuda/present/swapchain_choice.h), a unit with no
// driver: the sRGB byte order where offered and the copied frame's format
// beside it, FIFO, the surface's extent or the window's clamped, one image
// more than the least, and the refusals.

#include "cuda/present/swapchain_choice.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <limits>

#include "cuda/device/error.h"

namespace serenity::cuda {
namespace {

constexpr std::uint32_t any = std::numeric_limits<std::uint32_t>::max();

VkSurfaceFormatKHR srgb_space(VkFormat format) {
  return VkSurfaceFormatKHR{format, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
}

// A surface offering `formats`, FIFO and mailbox, its extent `extent`
// (any: the window's), 2 to 8 images.
SurfaceOffer offer(std::vector<VkSurfaceFormatKHR> formats, VkExtent2D extent) {
  SurfaceOffer o;
  o.formats = std::move(formats);
  o.modes = {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_FIFO_KHR};
  o.capabilities.currentExtent = extent;
  o.capabilities.minImageExtent = VkExtent2D{1, 1};
  o.capabilities.maxImageExtent = VkExtent2D{4096, 4096};
  o.capabilities.minImageCount = 2;
  o.capabilities.maxImageCount = 8;
  return o;
}

TEST_CASE("the sRGB byte order where offered, the copied frame's beside it") {
  const SwapchainChoice srgb =
      choose_swapchain(offer({srgb_space(VK_FORMAT_A2R10G10B10_UNORM_PACK32),
                              srgb_space(VK_FORMAT_B8G8R8A8_UNORM),
                              srgb_space(VK_FORMAT_B8G8R8A8_SRGB)},
                             VkExtent2D{3840, 1080}),
                       frame::Extent{3840, 1080});
  CHECK(srgb.format.format == VK_FORMAT_B8G8R8A8_SRGB);
  CHECK(srgb.staging == VK_FORMAT_R8G8B8A8_SRGB);
  const SwapchainChoice unorm = choose_swapchain(
      offer({srgb_space(VK_FORMAT_B8G8R8A8_UNORM)}, VkExtent2D{640, 480}),
      frame::Extent{640, 480});
  CHECK(unorm.format.format == VK_FORMAT_B8G8R8A8_UNORM);
  CHECK(unorm.staging == VK_FORMAT_R8G8B8A8_UNORM);
  // Not in the sRGB colour space: not taken.
  CHECK_THROWS_AS(
      choose_swapchain(
          offer({VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_SRGB,
                                    VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT}},
                VkExtent2D{640, 480}),
          frame::Extent{640, 480}),
      CudaError);
}

TEST_CASE("FIFO, and a surface without it refused") {
  SurfaceOffer o =
      offer({srgb_space(VK_FORMAT_B8G8R8A8_SRGB)}, VkExtent2D{640, 480});
  CHECK(choose_swapchain(o, frame::Extent{640, 480}).mode ==
        VK_PRESENT_MODE_FIFO_KHR);
  o.modes = {VK_PRESENT_MODE_MAILBOX_KHR};
  CHECK_THROWS_AS(choose_swapchain(o, frame::Extent{640, 480}), CudaError);
}

TEST_CASE("the surface's extent, or the window's clamped to what it allows") {
  const auto extent_for = [](VkExtent2D current, frame::Extent window) {
    return choose_swapchain(
               offer({srgb_space(VK_FORMAT_B8G8R8A8_SRGB)}, current), window)
        .extent;
  };
  const VkExtent2D given = extent_for(VkExtent2D{1920, 1080}, {800, 600});
  CHECK(given.width == 1920);
  CHECK(given.height == 1080);
  const VkExtent2D window = extent_for(VkExtent2D{any, any}, {800, 600});
  CHECK(window.width == 800);
  CHECK(window.height == 600);
  const VkExtent2D clamped = extent_for(VkExtent2D{any, any}, {9000, 600});
  CHECK(clamped.width == 4096);
  CHECK(clamped.height == 600);
  CHECK_THROWS_AS(extent_for(VkExtent2D{640, 480}, {0, 480}), CudaError);
}

TEST_CASE("one image more than the least, at most the most") {
  SurfaceOffer o =
      offer({srgb_space(VK_FORMAT_B8G8R8A8_SRGB)}, VkExtent2D{640, 480});
  CHECK(choose_swapchain(o, frame::Extent{640, 480}).images == 3);
  o.capabilities.maxImageCount = 2;
  CHECK(choose_swapchain(o, frame::Extent{640, 480}).images == 2);
  o.capabilities.maxImageCount = 0;  // no most
  o.capabilities.minImageCount = 5;
  CHECK(choose_swapchain(o, frame::Extent{640, 480}).images == 6);
}

}  // namespace
}  // namespace serenity::cuda
