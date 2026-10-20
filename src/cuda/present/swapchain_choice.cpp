#include "cuda/present/swapchain_choice.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>

#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

std::optional<VkSurfaceFormatKHR> offered(const SurfaceOffer& offer,
                                          VkFormat format) {
  for (const VkSurfaceFormatKHR& f : offer.formats) {
    if (f.format == format &&
        f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return f;
    }
  }
  return std::nullopt;
}

}  // namespace

SwapchainChoice choose_swapchain(const SurfaceOffer& offer,
                                 frame::Extent window) {
  if (window.width == 0 || window.height == 0) {
    throw CudaError("choose_swapchain: a window of no pixels");
  }
  SwapchainChoice choice;
  if (const auto srgb = offered(offer, VK_FORMAT_B8G8R8A8_SRGB)) {
    choice.format = *srgb;
    choice.staging = VK_FORMAT_R8G8B8A8_SRGB;
  } else if (const auto unorm = offered(offer, VK_FORMAT_B8G8R8A8_UNORM)) {
    choice.format = *unorm;
    choice.staging = VK_FORMAT_R8G8B8A8_UNORM;
  } else {
    throw CudaError(
        "choose_swapchain: the surface offers neither B8G8R8A8_SRGB nor "
        "B8G8R8A8_UNORM in the sRGB colour space");
  }
  if (std::ranges::find(offer.modes, VK_PRESENT_MODE_FIFO_KHR) ==
      offer.modes.end()) {
    throw CudaError(
        "choose_swapchain: the surface offers no FIFO present mode, which "
        "Vulkan requires of every surface");
  }
  choice.mode = VK_PRESENT_MODE_FIFO_KHR;

  const VkSurfaceCapabilitiesKHR& caps = offer.capabilities;
  constexpr std::uint32_t any = std::numeric_limits<std::uint32_t>::max();
  if (caps.currentExtent.width != any) {
    choice.extent = caps.currentExtent;
  } else {
    choice.extent =
        VkExtent2D{std::clamp(window.width, caps.minImageExtent.width,
                              caps.maxImageExtent.width),
                   std::clamp(window.height, caps.minImageExtent.height,
                              caps.maxImageExtent.height)};
  }
  // A most of 0 means none (VkSurfaceCapabilitiesKHR).
  choice.images = caps.minImageCount + 1;
  if (caps.maxImageCount != 0) {
    choice.images = std::min(choice.images, caps.maxImageCount);
  }
  return choice;
}

}  // namespace serenity::cuda
