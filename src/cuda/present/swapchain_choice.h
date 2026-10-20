#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

#include "core/frame/extent.h"

namespace serenity::cuda {

// Axis: GPU backend (the window's presentation:
// docs/research/2026-10-20-cuda-window.md).
//
// What the presenter (present/presenter.h) asks of a window's surface, as
// a function of what the surface offers, apart from the driver (F.8: a
// pure function of its arguments, tested as one, A.1):
//
//   - the format: B8G8R8A8_SRGB in the sRGB colour space where offered,
//     else B8G8R8A8_UNORM, the byte orders an X11 surface offers on
//     NVIDIA's driver. The frame the tone map writes is sRGB-encoded bytes
//     already (cuda/passes/tone_map/), so the image it is copied into
//     before the blit is R8G8B8A8_SRGB beside an sRGB swapchain and
//     R8G8B8A8_UNORM beside a UNORM one: the blit then decodes and encodes
//     the same, or neither, and filters in linear light where it scales
//     (`staging`);
//   - the present mode: FIFO, which every surface offers (the Vulkan
//     specification, VkPresentModeKHR), so frames come at the display's
//     rate, as the Mac's display link paces them;
//   - the extent: the surface's current extent where it gives one, else
//     the window's pixels, clamped to what the surface allows;
//   - the images: one more than the least, at most the most, so the
//     presenter acquires one while the display shows another.
//
// Throws CudaError (cuda/device/error.h) for a surface offering neither
// format, no FIFO (which would be the driver's fault), or a window of no
// pixels (I.5).
struct SurfaceOffer {
  VkSurfaceCapabilitiesKHR capabilities{};
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> modes;
};

struct SwapchainChoice {
  VkSurfaceFormatKHR format{};
  VkFormat staging = VK_FORMAT_UNDEFINED;  // the copied frame's (above)
  VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
  VkExtent2D extent{};
  std::uint32_t images = 0;
};

SwapchainChoice choose_swapchain(const SurfaceOffer& offer,
                                 frame::Extent window);

}  // namespace serenity::cuda
