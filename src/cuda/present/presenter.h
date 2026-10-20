#pragma once

#include <cuda.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "core/frame/extent.h"
#include "cuda/device/device.h"
#include "cuda/device/frame_sequence.h"
#include "cuda/device/offscreen.h"
#include "cuda/device/submission.h"
#include "cuda/present/swapchain_choice.h"

namespace serenity::cuda {

// What the window gives the presenter, by opaque handles, as it gives
// Metal its layer (app/window.h; the app names no Vulkan type): the
// instance extensions its surface needs, and a function making the surface
// for an instance, a VkInstance passed as a pointer and the VkSurfaceKHR
// returned as its 64 bits, which the presenter then owns. Called once, at
// construction, so the std::function's possible allocation is init's
// (MEM.9).
struct SurfaceSource {
  std::vector<const char*> instance_extensions;
  std::function<std::uint64_t(void* instance)> make_surface;
};

// A swapchain image acquired for a frame, and the frame slot's target the
// frame renders into (below).
struct WindowImage {
  std::uint32_t image = 0;
  std::uint32_t slot = 0;
  RenderTarget target;
};

// Axis: GPU backend (the window's presentation:
// docs/research/2026-10-20-cuda-window.md).
//
// Vulkan, for presenting CUDA's frames and nothing else: no pipeline and
// no shader. What it owns, made at construction (MEM.9, GPU.9) and
// released in reverse at destruction (R.1, C.31), the GPU idle first:
//
//   - the instance, with the window's surface extensions; the surface;
//   - the device whose UUID is the CUDA device's, the RTX 3090 Ti's, with
//     one queue that does graphics, transfers and presents, and the
//     external memory and semaphore extensions for file descriptors;
//   - the swapchain, as choose_swapchain() picks (swapchain_choice.h), its
//     images at the window's size;
//   - a frame slot each (frames_in_flight): a buffer of the render size,
//     RGBA8, its memory exported and imported into CUDA once and kept
//     mapped, as Blender Cycles' viewport takes Vulkan's buffer
//     (intern/cycles/device/cuda/graphics_interop.cpp), which is the slot's
//     RenderTarget; its memory a dedicated allocation, since CUDA imports
//     the exported memory whole (GPU.9's exception: the API's requirement,
//     and two buffers, made once); an image the buffer is copied into, the
//     swapchain's byte order and sRGB-ness as `staging` says; a command buffer;
//     and the semaphore its acquire signals;
//   - a semaphore a swapchain image, which its copy signals and its
//     present waits for;
//   - two timeline semaphores, exported to CUDA, as NVIDIA's CUDA samples
//     share theirs (simpleVulkan): `rendered`, which CUDA signals with a
//     frame's sequence + 1 after its last work and Vulkan's copy waits for;
//     `copied`, which Vulkan signals with the same after the copy and the
//     next frame to write that slot's buffer waits for on CUDA's stream.
//
// A frame, in order (cuda/frame/swapchain_frames.h): acquire(), for the
// slot the frame will take, before its submission begins, so a swapchain
// found out of date begins no frame; it waits on the CPU only for the
// slot's copy two frames back, which has long finished; wait_for_slot()
// on the stream that writes the target,
// before the frame is recorded; the frame recorded into the target; then
// present() with the stream its last work is on: CUDA's signal, Vulkan's
// copy into the slot's image and blit onto the swapchain image (scaling
// the render size to the window's, linear, and swapping RGBA to BGRA),
// and the present. So neither side waits for the other on the CPU, and
// frame N + 1's light half runs beside frame N's copy and present (GPU.7;
// GPU.8: each wait names its hazard, the buffer's write and its read).
// The buffer is owned by the external queue family between the two
// (VK_QUEUE_FAMILY_EXTERNAL): the copy acquires it, and releases it back.
//
// Sizes. The swapchain follows the surface's current extent: an acquire or
// present that finds it out of date or suboptimal marks it, and the next
// acquire remakes it, the device idle first. The slots' buffers and images
// follow the render size asked for (resize()), remade, with the swapchain,
// after the frames in flight (Submission::finish). Neither happens in a
// steady frame.
//
// Throws CudaError, with Vulkan's result named, for any failure; at
// construction, if no Vulkan device is the CUDA device, or the surface
// offers nothing to present (swapchain_choice.h).
class Presenter {
 public:
  Presenter(const Device& device, Submission& submission,
            const SurfaceSource& surface, frame::Extent render);

  Presenter(const Presenter&) = delete;
  Presenter& operator=(const Presenter&) = delete;
  Presenter(Presenter&&) = delete;
  Presenter& operator=(Presenter&&) = delete;
  // Waits for Vulkan's queue and CUDA's streams, then releases everything
  // in reverse (above).
  ~Presenter();

  // Follows the window to `render`, the size frames are rendered at:
  // remakes the slots' buffers and images, and the swapchain at the
  // surface's extent, when it changes. A side of 0 (a minimized window)
  // changes nothing.
  void resize(frame::Extent render);
  // The size frames are rendered at: the last asked for.
  frame::Extent size() const noexcept { return render_; }

  // Acquires the next swapchain image for the frame that will take slot
  // `slot`; none if the swapchain was out of date, then remade for the next
  // frame.
  std::optional<WindowImage> acquire(std::uint32_t slot);

  // Puts on `stream`, before the frame is recorded onto it, CUDA's wait
  // for Vulkan's last copy out of `image`'s slot's buffer.
  void wait_for_slot(CUstream stream, const WindowImage& image);

  // Signals on `stream`, after the frame's last work, that frame
  // `sequence` is rendered into `image`'s target; then copies, blits and
  // presents it (above).
  void present(CUstream stream, const WindowImage& image,
               std::uint64_t sequence);

 private:
  struct Slot {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
    CUexternalMemory imported = nullptr;
    CUdeviceptr mapped = 0;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory image_memory = VK_NULL_HANDLE;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
    std::uint64_t last_copy = 0;  // `copied`'s value its last copy signals
  };
  struct Timeline {
    VkSemaphore semaphore = VK_NULL_HANDLE;
    CUexternalSemaphore imported = nullptr;
  };

  void make_instance(const SurfaceSource& surface);
  void make_device(const Device& device);
  Timeline make_timeline();
  void make_swapchain();
  void free_swapchain() noexcept;
  void make_slots(frame::Extent render);
  void free_slots() noexcept;
  void record_copy(const Slot& slot, std::uint32_t image) const;
  // Everything made, released in reverse, the GPU idle first: the
  // destructor's, and a constructor's that fails (E.6).
  void release() noexcept;

  Submission& submission_;
  frame::Extent render_;

  VkInstance instance_ = VK_NULL_HANDLE;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  std::uint32_t family_ = 0;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  PFN_vkGetMemoryFdKHR get_memory_fd_ = nullptr;
  PFN_vkGetSemaphoreFdKHR get_semaphore_fd_ = nullptr;

  SwapchainChoice choice_;
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  std::vector<VkImage> images_;
  std::vector<VkSemaphore> copied_into_;  // a swapchain image each
  bool stale_ = false;

  Timeline rendered_;
  Timeline copied_;
  std::array<Slot, frames_in_flight> slots_{};
};

}  // namespace serenity::cuda
