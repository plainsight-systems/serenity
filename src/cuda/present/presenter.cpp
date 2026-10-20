#include "cuda/present/presenter.h"

#include <unistd.h>

#include <cstring>
#include <iterator>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

// Vulkan's failure, thrown as this backend's error, naming the call and
// the result's value (vulkan_core.h, VkResult).
void vk_check(VkResult result, std::string_view what) {
  if (result != VK_SUCCESS) {
    throw CudaError("Presenter: " + std::string(what) + " failed (VkResult " +
                    std::to_string(static_cast<int>(result)) + ")");
  }
}

// A Vulkan structure of type `T` and `type`, every other field zero (its
// pNext null): Vulkan's own idiom, written so no field is left out.
template <typename T>
T vk(VkStructureType type) {
  T value{};
  value.sType = type;
  return value;
}

// The count and the items of a Vulkan enumeration, in two calls.
template <typename T, typename F>
std::vector<T> enumerate(const F& call, std::string_view what) {
  std::uint32_t count = 0;
  vk_check(call(&count, nullptr), what);
  std::vector<T> items(count);
  vk_check(call(&count, items.data()), what);
  items.resize(count);
  return items;
}

// The memory type of `bits` with `wanted`, on `physical`; CudaError for
// none.
std::uint32_t memory_type(VkPhysicalDevice physical, std::uint32_t bits,
                          VkMemoryPropertyFlags wanted) {
  VkPhysicalDeviceMemoryProperties properties{};
  vkGetPhysicalDeviceMemoryProperties(physical, &properties);
  const std::span<const VkMemoryType> types =
      std::span(properties.memoryTypes).first(properties.memoryTypeCount);
  for (std::uint32_t i = 0; i < types.size(); ++i) {
    if ((bits & (1u << i)) != 0 &&
        (types[i].propertyFlags & wanted) == wanted) {
      return i;
    }
  }
  throw CudaError("Presenter: no memory type for an image or a buffer");
}

// The whole of an image's one colour level.
constexpr VkImageSubresourceRange colour_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                                               0, 1};

VkImageMemoryBarrier2 image_barrier(VkImage image, VkImageLayout from,
                                    VkImageLayout to,
                                    VkPipelineStageFlags2 src_stage,
                                    VkAccessFlags2 src_access,
                                    VkPipelineStageFlags2 dst_stage,
                                    VkAccessFlags2 dst_access) {
  auto barrier =
      vk<VkImageMemoryBarrier2>(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2);
  barrier.srcStageMask = src_stage;
  barrier.srcAccessMask = src_access;
  barrier.dstStageMask = dst_stage;
  barrier.dstAccessMask = dst_access;
  barrier.oldLayout = from;
  barrier.newLayout = to;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange = colour_range;
  return barrier;
}

}  // namespace

Presenter::Presenter(const Device& device, Submission& submission,
                     const SurfaceSource& surface, frame::Extent render)
    : submission_(submission), render_(render) {
  try {
    make_instance(surface);
    make_device(device);
    rendered_ = make_timeline();
    copied_ = make_timeline();
    make_swapchain();
    make_slots(render);
  } catch (...) {
    // What was made, released in reverse, as the destructor does (E.6).
    release();
    throw;
  }
}

Presenter::~Presenter() { release(); }

void Presenter::release() noexcept {
  submission_.wait_idle();
  if (device_ != VK_NULL_HANDLE) {
    (void)vkDeviceWaitIdle(device_);
  }
  free_slots();
  free_swapchain();
  for (Timeline* t : {&copied_, &rendered_}) {
    if (t->imported != nullptr) {
      (void)cuDestroyExternalSemaphore(t->imported);
      t->imported = nullptr;
    }
    if (t->semaphore != VK_NULL_HANDLE) {
      vkDestroySemaphore(device_, t->semaphore, nullptr);
      t->semaphore = VK_NULL_HANDLE;
    }
  }
  if (pool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device_, pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
  }
  if (device_ != VK_NULL_HANDLE) {
    vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;
  }
  if (surface_ != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    surface_ = VK_NULL_HANDLE;
  }
  if (instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
  }
}

void Presenter::make_instance(const SurfaceSource& surface) {
  auto app = vk<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
  app.pApplicationName = "serenity";
  // Timeline semaphores and the synchronization2 barriers are core in 1.3.
  app.apiVersion = VK_API_VERSION_1_3;
  auto info = vk<VkInstanceCreateInfo>(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
  info.pApplicationInfo = &app;
  info.enabledExtensionCount =
      static_cast<std::uint32_t>(surface.instance_extensions.size());
  info.ppEnabledExtensionNames = surface.instance_extensions.data();
  vk_check(vkCreateInstance(&info, nullptr, &instance_), "vkCreateInstance");
  // Vulkan's handle, its 64 bits back (presenter.h, SurfaceSource).
  // NOLINTNEXTLINE(performance-no-int-to-ptr,cppcoreguidelines-pro-type-reinterpret-cast)
  surface_ = reinterpret_cast<VkSurfaceKHR>(surface.make_surface(instance_));
  if (surface_ == VK_NULL_HANDLE) {
    throw CudaError("Presenter: the window made no Vulkan surface");
  }
}

void Presenter::make_device(const Device& device) {
  CUuuid uuid{};
  check(cuDeviceGetUuid(&uuid, device.handle()), "cuDeviceGetUuid");
  const std::vector<VkPhysicalDevice> physicals = enumerate<VkPhysicalDevice>(
      [&](std::uint32_t* n, VkPhysicalDevice* out) {
        return vkEnumeratePhysicalDevices(instance_, n, out);
      },
      "vkEnumeratePhysicalDevices");
  for (VkPhysicalDevice candidate : physicals) {
    auto id = vk<VkPhysicalDeviceIDProperties>(
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES);
    auto properties = vk<VkPhysicalDeviceProperties2>(
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
    properties.pNext = &id;
    vkGetPhysicalDeviceProperties2(candidate, &properties);
    static_assert(sizeof(id.deviceUUID) == sizeof(uuid.bytes));
    if (std::memcmp(std::data(id.deviceUUID), std::data(uuid.bytes),
                    VK_UUID_SIZE) == 0) {
      physical_ = candidate;
      break;
    }
  }
  if (physical_ == VK_NULL_HANDLE) {
    throw CudaError(
        "Presenter: no Vulkan device is the CUDA device (by their UUIDs)");
  }
  // One family that does graphics (the blit), transfers and presents.
  std::uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &count, families.data());
  bool found = false;
  for (std::uint32_t f = 0; f < count && !found; ++f) {
    VkBool32 presents = VK_FALSE;
    vk_check(
        vkGetPhysicalDeviceSurfaceSupportKHR(physical_, f, surface_, &presents),
        "vkGetPhysicalDeviceSurfaceSupportKHR");
    if ((families[f].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0 &&
        presents == VK_TRUE) {
      family_ = f;
      found = true;
    }
  }
  if (!found) {
    throw CudaError(
        "Presenter: the device has no queue that both draws and presents to "
        "the window");
  }
  const float priority = 1.0f;
  auto queue =
      vk<VkDeviceQueueCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
  queue.queueFamilyIndex = family_;
  queue.queueCount = 1;
  queue.pQueuePriorities = &priority;
  const std::array<const char*, 3> extensions{
      VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME};
  auto v13 = vk<VkPhysicalDeviceVulkan13Features>(
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
  v13.synchronization2 = VK_TRUE;
  auto v12 = vk<VkPhysicalDeviceVulkan12Features>(
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
  v12.timelineSemaphore = VK_TRUE;
  v12.pNext = &v13;
  auto info = vk<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
  info.pNext = &v12;
  info.queueCreateInfoCount = 1;
  info.pQueueCreateInfos = &queue;
  info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
  info.ppEnabledExtensionNames = extensions.data();
  vk_check(vkCreateDevice(physical_, &info, nullptr, &device_),
           "vkCreateDevice");
  vkGetDeviceQueue(device_, family_, 0, &queue_);
  // The loader returns every function as PFN_vkVoidFunction, to be cast to
  // its own type (the Vulkan specification, vkGetDeviceProcAddr).
  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  get_memory_fd_ = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
      vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR"));
  get_semaphore_fd_ = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
      vkGetDeviceProcAddr(device_, "vkGetSemaphoreFdKHR"));
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
  if (get_memory_fd_ == nullptr || get_semaphore_fd_ == nullptr) {
    throw CudaError(
        "Presenter: the device does not export memory or semaphores as file "
        "descriptors");
  }
  auto pool =
      vk<VkCommandPoolCreateInfo>(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
  pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool.queueFamilyIndex = family_;
  vk_check(vkCreateCommandPool(device_, &pool, nullptr, &pool_),
           "vkCreateCommandPool");
}

Presenter::Timeline Presenter::make_timeline() {
  Timeline t;
  auto exported = vk<VkExportSemaphoreCreateInfo>(
      VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO);
  exported.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
  auto type = vk<VkSemaphoreTypeCreateInfo>(
      VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO);
  type.pNext = &exported;
  type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  type.initialValue = 0;
  auto info =
      vk<VkSemaphoreCreateInfo>(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
  info.pNext = &type;
  vk_check(vkCreateSemaphore(device_, &info, nullptr, &t.semaphore),
           "vkCreateSemaphore");
  auto get =
      vk<VkSemaphoreGetFdInfoKHR>(VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR);
  get.semaphore = t.semaphore;
  get.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
  int fd = -1;
  vk_check(get_semaphore_fd_(device_, &get, &fd), "vkGetSemaphoreFdKHR");
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc{};
  desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD;
  // The C API's descriptor is a union, filled by the member its type names.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
  desc.handle.fd = fd;
  // CUDA takes the descriptor's ownership on success, not on failure.
  if (const CUresult imported = cuImportExternalSemaphore(&t.imported, &desc);
      imported != CUDA_SUCCESS) {
    (void)close(fd);
    vkDestroySemaphore(device_, t.semaphore, nullptr);
    check(imported, "cuImportExternalSemaphore");
  }
  return t;
}

void Presenter::make_swapchain() {
  SurfaceOffer offer;
  vk_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface_,
                                                     &offer.capabilities),
           "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
  offer.formats = enumerate<VkSurfaceFormatKHR>(
      [&](std::uint32_t* n, VkSurfaceFormatKHR* out) {
        return vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, n,
                                                    out);
      },
      "vkGetPhysicalDeviceSurfaceFormatsKHR");
  offer.modes = enumerate<VkPresentModeKHR>(
      [&](std::uint32_t* n, VkPresentModeKHR* out) {
        return vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface_, n,
                                                         out);
      },
      "vkGetPhysicalDeviceSurfacePresentModesKHR");
  // The render size where the surface gives no extent of its own.
  choice_ = choose_swapchain(offer, render_);
  auto info =
      vk<VkSwapchainCreateInfoKHR>(VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR);
  info.surface = surface_;
  info.minImageCount = choice_.images;
  info.imageFormat = choice_.format.format;
  info.imageColorSpace = choice_.format.colorSpace;
  info.imageExtent = choice_.extent;
  info.imageArrayLayers = 1;
  info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.preTransform = offer.capabilities.currentTransform;
  info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  info.presentMode = choice_.mode;
  info.clipped = VK_TRUE;
  vk_check(vkCreateSwapchainKHR(device_, &info, nullptr, &swapchain_),
           "vkCreateSwapchainKHR");
  images_ = enumerate<VkImage>(
      [&](std::uint32_t* n, VkImage* out) {
        return vkGetSwapchainImagesKHR(device_, swapchain_, n, out);
      },
      "vkGetSwapchainImagesKHR");
  copied_into_.assign(images_.size(), VK_NULL_HANDLE);
  for (VkSemaphore& s : copied_into_) {
    auto binary =
        vk<VkSemaphoreCreateInfo>(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    vk_check(vkCreateSemaphore(device_, &binary, nullptr, &s),
             "vkCreateSemaphore");
  }
  stale_ = false;
}

void Presenter::free_swapchain() noexcept {
  for (VkSemaphore s : copied_into_) {
    if (s != VK_NULL_HANDLE) {
      vkDestroySemaphore(device_, s, nullptr);
    }
  }
  copied_into_.clear();
  images_.clear();
  if (swapchain_ != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
  }
}

void Presenter::make_slots(frame::Extent render) {
  const VkDeviceSize bytes =
      VkDeviceSize{render.width} * render.height * 4;  // RGBA8
  for (Slot& slot : slots_) {
    // The buffer, its memory exported (above).
    auto external = vk<VkExternalMemoryBufferCreateInfo>(
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    auto buffer = vk<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    buffer.pNext = &external;
    buffer.size = bytes;
    buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(device_, &buffer, nullptr, &slot.buffer),
             "vkCreateBuffer");
    VkMemoryRequirements needs{};
    vkGetBufferMemoryRequirements(device_, slot.buffer, &needs);
    auto dedicated = vk<VkMemoryDedicatedAllocateInfo>(
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO);
    dedicated.buffer = slot.buffer;
    auto export_info = vk<VkExportMemoryAllocateInfo>(
        VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);
    export_info.pNext = &dedicated;
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    auto allocate =
        vk<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocate.pNext = &export_info;
    allocate.allocationSize = needs.size;
    allocate.memoryTypeIndex = memory_type(physical_, needs.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_check(vkAllocateMemory(device_, &allocate, nullptr, &slot.buffer_memory),
             "vkAllocateMemory");
    vk_check(vkBindBufferMemory(device_, slot.buffer, slot.buffer_memory, 0),
             "vkBindBufferMemory");
    auto get =
        vk<VkMemoryGetFdInfoKHR>(VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR);
    get.memory = slot.buffer_memory;
    get.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd = -1;
    vk_check(get_memory_fd_(device_, &get, &fd), "vkGetMemoryFdKHR");
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC desc{};
    desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    // The descriptor's union, by the member its type names (above).
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
    desc.handle.fd = fd;
    desc.size = needs.size;
    desc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
    // CUDA takes the descriptor's ownership on success, not on failure.
    if (const CUresult imported = cuImportExternalMemory(&slot.imported, &desc);
        imported != CUDA_SUCCESS) {
      (void)close(fd);
      check(imported, "cuImportExternalMemory");
    }
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC mapping{};
    mapping.offset = 0;
    mapping.size = bytes;
    check(
        cuExternalMemoryGetMappedBuffer(&slot.mapped, slot.imported, &mapping),
        "cuExternalMemoryGetMappedBuffer");

    // The image it is copied into before the blit (swapchain_choice.h).
    auto image = vk<VkImageCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = choice_.staging;
    image.extent = VkExtent3D{render.width, render.height, 1};
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vk_check(vkCreateImage(device_, &image, nullptr, &slot.image),
             "vkCreateImage");
    VkMemoryRequirements image_needs{};
    vkGetImageMemoryRequirements(device_, slot.image, &image_needs);
    auto image_allocate =
        vk<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    image_allocate.allocationSize = image_needs.size;
    image_allocate.memoryTypeIndex =
        memory_type(physical_, image_needs.memoryTypeBits,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_check(
        vkAllocateMemory(device_, &image_allocate, nullptr, &slot.image_memory),
        "vkAllocateMemory");
    vk_check(vkBindImageMemory(device_, slot.image, slot.image_memory, 0),
             "vkBindImageMemory");

    auto commands = vk<VkCommandBufferAllocateInfo>(
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    commands.commandPool = pool_;
    commands.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commands.commandBufferCount = 1;
    vk_check(vkAllocateCommandBuffers(device_, &commands, &slot.commands),
             "vkAllocateCommandBuffers");
    auto binary =
        vk<VkSemaphoreCreateInfo>(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    vk_check(vkCreateSemaphore(device_, &binary, nullptr, &slot.acquired),
             "vkCreateSemaphore");
  }
  render_ = render;
}

void Presenter::free_slots() noexcept {
  for (Slot& slot : slots_) {
    if (slot.mapped != 0) {
      (void)cuMemFree(slot.mapped);  // the mapping, as CUDA requires
    }
    if (slot.imported != nullptr) {
      (void)cuDestroyExternalMemory(slot.imported);
    }
    if (device_ != VK_NULL_HANDLE) {
      if (slot.acquired != VK_NULL_HANDLE) {
        vkDestroySemaphore(device_, slot.acquired, nullptr);
      }
      if (slot.commands != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device_, pool_, 1, &slot.commands);
      }
      if (slot.image != VK_NULL_HANDLE) {
        vkDestroyImage(device_, slot.image, nullptr);
      }
      if (slot.image_memory != VK_NULL_HANDLE) {
        vkFreeMemory(device_, slot.image_memory, nullptr);
      }
      if (slot.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, slot.buffer, nullptr);
      }
      if (slot.buffer_memory != VK_NULL_HANDLE) {
        vkFreeMemory(device_, slot.buffer_memory, nullptr);
      }
    }
    const std::uint64_t last = slot.last_copy;
    slot = Slot{};
    slot.last_copy = last;  // the timeline keeps counting across sizes
  }
}

void Presenter::resize(frame::Extent render) {
  if (render.width == 0 || render.height == 0 ||
      (render == render_ && !stale_)) {
    return;
  }
  // The frames in flight still use what is remade (GPU.9).
  submission_.finish();
  vk_check(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle");
  free_swapchain();
  if (!(render == render_)) {
    free_slots();
    render_ = render;
    make_slots(render);
  }
  make_swapchain();
}

std::optional<WindowImage> Presenter::acquire(std::uint32_t slot_index) {
  if (stale_) {
    resize(render_);  // the swapchain remade, at the surface's extent
  }
  Slot& slot = slots_.at(slot_index);
  // Its command buffer's last copy, two frames back, long finished.
  if (slot.last_copy > 0) {
    auto wait = vk<VkSemaphoreWaitInfo>(VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO);
    wait.semaphoreCount = 1;
    wait.pSemaphores = &copied_.semaphore;
    wait.pValues = &slot.last_copy;
    vk_check(vkWaitSemaphores(device_, &wait,
                              std::numeric_limits<std::uint64_t>::max()),
             "vkWaitSemaphores");
  }
  std::uint32_t image = 0;
  const VkResult acquired = vkAcquireNextImageKHR(
      device_, swapchain_, std::numeric_limits<std::uint64_t>::max(),
      slot.acquired, VK_NULL_HANDLE, &image);
  if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
    stale_ = true;
    return std::nullopt;
  }
  if (acquired == VK_SUBOPTIMAL_KHR) {
    stale_ = true;  // presented this once, remade for the next
  } else {
    vk_check(acquired, "vkAcquireNextImageKHR");
  }
  return WindowImage{image, slot_index, RenderTarget{slot.mapped, render_}};
}

void Presenter::wait_for_slot(CUstream stream, const WindowImage& image) {
  const Slot& slot = slots_.at(image.slot);
  if (slot.last_copy == 0) {
    return;  // never copied from
  }
  CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wait{};
  wait.params.fence.value = slot.last_copy;
  check(cuWaitExternalSemaphoresAsync(&copied_.imported, &wait, 1, stream),
        "cuWaitExternalSemaphoresAsync");
}

void Presenter::record_copy(const Slot& slot, std::uint32_t image) const {
  auto begin =
      vk<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk_check(vkResetCommandBuffer(slot.commands, 0), "vkResetCommandBuffer");
  vk_check(vkBeginCommandBuffer(slot.commands, &begin), "vkBeginCommandBuffer");
  // The buffer from CUDA, the external queue family (presenter.h); the
  // slot's image and the swapchain's for transfers.
  auto from_cuda =
      vk<VkBufferMemoryBarrier2>(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2);
  from_cuda.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
  from_cuda.srcAccessMask = VK_ACCESS_2_NONE;
  from_cuda.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  from_cuda.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  from_cuda.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
  from_cuda.dstQueueFamilyIndex = family_;
  from_cuda.buffer = slot.buffer;
  from_cuda.size = VK_WHOLE_SIZE;
  const std::array<VkImageMemoryBarrier2, 2> to_transfer{
      image_barrier(slot.image, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT),
      image_barrier(images_.at(image), VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_BLIT_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT)};
  auto first = vk<VkDependencyInfo>(VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
  first.bufferMemoryBarrierCount = 1;
  first.pBufferMemoryBarriers = &from_cuda;
  first.imageMemoryBarrierCount =
      static_cast<std::uint32_t>(to_transfer.size());
  first.pImageMemoryBarriers = to_transfer.data();
  vkCmdPipelineBarrier2(slot.commands, &first);

  VkBufferImageCopy region{};
  region.bufferRowLength = 0;  // tightly packed (offscreen.h, RenderTarget)
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = VkExtent3D{render_.width, render_.height, 1};
  vkCmdCopyBufferToImage(slot.commands, slot.buffer, slot.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  // The buffer back to CUDA; the slot's image from copy to blit.
  VkBufferMemoryBarrier2 to_cuda = from_cuda;
  to_cuda.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  to_cuda.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  to_cuda.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
  to_cuda.dstAccessMask = VK_ACCESS_2_NONE;
  to_cuda.srcQueueFamilyIndex = family_;
  to_cuda.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
  const VkImageMemoryBarrier2 to_blit = image_barrier(
      slot.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
      VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
      VK_ACCESS_2_TRANSFER_READ_BIT);
  auto second = vk<VkDependencyInfo>(VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
  second.bufferMemoryBarrierCount = 1;
  second.pBufferMemoryBarriers = &to_cuda;
  second.imageMemoryBarrierCount = 1;
  second.pImageMemoryBarriers = &to_blit;
  vkCmdPipelineBarrier2(slot.commands, &second);

  // The render size onto the window's, linear (swapchain_choice.h).
  VkImageBlit blit{};
  blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.srcOffsets[1] = VkOffset3D{static_cast<std::int32_t>(render_.width),
                                  static_cast<std::int32_t>(render_.height), 1};
  blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.dstOffsets[1] =
      VkOffset3D{static_cast<std::int32_t>(choice_.extent.width),
                 static_cast<std::int32_t>(choice_.extent.height), 1};
  vkCmdBlitImage(slot.commands, slot.image,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, images_.at(image),
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                 VK_FILTER_LINEAR);

  const VkImageMemoryBarrier2 to_present = image_barrier(
      images_.at(image), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_BLIT_BIT,
      VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE,
      VK_ACCESS_2_NONE);
  auto third = vk<VkDependencyInfo>(VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
  third.imageMemoryBarrierCount = 1;
  third.pImageMemoryBarriers = &to_present;
  vkCmdPipelineBarrier2(slot.commands, &third);
  vk_check(vkEndCommandBuffer(slot.commands), "vkEndCommandBuffer");
}

void Presenter::present(CUstream stream, const WindowImage& image,
                        std::uint64_t sequence) {
  Slot& slot = slots_.at(image.slot);
  const std::uint64_t value = sequence + 1;  // a frame's, from 1
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal{};
  signal.params.fence.value = value;
  check(
      cuSignalExternalSemaphoresAsync(&rendered_.imported, &signal, 1, stream),
      "cuSignalExternalSemaphoresAsync");

  record_copy(slot, image.image);
  const std::array<VkSemaphoreSubmitInfo, 2> waits{
      VkSemaphoreSubmitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr,
                            slot.acquired, 0, VK_PIPELINE_STAGE_2_BLIT_BIT, 0},
      VkSemaphoreSubmitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr,
                            rendered_.semaphore, value,
                            VK_PIPELINE_STAGE_2_COPY_BIT, 0}};
  const std::array<VkSemaphoreSubmitInfo, 2> signals{
      VkSemaphoreSubmitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr,
                            copied_into_.at(image.image), 0,
                            VK_PIPELINE_STAGE_2_BLIT_BIT, 0},
      VkSemaphoreSubmitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr,
                            copied_.semaphore, value,
                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0}};
  auto commands = vk<VkCommandBufferSubmitInfo>(
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO);
  commands.commandBuffer = slot.commands;
  auto submit = vk<VkSubmitInfo2>(VK_STRUCTURE_TYPE_SUBMIT_INFO_2);
  submit.waitSemaphoreInfoCount = static_cast<std::uint32_t>(waits.size());
  submit.pWaitSemaphoreInfos = waits.data();
  submit.commandBufferInfoCount = 1;
  submit.pCommandBufferInfos = &commands;
  submit.signalSemaphoreInfoCount = static_cast<std::uint32_t>(signals.size());
  submit.pSignalSemaphoreInfos = signals.data();
  vk_check(vkQueueSubmit2(queue_, 1, &submit, VK_NULL_HANDLE),
           "vkQueueSubmit2");
  slot.last_copy = value;

  auto present = vk<VkPresentInfoKHR>(VK_STRUCTURE_TYPE_PRESENT_INFO_KHR);
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores = &copied_into_.at(image.image);
  present.swapchainCount = 1;
  present.pSwapchains = &swapchain_;
  present.pImageIndices = &image.image;
  const VkResult presented = vkQueuePresentKHR(queue_, &present);
  if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) {
    stale_ = true;
  } else {
    vk_check(presented, "vkQueuePresentKHR");
  }
}

}  // namespace serenity::cuda
