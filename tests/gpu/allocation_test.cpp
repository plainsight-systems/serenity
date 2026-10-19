// Recording a frame allocates nothing of ours (MEM.9), as metal/frame/
// renderer.h and metal/acceleration/scene_acceleration.h claim: every
// buffer, structure and descriptor a frame uses is made at construction or
// in prepare(), and Renderer::record(), with the shapes' placing and the
// structure's update inside it, takes no memory from the C++ heap. For
// every pass type, in graphs this test builds: the test pattern, the path
// tracer, ReSTIR PT with the features beside it, the features alone, and
// the denoised pipeline on the test fixture's network
// (gpu/support/rendering.h). An integration test.
//
// Counted by replacing the program's operator new: while a scope of
// Counting is open on a thread, each allocation that thread makes through
// operator new from our code is counted. Ours: the first caller on the
// stack, past the C++ and C runtimes' own functions (a std::string's growth
// is the runtime's code, made for ours), is in this program's image, where
// the core, the backend and the test are linked. Not ours, and not counted:
// what Metal allocates for itself, its validation layer's bookkeeping above
// all (MTL_DEBUG_LAYER), and what Submission::commit()'s feedback handler
// costs, which is outside record() and its header states.

#include <dlfcn.h>
#include <doctest/doctest.h>
#include <execinfo.h>

#include <Foundation/Foundation.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <new>
#include <string>
#include <vector>

#include "core/frame/schedule.h"
#include "core/scene/scene.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"
#include "support/scene_text.h"
#include "support/schedules.h"
#include "support/text.h"

namespace {

// The allocations counted on this thread, and whether they are being
// counted. Plain thread-local integers: reading them allocates nothing.
// Mutable and global because operator new, which the standard calls, can
// reach no other state: the instrument this test is.
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
thread_local std::size_t allocations = 0;
thread_local bool counting = false;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

// An object in this program's image, to find the image by; and the image,
// found once, by the first Counting (below), before anything is counted.
const int in_this_image = 0;
const void* this_image() {
  static const void* const image = [] {
    Dl_info info{};
    return dladdr(&in_this_image, &info) != 0 ? info.dli_fbase : nullptr;
  }();
  return image;
}

// Whether `path` is one of the runtimes' libraries, through which our code
// allocates as much as anyone's: libc++, libc++abi, libSystem's parts.
bool runtime(const char* path) {
  return std::strstr(path, "/usr/lib/libc++") != nullptr ||
         std::strstr(path, "/usr/lib/system/") != nullptr;
}

// Counts the allocation being made if our code made it (see above). Inlined
// into operator new, so the walk's first frame is operator new's own.
[[gnu::always_inline]] inline void count_if_ours() {
  counting = false;  // nothing the walk itself allocates is counted
  std::array<void*, 32> frames{};
  const int depth = backtrace(frames.data(), static_cast<int>(frames.size()));
  for (std::size_t i = 1; i < static_cast<std::size_t>(depth); ++i) {
    Dl_info info{};
    if (dladdr(frames.at(i), &info) == 0 || info.dli_fname == nullptr) {
      continue;
    }
    if (info.dli_fbase == this_image()) {
      ++allocations;
      break;
    }
    if (!runtime(info.dli_fname)) {
      break;  // a framework's own
    }
  }
  counting = true;
}

// Counts this thread's allocations by our code for as long as it lives.
class Counting {
 public:
  Counting() noexcept {
    (void)this_image();  // found now, while nothing is counted
    allocations = 0;
    counting = true;
  }
  ~Counting() { counting = false; }
  Counting(const Counting&) = delete;
  Counting& operator=(const Counting&) = delete;
  Counting(Counting&&) = delete;
  Counting& operator=(Counting&&) = delete;

  std::size_t count() const noexcept { return allocations; }
};

// The replacement operator new's memory: the C heap is the only memory
// under the C++ one, so it is managed by hand here, and only here.
// NOLINTBEGIN(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
void* allocate(std::size_t size, std::align_val_t aligned) {
  const auto alignment = static_cast<std::size_t>(aligned);
  const std::size_t bytes = size == 0 ? 1 : size;
  void* memory = alignment <= alignof(std::max_align_t)
                     ? std::malloc(bytes)
                     : std::aligned_alloc(alignment, (bytes + alignment - 1) /
                                                         alignment * alignment);
  if (memory == nullptr) {
    throw std::bad_alloc();
  }
  return memory;
}

}  // namespace

// The replaceable allocation functions every other form of new and delete
// forwards to (the nothrow, array and sized forms, in the standard library).
void* operator new(std::size_t size) {
  if (counting) {
    count_if_ours();
  }
  return allocate(size, std::align_val_t{alignof(std::max_align_t)});
}

void* operator new(std::size_t size, std::align_val_t alignment) {
  if (counting) {
    count_if_ours();
  }
  return allocate(size, alignment);
}

void operator delete(void* memory) noexcept { std::free(memory); }

void operator delete(void* memory, std::align_val_t) noexcept {
  std::free(memory);
}
// NOLINTEND(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)

namespace animation = serenity::animation;
namespace frame = serenity::frame;
namespace metal = serenity::metal;
namespace scene = serenity::scene;
namespace tests = serenity::tests;

namespace {

// Records frames 0 to `frames` - 1 of `description` through `schedule`,
// each starting its image over at its own time, its reuse history kept
// from frame 0, the shapes moving if the scene's do; returns the most any
// one record() allocated.
std::size_t most_allocated_by_record(const scene::SceneDescription& description,
                                     const frame::Schedule& schedule,
                                     std::uint64_t frames) {
  constexpr frame::Extent size{64, 48};
  metal::Device device;
  metal::Submission submission(device);
  metal::Offscreen target(device, submission, size);
  metal::Renderer renderer(device, submission, schedule, &description);
  std::size_t most = 0;
  std::uint64_t last = 0;
  for (std::uint64_t i = 0; i < frames; ++i) {
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    const frame::FrameInputs inputs = tests::frame_at(
        i, i, 0.25 * static_cast<double>(i), description.camera);
    renderer.prepare(inputs, size);
    const metal::FrameSlot slot = submission.begin();
    {
      const Counting counted;
      renderer.record(slot, inputs, target.texture(), size);
      most = std::max(most, counted.count());
    }
    submission.commit();
    last = slot.sequence;
  }
  (void)submission.wait_until_complete(last);
  CHECK(renderer.non_finite_samples() == 0);
  return most;
}

}  // namespace

TEST_CASE("the counter counts: an allocation inside a Counting scope is seen") {
  // The function itself, not a new-expression, which a compiler may elide.
  void* memory = nullptr;
  {
    const Counting counted;
    memory = ::operator new(16);
    CHECK(counted.count() == 1);
  }
  ::operator delete(memory);
}

TEST_CASE(
    "recording a frame allocates nothing of ours, in every kind of graph, the "
    "structure's update and all") {
  using frame::PassKind;
  std::vector<PassKind> light = tests::restir_pt_passes;
  light.push_back(PassKind::features);
  light.push_back(PassKind::tone_map);
  frame::Schedule restir_pt_features{light, tests::look};
  restir_pt_features.restir_pt = tests::written_restir_pt(light);
  const std::vector<std::pair<const char*, frame::Schedule>> graphs = {
      {"the test pattern",
       frame::Schedule{{PassKind::test_pattern}, std::nullopt}},
      {"the path tracer", tests::path_graph()},
      {"ReSTIR PT with the features", restir_pt_features},
      {"the features alone",
       frame::Schedule{{PassKind::features}, std::nullopt}},
      {"the denoised pipeline", tests::fixture_denoised_graph()},
  };
  SUBCASE(
      "fireflies that fly and flash: every frame places shapes, rebuilds the "
      "structure, lights glows") {
    // Through a lens, so the denoised pipeline's depth of field records
    // its dispatches (a pinhole's records none).
    const scene::SceneDescription flying =
        scene::parse(tests::replaced(tests::flying_fireflies(),
                                     "vertical_fov_degrees = 38\n",
                                     "vertical_fov_degrees = 38\nlens = { "
                                     "radius = 0.02, focus = 4.0 }\n"),
                     "flying fireflies");
    REQUIRE(flying.camera.lens_radius > 0.0f);
    REQUIRE(animation::moves(flying.animation));
    REQUIRE_FALSE(flying.animation.glowers.empty());
    for (const auto& [name, graph] : graphs) {
      INFO(name);
      CHECK(most_allocated_by_record(flying, graph, 6) == 0);
    }
  }
  SUBCASE("a still scene") {
    const scene::SceneDescription still =
        scene::parse(tests::still_fireflies(), "still fireflies");
    REQUIRE_FALSE(scene::changes(still));
    CHECK(most_allocated_by_record(still, tests::path_graph(), 4) == 0);
    CHECK(most_allocated_by_record(still, tests::fixture_denoised_graph(), 4) ==
          0);
  }
}
