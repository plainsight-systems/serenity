// Recording a frame allocates nothing of ours (MEM.9), as metal/frame/
// renderer.h and metal/acceleration/scene_acceleration.h claim: every
// buffer, structure and descriptor a frame uses is made at construction or
// in prepare(), and Renderer::record(), with the shapes' placing and the
// structure's update inside it, takes no memory from the C++ heap. For
// every pass type: every graph in graphs/ that runs from the checkout
// alone, and the denoised graph with the test fixture's network in place
// of the one beside the checkout (gpu/support/rendering.h).
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
#include <execinfo.h>

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

#include <doctest/doctest.h>

#include <Foundation/Foundation.hpp>

#include "core/frame/graph_file.h"
#include "core/scene/scene.h"
#include "gpu/support/rendering.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"
#include "test_paths.h"

namespace {

// The allocations counted on this thread, and whether they are being
// counted. Plain thread-local integers: reading them allocates nothing.
thread_local std::size_t allocations = 0;
thread_local bool counting = false;

// An object in this program's image, to find the image by; and the image.
const int in_this_image = 0;
const void* this_image = nullptr;

// Whether `path` is one of the runtimes' libraries, through which our code
// allocates as much as anyone's: libc++, libc++abi, libSystem's parts.
bool runtime(const char* path) {
    return std::strstr(path, "/usr/lib/libc++") != nullptr || std::strstr(path, "/usr/lib/system/") != nullptr;
}

// Counts the allocation being made if our code made it (see above). Inlined
// into operator new, so the walk's first frame is operator new's own.
[[gnu::always_inline]] inline void count_if_ours() {
    counting = false;  // nothing the walk itself allocates is counted
    std::array<void*, 32> frames{};
    const int depth = backtrace(frames.data(), static_cast<int>(frames.size()));
    for (std::size_t i = 1; i < static_cast<std::size_t>(depth); ++i) {
        Dl_info info{};
        if (dladdr(frames[i], &info) == 0 || info.dli_fname == nullptr) {
            continue;
        }
        if (info.dli_fbase == this_image) {
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
        if (this_image == nullptr) {
            Dl_info info{};
            if (dladdr(&in_this_image, &info) != 0) {
                this_image = info.dli_fbase;
            }
        }
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

void* allocate(std::size_t size, std::size_t alignment) {
    const std::size_t bytes = size == 0 ? 1 : size;
    void* memory = alignment <= alignof(std::max_align_t)
                       ? std::malloc(bytes)
                       : std::aligned_alloc(alignment, (bytes + alignment - 1) / alignment * alignment);
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
    return allocate(size, alignof(std::max_align_t));
}

void* operator new(std::size_t size, std::align_val_t alignment) {
    if (counting) {
        count_if_ours();
    }
    return allocate(size, static_cast<std::size_t>(alignment));
}

void operator delete(void* memory) noexcept {
    std::free(memory);
}

void operator delete(void* memory, std::align_val_t) noexcept {
    std::free(memory);
}

using namespace serenity;

namespace {

// Records frames 0 to `frames` - 1 of `description` through `schedule`,
// each starting its image over at its own time, its reuse history kept
// from frame 0, the shapes moving if the scene's do; returns the most any
// one record() allocated.
std::size_t most_allocated_by_record(const scene::SceneDescription& description, const frame::Schedule& schedule,
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
        const frame::FrameInputs inputs = tests::frame_at(i, i, 0.25 * static_cast<double>(i), description.camera);
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

TEST_CASE("recording a frame of moving, blinking fireflies allocates nothing of ours, in every graph, the "
          "structure's update and all") {
    const scene::SceneDescription flying = scene::load(tests::scenes_dir / "brass_sphere_flight.toml");
    REQUIRE(animation::moves(flying.animation));
    // Every graph in graphs/, by name; those whose network is beside the
    // checkout replaced by the denoised graph with the fixture's.
    std::vector<std::filesystem::path> files;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(tests::graphs_dir)) {
        if (entry.path().extension() == ".toml") {
            files.push_back(entry.path());
        }
    }
    std::ranges::sort(files);
    std::size_t denoised = 0;
    for (const std::filesystem::path& file : files) {
        const frame::Schedule schedule = frame::load_schedule(file);
        if (schedule.denoise.has_value()) {
            ++denoised;
            continue;
        }
        INFO(file.filename().string());
        CHECK(most_allocated_by_record(flying, schedule, 6) == 0);
    }
    REQUIRE(files.size() > 10);  // graphs/ found, not an empty directory
    REQUIRE(denoised > 0);
    INFO("restir_pt_denoised.toml, the fixture's network");
    CHECK(most_allocated_by_record(flying, tests::fixture_denoised_graph(), 6) == 0);
}

TEST_CASE("recording a frame of a still scene allocates nothing of ours") {
    const scene::SceneDescription still = scene::load(tests::scenes_dir / "brass_sphere.toml");
    REQUIRE_FALSE(scene::changes(still));
    CHECK(most_allocated_by_record(still, frame::load_schedule(tests::graphs_dir / "path.toml"), 4) == 0);
}
