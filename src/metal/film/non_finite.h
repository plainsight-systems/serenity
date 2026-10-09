#pragma once

#include <cstdint>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Film (what a pixel accumulates: its failures).
//
// The count of samples a converging pass left out of its pixels for not
// being finite (metal/film/accumulate.metal.h): one 32-bit counter in shared
// memory, which the pass adds to on the GPU and the program reads on the CPU.
// Film decides what a failed sample is and keeps the count; the frame graph
// only binds it (metal/frame/renderer.h), and the window and the headless
// renderer report it.
//
// count() is exact for every frame that has completed
// (Submission::wait_until_complete, drain or finish) and a lower bound while
// frames are in flight. It counts from construction, over every frame and
// every size of image since.
//
// Throws Error if the device cannot make the counter. Not
// performance-sensitive: one 4-byte read when asked.
class NonFinite {
public:
    NonFinite(const Device& device, Submission& submission);

    NonFinite(const NonFinite&) = delete;
    NonFinite& operator=(const NonFinite&) = delete;
    NonFinite(NonFinite&&) = delete;
    NonFinite& operator=(NonFinite&&) = delete;
    ~NonFinite() = default;

    // The counter, as a shader binds it.
    MTL::GPUAddress address() const { return counter_->gpuAddress(); }

    std::uint32_t count() const;

private:
    NS::SharedPtr<MTL::Buffer> counter_;  // one atomic uint
};

}  // namespace serenity::metal
