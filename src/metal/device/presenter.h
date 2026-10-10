#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "core/frame/extent.h"
#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: GPU backend.
//
// The window's surface, as Metal sees it: a CAMetalLayer, configured once,
// and the drawable a frame renders into.
//
// The layer comes from the window library (SDL3, in src/app/) as an opaque
// pointer, so the app hands it over without naming any Metal type; this is
// the one place it becomes a CA::MetalLayer. LayerHandle exists so that a
// pointer to anything else cannot be passed by accident (I.4).
//
// The layer is configured for compute: frames write the drawable's texture
// from a kernel, so it is not framebuffer-only; its pixel format is
// BGRA8Unorm, the format the passes write; three drawables, display sync
// on. Its residency set is added to the queue once, at construction, so every
// drawable it hands out is resident, and taken off at destruction, once the
// GPU is done with the frames in flight (submission.h, Lifetime).
//
// Pacing comes from here. acquire() blocks until the display frees a
// drawable, which holds the loop to the display's refresh without a timer.
// Core Animation gives up after a second and returns none, when the window
// cannot be shown; acquire() then returns null and the caller skips the
// frame. That is not an error: nothing was owed to a hidden window.
//
// The drawable size follows the window in pixels. resize() is called from the
// window's resize event, never per frame.
//
// Not performance-sensitive in itself: acquire() is one call a frame, and
// its wait is the display's, not work.
struct LayerHandle {
    void* ca_metal_layer = nullptr;  // a CAMetalLayer; must not be null
};

class Presenter {
public:
    // Configures the layer for `device` at `size`, and adds its residency set
    // to `submission`'s queue. Throws MetalError if the handle is null, or `size`
    // is not one Metal makes an image of (device.h, check_texture_size).
    Presenter(const Device& device, Submission& submission, LayerHandle layer, frame::Extent size);

    Presenter(const Presenter&) = delete;
    Presenter& operator=(const Presenter&) = delete;
    Presenter(Presenter&&) = delete;
    Presenter& operator=(Presenter&&) = delete;
    // Waits for the GPU, then takes the layer's residency set off the queue
    // (submission.h, Lifetime).
    ~Presenter();

    void resize(frame::Extent size);
    frame::Extent size() const noexcept { return size_; }

    // The next drawable, or null if Core Animation had none to give. Owned
    // by Core Animation; valid until it is presented (Submission::present).
    CA::MetalDrawable* acquire();

private:
    Submission& submission_;
    NS::SharedPtr<CA::MetalLayer> layer_;
    NS::SharedPtr<MTL::ResidencySet> drawables_;  // the layer's, on the queue
    frame::Extent size_;
};

}  // namespace serenity::metal
