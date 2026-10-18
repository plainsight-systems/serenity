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
// and the size of the drawables it makes.
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
// Pacing is not here: the drawables reach the frame through the display
// link on this layer (display_link.h), which calls the frame when the
// display wants one, as Apple's Metal apps are paced; nothing here waits
// for the display. Three drawables, as Apple's sample leaves the layer
// (its default): one on screen, one waiting to be, one being rendered.
//
// The drawable size follows the window in pixels, through resize(), which
// the window calls each frame with the size it wants. It sets the layer's
// drawable size only when the layer's is not that size: when the window's
// size changed, and when something else set the layer's (SDL sets it to the
// window's full pixels on every resize, src/app/window.h, before the
// window's frame has asked for its own). Reading the layer's size is one
// property read a frame; setting it, only on a change, as Apple's sample
// does. A drawable made before a change has the old size; the frame skips
// it (metal/frame/renderer.h, render_to_window).
//
// Not performance-sensitive: one property read a frame.
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

    // Follows the window to `size`, setting the layer's drawable size if the
    // layer's is not `size` (above); a side of 0 (a minimized window) keeps
    // the last size and changes nothing. Throws MetalError for a size Metal
    // makes no image of (device.h, check_texture_size), as the constructor
    // does.
    void resize(frame::Extent size);
    // The size drawables are rendered at: the last asked for.
    frame::Extent size() const noexcept { return size_; }

    // The layer, for the display link on it (display_link.h); configured,
    // and retained for this object's life.
    CA::MetalLayer* layer() const noexcept { return layer_.get(); }

private:
    Submission& submission_;
    NS::SharedPtr<CA::MetalLayer> layer_;
    NS::SharedPtr<MTL::ResidencySet> drawables_;  // the layer's, on the queue
    frame::Extent size_;
};

}  // namespace serenity::metal
