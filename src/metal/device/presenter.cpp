#include "metal/device/presenter.h"

#include "metal/device/error.h"

namespace serenity::metal {

namespace {

// Drawables the layer may have out at once (presenter.h): one on screen,
// one waiting to be, and one being rendered.
constexpr NS::UInteger drawable_count = 3;

NS::SharedPtr<CA::MetalLayer> retained(LayerHandle layer) {
    if (layer.ca_metal_layer == nullptr) {
        throw MetalError("Presenter: the layer handle is null");
    }
    return NS::RetainPtr(static_cast<CA::MetalLayer*>(layer.ca_metal_layer));
}

}  // namespace

Presenter::Presenter(const Device& device, Submission& submission, LayerHandle layer, frame::Extent size)
    : submission_(submission), layer_(retained(layer)) {
    // A size to render at from the first frame (C.41); resize() keeps the
    // last one for a window minimized later.
    check_texture_size(size, "Presenter");
    layer_->setDevice(device.handle());
    layer_->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    layer_->setFramebufferOnly(false);
    layer_->setMaximumDrawableCount(drawable_count);
    layer_->setDisplaySyncEnabled(true);
    resize(size);

    MTL::ResidencySet* drawables = layer_->residencySet();
    if (drawables == nullptr) {
        throw MetalError("Presenter: the layer has no residency set");
    }
    submission.add_residency_set(drawables);
    drawables_ = NS::RetainPtr(drawables);
}

Presenter::~Presenter() {
    if (drawables_) {
        submission_.remove_residency_set(drawables_.get());
    }
}

void Presenter::resize(frame::Extent size) {
    if (size.width == 0 || size.height == 0) {
        return;  // a minimized window: keep the last size, render nothing new
    }
    layer_->setDrawableSize(CGSize{static_cast<CGFloat>(size.width), static_cast<CGFloat>(size.height)});
    size_ = size;
}

CA::MetalDrawable* Presenter::acquire() {
    return layer_->nextDrawable();
}

}  // namespace serenity::metal
