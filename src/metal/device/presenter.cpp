#include "metal/device/presenter.h"

#include "metal/device/error.h"

namespace serenity::metal {

Presenter::Presenter(const Device& device, Submission& submission, LayerHandle layer, frame::Extent size)
    : submission_(submission) {
    if (layer.ca_metal_layer == nullptr) {
        throw Error("Presenter: the layer handle is null");
    }
    // A size to render at from the first frame (C.41); resize() keeps the
    // last one for a window minimized later.
    check_texture_size(size, "Presenter");
    layer_ = NS::RetainPtr(static_cast<CA::MetalLayer*>(layer.ca_metal_layer));
    layer_->setDevice(device.handle());
    layer_->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    layer_->setFramebufferOnly(false);
    layer_->setMaximumDrawableCount(3);
    layer_->setDisplaySyncEnabled(true);
    resize(size);

    MTL::ResidencySet* drawables = layer_->residencySet();
    if (drawables == nullptr) {
        throw Error("Presenter: the layer has no residency set");
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
