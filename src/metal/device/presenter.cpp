#include "metal/device/presenter.h"

#include "metal/device/error.h"

namespace serenity::metal {

Presenter::Presenter(const Device& device, Submission& submission, LayerHandle layer, frame::Extent size) {
    if (layer.ca_metal_layer == nullptr) {
        throw Error("Presenter: the layer handle is null");
    }
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
    submission.queue()->addResidencySet(drawables);
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
