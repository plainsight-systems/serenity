#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/contracts/linear_image.h"
#include "core/frame/extent.h"
#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Frame graph (the denoiser's feature images: contract 17).
//
// The four images the features pass writes (core/contracts/features.h
// states each and its format): albedo, normal and motion RGBA16Float, depth
// R32Float, at the frame's size. Made only for a graph with the features
// pass, so every other graph allocates what it did. Made when the frame's
// size is first known and remade when it changes, before the frame's
// submission begins, draining the submission first, as the images between
// passes are (frame_images.h, FrameImages, whose reasons hold here: one set
// the frames in flight share, private storage, made resident through the
// submission, made outside the frame loop, GPU.9).
//
// read() copies them back for the headless renderer's --features: a
// readback (GPU.1), not for a frame loop; it waits for its copy. Each image
// becomes a LinearImage (contract 13), three floats a pixel, the halves
// widened exactly: albedo and normal their RGB; motion its (dx, dy, dz);
// depth its one value in all three, so it opens as gray. Traffic, at P
// pixels: 28 P bytes copied to a shared buffer and out of it, as the values
// they are (ES.48), then 48 P written as floats: 14.5, 14.5 and 25 MB at
// 960 x 540, once a written frame (GPU.1: a readback at the frame's
// boundary, for files; the denoiser will read the images where they are).
//
// Throws MetalError if the frame's size is not one Metal makes an image of
// (metal/device/device.h, max_texture_side), or the device cannot make one.
class FeatureImages {
public:
    FeatureImages(const Device& device, Submission& submission);

    FeatureImages(const FeatureImages&) = delete;
    FeatureImages& operator=(const FeatureImages&) = delete;
    FeatureImages(FeatureImages&&) = delete;
    FeatureImages& operator=(FeatureImages&&) = delete;
    ~FeatureImages() = default;

    // Ready for a frame of `size`: made, or remade, if not made at it.
    void prepare(frame::Extent size);

    // Null before the first prepare().
    MTL::Texture* albedo() const noexcept { return albedo_.texture.get(); }
    MTL::Texture* normal() const noexcept { return normal_.texture.get(); }
    MTL::Texture* depth() const noexcept { return depth_.texture.get(); }
    MTL::Texture* motion() const noexcept { return motion_.texture.get(); }

    // The four images as they are, read back (above).
    struct Read {
        contracts::LinearImage albedo;
        contracts::LinearImage normal;
        contracts::LinearImage depth;
        contracts::LinearImage motion;
    };

    // Reads the images back, through the submission. The frames that
    // wrote them must have completed (Submission::wait_until_complete).
    // Throws MetalError before the first prepare().
    Read read() const;

private:
    NS::SharedPtr<MTL::Device> device_;
    Submission& submission_;
    frame::Extent size_;
    // Each image with its residency, released after the GPU is done with it
    // (submission.h, Lifetime).
    struct Image {
        NS::SharedPtr<MTL::Texture> texture;
        Resident resident;
    };
    Image albedo_;
    Image normal_;
    Image depth_;
    Image motion_;
};

}  // namespace serenity::metal
