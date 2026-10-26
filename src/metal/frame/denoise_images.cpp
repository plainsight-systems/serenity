#include "metal/frame/denoise_images.h"

#include <array>
#include <cstdint>
#include <string>

#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

DenoiseImages::DenoiseImages(const Device& device, Submission& submission,
                             const denoise::Widths& widths)
    : device_(NS::RetainPtr(device.handle())),
      submission_(submission),
      widths_(widths) {}

void DenoiseImages::make(Buffer& buffer, std::size_t bytes,
                         std::string_view what) {
  buffer = make_resident_buffer(
      device_.get(), submission_, bytes, MTL::ResourceStorageModeShared,
      "the denoiser's " + std::string(what), Zeroed::yes);
}

NS::SharedPtr<MTL::Tensor> DenoiseImages::tensor_over(const Buffer& buffer,
                                                      std::uint32_t channels,
                                                      Resident& resident) {
  // (W, H, C, 1), innermost first, planar: the innermost stride 1, a row's
  // its pitch (contract 18's tensor_pitch()), a channel's pitch x H
  // (above).
  const std::array<NS::Integer, 4> dims{padded_.width, padded_.height, channels,
                                        1};
  const NS::Integer pitch = contracts::tensor_pitch(padded_.width);
  const NS::Integer plane = pitch * padded_.height;
  const std::array<NS::Integer, 4> strides{1, pitch, plane, plane * channels};
  auto descriptor = NS::TransferPtr(MTL::TensorDescriptor::alloc()->init());
  descriptor->setDataType(MTL::TensorDataTypeFloat16);
  descriptor->setDimensions(
      NS::TransferPtr(MTL::TensorExtents::alloc()->init(4, dims.data())).get());
  descriptor->setStrides(
      NS::TransferPtr(MTL::TensorExtents::alloc()->init(4, strides.data()))
          .get());
  descriptor->setUsage(MTL::TensorUsageMachineLearning);
  NS::Error* error = nullptr;
  auto tensor =
      NS::TransferPtr(buffer.buffer->newTensor(descriptor.get(), 0, &error));
  if (!tensor) {
    throw MetalError("the denoiser's tensor: " + describe(error));
  }
  resident = submission_.keep_resident(tensor.get());
  return tensor;
}

bool DenoiseImages::prepare(const frame::FrameInputs& inputs,
                            frame::Extent light) {
  check_texture_size(light, "the denoiser's buffers");
  // Joined on a copy, kept once the buffers exist (E.4; frame_images.cpp,
  // PathReuseImages::prepare).
  frame::ReuseHistory joined = history_;
  std::uint64_t held = 0;
  try {
    held = joined.join(inputs, light).held;
  } catch (const frame::HistoryError& refused) {
    throw MetalError(std::string("the denoiser's history: ") + refused.what());
  }
  if (light == light_) {
    history_ = joined;
    index_ = inputs.index;
    return held > 0;
  }
  // A new size starts the history over (the rule refuses one that does
  // not); frames in flight may still read the old buffers.
  if (demod_) {
    submission_.drain();
  }
  light_ = {};
  const auto pool = scoped_pool();
  padded_ = denoise::unet(widths_, light).padded;
  allocate(light);
  light_ = light;
  history_ = joined;
  index_ = inputs.index;
  return held > 0;
}

void DenoiseImages::allocate(frame::Extent light) {
  const std::size_t input_pixels = std::size_t{padded_.width} * padded_.height;
  // The network's input and features, planar half floats a pitch a row,
  // and their tensors (above).
  const std::size_t plane =
      std::size_t{contracts::tensor_pitch(padded_.width)} * padded_.height;
  input_tensor_.reset();
  features_tensor_.reset();
  make(input_, plane * contracts::network_inputs * sizeof(std::uint16_t),
       "network input");
  make(features_, plane * widths_.dec.back() * sizeof(std::uint16_t),
       "network features");
  input_tensor_ =
      tensor_over(input_, contracts::network_inputs, input_tensor_resident_);
  features_tensor_ =
      tensor_over(features_, widths_.dec.back(), features_tensor_resident_);
  // The demodulated light and the histories: a float4 a pixel.
  make(demod_, input_pixels * 4 * sizeof(float), "demodulated light");
  for (Buffer& h : history_buffers_) {
    make(h, input_pixels * contracts::denoise_subpixels * 4 * sizeof(float),
         "history");
  }
  make(histogram_, contracts::median_bins * sizeof(std::uint32_t),
       "depth histogram");
  make(median_, sizeof(contracts::MedianState), "depth median");
  // The hand-off, at the light's own size, the accumulated image's format.
  for (ResidentTexture& image : light_radiance_) {
    image.reset();
    image = make_resident_texture(device_.get(), submission_,
                                  MTL::PixelFormatRGBA32Float, light,
                                  "light radiance hand-off");
  }
}

MTL::Texture* DenoiseImages::light_radiance(std::uint32_t slot) const {
  if (slot >= light_radiance_.size()) {
    throw MetalError("DenoiseImages::light_radiance: slot " +
                     std::to_string(slot) + " is not one");
  }
  return light_radiance_.at(slot).texture.get();
}

MTL::GPUAddress DenoiseImages::demod() const noexcept {
  return demod_.address();
}
MTL::GPUAddress DenoiseImages::histogram() const noexcept {
  return histogram_.address();
}
MTL::GPUAddress DenoiseImages::median() const noexcept {
  return median_.address();
}
MTL::GPUAddress DenoiseImages::history() const noexcept {
  return history_buffers_.at((index_ + 1) % 2).address();
}
MTL::GPUAddress DenoiseImages::next_history() const noexcept {
  return history_buffers_.at(index_ % 2).address();
}

}  // namespace serenity::metal
