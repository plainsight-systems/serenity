#include "cuda/frame/denoise_images.h"

#include <string>

#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

// Pixels past a bordered level's last row that a convolution's tiles may
// reach: its 32-pixel rows run past the level's right edge, by up to 31
// pixels and the border, into the next row; at the last row, past the end.
// Read, never used (denoise.cu), so only there to be read inside the
// buffer. Metal's.
constexpr std::size_t slack_pixels = 64;

}  // namespace

std::uint32_t chunk_blocks(std::uint32_t out_blocks) {
  for (std::uint32_t nt = 4; nt > 1; --nt) {
    if (out_blocks % nt == 0) {
      return nt;
    }
  }
  return 1;
}

DenoiseImages::DenoiseImages(const Device& device, Submission& submission,
                             const denoise::Widths& widths)
    : device_(device), submission_(submission), widths_(widths) {}

DeviceBuffer DenoiseImages::zeroed(std::size_t bytes) {
  DeviceBuffer buffer(bytes);
  zero_now(buffer.address(), bytes, "DenoiseImages");
  return buffer;
}

void DenoiseImages::make(Activations& a, frame::Extent size,
                         std::uint32_t stride) {
  a.size = size;
  a.pitch = size.width + 2u;
  a.stride = stride;
  const std::size_t pixels =
      std::size_t{size.height + 2u} * a.pitch + slack_pixels;
  a.memory = zeroed(pixels * stride * sizeof(std::uint16_t));
}

bool DenoiseImages::prepare(const frame::FrameInputs& inputs,
                            frame::Extent light) {
  if (light.width == 0 || light.height == 0) {
    throw CudaError("the denoiser's buffers: a light of no pixels");
  }
  // Joined on a copy, kept once the buffers exist (E.4).
  frame::ReuseHistory joined = history_;
  std::uint64_t held = 0;
  try {
    held = joined.join(inputs, light).held;
  } catch (const frame::HistoryError& refused) {
    throw CudaError(std::string("the denoiser's history: ") + refused.what());
  }
  if (light == light_) {
    history_ = joined;
    index_ = inputs.index;
    return held > 0;
  }
  // A new size starts the history over (the rule refuses one that does
  // not); frames in flight may still read the old buffers.
  if (demod_.address() != 0) {
    submission_.finish();
  }
  light_ = {};
  const denoise::UNet net = denoise::unet(widths_, light);
  padded_ = net.padded;
  allocate(net, light);
  plan(net);
  light_ = light;
  history_ = joined;
  index_ = inputs.index;
  return held > 0;
}

void DenoiseImages::allocate(const denoise::UNet& net, frame::Extent light) {
  for (std::size_t i = 0; i < activations_.size(); ++i) {
    const denoise::Activation& a = net.activations.at(i);
    make(activations_.at(i), a.size, a.channels);
  }
  // The demodulated light and the histories: four floats a pixel.
  const std::size_t input_pixels = std::size_t{padded_.width} * padded_.height;
  demod_ = zeroed(input_pixels * 4 * sizeof(float));
  for (DeviceBuffer& h : history_buffers_) {
    h = zeroed(input_pixels * contracts::denoise_subpixels * 4 * sizeof(float));
  }
  histogram_ = zeroed(contracts::median_bins * sizeof(std::uint32_t));
  median_ = zeroed(sizeof(contracts::MedianState));
  // The hand-off, at the light's own size, the accumulated image's format.
  for (std::unique_ptr<PitchedImage>& image : light_radiance_) {
    image.reset();
    image = std::make_unique<PitchedImage>(device_, submission_, light);
  }
}

const PitchedImage& DenoiseImages::light_radiance(std::uint32_t slot) const {
  if (slot >= light_radiance_.size() || !light_radiance_.at(slot)) {
    throw CudaError("DenoiseImages::light_radiance: slot " +
                    std::to_string(slot) + " is not one");
  }
  return *light_radiance_.at(slot);
}

void DenoiseImages::plan(const denoise::UNet& net) {
  const auto activation = [&](std::size_t i) -> const Activations& {
    return activations_.at(i);
  };
  // Each convolution and pool of the core's network as this backend's
  // dispatch: its parameters from where each activation lies (above).
  for (std::size_t i = 0; i < convs_.size(); ++i) {
    const denoise::Convolution3x3& n = net.convolutions.at(i);
    const Activations& in = activation(n.input);
    const Activations& out = activation(n.output);
    Conv c{};
    c.params.width = in.size.width;
    c.params.height = in.size.height;
    c.params.in_pitch = in.pitch;
    c.params.in_stride = in.stride;
    c.params.in_blocks = n.in_channels / contracts::channel_block;
    c.params.out_pitch = out.pitch;
    c.params.out_stride = out.stride;
    c.params.out_blocks = n.out_channels / contracts::channel_block;
    c.params.chunk_blocks = chunk_blocks(c.params.out_blocks);
    c.params.relu = 1u;
    if (n.up_channels != 0) {
      c.params.up_blocks = n.up_channels / contracts::channel_block;
      c.params.up_pitch = activation(n.coarse).pitch;
      c.params.up_stride = activation(n.coarse).stride;
    }
    c.input = in.memory.address();
    c.output = out.memory.address();
    c.coarse = activation(n.coarse).memory.address();
    convs_.at(i) = c;
  }
  for (std::size_t i = 0; i < pools_.size(); ++i) {
    const denoise::MaxPool& n = net.pools.at(i);
    const Activations& in = activation(n.input);
    const Activations& out = activation(n.output);
    Resample r{};
    r.params.width = out.size.width;
    r.params.height = out.size.height;
    r.params.channels = n.channels;
    r.params.in_pitch = in.pitch;
    r.params.in_stride = in.stride;
    r.params.out_pitch = out.pitch;
    r.params.out_stride = out.stride;
    r.input = in.memory.address();
    r.output = out.memory.address();
    pools_.at(i) = r;
  }
}

CUdeviceptr DenoiseImages::input() const noexcept {
  return activations_.at(denoise::input_activation).memory.address();
}
CUdeviceptr DenoiseImages::features() const noexcept {
  return activations_.at(denoise::final_activation).memory.address();
}
CUdeviceptr DenoiseImages::history() const noexcept {
  return history_buffers_.at((index_ + 1) % 2).address();
}
CUdeviceptr DenoiseImages::next_history() const noexcept {
  return history_buffers_.at(index_ % 2).address();
}

}  // namespace serenity::cuda
