#include "metal/frame/denoise_images.h"

#include <cstring>
#include <string>
#include <type_traits>

#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

namespace {

constexpr std::uint32_t align(std::uint32_t side) {
  return (side + contracts::unet_alignment - 1u) / contracts::unet_alignment *
         contracts::unet_alignment;
}

// Pixels past a bordered level's last row that a convolution's matrix loads
// may reach: its 32-pixel rows run past the level's right edge, by up to 31
// pixels and the border, into the next row; at the last row, past the end.
// Read, never used (conv3x3), so only there to be read inside the buffer.
constexpr std::size_t slack_pixels = 64;

static_assert(std::is_trivially_copyable_v<contracts::ConvParams>);
static_assert(std::is_trivially_copyable_v<contracts::ResampleParams>);

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
    : device_(NS::RetainPtr(device.handle())),
      submission_(submission),
      widths_(widths) {}

void DenoiseImages::make(Buffer& buffer, std::size_t bytes,
                         std::string_view what) {
  buffer = make_resident_buffer(
      device_.get(), submission_, bytes, MTL::ResourceStorageModeShared,
      "the denoiser's " + std::string(what), Zeroed::yes);
}

void DenoiseImages::make(Activations& a, frame::Extent size,
                         std::uint32_t stride, std::string_view what) {
  a.size = size;
  a.pitch = size.width + 2u;
  a.stride = stride;
  const std::size_t pixels =
      std::size_t{size.height + 2u} * a.pitch + slack_pixels;
  make(a.memory, pixels * stride * sizeof(std::uint16_t), what);
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
  if (params_) {
    submission_.drain();
  }
  light_ = {};
  const auto pool = scoped_pool();
  padded_ = {align(light.width), align(light.height)};
  allocate(light);
  plan();
  light_ = light;
  history_ = joined;
  index_ = inputs.index;
  return held > 0;
}

void DenoiseImages::allocate(frame::Extent light) {
  const denoise::Widths& w = widths_;
  const auto level = [&](std::uint32_t l) {
    return frame::Extent{padded_.width >> l, padded_.height >> l};
  };
  constexpr std::uint32_t bottom =
      contracts::unet_levels - 1u;  // the lowest level with a skip

  make(input_, level(0), contracts::input_channels, "network's input");
  for (std::uint32_t l = 0; l < levels; ++l) {
    make(skip_[l], level(l), w.enc[l], "skip");
    make(pooled_[l], level(l + 1), w.enc[l], "max pool");
    make(decoded_[l], level(l), w.dec[bottom - l], "decoder's output");
  }
  make(bottleneck_a_, level(contracts::unet_levels), w.bottleneck,
       "bottleneck");
  make(bottleneck_b_, level(contracts::unet_levels), w.bottleneck,
       "bottleneck");
  make(final_, level(0), w.dec[bottom], "last features");
  // The demodulated light and the histories: a float4 a pixel.
  const std::size_t input_pixels = std::size_t{padded_.width} * padded_.height;
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
  return light_radiance_[slot].texture.get();
}

void DenoiseImages::plan() {
  const denoise::Widths& w = widths_;
  constexpr std::uint32_t bottom = contracts::unet_levels - 1u;
  // The channels upsampled into level l's decoder: the bottleneck's at
  // the bottom level, the decoder's below it otherwise.
  const auto up = [&](std::uint32_t l) {
    return l == bottom ? w.bottleneck : w.dec[bottom - 1 - l];
  };

  const auto address = [](const Activations& a) { return a.memory.address(); };
  // No concatenation (denoise_images.h): every input and output from its
  // channel 0, the parameters' offsets left 0.
  const auto make_conv = [&](const Activations& in, std::uint32_t in_channels,
                             const Activations& out, std::uint32_t outputs) {
    Conv c{};
    c.params.width = in.size.width;
    c.params.height = in.size.height;
    c.params.in_pitch = in.pitch;
    c.params.in_stride = in.stride;
    c.params.in_blocks = in_channels / contracts::channel_block;
    c.params.out_pitch = out.pitch;
    c.params.out_stride = out.stride;
    c.params.out_blocks = outputs / contracts::channel_block;
    c.params.chunk_blocks = chunk_blocks(c.params.out_blocks);
    c.params.relu = 1u;
    c.input = address(in);
    c.output = address(out);
    c.coarse = address(in);
    return c;
  };
  const auto make_pool = [&](const Activations& in, const Activations& out,
                             std::uint32_t channels) {
    Resample r{};
    r.params.width = out.size.width;
    r.params.height = out.size.height;
    r.params.channels = channels;
    r.params.in_pitch = in.pitch;
    r.params.in_stride = in.stride;
    r.params.out_pitch = out.pitch;
    r.params.out_stride = out.stride;
    r.input = address(in);
    r.output = address(out);
    return r;
  };
  // Encoders: each into its skip.
  convs_[0] = make_conv(input_, contracts::input_channels, skip_[0], w.enc[0]);
  for (std::uint32_t l = 1; l < levels; ++l) {
    convs_[l] = make_conv(pooled_[l - 1], w.enc[l - 1], skip_[l], w.enc[l]);
  }
  for (std::uint32_t l = 0; l < levels; ++l) {
    pools_[l] = make_pool(skip_[l], pooled_[l], w.enc[l]);
  }
  convs_[first_bottleneck] =
      make_conv(pooled_[bottom], w.enc[bottom], bottleneck_a_, w.bottleneck);
  convs_[first_bottleneck + 1] =
      make_conv(bottleneck_a_, w.bottleneck, bottleneck_b_, w.bottleneck);
  // Decoders: dec i at level bottom - i, from the level below's output.
  for (std::uint32_t i = 0; i < levels; ++i) {
    const std::uint32_t l = bottom - i;
    const Activations& below = i == 0 ? bottleneck_b_ : decoded_[l + 1];
    Conv& c = convs_[first_decoder + i];
    c = make_conv(skip_[l], up(l) + w.enc[l], decoded_[l], w.dec[i]);
    c.params.up_blocks = up(l) / contracts::channel_block;
    c.params.up_pitch = below.pitch;
    c.params.up_stride = below.stride;
    c.coarse = address(below);
  }
  convs_[last_convolution] =
      make_conv(decoded_[0], w.dec[bottom], final_, w.dec[bottom]);

  // Every dispatch's parameters, in one buffer, each at a stride bound as
  // `constant` (Metal's constant offset alignment, support.h).
  constexpr std::size_t param_stride = 64;
  static_assert(sizeof(contracts::ConvParams) <= param_stride &&
                sizeof(contracts::ResampleParams) <= param_stride);
  static_assert(param_stride % constant_offset_alignment == 0 &&
                param_stride % alignof(contracts::ConvParams) == 0 &&
                param_stride % alignof(contracts::ResampleParams) == 0);
  make(params_, param_stride * (convolutions + resamplings),
       "dispatch parameters");
  const std::span<std::byte> bytes(
      static_cast<std::byte*>(params_.buffer->contents()),
      param_stride * (convolutions + resamplings));
  std::size_t at = 0;
  const auto place = [&](auto& dispatch) {
    std::memcpy(bytes.subspan(at, sizeof(dispatch.params)).data(),
                &dispatch.params, sizeof(dispatch.params));
    dispatch.params_address = params_.buffer->gpuAddress() + at;
    at += param_stride;
  };
  for (Conv& c : convs_) {
    place(c);
  }
  for (Resample& r : pools_) {
    place(r);
  }
}

MTL::GPUAddress DenoiseImages::input() const noexcept {
  return input_.memory.address();
}
MTL::GPUAddress DenoiseImages::features() const noexcept {
  return final_.memory.address();
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
  return history_buffers_[(index_ + 1) % 2].address();
}
MTL::GPUAddress DenoiseImages::next_history() const noexcept {
  return history_buffers_[index_ % 2].address();
}

}  // namespace serenity::metal
