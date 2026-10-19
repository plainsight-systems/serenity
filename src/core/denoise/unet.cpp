#include "core/denoise/unet.h"

namespace serenity::denoise {

namespace {

constexpr std::uint32_t align(std::uint32_t side) {
  return (side + contracts::unet_alignment - 1u) / contracts::unet_alignment *
         contracts::unet_alignment;
}

}  // namespace

frame::Extent unet_padded(frame::Extent light) noexcept {
  return {align(light.width), align(light.height)};
}

UNet unet(const Widths& widths, frame::Extent light) {
  const Widths& w = widths;
  UNet net;
  net.padded = unet_padded(light);
  const auto level = [&](std::size_t l) {
    return frame::Extent{net.padded.width >> l, net.padded.height >> l};
  };
  // the lowest level with a skip
  constexpr std::size_t bottom = unet_levels - 1u;

  auto& a = net.activations;
  a.at(input_activation) = {level(0), contracts::input_channels};
  for (std::size_t l = 0; l < unet_levels; ++l) {
    a.at(skip_activation(l)) = {level(l), w.enc.at(l)};
    a.at(pooled_activation(l)) = {level(l + 1), w.enc.at(l)};
    a.at(decoded_activation(l)) = {level(l), w.dec.at(bottom - l)};
  }
  a.at(bottleneck_a_activation) = {level(unet_levels), w.bottleneck};
  a.at(bottleneck_b_activation) = {level(unet_levels), w.bottleneck};
  a.at(final_activation) = {level(0), w.dec.at(bottom)};

  // The channels upsampled into level l's decoder: the bottleneck's at
  // the bottom level, the decoder's below it otherwise.
  const auto up = [&](std::size_t l) {
    return l == bottom ? w.bottleneck : w.dec.at(bottom - 1 - l);
  };
  const auto conv = [](std::size_t in, std::uint32_t in_channels,
                       std::size_t out, std::uint32_t outputs) {
    return Convolution3x3{in, in, out, in_channels, 0u, outputs};
  };
  auto& c = net.convolutions;
  // Encoders: each into its skip.
  c.at(0) = conv(input_activation, contracts::input_channels,
                 skip_activation(0), w.enc.at(0));
  for (std::size_t l = 1; l < unet_levels; ++l) {
    c.at(l) = conv(pooled_activation(l - 1), w.enc.at(l - 1),
                   skip_activation(l), w.enc.at(l));
  }
  for (std::size_t l = 0; l < unet_levels; ++l) {
    net.pools.at(l) = {skip_activation(l), pooled_activation(l), w.enc.at(l)};
  }
  c.at(first_bottleneck) = conv(pooled_activation(bottom), w.enc.at(bottom),
                                bottleneck_a_activation, w.bottleneck);
  c.at(first_bottleneck + 1) = conv(bottleneck_a_activation, w.bottleneck,
                                    bottleneck_b_activation, w.bottleneck);
  // Decoders: dec i at level bottom - i, from the level below's output.
  for (std::size_t i = 0; i < unet_levels; ++i) {
    const std::size_t l = bottom - i;
    const std::size_t below =
        i == 0 ? bottleneck_b_activation : decoded_activation(l + 1);
    Convolution3x3& d = c.at(first_decoder + i);
    d = conv(skip_activation(l), up(l) + w.enc.at(l), decoded_activation(l),
             w.dec.at(i));
    d.up_channels = up(l);
    d.coarse = below;
  }
  c.at(last_convolution) = conv(decoded_activation(0), w.dec.at(bottom),
                                final_activation, w.dec.at(bottom));
  return net;
}

std::array<const Convolution*, unet_convolutions> in_order(
    const Network& network) {
  std::array<const Convolution*, unet_convolutions> convs{};
  for (std::size_t l = 0; l < unet_levels; ++l) {
    convs.at(l) = &network.enc.at(l);
    convs.at(first_decoder + l) = &network.dec.at(l);
  }
  convs.at(first_bottleneck) = &network.bottleneck_a;
  convs.at(first_bottleneck + 1) = &network.bottleneck_b;
  convs.at(last_convolution) = &network.dec_final;
  return convs;
}

}  // namespace serenity::denoise
