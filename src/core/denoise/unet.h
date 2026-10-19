#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "core/contracts/denoise.h"
#include "core/denoise/network.h"
#include "core/frame/extent.h"

namespace serenity::denoise {

// Axis: Pass (denoise: the network as dispatches, for every backend).
//
// The U-Net (contract 18; network.h) as the dispatches that run it at a
// light of a given size: its padded grid, each activation's size and
// channels, and, for each convolution and max pool in the order they run,
// which activations it reads and writes. No concatenation and no upsampled
// copy: a decoder convolution reads the level below, upsampled, and the
// skip, each where it is, so each activation is written once and read
// where it lies.
//
// What the network computes and in what order is the same on every
// backend, so decided here (A.1), a pure function (F.8). How each
// activation is laid out in memory (its pitch, borders, stride and any
// slack past its end), how a convolution is tiled, and how the weights are
// laid out for the backend's matrix units are each backend's (Metal's:
// metal/frame/denoise_images.h, metal/passes/denoise/denoise.h).
//
// The padded grid is the light's size rounded up to contracts::
// unet_alignment on each axis, as training/evaluate.py pads; level l is it
// halved l times.

// The convolutions, in the order they run: enc 0 .. 3, bottleneck a and
// b, dec 0 .. 3, the last. The pools, levels 1 .. 4, each after its
// encoder.
inline constexpr std::size_t unet_levels = contracts::unet_levels;
inline constexpr std::size_t first_bottleneck = unet_levels;
inline constexpr std::size_t first_decoder = first_bottleneck + 2;
inline constexpr std::size_t last_convolution = first_decoder + unet_levels;
inline constexpr std::size_t unet_convolutions = last_convolution + 1;
inline constexpr std::size_t unet_pools = unet_levels;

// The activations, by index: the network's input; the encoders' outputs,
// each a skip, levels 0 .. 3; the max pools' outputs, levels 1 .. 4; the
// bottleneck's two; the decoders' outputs, levels 0 .. 3 (dec 3 .. 0's);
// and the last convolution's.
inline constexpr std::size_t input_activation = 0;
constexpr std::size_t skip_activation(std::size_t level) noexcept {
  return 1 + level;
}
constexpr std::size_t pooled_activation(std::size_t level) noexcept {
  return 1 + unet_levels + level;  // the pool of level `level`'s skip
}
inline constexpr std::size_t bottleneck_a_activation = 1 + 2 * unet_levels;
inline constexpr std::size_t bottleneck_b_activation =
    bottleneck_a_activation + 1;
constexpr std::size_t decoded_activation(std::size_t level) noexcept {
  return bottleneck_b_activation + 1 + level;
}
inline constexpr std::size_t final_activation = decoded_activation(unet_levels);
inline constexpr std::size_t unet_activations = final_activation + 1;

struct Activation {
  frame::Extent size;  // its level's, unbordered
  std::uint32_t channels = 0;
};

// A convolution: `input`'s `in_channels`, of which the first
// `up_channels` are `coarse`'s, read upsampled from the level below (a
// decoder's; 0 for any other, whose `coarse` is its `input`, unread), into
// `output`'s `out_channels`, a ReLU after the bias.
struct Convolution3x3 {
  std::size_t input = 0;
  std::size_t coarse = 0;
  std::size_t output = 0;
  std::uint32_t in_channels = 0;
  std::uint32_t up_channels = 0;
  std::uint32_t out_channels = 0;
};

// A 2 x 2 max pool of `input`'s `channels` into `output`, half its size.
struct MaxPool {
  std::size_t input = 0;
  std::size_t output = 0;
  std::uint32_t channels = 0;
};

struct UNet {
  frame::Extent padded;
  std::array<Activation, unet_activations> activations{};
  std::array<Convolution3x3, unet_convolutions> convolutions{};
  std::array<MaxPool, unet_pools> pools{};
};

// The light's size rounded up on each axis to the U-Net's alignment.
frame::Extent unet_padded(frame::Extent light) noexcept;

// The network of `widths` run at a light of `light`.
UNet unet(const Widths& widths, frame::Extent light);

// `network`'s convolutions in the order UNet's run (above), each
// convolution's weights and bias the network's; valid while `network` is.
std::array<const Convolution*, unet_convolutions> in_order(
    const Network& network);

}  // namespace serenity::denoise
