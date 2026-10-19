// The denoiser's network as dispatches (core/denoise/unet.h): every
// convolution reads and writes activations of its channels and sizes, a
// decoder's coarse input is the level below, every activation but the
// network's input is written once, and the network's convolutions are
// handed over in the order the dispatches run.

#include "core/denoise/unet.h"

#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace serenity::denoise {
namespace {

// Widths of no network in particular: each a multiple of a channel block,
// all different, so a channel count read from the wrong layer shows.
Widths widths() {
  Widths w;
  w.enc = {24, 40, 56, 72};
  w.bottleneck = 88;
  w.dec = {80, 64, 48, 32};  // levels 3, 2, 1, 0
  return w;
}

frame::Extent half(frame::Extent e) { return {e.width / 2, e.height / 2}; }

TEST_CASE("the padded grid is the light rounded up to the U-Net's alignment") {
  for (const frame::Extent light :
       {frame::Extent{1, 1}, frame::Extent{16, 16}, frame::Extent{17, 33},
        frame::Extent{1728, 1117}}) {
    const frame::Extent padded = unet_padded(light);
    CHECK(padded.width % contracts::unet_alignment == 0);
    CHECK(padded.height % contracts::unet_alignment == 0);
    CHECK(padded.width >= light.width);
    CHECK(padded.width < light.width + contracts::unet_alignment);
    CHECK(padded.height >= light.height);
    CHECK(padded.height < light.height + contracts::unet_alignment);
  }
}

TEST_CASE("every dispatch reads and writes activations of its channels") {
  const UNet net = unet(widths(), {1728, 1117});
  const auto& a = net.activations;
  CHECK(a.at(input_activation).size == net.padded);
  CHECK(a.at(input_activation).channels == contracts::input_channels);
  for (std::size_t i = 0; i < unet_convolutions; ++i) {
    CAPTURE(i);
    const Convolution3x3& c = net.convolutions.at(i);
    CHECK(c.in_channels == a.at(c.input).channels + c.up_channels);
    CHECK(c.out_channels == a.at(c.output).channels);
    CHECK(a.at(c.output).size == a.at(c.input).size);
    const bool decoder = i >= first_decoder && i < last_convolution;
    if (decoder) {
      CHECK(c.up_channels == a.at(c.coarse).channels);
      CHECK(half(a.at(c.input).size) == a.at(c.coarse).size);
    } else {
      CHECK(c.up_channels == 0);
      CHECK(c.coarse == c.input);
    }
  }
  for (const MaxPool& p : net.pools) {
    CHECK(p.channels == a.at(p.input).channels);
    CHECK(p.channels == a.at(p.output).channels);
    CHECK(half(a.at(p.input).size) == a.at(p.output).size);
  }
}

TEST_CASE("every activation but the input is written once, the last last") {
  const UNet net = unet(widths(), {640, 360});
  std::array<int, unet_activations> writes{};
  for (const Convolution3x3& c : net.convolutions) {
    ++writes.at(c.output);
  }
  for (const MaxPool& p : net.pools) {
    ++writes.at(p.output);
  }
  CHECK(writes.at(input_activation) == 0);
  for (std::size_t i = 1; i < unet_activations; ++i) {
    CAPTURE(i);
    CHECK(writes.at(i) == 1);
  }
  CHECK(net.convolutions.at(last_convolution).output == final_activation);
}

TEST_CASE("the network's convolutions come in the order the dispatches run") {
  const Widths w = widths();
  const UNet net = unet(w, {640, 360});
  // A network whose convolutions carry the dispatches' channel counts, so
  // each handed over where another belongs shows by its counts.
  Network network;
  network.widths = w;
  const auto dims = [&](std::size_t i) {
    Convolution c;
    c.inputs = net.convolutions.at(i).in_channels;
    c.outputs = net.convolutions.at(i).out_channels;
    return c;
  };
  for (std::size_t l = 0; l < unet_levels; ++l) {
    network.enc.at(l) = dims(l);
    network.dec.at(l) = dims(first_decoder + l);
  }
  network.bottleneck_a = dims(first_bottleneck);
  network.bottleneck_b = dims(first_bottleneck + 1);
  network.dec_final = dims(last_convolution);
  const auto ordered = in_order(network);
  for (std::size_t i = 0; i < unet_convolutions; ++i) {
    CAPTURE(i);
    CHECK(ordered.at(i)->inputs == net.convolutions.at(i).in_channels);
    CHECK(ordered.at(i)->outputs == net.convolutions.at(i).out_channels);
  }
}

}  // namespace
}  // namespace serenity::denoise
