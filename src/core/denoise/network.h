#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/contracts/denoise.h"

namespace serenity::denoise {

// Axis: Pass (the denoiser's network: its weights, read from its export).
//
// The trained network as training/export.py writes it, into a directory:
// weights.bin, every parameter as little-endian float32 in PyTorch's order
// (a convolution's weight [out][in][ky][kx], its bias [out]), and
// layout.json, each tensor's name, shape, byte offset and count, with the
// network's widths and the constants its pre- and post-processing use.
// Read here, in the core, once, at start-up, so every backend runs the same
// numbers; a backend lays them out for its own kernels
// (metal/passes/denoise/denoise.h).
//
// The structure is training/model.py's, fixed (core/contracts/denoise.h):
// four encoder convolutions (3 x 3, ReLU, each kept as a skip and max
// pooled), two at the bottleneck, four decoder convolutions on each
// upsampled level concatenated with its skip, one more at the input's
// level, and the 1 x 1 head of contracts::head_outputs scores. Only the
// widths vary, read from the layout: the demo's network and the test
// fixture's small random one load alike (the graph's [denoise] weights,
// core/passes/denoise.h).
//
// What is checked, each a NetworkError naming the file and what is wrong,
// never a network half-loaded or run on other constants (I.5, E.2):
//
//   - layout.json is JSON (json.h), of format "serenity-denoiser-weights/1",
//     float32, little-endian;
//   - its constants are contract 18's: 30 input channels (a network
//     trained before the camera's jitter was an input has 28, and is
//     refused by this, named: inputs.channels), scale 2, 124
//     scores, spacings 1, 2 and 4, the masked score, the albedo floor and
//     PU's ten constants, its normalization among them, each equal as a
//     float;
//   - its widths are positive multiples of contracts::channel_block, which
//     the kernels' blocks of channels need (both trained networks' are);
//   - every tensor the structure needs is listed once, with the shape its
//     widths give, and no other; each lies inside weights.bin, whose size
//     is their sum exactly; every value is finite.
//
// The export's SHA256SUMS is not checked: the file sizes and shapes are,
// and a network that is the wrong one but well-formed is the graph's to
// name, as a scene file is.
//
// Not performance-sensitive: about 2 MB read once.

class NetworkError : public std::runtime_error {
 public:
  explicit NetworkError(const std::string& what) : std::runtime_error(what) {}
};

// One convolution's parameters, in PyTorch's order.
struct Convolution {
  std::uint32_t inputs = 0;
  std::uint32_t outputs = 0;
  std::uint32_t kernel = 0;   // 3, or 1 for the head
  std::vector<float> weight;  // [outputs][inputs][kernel][kernel]
  std::vector<float> bias;    // [outputs]
};

// training/model.py's Widths: one encoder and one decoder width a level
// (contracts::unet_levels).
struct Widths {
  std::array<std::uint32_t, contracts::unet_levels> enc{};
  std::uint32_t bottleneck = 0;
  std::array<std::uint32_t, contracts::unet_levels> dec{};  // levels 3, 2, 1, 0
};

struct Network {
  Widths widths;
  std::array<Convolution, contracts::unet_levels> enc;
  Convolution bottleneck_a;
  Convolution bottleneck_b;
  std::array<Convolution, contracts::unet_levels> dec;
  Convolution dec_final;
  Convolution head;
  std::string source;  // the directory read, for messages
};

// Reads the network exported into `directory` (above). Throws NetworkError.
Network load_network(const std::filesystem::path& directory);

// Reads a network from the two files' contents, naming it `source` in
// errors: for tests, and for load_network() itself.
Network parse_network(const std::string& layout_json,
                      std::span<const std::byte> weights,
                      const std::string& source);

}  // namespace serenity::denoise
