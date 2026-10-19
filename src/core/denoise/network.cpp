#include "core/denoise/network.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>
#include <utility>

#include "core/contracts/denoise.h"
#include "core/denoise/json.h"

namespace serenity::denoise {

namespace {

static_assert(std::endian::native == std::endian::little,
              "weights.bin is little-endian float32, read as it is");
static_assert(std::numeric_limits<float>::is_iec559,
              "weights.bin's float32 is IEEE 754 binary32");

constexpr std::string_view format = "serenity-denoiser-weights/1";

// The largest count a double holds with every integer below it exact, 2^53.
constexpr double largest_exact_count = 9007199254740992.0;

// The widest layer the loader takes: far past both trained networks' 96,
// low enough that no shape's product can overflow.
constexpr std::uint64_t max_width = 4096;

[[noreturn]] void refuse(const std::string& source, const std::string& what) {
  throw NetworkError("the denoiser's network " + source + ": " + what);
}

// A count the layout writes: a non-negative integer a double holds exactly.
std::uint64_t whole(const Json& node, std::string_view what,
                    const std::string& source) {
  const double n = node.number(what);
  if (!(n >= 0.0 && n <= largest_exact_count && std::floor(n) == n)) {
    refuse(source, std::string(what) + " is not a count");
  }
  return static_cast<std::uint64_t>(n);
}

std::uint32_t width(const Json& node, std::string_view what,
                    const std::string& source) {
  const std::uint64_t n = whole(node, what, source);
  if (n == 0 || n > max_width || n % contracts::channel_block != 0) {
    refuse(source, std::string(what) + ", " + std::to_string(n) +
                       ", is not a positive multiple of " +
                       std::to_string(contracts::channel_block) + " up to " +
                       std::to_string(max_width) +
                       ", which the kernels' blocks of channels need");
  }
  return static_cast<std::uint32_t>(n);
}

Widths read_widths(const Json& config, const std::string& source) {
  const Json& widths = config.at("widths");
  Widths w;
  const JsonArray& enc = widths.at("enc").array("widths.enc");
  const JsonArray& dec = widths.at("dec").array("widths.dec");
  if (enc.size() != contracts::unet_levels ||
      dec.size() != contracts::unet_levels) {
    refuse(source, "widths.enc and widths.dec must have " +
                       std::to_string(contracts::unet_levels) +
                       " entries each");
  }
  for (std::size_t i = 0; i < contracts::unet_levels; ++i) {
    w.enc.at(i) = width(enc[i], "a widths.enc entry", source);
    w.dec.at(i) = width(dec[i], "a widths.dec entry", source);
  }
  w.bottleneck = width(widths.at("bottleneck"), "widths.bottleneck", source);
  return w;
}

// The constants the layout writes, each contract 18's (network.h).
void check_constants(const Json& layout, const std::string& source) {
  const auto same = [&](const Json& node, float expected,
                        std::string_view what) {
    if (static_cast<float>(node.number(what)) != expected) {
      refuse(source, std::string(what) + " is not the " +
                         std::format("{}", expected) +
                         " this renderer runs with");
    }
  };
  if (layout.at("format").string("format") != format) {
    refuse(source, "its format is not " + std::string(format));
  }
  if (layout.at("dtype").string("dtype") != "float32" ||
      layout.at("byte_order").string("byte_order") != "little") {
    refuse(source, "its weights are not little-endian float32");
  }
  same(layout.at("output_scale"), static_cast<float>(contracts::denoise_scale),
       "output_scale");
  const Json& inputs = layout.at("inputs");
  same(inputs.at("channels"), static_cast<float>(contracts::network_inputs),
       "inputs.channels");
  same(inputs.at("albedo_floor"), contracts::albedo_floor,
       "inputs.albedo_floor");
  const Json& pu = inputs.at("pu");
  same(pu.at("a"), contracts::pu_a, "pu.a");
  same(pu.at("b"), contracts::pu_b, "pu.b");
  same(pu.at("c"), contracts::pu_c, "pu.c");
  same(pu.at("d"), contracts::pu_d, "pu.d");
  same(pu.at("e"), contracts::pu_e, "pu.e");
  same(pu.at("f"), contracts::pu_f, "pu.f");
  same(pu.at("g"), contracts::pu_g, "pu.g");
  same(pu.at("y0"), contracts::pu_y0, "pu.y0");
  same(pu.at("y1"), contracts::pu_y1, "pu.y1");
  same(pu.at("norm_scale"), contracts::pu_norm_scale, "pu.norm_scale");
  const Json& scores = layout.at("scores");
  same(scores.at("channels"), static_cast<float>(contracts::head_outputs),
       "scores.channels");
  same(scores.at("masked_tap_score"), contracts::masked_score,
       "scores.masked_tap_score");
  const JsonArray& spacings = scores.at("spacings").array("scores.spacings");
  // The contract's list, a C array for the shaders (contracts/denoise.h), as
  // a std::array here, to read with bounds checked (Bounds.2).
  constexpr auto spacing_pixels = std::to_array(contracts::spacing_pixels);
  bool same_spacings = spacings.size() == contracts::denoise_spacings;
  for (std::size_t s = 0; same_spacings && s < spacings.size(); ++s) {
    same_spacings = spacings[s].number("a spacing") ==
                    static_cast<double>(spacing_pixels.at(s));
  }
  if (!same_spacings) {
    std::string listed;
    for (std::size_t s = 0; s < contracts::denoise_spacings; ++s) {
      listed += (s == 0                                 ? ""
                 : s + 1 == contracts::denoise_spacings ? " and "
                                                        : ", ") +
                std::to_string(spacing_pixels.at(s));
    }
    refuse(source, "its spacings are not contract 18's " + listed);
  }
}

// What the structure needs, by name, in the order model.py's state_dict
// lists it: each convolution's (name, inputs, outputs, kernel).
struct Needed {
  std::string name;
  std::uint32_t inputs;
  std::uint32_t outputs;
  std::uint32_t kernel;
  Convolution* into;
};

std::vector<Needed> needed(Network& n) {
  const Widths& w = n.widths;
  // Each level's convolution both ways, and four more: the bottleneck's two,
  // the decoder's last and the head.
  constexpr std::size_t convolutions = 2 * contracts::unet_levels + 4;
  std::vector<Needed> list;
  list.reserve(convolutions);
  constexpr std::size_t deepest =
      contracts::unet_levels -
      1;  // the encoder's last level, the decoder's first
  for (std::size_t i = 0; i < contracts::unet_levels; ++i) {
    list.push_back({"unet.enc." + std::to_string(i),
                    i == 0 ? contracts::network_inputs : w.enc.at(i - 1),
                    w.enc.at(i), 3, &n.enc.at(i)});
  }
  list.push_back(
      {"unet.bottleneck_a", w.enc[deepest], w.bottleneck, 3, &n.bottleneck_a});
  list.push_back(
      {"unet.bottleneck_b", w.bottleneck, w.bottleneck, 3, &n.bottleneck_b});
  std::uint32_t previous = w.bottleneck;
  for (std::size_t i = 0; i < contracts::unet_levels; ++i) {
    list.push_back({"unet.dec." + std::to_string(i),
                    previous + w.enc.at(deepest - i), w.dec.at(i), 3,
                    &n.dec.at(i)});
    previous = w.dec.at(i);
  }
  list.push_back(
      {"unet.dec_final", w.dec[deepest], w.dec[deepest], 3, &n.dec_final});
  list.push_back(
      {"unet.head", w.dec[deepest], contracts::head_outputs, 1, &n.head});
  return list;
}

// The tensor `name` of `tensors`, read from `weights`, its shape checked
// against `shape`; each tensor is taken once.
std::vector<float> tensor(const JsonArray& tensors, std::vector<bool>& taken,
                          const std::string& name,
                          const std::vector<std::uint64_t>& shape,
                          std::span<const std::byte> weights,
                          const std::string& source) {
  for (std::size_t t = 0; t < tensors.size(); ++t) {
    if (tensors[t].at("name").string("a tensor's name") != name) {
      continue;
    }
    if (taken[t]) {
      refuse(source, "the tensor " + name + " is listed twice");
    }
    taken[t] = true;
    const JsonArray& dims = tensors[t].at("shape").array(name + "'s shape");
    std::uint64_t count = 1;
    bool matches = dims.size() == shape.size();
    for (std::size_t d = 0; matches && d < dims.size(); ++d) {
      matches = whole(dims[d], name + "'s shape", source) == shape[d];
      count *= shape[d];
    }
    if (!matches) {
      refuse(source,
             "the tensor " + name + " has another shape than its widths give");
    }
    const std::uint64_t offset =
        whole(tensors[t].at("offset"), name + "'s offset", source);
    if (whole(tensors[t].at("count"), name + "'s count", source) != count ||
        offset % sizeof(float) != 0 || offset > weights.size() ||
        count > (weights.size() - offset) / sizeof(float)) {
      refuse(source, "the tensor " + name +
                         " does not lie inside weights.bin as its count says");
    }
    std::vector<float> values(static_cast<std::size_t>(count));
    const std::span<const std::byte> tensor = weights.subspan(
        static_cast<std::size_t>(offset), values.size() * sizeof(float));
    std::memcpy(values.data(), tensor.data(), tensor.size());
    for (float v : values) {
      if (!std::isfinite(v)) {
        refuse(source,
               "the tensor " + name + " holds a value that is not finite");
      }
    }
    return values;
  }
  refuse(source, "it has no tensor " + name);
}

// The file's bytes, read once; the weights are viewed as bytes where they
// are (std::as_bytes), not copied again. Sized from the file, and read
// whole or refused, as load_schedule() and scene::load() read theirs: a
// read that fails, a directory, or a file that ends early is "cannot
// read", never a shorter file that surfaces later as a size or JSON
// mistake it is not (I.10, SL.io.2).
std::string read_file(const std::filesystem::path& path,
                      const std::string& source) {
  const std::string cannot = "cannot read " + path.filename().string();
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  std::ifstream file{path, std::ios::binary};
  if (error || !file ||
      size > static_cast<std::uintmax_t>(
                 std::numeric_limits<std::streamsize>::max())) {
    refuse(source, cannot);
  }
  std::string bytes(static_cast<std::size_t>(size), '\0');
  file.read(bytes.data(), static_cast<std::streamsize>(size));
  if (!file || file.gcount() != static_cast<std::streamsize>(size)) {
    refuse(source, cannot);
  }
  return bytes;
}

}  // namespace

Network parse_network(const std::string& layout_json,
                      std::span<const std::byte> weights,
                      const std::string& source) {
  Json layout;
  try {
    layout = parse_json(layout_json);
  } catch (const JsonError& error) {
    refuse(source, std::string("layout.json: ") + error.what());
  }
  try {
    check_constants(layout, source);
    Network network;
    network.source = source;
    network.widths = read_widths(layout.at("config"), source);
    const JsonArray& tensors = layout.at("tensors").array("tensors");
    std::vector<bool> taken(tensors.size(), false);
    std::uint64_t total = 0;
    for (const Needed& need : needed(network)) {
      Convolution& c = *need.into;
      c.inputs = need.inputs;
      c.outputs = need.outputs;
      c.kernel = need.kernel;
      c.weight = tensor(tensors, taken, need.name + ".weight",
                        {need.outputs, need.inputs, need.kernel, need.kernel},
                        weights, source);
      c.bias = tensor(tensors, taken, need.name + ".bias", {need.outputs},
                      weights, source);
      total += (c.weight.size() + c.bias.size()) * sizeof(float);
    }
    for (std::size_t t = 0; t < tensors.size(); ++t) {
      if (!taken[t]) {
        refuse(source, "it lists a tensor the structure has no place for, " +
                           tensors[t].at("name").string("a tensor's name"));
      }
    }
    if (total != weights.size()) {
      refuse(source, "weights.bin is " + std::to_string(weights.size()) +
                         " bytes, and its tensors " + std::to_string(total));
    }
    return network;
  } catch (const JsonError& error) {
    refuse(source, std::string("layout.json: ") + error.what());
  }
}

Network load_network(const std::filesystem::path& directory) {
  const std::string source = directory.string();
  const std::string layout = read_file(directory / "layout.json", source);
  const std::string weights = read_file(directory / "weights.bin", source);
  return parse_network(layout, std::as_bytes(std::span(weights)), source);
}

}  // namespace serenity::denoise
