#include "metal/frame/denoise_images.h"

#include <cstring>
#include <string>
#include <type_traits>

#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

namespace {

constexpr std::uint32_t align(std::uint32_t side) {
    return (side + contracts::unet_alignment - 1u) / contracts::unet_alignment * contracts::unet_alignment;
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

DenoiseImages::DenoiseImages(const Device& device, Submission& submission, const denoise::Widths& widths)
    : device_(NS::RetainPtr(device.handle())), submission_(submission), widths_(widths) {}

void DenoiseImages::make(Buffer& buffer, std::size_t bytes, const char* what) {
    buffer.buffer = NS::TransferPtr(device_->newBuffer(bytes, MTL::ResourceStorageModeShared));
    if (!buffer.buffer) {
        throw MetalError(std::string("DenoiseImages: the device made no buffer for the ") + what);
    }
    std::memset(buffer.buffer->contents(), 0, bytes);
    buffer.resident = submission_.keep_resident(buffer.buffer.get());
}

void DenoiseImages::make(Activations& a, frame::Extent size, std::uint32_t stride, const char* what) {
    a.size = size;
    a.pitch = size.width + 2u;
    a.stride = stride;
    const std::size_t pixels = std::size_t{size.height + 2u} * a.pitch + slack_pixels;
    make(a.memory, pixels * stride * sizeof(std::uint16_t), what);
}

bool DenoiseImages::prepare(const frame::FrameInputs& inputs, frame::Extent light) {
    check_texture_size(light, "the denoiser's buffers");
    std::uint64_t held = 0;
    try {
        held = history_.join(inputs, light).held;
    } catch (const frame::HistoryError& refused) {
        throw MetalError(std::string("the denoiser's history: ") + refused.what());
    }
    index_ = inputs.index;
    if (light == light_) {
        return held > 0;
    }
    // A new size starts the history over (the rule refuses one that does
    // not); frames in flight may still read the old buffers.
    if (params_.buffer) {
        submission_.drain();
    }
    light_ = {};
    const auto pool = scoped_pool();
    padded_ = {align(light.width), align(light.height)};
    layout();
    light_ = light;
    return held > 0;
}

void DenoiseImages::layout() {
    const denoise::Widths& w = widths_;
    const auto level = [&](std::uint32_t l) { return frame::Extent{padded_.width >> l, padded_.height >> l}; };
    // The channels upsampled into level l's decoder: the bottleneck's at
    // level 3, the decoder's below it otherwise.
    const auto up = [&](std::uint32_t l) { return l == 3 ? w.bottleneck : w.dec[2 - l]; };

    make(input_, level(0), contracts::input_channels, "network's input");
    for (std::uint32_t l = 0; l < 4; ++l) {
        make(skip_[l], level(l), w.enc[l], "skip");
        make(pooled_[l], level(l + 1), w.enc[l], "max pool");
        make(decoded_[l], level(l), w.dec[3 - l], "decoder's output");
    }
    make(bottleneck_a_, level(4), w.bottleneck, "bottleneck");
    make(bottleneck_b_, level(4), w.bottleneck, "bottleneck");
    make(final_, level(0), w.dec[3], "last features");
    const std::size_t input_pixels = std::size_t{padded_.width} * padded_.height;
    make(demod_, input_pixels * 4 * sizeof(float), "demodulated light");
    for (Buffer& h : history_buffers_) {
        make(h, input_pixels * contracts::denoise_subpixels * 4 * sizeof(float), "history");
    }
    make(histogram_, contracts::median_bins * sizeof(std::uint32_t), "depth histogram");
    make(median_, sizeof(contracts::MedianState), "depth median");

    const auto address = [](const Activations& a) { return a.memory.buffer->gpuAddress(); };
    const auto conv = [&](const Activations& in, std::uint32_t in_channels, const Activations& out,
                          std::uint32_t offset, std::uint32_t outputs) {
        Conv c{};
        c.params.width = in.size.width;
        c.params.height = in.size.height;
        c.params.in_pitch = in.pitch;
        c.params.in_stride = in.stride;
        c.params.in_blocks = in_channels / contracts::channel_block;
        c.params.out_pitch = out.pitch;
        c.params.out_stride = out.stride;
        c.params.out_offset = offset;
        c.params.out_blocks = outputs / contracts::channel_block;
        c.params.chunk_blocks = chunk_blocks(c.params.out_blocks);
        c.params.relu = 1u;
        c.input = address(in);
        c.output = address(out);
        c.coarse = address(in);
        return c;
    };
    const auto resample = [&](const Activations& in, std::uint32_t in_offset, const Activations& out,
                              std::uint32_t out_offset, std::uint32_t channels) {
        Resample r{};
        r.params.width = out.size.width;
        r.params.height = out.size.height;
        r.params.channels = channels;
        r.params.in_pitch = in.pitch;
        r.params.in_stride = in.stride;
        r.params.in_offset = in_offset;
        r.params.out_pitch = out.pitch;
        r.params.out_stride = out.stride;
        r.params.out_offset = out_offset;
        r.input = address(in);
        r.output = address(out);
        return r;
    };
    // Encoders: each into its skip.
    convs_[0] = conv(input_, contracts::input_channels, skip_[0], 0, w.enc[0]);
    for (std::uint32_t l = 1; l < 4; ++l) {
        convs_[l] = conv(pooled_[l - 1], w.enc[l - 1], skip_[l], 0, w.enc[l]);
    }
    for (std::uint32_t l = 0; l < 4; ++l) {
        pools_[l] = resample(skip_[l], 0, pooled_[l], 0, w.enc[l]);
    }
    convs_[4] = conv(pooled_[3], w.enc[3], bottleneck_a_, 0, w.bottleneck);
    convs_[5] = conv(bottleneck_a_, w.bottleneck, bottleneck_b_, 0, w.bottleneck);
    // Decoders: dec i at level 3 - i, from the level below's output.
    for (std::uint32_t i = 0; i < 4; ++i) {
        const std::uint32_t l = 3 - i;
        const Activations& below = i == 0 ? bottleneck_b_ : decoded_[l + 1];
        Conv& c = convs_[6 + i];
        c = conv(skip_[l], up(l) + w.enc[l], decoded_[l], 0, w.dec[i]);
        c.params.up_blocks = up(l) / contracts::channel_block;
        c.params.up_pitch = below.pitch;
        c.params.up_stride = below.stride;
        c.coarse = address(below);
    }
    convs_[10] = conv(decoded_[0], w.dec[3], final_, 0, w.dec[3]);

    // Every dispatch's parameters, in one buffer, each at an aligned place.
    constexpr std::size_t slot = 64;
    static_assert(sizeof(contracts::ConvParams) <= slot && sizeof(contracts::ResampleParams) <= slot);
    make(params_, slot * (convolutions + resamplings), "dispatch parameters");
    std::byte* bytes = static_cast<std::byte*>(params_.buffer->contents());
    std::size_t at = 0;
    const auto place = [&](auto& dispatch) {
        std::memcpy(bytes + at, &dispatch.params, sizeof(dispatch.params));
        dispatch.params_address = params_.buffer->gpuAddress() + at;
        at += slot;
    };
    for (Conv& c : convs_) {
        place(c);
    }
    for (Resample& r : pools_) {
        place(r);
    }
}

MTL::GPUAddress DenoiseImages::input() const noexcept {
    return input_.memory.buffer ? input_.memory.buffer->gpuAddress() : 0;
}
MTL::GPUAddress DenoiseImages::features() const noexcept {
    return final_.memory.buffer ? final_.memory.buffer->gpuAddress() : 0;
}
MTL::GPUAddress DenoiseImages::demod() const noexcept {
    return demod_.buffer ? demod_.buffer->gpuAddress() : 0;
}
MTL::GPUAddress DenoiseImages::histogram() const noexcept {
    return histogram_.buffer ? histogram_.buffer->gpuAddress() : 0;
}
MTL::GPUAddress DenoiseImages::median() const noexcept {
    return median_.buffer ? median_.buffer->gpuAddress() : 0;
}
MTL::GPUAddress DenoiseImages::history() const noexcept {
    const Buffer& b = history_buffers_[(index_ + 1) % 2];
    return b.buffer ? b.buffer->gpuAddress() : 0;
}
MTL::GPUAddress DenoiseImages::next_history() const noexcept {
    const Buffer& b = history_buffers_[index_ % 2];
    return b.buffer ? b.buffer->gpuAddress() : 0;
}

}  // namespace serenity::metal
