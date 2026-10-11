#pragma once

#include <array>
#include <cstdint>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/history.h"
#include "core/passes/tone_map.h"
#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Frame graph (the images between passes: contract 6).
//
// The images a frame's passes hand one another within the frame, as the
// schedule says they do (core/frame/schedule.h):
//
//   - the radiance image: the frame's linear radiance, RGBA32Float at the
//     frame's size, written by the pass that computes light (preview) and
//     read by the pass that presents it (display, tone_map). 32-bit, not
//     16: the display pass must show exactly what the light pass computed,
//     as the passes did when they encoded the target themselves, and a
//     firefly's radiance of hundreds keeps its digits. A light pass that
//     accumulates (path) has none here: its accumulated image, the same
//     format at the same size, is the frame's radiance (accumulation.h,
//     passes/path/path.h), so the mean is written once (GDSA.6);
//   - the bloom pyramid, for the tone-map pass alone: passes::bloom_levels
//     levels (core/passes/tone_map.h), each max(1, ceil(previous / 2)) on
//     each axis from the frame's size, so any frame has every level;
//     RGBA16Float, which a blur of light needs no more than (half's 11 bits
//     are 3 decimal digits), the light kept within its range
//     (core/passes/tone_map.h, steps 1 to 3).
//
// Made when the frame's size is first known and remade when it changes,
// before the frame's submission begins (Renderer::prepare), draining the
// submission first, as the accumulated image is (accumulation.h): frames in
// flight may still read the old images. Private storage, the GPU's alone,
// made resident through the submission. Only what the schedule uses is
// made: a schedule with no light pass makes none; one without tone_map, no
// pyramid.
//
// One set, not one per frame in flight: the frames in flight share it, and
// what keeps one frame's passes from writing it while the frame before still
// reads it is a barrier the renderer records before the first pass of a
// frame that writes it, waiting for the queue's earlier dispatches
// (metal/frame/renderer.h). So the CPU still records a frame while the GPU
// runs the one before; what the barrier gives up is only the GPU starting a
// frame's first dispatch before the last frame's final one ends. A copy per
// frame in flight would keep that sliver of overlap at twice the memory,
// 290 MB at the display's size, and the overlap is no gain: two frames'
// passes running at once take longer than one after the other (measured,
// docs/research/2026-10-09-pass-costs.md).
//
// Made outside the frame loop, at a size, and kept until the size changes
// (GPU.9); nothing else lives only within a frame for them to share memory
// with yet. The radiance image is an intermediate the frame writes and reads
// back, which GDSA.16 says to stream through on-chip memory instead: the
// tone map cannot, since its bloom reads every pixel's neighbourhood as far
// as B_5's 64 pixels; the display pass could fold back into the light pass,
// as it was, and is kept apart so a graph has one presenting pass
// (passes/display/display.h), at the cost derived below; it has not been
// measured on its own (docs/research/2026-10-09-pass-costs.md).
//
// Cost: at 3456 x 2234 (P = 7.7 M pixels), the radiance image is 123 MB,
// when the light pass does not accumulate, and the pyramid 21 MB. Per
// frame, the light pass writes the radiance image once, P texels; the
// display pass reads it once, P texels, 247 MB of traffic in all, some
// 0.6 ms of the M3 Max's 400 GB/s. The tone-map pass reads it twice, in two
// dispatches ten apart, too far for one read to leave it cached for the
// other: 13 filtered samples for each of B_0's P / 4 texels, then P exact
// reads (passes/tone_map/tone_map.h); what that costs in memory traffic is
// the pass's measured time, not a count here.
//
// Throws MetalError if the frame's size is not one Metal makes an image of
// (metal/device/device.h, max_texture_side), or the device cannot make one.
class FrameImages {
public:
    // Whether the frame's radiance is an image of its own, or the light
    // pass's accumulated image (accumulation.h), which this then does not
    // make. Types, not bools, so the call says which (I.4).
    enum class Radiance { image, accumulated };

    // Whether the schedule has the tone-map pass, which needs the pyramid.
    enum class Bloom { none, pyramid };

    // The radiance image as `radiance` says, and the pyramid as `bloom`
    // says (a schedule with no light pass makes no FrameImages).
    FrameImages(const Device& device, Submission& submission, Radiance radiance, Bloom bloom);

    FrameImages(const FrameImages&) = delete;
    FrameImages& operator=(const FrameImages&) = delete;
    FrameImages(FrameImages&&) = delete;
    FrameImages& operator=(FrameImages&&) = delete;
    ~FrameImages() = default;

    // Ready for a frame of `size`: made, or remade, if not made at it.
    void prepare(frame::Extent size);

    // Null when the schedule does not use it, or before the first prepare().
    // bloom() throws MetalError for a level past passes::bloom_levels.
    MTL::Texture* radiance() const noexcept { return radiance_.texture.get(); }
    MTL::Texture* bloom(std::uint32_t level) const;

private:
    NS::SharedPtr<MTL::Device> device_;
    Submission& submission_;
    Radiance radiance_kind_;
    Bloom bloom_;
    frame::Extent size_;
    // Each image with its residency, released after the GPU is done with it
    // (submission.h, Lifetime).
    struct Image {
        NS::SharedPtr<MTL::Texture> texture;
        Resident resident;
    };
    Image radiance_;
    std::array<Image, passes::bloom_levels> pyramid_;
};

// Axis: Frame graph (ReSTIR DI's images between passes and across frames:
// contract 6, holding contracts 5 and 14).
//
// The arrays ReSTIR DI's passes hand one another (core/frame/schedule.h),
// made only for a graph that has those passes, so a graph without them
// allocates what it did and renders what it did
// (docs/research/2026-10-10-restir-di-layout.md, "Additive"). Buffers, not
// textures: each element is a shared layout of 12 to 80 bytes, read whole,
// flat in row-major pixel order (contracts/surface_record.h,
// contracts/reservoir.h; GPU.2).
//
//   records             contract 5's SurfaceRecord, 80 bytes a pixel; two,
//                       ping-ponged, in a graph with a temporal merge: this
//                       frame's, which the initial pass writes, and last
//                       frame's, which the temporal merge reads (MEM.4: data
//                       that crosses one frame, double-buffered); one
//                       otherwise.
//   path states         contract 5's PathState, 64 bytes a pixel, one.
//   previous positions  a Float3, 12 bytes a pixel, one, only in a graph
//                       with a temporal merge.
//   working reservoirs  contract 14, 32 bytes a pixel, one, only in a graph
//                       with a merge: written by the initial pass, merged in
//                       place by the temporal merge, read at neighbors by
//                       the spatial merge.
//   final reservoirs    contract 14, 32 bytes a pixel: what the last pass
//                       before the shade writes, which the shade reads; two,
//                       ping-ponged, in a graph with a temporal merge, so
//                       last frame's survives until this frame's temporal
//                       merge has read it, whichever pass writes this
//                       frame's (a temporal merge with no spatial merge
//                       after it writes its output there while it reads
//                       last frame's at other pixels); one otherwise.
//
// Which half of a ping-pong is this frame's is the frame's index's parity:
// the reuse history holds only consecutive frames (core/frame/history.h),
// so last frame's half is always the other.
//
// The history. Which frames the records and final reservoirs hold is the
// core's rule (frame::ReuseHistory), kept by the one this object owns, as
// Accumulation keeps frame::History (accumulation.h): prepare() asks it
// whether the frame joins, refuses by MetalError with the rule's reason
// what it refuses, and returns whether the frame has a previous one to
// merge, for the reuse constants (core/contracts/reuse_constants.h).
// Where the rule says the history starts over at another size, the arrays
// are made anew: first draining the submission, as the accumulated image
// is remade, since frames in flight may still read the old ones; before
// the frame's submission begins (Renderer::prepare). Private storage, the
// GPU's alone, made resident through the submission, made outside the
// frame loop and kept until the size changes (GPU.9, MEM.9). Nothing in
// them is cleared when made: a frame that starts the history over reads no
// last frame's half, and every pass writes every pixel of what it writes.
//
// One set, not one per frame in flight, as FrameImages: the renderer's
// queue barrier before the initial pass keeps a frame from writing what
// the frame before still reads (metal/frame/renderer.h).
//
// Cost, memory, at P pixels, for a graph with all four passes: 160 + 64 +
// 12 + 32 + 64 = 332 bytes a pixel, 2.56 GB at the display's 3456 x 2234
// and 641 MB at a traced 1728 x 1117; the surface records are half of it.
// Per frame, what the passes read and write of it is counted in
// metal/passes/restir_di/restir_di.metal.h. No CPU-GPU crossing: the CPU never
// reads them.
//
// Throws MetalError if the frame's size is not one Metal makes an image of
// (metal/device/device.h, max_texture_side; the arrays are sized from it,
// as the images are), or the device cannot make a buffer.
class ReuseImages {
public:
    // Whether the graph has restir_temporal: two halves of the records and
    // final reservoirs, and the previous positions. Types, not bools (I.4).
    enum class Temporal { none, merged };

    // Whether a merge follows the initial pass: the working reservoirs.
    enum class Working { none, needed };

    ReuseImages(const Device& device, Submission& submission, Temporal temporal, Working working);

    ReuseImages(const ReuseImages&) = delete;
    ReuseImages& operator=(const ReuseImages&) = delete;
    ReuseImages(ReuseImages&&) = delete;
    ReuseImages& operator=(ReuseImages&&) = delete;
    ~ReuseImages() = default;

    // Ready for frame `inputs` at `size`; see above. Returns the frames the
    // history holds before this one (0: it starts over, and last frame's
    // halves are not to be read). Throws MetalError if the core's rule
    // refuses the frame, or an array cannot be made.
    std::uint64_t prepare(const frame::FrameInputs& inputs, frame::Extent size);

    // The arrays frame `index` reads and writes, as a shader binds them:
    // this frame's halves and last frame's by its parity; 0 for what the
    // graph does not have, or before the first prepare().
    struct Addresses {
        MTL::GPUAddress records = 0;
        MTL::GPUAddress last_records = 0;
        MTL::GPUAddress path_states = 0;
        MTL::GPUAddress previous_positions = 0;
        MTL::GPUAddress working = 0;
        MTL::GPUAddress final_reservoirs = 0;
        MTL::GPUAddress last_final = 0;
    };
    Addresses addresses(std::uint64_t index) const noexcept;

private:
    struct Array {
        NS::SharedPtr<MTL::Buffer> buffer;
        Resident resident;  // released after the GPU is done with it (submission.h, Lifetime)
    };

    NS::SharedPtr<MTL::Device> device_;
    Submission& submission_;
    Temporal temporal_;
    Working working_;
    std::array<Array, 2> records_;  // the second only with a temporal merge
    Array path_states_;
    Array previous_positions_;
    Array working_reservoirs_;
    std::array<Array, 2> final_reservoirs_;  // the second only with a temporal merge
    frame::ReuseHistory history_;  // the core's rule, and what the arrays hold
};

}  // namespace serenity::metal
