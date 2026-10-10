#include "metal/scene/scene_buffers.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>

#include "metal/device/error.h"

namespace serenity::metal {

namespace {

using Block = gpu::SceneBlock;
using Scene = scene::SceneDescription;

// The scene's layouts are copied as bytes, so each must be one whose bytes
// are its value (SL.con.4, COPY.6).
template <typename T>
std::span<const std::byte> bytes(const std::vector<T>& array) {
    static_assert(std::is_trivially_copyable_v<T>, "a scene array is copied to the GPU as its bytes");
    return std::as_bytes(std::span(array));
}

template <typename T>
std::span<const std::byte> bytes(const T& one) {
    static_assert(std::is_trivially_copyable_v<T>, "a scene record is copied to the GPU as its bytes");
    return std::as_bytes(std::span(&one, 1));
}

// The one list of the scene's arrays: each with the field of the scene's
// block (scene_block.h) that holds its address, named, so no array can be
// given another's address. Checked at compile time rather than at run time
// (P.5), below: there is an entry for every field of the block, since the
// list's length is the count of entries written and the block holds that
// many addresses; and no entry is empty or names a field another names.
struct Entry {
    std::uint64_t Block::*field;
    std::span<const std::byte> (*bytes)(const Scene&);
};

constexpr auto entries = std::to_array<Entry>({
    {&Block::environment, [](const Scene& s) { return bytes(s.environment); }},
    {&Block::textures, [](const Scene& s) { return bytes(s.textures); }},
    {&Block::checkers, [](const Scene& s) { return bytes(s.checkers); }},
    {&Block::woods, [](const Scene& s) { return bytes(s.woods); }},
    {&Block::swirls, [](const Scene& s) { return bytes(s.swirls); }},
    {&Block::materials, [](const Scene& s) { return bytes(s.materials); }},
    {&Block::roughs, [](const Scene& s) { return bytes(s.roughs); }},
    {&Block::dielectrics, [](const Scene& s) { return bytes(s.dielectrics); }},
    {&Block::conductors, [](const Scene& s) { return bytes(s.conductors); }},
    {&Block::emissives, [](const Scene& s) { return bytes(s.emissives); }},
    {&Block::coateds, [](const Scene& s) { return bytes(s.coateds); }},
    {&Block::media, [](const Scene& s) { return bytes(s.media); }},
    {&Block::absorbings, [](const Scene& s) { return bytes(s.absorbings); }},
    {&Block::shapes, [](const Scene& s) { return bytes(s.shapes.records); }},
    {&Block::boxes, [](const Scene& s) { return bytes(s.shapes.boxes); }},
    {&Block::light_records, [](const Scene& s) { return bytes(s.lights); }},
    {&Block::shape_lights, [](const Scene& s) { return bytes(s.shape_lights); }},
    {&Block::sphere_lights, [](const Scene& s) { return bytes(s.sphere_lights); }},
    {&Block::light_counts, [](const Scene& s) { return bytes(s.light_counts); }},
});

constexpr bool every_entry_its_own() {
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].field == nullptr || entries[i].bytes == nullptr) {
            return false;
        }
        for (std::size_t j = i + 1; j < entries.size(); ++j) {
            if (entries[i].field == entries[j].field) {
                return false;
            }
        }
    }
    return true;
}

static_assert(entries.size() * sizeof(std::uint64_t) == sizeof(Block),
              "every field of gpu::SceneBlock needs its entry in `entries`");
static_assert(every_entry_its_own(), "an entry of `entries` is empty, or names a field another names");
static_assert(std::is_trivially_copyable_v<Block>, "the block is copied to the GPU as its bytes");

std::array<std::span<const std::byte>, entries.size()> arrays_of(const Scene& scene) {
    std::array<std::span<const std::byte>, entries.size()> arrays;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        arrays[i] = entries[i].bytes(scene);
    }
    return arrays;
}

}  // namespace

SceneBuffers::SceneBuffers(const Device& device, Submission& submission, const scene::SceneDescription& scene)
    : arrays_(device, submission, arrays_of(scene)) {
    for (std::size_t i = 0; i < entries.size(); ++i) {
        block_.*entries[i].field = arrays_.address(i);
    }
    // The block itself, written once (scene_block.h); shared storage, which
    // Apple silicon's unified memory lets the GPU read where the CPU wrote.
    block_buffer_ = NS::TransferPtr(device.handle()->newBuffer(sizeof(block_), MTL::ResourceStorageModeShared));
    if (!block_buffer_) {
        throw MetalError("SceneBuffers: the device made no buffer for the scene's block");
    }
    std::memcpy(block_buffer_->contents(), &block_, sizeof(block_));
    resident_ = submission.keep_resident(block_buffer_.get());
}

}  // namespace serenity::metal
