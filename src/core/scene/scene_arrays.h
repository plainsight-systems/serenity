#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "core/scene/scene.h"
#include "core/scene/scene_block.h"

namespace serenity::scene {

// Axis: Scene content (which of the scene's arrays the GPU holds).
//
// The one list of the scene description's arrays (scene.h) that a backend
// puts on the GPU, in the order of the scene's block (scene_block.h), each
// with the block's field that holds its address. A new kind's array is
// added here and to the block, as it is to the description, and no
// backend's pass changes; how a backend allocates, aligns and keeps the
// arrays is its own (Metal's: metal/device/static_arrays.h). Falcor's
// Scene keeps the same list, in its scene's parameter block.
//
// The core's (A.1): which arrays the scene has is the same on every
// backend, so no backend writes the list again. Checked at compile time
// (P.5): there is an entry for every field of the block, and no two
// entries name one field (scene_arrays.cpp).
//
// The layouts are the core's shared ones, so the bytes are copied as they
// are: nothing is converted, and the shaders read exactly what the scene
// reader wrote (SL.con.4, COPY.6: each is trivially copyable, asserted
// where its bytes are taken). An array with no elements is an empty span;
// a backend gives it an address all the same (Metal's, a zeroed block):
// a shader indexes it only through a record that names its kind, and no
// record names a kind the scene has none of.

// How many arrays the block names, one address each.
inline constexpr std::size_t scene_array_count =
    sizeof(gpu::SceneBlock) / sizeof(std::uint64_t);

// The scene's arrays as bytes, in the block's order: views into `scene`,
// valid while it is.
std::array<std::span<const std::byte>, scene_array_count> scene_arrays(
    const SceneDescription& scene);

// The block whose fields are `addresses`, given in the block's order: the
// address a backend put scene_arrays()'s array i at, at i.
gpu::SceneBlock scene_block(
    const std::array<std::uint64_t, scene_array_count>& addresses);

}  // namespace serenity::scene
