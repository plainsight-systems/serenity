#pragma once

// Axis: Pass (every pass): where each kernel's arguments are bound.
//
// The argument-table index of every buffer and texture each pass's kernels
// take, named once and read by both sides: a kernel's parameter list in its
// .metal file ([[buffer(n)]], [[texture(n)]]) and its pass's record() in
// its .cpp (MTL4::ArgumentTable::setAddress, setResource, setTexture). A
// binding moved here moves on both sides, so they cannot drift apart
// (ES.45, P.5); before this file each side wrote its numbers and a comment
// said they matched.
//
// Plain enumerations, one namespace per kernel, of a fixed underlying type:
// the shading language's attributes and Metal's setters both take an index
// as an integer, which a scoped enumeration would need a cast for at every
// use; the values are the binding layout (Enum.8). Buffers and textures are
// numbered apart, as Metal numbers them.

namespace serenity {
namespace bindings {

namespace test_pattern {
enum Buffer : unsigned { constants = 0 };
enum Texture : unsigned { target = 0 };
}  // namespace test_pattern

// The preview and the path tracer read the scene alike: the frame's
// constants and camera, the structure, the scene's block whole
// (metal/scene/scene_block.h), and what changes per frame beside it, the
// sphere lights' glows and the shapes' transforms.
namespace preview {
enum Buffer : unsigned { constants = 0, camera = 1, structure = 2, scene = 3, glows = 4, transforms = 5 };
enum Texture : unsigned { radiance = 0 };
}  // namespace preview

namespace path {
enum Buffer : unsigned {
    constants = 0,
    camera = 1,
    structure = 2,
    scene = 3,
    glows = 4,
    transforms = 5,
    non_finite = 6,  // the frame's count of samples not finite (metal/film/non_finite.h)
};
enum Texture : unsigned { accumulated = 0, radiance = 1 };
}  // namespace path

namespace display {
enum Buffer : unsigned { constants = 0 };
enum Texture : unsigned { radiance = 0, target = 1 };
}  // namespace display

// The tone map's four kernels (passes/tone_map/tone_map.h): each reads its
// input at texture 0 and writes at 1; the last reads B_0 at 1 and writes
// the target at 2.
namespace tone_map {
enum Buffer : unsigned { settings = 0 };
enum Texture : unsigned { input = 0, level = 1, bloom = 1, target = 2 };
}  // namespace tone_map

}  // namespace bindings
}  // namespace serenity
