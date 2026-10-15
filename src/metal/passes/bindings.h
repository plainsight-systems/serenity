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
    mnee_requests = 7,  // with mnee: MNEE's work list (contract 16)
    mnee_header = 8,    // its count and capacity
};
enum Texture : unsigned { accumulated = 0 };  // also the frame's radiance (path/path.h)
}  // namespace path

// The MNEE pass's three kernels (passes/mnee/mnee.h), recorded by the path
// pass and ReSTIR PT's initial pass after their own: the scene at the path
// tracer's indices, then the work list, its header, its indirect grid, and
// the light each request adds to its pixel (a float4 a pixel: the path
// pass's MNEE light, folded into the accumulated image at texture 0, or
// ReSTIR PT's light buffer).
namespace mnee {
enum Buffer : unsigned {
    constants = 0,
    structure = 2,
    scene = 3,
    glows = 4,
    transforms = 5,
    non_finite = 6,
    requests = 7,
    header = 8,
    grid = 9,
    light = 10,
};
enum Texture : unsigned { accumulated = 0 };
}  // namespace mnee

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

// ReSTIR DI's four kernels (core/frame/schedule.h). Each reads the scene as
// the path tracer does, at the same indices 0 to 5 (the camera bound only
// by those that read it: initial, temporal and spatial), and the reuse
// constants (core/contracts/reuse_constants.h) at 6; then the records and
// reservoirs it reads and writes, flat arrays in buffers (contracts 5 and
// 14), each at its own index. The temporal merge also reads last frame's
// scene: its transforms and glows, and in raytraced mode its structure
// (bound only then). The shade writes its final reservoirs back in a graph
// with a temporal merge (restir_shade.h). Which reservoir array a pass
// writes, the working one or the frame's final one, the renderer binds by
// the graph (metal/frame/frame_images.h); the kernel names one output.
namespace restir_initial {
enum Buffer : unsigned {
    constants = 0,
    camera = 1,
    structure = 2,
    scene = 3,
    glows = 4,
    transforms = 5,
    reuse = 6,
    records = 7,             // written
    path_states = 8,         // written
    previous_positions = 9,  // written, in a graph with a temporal merge
    last_transforms = 10,    // last frame's, in a graph with a temporal merge
    output = 11,             // the working reservoirs, or the final ones in a graph with no merge
};
}  // namespace restir_initial

namespace restir_temporal {
enum Buffer : unsigned {
    constants = 0,
    camera = 1,
    structure = 2,
    scene = 3,
    glows = 4,
    transforms = 5,
    reuse = 6,
    records = 7,
    previous_positions = 8,
    last_records = 9,
    last_final = 10,      // last frame's final reservoirs
    input = 11,           // the initial reservoirs
    output = 12,          // the same array, in place, when a spatial merge follows; else this frame's final
    last_structure = 13,  // last frame's, in raytraced mode: refused with a temporal merge until it is kept
    last_glows = 14,
    last_transforms = 15,
};
}  // namespace restir_temporal

namespace restir_spatial {
enum Buffer : unsigned {
    constants = 0,
    camera = 1,
    structure = 2,
    scene = 3,
    glows = 4,
    transforms = 5,
    reuse = 6,
    records = 7,
    input = 8,   // the reservoirs the pass before wrote: read at this pixel and its neighbors
    output = 9,  // this frame's final reservoirs
};
}  // namespace restir_spatial

namespace restir_shade {
enum Buffer : unsigned {
    constants = 0,
    camera = 1,
    structure = 2,
    scene = 3,
    glows = 4,
    transforms = 5,
    reuse = 6,
    records = 7,
    path_states = 8,
    final_reservoirs = 9,  // read; its tested reservoirs written back in a graph with a temporal merge
    non_finite = 10,  // the frame's count of samples not finite (metal/film/non_finite.h)
};
enum Texture : unsigned { accumulated = 0 };  // also the frame's radiance, as the path pass's
}  // namespace restir_shade

// ReSTIR PT's passes (metal/passes/restir_pt/restir_pt.metal.h) share one
// layout, each kernel binding the part it reads or writes.
namespace restir_pt {
enum Buffer : unsigned {
    constants = 0,
    camera = 1,
    structure = 2,
    scene = 3,
    glows = 4,
    transforms = 5,
    settings = 6,    // contracts::PathReuseSettings
    reservoirs = 7,  // the path reservoirs, a pixel's each (contract 15)
    primaries = 8,   // each pixel's primary hit, for the merge's similarity test
    light = 9,       // the frame's light buffer, a float4 a pixel
    pairing = 10,    // the reuse textures (core/sampling/pairing.h)
    shifts = 11,     // the shifted paths, 3 a pixel
    replays = 12,    // the pairs that replay, compacted
    counter = 13,    // their count
    dispatch = 14,   // the replay's indirect grid
    non_finite = 15, // the frame's count of samples not finite (metal/film/non_finite.h)
    light_tiles = 16,  // the frame's light tiles (contracts/light_tile.h)
    path_constants = 17,      // contracts::PathReuseConstants: last frame's camera, whether it is history
    last_primaries = 18,      // last frame's primary hits
    finals = 19,              // this frame's final path reservoirs, which the next frame's temporal reuse reads
    last_finals = 20,         // last frame's
    previous_positions = 21,  // where each pixel's primary hit was last frame, a Float3
    previous_pixels = 22,     // the temporal input's pixel, a uint a pixel
    duplication = 23,         // the duplication map, D a pixel
    last_transforms = 24,     // last frame's shape transforms
    last_glows = 25,          // and glows
    mnee_requests = 26,       // with mnee: MNEE's work list (contract 16), the initial pass's
    mnee_header = 27,         // its count and capacity
};
enum Texture : unsigned { accumulated = 0 };  // the shade's: also the frame's radiance, as the path pass's
}  // namespace restir_pt

// The features pass (passes/features/features.h): the scene at the path
// tracer's indices, its constants (contract 17's FeatureConstants) and last
// frame's transforms; the four feature images it writes.
namespace features {
enum Buffer : unsigned {
    constants = 0,
    camera = 1,
    structure = 2,
    scene = 3,
    glows = 4,
    transforms = 5,
    feature_constants = 6,
    last_transforms = 7,
};
enum Texture : unsigned { albedo = 0, normal = 1, depth = 2, motion = 3 };
}  // namespace features

// The denoise pass's kernels (passes/denoise/denoise.h), one layout for all
// of them, each taking what it reads: its constants (contract 18's
// DenoiseConstants) and a dispatch's ConvParams or ResampleParams; the
// buffer it reads, the weights and biases, and the buffer it writes; the
// depth median's histogram and state; the demodulated light; last frame's
// output and this frame's. Its textures: the light pass's accumulated
// image and the four features at the light size, the full-size albedo, and
// the radiance image it writes.
namespace denoise {
enum Buffer : unsigned {
    constants = 0,
    params = 1,
    input = 2,
    weights = 3,
    bias = 4,
    output = 5,
    histogram = 6,
    median = 7,
    demod = 8,
    history = 9,
    next_history = 10,
};
enum Texture : unsigned { light = 0, albedo = 1, normal = 2, depth = 3, motion = 4, full_albedo = 5, radiance = 6 };
}  // namespace denoise

}  // namespace bindings
}  // namespace serenity
