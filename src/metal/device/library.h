#pragma once

#include <span>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/device.h"

namespace serenity::metal {

// Axis: the Metal API surface.
//
// A compiled Metal library, loaded from the bytes of a .metallib, and the
// compute pipelines made from its kernels.
//
// Shaders are compiled when the project builds, never when it runs: each
// library's .metallib is built with the flags pinned in
// cmake/MetalLibrary.cmake, by the toolchain pinned in cmake/toolchain.json,
// and its bytes are compiled into the executable. What a shader computes is
// then fixed by the source revision, and a shader that does not compile fails
// the build rather than a run.
//
// Failures throw Error, naming the cause and Metal's own description where it
// gave one (E.2, E.14): bytes Metal will not load as a library, a function
// the library does not have, a function that cannot be a compute pipeline.
//
// Every call that can receive an autoreleased object from Metal (an
// NS::Error) drains a pool of its own before it returns, so callers need no
// autorelease pool of their own. Objects returned to the caller are owned by
// the caller through NS::SharedPtr (R.1, I.11).
//
// Not performance-sensitive: a library is loaded and its pipelines are built
// once, at start-up.
class Library {
public:
    // Loads the library in `metallib`, on `device`. The bytes are copied, so
    // the span need not outlive the call.
    Library(const Device& device, std::span<const unsigned char> metallib);

    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
    Library(Library&&) = delete;
    Library& operator=(Library&&) = delete;
    ~Library() = default;

    // A compute pipeline for the kernel named `function`, a null-terminated
    // name that must not be null (F.25). Built on the device the library was
    // loaded on.
    NS::SharedPtr<MTL::ComputePipelineState> compute_pipeline(const char* function) const;

    // Valid for the Library's lifetime; not retained for the caller.
    MTL::Library* handle() const { return library_.get(); }

private:
    NS::SharedPtr<MTL::Device> device_;
    NS::SharedPtr<MTL::Library> library_;
};

}  // namespace serenity::metal
