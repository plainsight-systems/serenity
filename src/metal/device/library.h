#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstddef>
#include <span>
#include <string_view>

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
// Failures throw MetalError, naming the cause and Metal's own description where
// it gave one (E.2, E.14): bytes Metal will not load as a library, a function
// the library does not have, a function that cannot be a compute pipeline.
//
// Every call that can receive an autoreleased object from Metal (an
// NS::Error) drains a pool of its own before it returns, so callers need no
// autorelease pool of their own. Objects returned to the caller are owned by
// the caller through NS::SharedPtr (R.1, I.11).
//
// Not performance-sensitive: a library is loaded and its pipelines are built
// once, at start-up.
// How a pipeline is built beyond its kernel, named fields (I.24):
// `group_threads`, the most threads a group it is built for (Metal's
// maxTotalThreadsPerThreadgroup, which the compiler sizes its registers
// by), 0 for as many as the device and the kernel allow; `constants`, the
// values its kernel is specialized by, every function constant it reads
// (passes/restir_pt/reuse_constants.h), or none for a kernel that reads
// none; `intersection_functions`, the library's intersection functions its
// kernel traces with, by name, linked into it (Library::intersection_table()
// makes their table). Not retained: each need only outlive the call.
struct PipelineOptions {
  NS::UInteger group_threads = 0;
  const MTL::FunctionConstantValues* constants = nullptr;
  std::span<const char* const> intersection_functions;
};

class Library {
 public:
  // Loads the library in `metallib`, on `device`. The bytes are copied, so
  // the span need not outlive the call.
  Library(const Device& device, std::span<const std::byte> metallib);

  Library(const Library&) = delete;
  Library& operator=(const Library&) = delete;
  Library(Library&&) = delete;
  Library& operator=(Library&&) = delete;
  ~Library() = default;

  // A compute pipeline for the kernel named `function` (a view, which can
  // be no null pointer: F.24, I.4), built as `options` say. Built on the
  // device the library was loaded on.
  NS::SharedPtr<MTL::ComputePipelineState> compute_pipeline(
      std::string_view function, PipelineOptions options = {}) const;

  // A render pipeline of the vertex function `vertex` and the fragment
  // function `fragment`, drawing one color attachment of `color` and a
  // depth attachment of `depth`. Throws MetalError if the library has no
  // such function or they make no pipeline.
  NS::SharedPtr<MTL::RenderPipelineState> render_pipeline(
      const char* vertex, const char* fragment, MTL::PixelFormat color,
      MTL::PixelFormat depth) const;

  // The table of the intersection functions `functions` linked into
  // `pipeline` (PipelineOptions), entry i the function named
  // `functions[i]`, for its kernel to trace with. Throws MetalError if the
  // library has no such function or Metal makes no table.
  NS::SharedPtr<MTL::IntersectionFunctionTable> intersection_table(
      MTL::ComputePipelineState* pipeline,
      std::span<const char* const> functions) const;

  // Valid for the Library's lifetime; not retained for the caller.
  MTL::Library* handle() const noexcept { return library_.get(); }

 private:
  // The library's function `name`; throws MetalError if it has none.
  NS::SharedPtr<MTL::Function> function_named(const char* name) const;

  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::Library> library_;
};

}  // namespace serenity::metal
