#pragma once

// The launch parameters of the CUDA backend's tracing probes
// (tests/cuda/support/traced_probe.h, kernels/traced_probes.cu): Metal's
// bindings, positional, as device addresses, binding i at bindings[i], the
// order each probe's comment lists (tests/gpu/kernels/<probe>.metal); and
// the structures, by value, which a Metal probe binds as a block and an
// OptiX program reads from its launch parameters. Compiled by both
// compilers: one type.

#include <cstdint>

#include "cuda/acceleration/structure_block.h"

namespace serenity::tests {

// The most bindings a tracing probe takes.
inline constexpr std::uint32_t probe_bindings = 12;

struct ProbeLaunch {
  // A C array: device code reads it, and std::array's subscript is a host
  // function there (the shared layouts' rule, core/contracts/
  // shared_layout.h).
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
  std::uint64_t bindings[probe_bindings];
  cuda::StructureBlock structures;
};

}  // namespace serenity::tests
