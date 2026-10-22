#pragma once

// Axis: GPU backend (device code): copies into shared memory that bypass
// a thread's registers.
//
// Ampere's cp.async: a thread starts a 16-byte copy from device memory into
// shared memory and goes on; the copies a thread started are committed as
// a group, and a later wait names how many of its newest groups may still
// be in flight. A copy goes from memory to shared memory without passing
// through the thread's registers, as CUTLASS's Ampere mainloops copy
// (cutlass/gemm/threadblock/mma_multistage.h); the denoiser's convolutions
// copy their tiles so (cuda/passes/denoise/denoise.cu;
// docs/research/2026-10-20-nvidia-hardware-design.md, section 8, change 7).
// The PTX ISA's "cp.async", "cp.async.commit_group" and
// "cp.async.wait_group", as inline PTX (SIMD.8, as matrix.cuh's, behind
// these functions: I.30): .cg, cached in L2 alone, as CUTLASS's 16-byte
// copies are, since each byte is read once into shared memory. A wait
// orders a thread's own copies only: the block's threads then meet at a
// barrier before reading one another's.

#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {
namespace async_copy {

// Starts copying the 16 bytes at `source` (device memory, 16-byte
// aligned) to `target` (shared memory, 16-byte aligned); with `read`
// false, writes 16 zero bytes instead and reads nothing (the ISA's
// src-size 0), so `source` need only be a valid address.
__device__ inline void copy16(void* target, const void* source, bool read) {
  const uint shared = static_cast<uint>(__cvta_generic_to_shared(target));
  const uint bytes = read ? 16u : 0u;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(shared),
               "l"(source), "r"(bytes)
               : "memory");
}

// The copies this thread started since the last commit, one group.
__device__ inline void commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

// Waits until at most `Pending` of this thread's newest groups are still
// in flight.
template <int Pending>
__device__ inline void wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(Pending) : "memory");
}

}  // namespace async_copy
}  // namespace shaders
}  // namespace serenity
