#pragma once

// Axis: GPU backend (device code): the tensor cores' matrix products.
//
// Metal's simdgroup matrices (simdgroup_half8x8, simdgroup_float8x8,
// simdgroup_load, simdgroup_store, simdgroup_multiply_accumulate: half in,
// float sums) as the denoiser's kernels use them (cuda/passes/denoise/
// denoise.cu), on NVIDIA's tensor cores: one warp's
//
//   mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
//
// D = A B + C, A 16 x 8 halves (rows: pixels; columns: an 8-channel input
// block), B 8 x 8 halves (an input block by an 8-channel output block), C
// and D 16 x 8 floats (docs/research/2026-10-19-cuda-backend.md, "Phase 5's
// design", which proved these forms on the RTX 3090 Ti against the CPU in
// double: 2.8e-7 at most, the float sum's rounding). A Metal row of 32
// pixels, four 8 x 8 blocks, is two of these 16-pixel tiles.
//
// Where each element lies in a warp is the PTX ISA's (9.4, "Matrix
// Fragments for mma.m16n8k8"), with g = lane / 4 and t = lane % 4:
//
//   A  a0: row g,     columns 2t, 2t + 1;  a1: row g + 8, the same columns
//   B  b0: rows 2t, 2t + 1, column g
//   C  c0, c1: row g, columns 2t, 2t + 1;  c2, c3: row g + 8
//
// two halves a 32-bit register, the lower index in the lower bits. Loaded
// from shared memory by ldmatrix (the ISA's "ldmatrix"), which hands each
// lane its elements of 8 x 8 blocks whose rows the lanes name, every row 16
// contiguous bytes at a 16-byte aligned address: A by .x2, lane L naming
// row L % 16; B by .x1.trans from a row-major [k][columns] array, lane L
// naming row k = L % 8, transposed into the column fragment. From device
// memory, where Metal's simdgroup_load reads a buffer, ldmatrix cannot
// read, so the fragments are loaded by the formula above (load_a_device,
// load_b_device). Stores are the formula too, by the caller, into the
// layout Metal's simdgroup_store gives (fragment_row, fragment_column).
//
// Inline PTX: CUDA has no intrinsic for m16n8k8 nor for ldmatrix (its wmma
// API has no k = 8 shape for halves, and its fragments' layout is not
// specified, so they could not be stored into Metal's staging layout), the
// case for assembly SIMD.8 names; kept here, behind these functions, so no
// kernel spells it (I.30, P.11). Each asm is volatile with a "memory"
// clobber: it reads shared memory the block's threads wrote, so the
// compiler moves no load or store across it. Every call is warp-wide: all
// 32 lanes, converged (".sync.aligned").
//
// Cost: one ldmatrix moves a 128- or 256-byte tile in one instruction; an
// mma is 2 x 16 x 8 x 8 multiply-adds a warp.

#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {
namespace matrix {

// A 16 x 8 tile of halves, two registers a lane (above).
struct FragmentA {
  uint a0;
  uint a1;
};

// An 8 x 8 tile of halves, one register a lane.
struct FragmentB {
  uint b0;
};

// A 16 x 8 tile of floats, four a lane: element i at fragment_row(lane, i),
// fragment_column(lane, i).
struct Accumulator {
  float c[4];
};

__device__ inline Accumulator zero_accumulator() {
  return Accumulator{{0.0f, 0.0f, 0.0f, 0.0f}};
}

// The row and column of a lane's accumulator element i (C's fragment).
__device__ inline uint fragment_row(uint lane, uint i) {
  return (lane >> 2u) + (i >= 2u ? 8u : 0u);
}
__device__ inline uint fragment_column(uint lane, uint i) {
  return 2u * (lane & 3u) + (i & 1u);
}

// A generic pointer into shared memory as the shared window's address,
// which ldmatrix takes.
__device__ inline uint shared_address(const void* p) {
  return static_cast<uint>(__cvta_generic_to_shared(p));
}

// A, 16 rows of 8 halves in shared memory: `row_of_lane` is the start of
// row lane % 16, 16-byte aligned.
__device__ inline FragmentA load_a(const void* row_of_lane) {
  FragmentA a;
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(a.a0), "=r"(a.a1)
               : "r"(shared_address(row_of_lane))
               : "memory");
  return a;
}

// B, 8 rows k of 8 halves of a row-major [k][columns] array in shared
// memory, transposed into B's fragment: `row_of_lane` is the start of row
// lane % 8's 8 halves, 16-byte aligned.
__device__ inline FragmentB load_b(const void* row_of_lane) {
  FragmentB b;
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
               : "=r"(b.b0)
               : "r"(shared_address(row_of_lane))
               : "memory");
  return b;
}

// Two adjacent halves at `p` (4-byte aligned) as one register, the first
// in the lower bits.
__device__ inline uint pair_at(const ushort* p) {
  return *reinterpret_cast<const uint*>(p);
}

// Two halves as one register, `low` in the lower bits.
__device__ inline uint pair_of(ushort low, ushort high) {
  return uint{low} | (uint{high} << 16u);
}

// A, 16 rows of 8 halves in device memory, row r at a + r * ld halves (ld
// even, a 4-byte aligned): each lane's two pairs by the fragment formula.
__device__ inline FragmentA load_a_device(const ushort* a, uint ld, uint lane) {
  const uint g = lane >> 2u;
  const uint t = lane & 3u;
  return FragmentA{pair_at(a + g * ld + 2u * t),
                   pair_at(a + (g + 8u) * ld + 2u * t)};
}

// B, 8 rows k of 8 halves of a row-major [k][columns] array in device
// memory, row k at b + k * ld halves: each lane's column g, rows 2t and
// 2t + 1.
__device__ inline FragmentB load_b_device(const ushort* b, uint ld, uint lane) {
  const uint g = lane >> 2u;
  const uint t = lane & 3u;
  return FragmentB{pair_of(b[(2u * t) * ld + g], b[(2u * t + 1u) * ld + g])};
}

// c = a b + c, float sums of half products.
__device__ inline void multiply_accumulate(Accumulator& c, FragmentA a,
                                           FragmentB b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
      : "+f"(c.c[0]), "+f"(c.c[1]), "+f"(c.c[2]), "+f"(c.c[3])
      : "r"(a.a0), "r"(a.a1), "r"(b.b0)
      : "memory");
}

}  // namespace matrix
}  // namespace shaders
}  // namespace serenity
