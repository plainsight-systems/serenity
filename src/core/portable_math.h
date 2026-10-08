#pragma once

// Math that gives the same bits on the CPU and on the GPU.
//
// Axis: none of the renderer's yet (docs/architecture/change-axes.md is
// written next). This file changes when the toolchain's floating-point
// behavior does, and with nothing else.
//
// A header written against this one compiles twice: as C++20, for the tests
// and any CPU path, and as the Metal shading language, for a kernel that
// includes it. The shading language is a dialect of C++14, so inline
// functions over float compile as both; what differs is the library each side
// calls, chosen below by __METAL_VERSION__. Nested namespaces are spelled out
// because C++14 has no `a::b` namespace definition.
//
// Equal bits are a property of the build as much as of the source. They hold
// under three conditions, each measured on the development machine
// (docs/research/2026-10-08-metal-math-modes.md) and each checked by
// tests/gpu/portable_math_test.cpp:
//
//   1. The shader is compiled with -fmetal-math-mode=safe and
//      -fmetal-math-fp32-functions=precise (cmake/MetalLibrary.cmake).
//      Metal's default is fast, under which division, sqrt and a reassociated
//      sum each differed from the CPU in a fifth to a third of inputs.
//   2. Neither side contracts a * b + c into a fused multiply-add:
//      -ffp-contract=off for the shader (cmake/MetalLibrary.cmake) and for
//      every C++ target that links the core (CMakeLists.txt). With
//      contraction on one side only, a * b + c differed in a quarter of
//      inputs. Where a fused multiply-add is wanted, call fma() below: it is
//      fused on both sides, whatever either compiler is allowed.
//   3. Only operations IEEE 754 rounds exactly are used: + - * /, sqrt() and
//      fma(). exp and sin differed in two inputs of five even under precise,
//      so they are not offered here. A function built on them needs its own
//      arithmetic from the exact operations, or a tolerance stated where it
//      is tested.
//
// The guarantee covers zero, normal, infinite and NaN values, not subnormal
// ones: the GPU flushes a subnormal float to zero, as an input and as a
// result, under every math mode tried, and -fdenormal-fp-math=ieee did not
// change it; the CPU keeps them. So equal bits need every input, every
// rounded intermediate and the result to be other than subnormal, that is,
// of magnitude 2^-126 or more, or zero. Making the CPU flush too would mean
// setting a thread-wide floating-point mode, ambient state every other
// computation on the thread would inherit, so the domain is narrowed instead.
//
// The guarantee is per operation, between this machine's CPU and GPU, on the
// pinned toolchain (cmake/toolchain.json). It says nothing about the order of
// a reduction: a sum over many elements is equal on both sides only if both
// add in the same order, and each reduction states its own determinism level
// (GDSA.2). NaN results are equal as NaN, not by payload.
//
// The cost is condition 1: precise functions and no reassociation for every
// shader built with these flags. A kernel that wants Metal's fast math opts
// out explicitly, labelled as an optimization with its measurement, and gives
// up sharing its math with the CPU bit for bit.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <cmath>
#endif

namespace serenity {
namespace portable {

// a * b + c, rounded once.
inline float fma(float a, float b, float c) {
#if defined(__METAL_VERSION__)
    return metal::fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

// The square root, correctly rounded.
inline float sqrt(float x) {
#if defined(__METAL_VERSION__)
    return metal::sqrt(x);
#else
    return std::sqrt(x);
#endif
}

}  // namespace portable
}  // namespace serenity
