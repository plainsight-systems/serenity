#pragma once

// How a layout crosses into a shader (architecture/file-mapping.md): the one
// place the differences between C++20, Metal's shading language and CUDA's
// device code are written, so every header that both compile spells them one
// way, here, rather than each its own (P.11, ES.30). Every shared header
// includes this first; nothing else does.
//
//   - The standard headers each side needs: metal_stdlib in a shader;
//     stdint.h's fixed-width integers and type_traits on the host.
//   - SERENITY_CONSTANT, a constant at namespace scope: `constant constexpr`
//     in a shader, whose namespace-scope constants must be in its constant
//     address space, and `inline constexpr` on the host, one definition
//     however many translation units include it. Defined once, here, and
//     never undefined: a header that includes another sees the same
//     definition, so no inner header can remove an outer one's.
//   - SERENITY_SHARED_FUNCTION, on each function a shared layout defines
//     (its encodings): `__host__ __device__` where nvcc compiles it
//     (__CUDACC__), so CUDA's device code may call it, and nothing
//     elsewhere, where every function is callable from both sides already.
//     nvcc otherwise lets device code call a constexpr function only under
//     its experimental --expt-relaxed-constexpr; the annotation is how
//     CUTLASS marks its own (docs/research/2026-10-19-cuda-backend.md,
//     "Phase 3's design"). A macro, since the annotation is not C++'s; one
//     prefixed name (ES.33), in the one header that holds the languages'
//     differences (P.11). nvcc compiles the host branch below otherwise:
//     CUDA's device code is C++.
//
// A shared layout is written by the host as bytes and read by a shader as
// its own type, so on the host each is asserted trivially copyable beside
// its size (SL.con.4: copying a type's bytes is defined only for such a
// type; COPY.6, LIFE.4). The shading language has no type_traits, so the
// assertion is the host's alone; the size is asserted on both sides.

#if defined(__CUDACC__)
#define SERENITY_SHARED_FUNCTION __host__ __device__
#else
#define SERENITY_SHARED_FUNCTION
#endif

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#define SERENITY_CONSTANT constant constexpr
#else
#include <stdint.h>

#include <type_traits>
#define SERENITY_CONSTANT inline constexpr
#endif
