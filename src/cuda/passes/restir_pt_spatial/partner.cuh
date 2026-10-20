#pragma once

// Axis: Pass (ReSTIR PT's spatial reuse: a pixel's paired partners).
//
// The paired neighbor of a pixel in reuse texture `t` this frame, and
// whether it is in the image; with the checkerboard, over the traced
// field's lattice (reuse/pairing.cuh, lattice_partner()). Metal's, in its
// spatial .metal (metal/passes/restir_pt_spatial/restir_pt_spatial.metal);
// here in a header, since the shift and the replay that read it are
// programs and the merge a kernel, two modules.

#include "cuda/math/vector.cuh"
#include "cuda/passes/restir_pt/restir_pt.cuh"
#include "cuda/reuse/pairing.cuh"

namespace serenity {
namespace shaders {
namespace spatial {

struct Partner {
  bool inside;
  uint2 pixel;
};

__device__ inline Partner partner_of(const ushort* pairing, uint2 pixel, uint t,
                                     restir_pt::Traced traced) {
  const pairing::Framing map = pairing::framing(traced.frame, t);
  const int2 p =
      traced.checkerboard
          ? pairing::lattice_partner(pairing,
                                     make_int2(int(pixel.x), int(pixel.y)), t,
                                     map, int(traced.frame & 1u))
          : pairing::partner(pairing, make_int2(int(pixel.x), int(pixel.y)), t,
                             map);
  const bool inside = p.x >= 0 && p.y >= 0 && p.x < int(traced.width) &&
                      p.y < int(traced.height);
  return Partner{
      inside, inside ? make_uint2(uint(p.x), uint(p.y)) : make_uint2(0u, 0u)};
}

}  // namespace spatial
}  // namespace shaders
}  // namespace serenity
