#pragma once

// Axis: Sample reuse (ReSTIR PT's paired neighbors).
//
// A pixel's paired neighbor through one of the reuse textures (core/
// sampling/pairing.h builds them; ReSTIR PT Enhanced §3.2): the offset the
// texture stores at the pixel's texel. Each texture is self-inverting, A's
// partner's partner A, so the texture must change each frame, or a pixel
// would reuse from the same neighbor every frame; rather than many textures,
// each frame flips, mirrors, transposes and offsets each one at random
// (§3.2: "empirically, this completely solves the problem"). The pixel is
// taken into the texture by that frame's map, and the offset read there
// brought back by the map's inverse, so the pairing stays an involution in
// the image: with L the map's flips and transpose, the pixel p reads the
// texel t = L(p) + o (mod width), its partner p' = p + L^-1(d(t)) reads
// t + d(t), whose offset is -d(t), and goes back to p. Held by a GPU test
// at every pixel of an image, frame by frame (tests/gpu/
// restir_pt_test.cpp).
//
// The frame's map is a function of the frame's index and the texture alone
// (pcg_hash, metal/math/hash.metal.h): every pass of a frame finds the same
// partner for the same pixel, and any frame can be rendered again (GDSA.3).
// A partner outside the image is no partner; the caller tests it.
//
// With checkerboard rendering (contract 20, core/contracts/
// checkerboard.h), a frame traces one field, and a partner must be a pixel
// it traces. The pairing is then taken over the field's lattice
// (lattice_partner()): a traced pixel (x, y) of the field of parity phi is
// the lattice point (u, v) = ((x + y - phi) / 2, (x - y - phi) / 2), whole
// numbers since x + y - phi is even, and back, x = u + v + phi, y = u - v;
// the partner found for (u, v) through the texture, as for a pixel above,
// is brought back to a pixel the same way. An involution over the lattice,
// so over the field: a traced pixel's partner is traced and its partner
// is the pixel. A lattice offset (du, dv) is the pixel offset
// (du + dv, du - dv), each axis sqrt(2) times as far, which the renderer
// meets by building the textures at sigma over the lattice's step
// (contracts::checkerboard_lattice_step). Considered and not taken:
// RTXDI's way for its independently drawn neighbors, moving the partner
// into the field by one pixel along the row
// (RTXDI_ActivateCheckerboardPixel), which breaks the pairing: the moved
// partner's own partner is not the pixel, so one shift would no longer
// serve both ends (core/sampling/pairing.h).
//
// Cost: a hash, one 2-byte load from a 323 KB array read by every pixel of
// a 254 x 254 window at once, so from the cache; integer arithmetic.

#include <metal_stdlib>

#include "core/contracts/path_reuse_settings.h"
#include "metal/device/layout.metal.h"
#include "metal/math/hash.metal.h"

namespace serenity {
namespace shaders {
namespace pairing {

// One frame's map of one texture (above).
struct Framing {
  bool flip_x;
  bool flip_y;
  bool transpose;
  int2 offset;  // within [0, width)
};

inline Framing framing(uint frame, uint texture) {
  const uint h = pcg_hash(pcg_hash(frame) ^ (texture * golden_ratio_32));
  const int width = int(serenity::contracts::pairing_widths[texture]);
  const uint g = pcg_hash(h);
  return Framing{
      (h & 1u) != 0u, (h & 2u) != 0u, (h & 4u) != 0u,
      int2(int((h >> 8u) % uint(width)), int((g >> 8u) % uint(width)))};
}

// `pixel`'s partner through texture `texture` of `textures`, the textures
// one after another (contracts/path_reuse_settings.h, pairing_starts),
// under `map`: possibly outside the image.
inline int2 partner(constant ushort* textures, int2 pixel, uint texture,
                    Framing map) {
  const int width = int(serenity::contracts::pairing_widths[texture]);
  int2 s = map.transpose ? pixel.yx : pixel;
  s.x = map.flip_x ? -s.x : s.x;
  s.y = map.flip_y ? -s.y : s.y;
  const int2 t = ((s + map.offset) % width + width) % width;
  const uint packed = textures[serenity::contracts::pairing_starts[texture] +
                               uint(t.y * width + t.x)];
  // Two signed 8-bit offsets, read as two's complement (core/sampling/
  // pairing.h).
  int2 d = int2(signed_byte(packed, 0u), signed_byte(packed, 8u));
  d.y = map.flip_y ? -d.y : d.y;
  d.x = map.flip_x ? -d.x : d.x;
  return pixel + (map.transpose ? d.yx : d);
}

// `pixel`'s partner through texture `texture` under `map`, over the
// lattice of the traced field of parity `phase` (above): possibly outside
// the image. Precondition: the field holds `pixel` (x + y - phase even).
inline int2 lattice_partner(constant ushort* textures, int2 pixel, uint texture,
                            Framing map, int phase) {
  const int2 lattice =
      int2((pixel.x + pixel.y - phase) / 2, (pixel.x - pixel.y - phase) / 2);
  const int2 found = partner(textures, lattice, texture, map);
  return int2(found.x + found.y + phase, found.x - found.y);
}

}  // namespace pairing
}  // namespace shaders
}  // namespace serenity
