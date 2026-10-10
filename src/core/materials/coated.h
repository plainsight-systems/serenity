#pragma once

// Axis: Material (coated).
//
// An opaque glossy surface: a smooth clear coat over a matte colored base.
// Opaque glass and porcelain marbles, glazed ceramic, plastic: a color lit
// by the light that reaches it, under a mirror-sharp gloss. On the marbles'
// table it is the surface that shows the light the estimators gather, the
// colored matte catching every firefly and the gloss showing each one as a
// sharp point (scenes/marbles.toml).
//
// Mitsuba 3's smooth `plastic` BSDF with its nonlinear option (Jakob et al.,
// Mitsuba 3 documentation; after Weidlich and Wilkie, "Arbitrarily Layered
// Micro-Facet Surfaces", 2007): of the light arriving, the coat reflects the
// share its Fresnel reflectance gives; the rest refracts into it, reaches
// the base and scatters there by Lambert's law; of what the base sends back
// up, the coat lets out what its Fresnel transmittance does and reflects the
// rest back down to the base, again and again, a geometric series. With
// ior the coat's index of refraction, against air outside, F(cos) the exact
// unpolarized Fresnel reflectance from air into it (the dielectric's,
// metal/materials/dielectric.metal.h), and rho the base's color:
//
//   coat   a delta reflection about the normal, F(cos theta_o) of the light.
//   base   a diffuse reflection,
//            f(wo, wi) = (1 - F(cos theta_i)) (1 - F(cos theta_o)) rho
//                        / (pi ior^2 (1 - rho F_in)),
//          F_in the coat's reflectance from inside, averaged over the
//          cosine-weighted hemisphere, total internal reflection included.
//          What is kept, computed once, at load, into the material's data,
//          is 1 - F_in, the escape, internal_escape(ior), and the
//          denominator is formed as (1 - rho) + rho (1 - F_in): a coat of
//          high ior lets out almost none of the base's light, and 1 - F_in
//          as a float stays above 0 where F_in itself rounds to 1 and 1 -
//          rho F_in would divide by 0.
//
// Energy: for a white base, rho = 1, the two lobes reflect all the light
// arriving, at every angle: the base's albedo, the integral of f |cos theta_i|,
// is (1 - F(cos theta_o)) (1 - F_out) / (ior^2 (1 - F_in)), F_out the coat's
// reflectance from outside averaged as F_in is, and 1 - F_out = ior^2 (1 -
// F_in) (reciprocity, Stokes), so it is 1 - F(cos theta_o), which with the
// coat's F(cos theta_o) makes 1. The GPU tests hold the BSDF to it, a white
// furnace: total reflectance 1 within sampling error at every angle.
//
// The four questions of contract 2, by these steps, which the shader half
// (metal/materials/coated.metal.h) carries by number:
//
//   Step 1  evaluate(wo, wi): the base's f where wo and wi are both above
//           the surface; 0 otherwise. The coat's is delta: 0.
//   Step 2  sample(wo, u): two-sided, as every reflecting kind here, the
//           normal turned to wo's side (metal/materials/bsdf.metal.h). With
//           probability F(cos theta_o), when u.x falls
//           below it, the coat: wi the mirror of wo, pdf F(cos theta_o),
//           value F(cos theta_o) / cos theta_i, so the weight is 1: the
//           coat's share was spent choosing it. Else the base: wi drawn
//           cosine-weighted from u.yz (metal/math/warp.metal.h), value f,
//           pdf as step 3's.
//   Step 3  pdf(wo, wi): (1 - F(cos theta_o)) cos theta_i / pi, the base's
//           density as sample() draws it; 0 below the surface.
//   Step 4  lobes(): reflection, diffuse and delta. Its diffuse lobe aims
//           at lights (aims_at_lights, contract 2); its delta lobe is found
//           only by sampling, as a mirror's is.
//
// It resolves (metal/materials/resolve.metal.h) to BsdfKind::coated: color
// rho, from its texture or its constant, as a rough surface's is
// (rough.h); ior; and 1 - F_in, as Bsdf::escape.
//
// internal_escape(ior), 1 - F_in, and internal_reflectance(ior), F_in: the
// latter the integral over the hemisphere inside
// the coat of F_inside(cos theta) 2 cos theta sin theta d theta, F_inside
// the Fresnel reflectance from the coat's side, 1 past the critical angle
// theta_c. The escape is computed directly, as the integral below theta_c
// of (1 - F_inside) 2 cos sin, by Simpson's rule over 4096 intervals of s,
// theta = theta_c - s^2, which smooths the reflectance's infinite slope at
// theta_c, in double; F_in is 1 less it, some 0.596 for ior 1.5. A pure function of the ior (F.8), computed once a material, at
// load, rather than at every hit that resolves it; the shaders read the
// result. CPU only.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "core/contracts/float3.h"
#include "core/contracts/texture_reference.h"

namespace serenity {
namespace materials {

struct CoatedData {
    contracts::Float3 color;              // the base's albedo, each in [0, 1]; used when texture.index is no_texture
    contracts::TextureReference texture;  // the base's albedo from a texture, or no_texture
    float ior;                            // the coat's index of refraction; greater than 1
    float escape;                         // internal_escape(ior), 1 - F_in, filled at load
    uint32_t padding[2];
};

static_assert(sizeof(CoatedData) == 32, "CoatedData must be the same 32 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
// 1 - F_in and F_in for a coat of `ior`, greater than 1: above.
double internal_escape(double ior);
double internal_reflectance(double ior);
#endif

}  // namespace materials
}  // namespace serenity
