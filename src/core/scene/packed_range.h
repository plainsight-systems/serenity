#pragma once

// Axis: Scene content (whether a scene's materials fit contract 2's packed
// form).
//
// ReSTIR PT stores each pixel's primary hit with its Bsdf packed
// (contracts/bsdf.h, pack_bsdf), whose precondition is that every color and
// scalar a material resolves to is finite and at most 65504 in magnitude,
// half's largest (contracts::packed_largest). The scene reader admits more: an
// ior is any finite number above 1, and a rough material's color, or a
// checker texture's, any finite number (scene.h). The naive path tracer
// packs nothing and shades them as they are; a renderer that packs asks
// this first, and refuses the scene by what it answers (metal/frame/
// renderer.h), so the scene's own checks, which every graph shares, do not
// change.
//
// The values asked, by the material's kind, as metal/materials/
// resolve.metal.h resolves it: rough, its color or its texture's colors;
// coated, its color or its texture's colors, its ior and its escape;
// conductor, its f0 (its alpha, at least least_alpha and at most 1, always
// fits); dielectric, its ior; emissive, nothing (it resolves to none). A
// texture's colors are each kind's bound on what it returns: a checker's
// two, a swirl's two, a wood's two times 1 + wood_board_shade
// (core/textures/wood.h). Here, beside the reader, because the reader is
// the place that lists the kinds (change-axes.md): a kind added there adds
// its values here.
//
// Pure (F.8); O(materials). Decided 2026-10-11, by the review of f86e7ae,
// which found an ior of 70000, valid for the scene reader, packed to
// infinity.

#include <optional>
#include <string>

#include "core/scene/scene.h"

namespace serenity::scene {

// The first value of `scene`'s materials outside contract 2's packed form,
// naming the material (by its file's name, or its record's index where the
// description has no names) and the value; none when every one fits.
std::optional<std::string> outside_packed_bsdf(const SceneDescription& scene);

}  // namespace serenity::scene
