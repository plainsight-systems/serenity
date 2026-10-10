#pragma once

// Axis: Medium (contract 12's shader side, contracts/medium.h).
//
// The media shapes are filled with, as a shader reads them: the records and
// one array per kind; and the contract's question, transmittance, answered
// by the medium's kind: the one place a shader dispatches on MediumKind. The
// switch has no default, so a kind added to MediumKind and not here fails to
// compile (-Werror).

#include <metal_stdlib>

#include "core/contracts/medium.h"
#include "core/media/absorbing.h"
#include "core/media/medium.h"
#include "metal/media/absorbing.metal.h"

namespace serenity {
namespace shaders {

struct Media {
    constant serenity::media::MediumRecord* records;
    constant serenity::media::AbsorbingData* absorbing;
};

// The share of light, per channel, kept over a stretch `t` long inside
// medium `which`, an index into the records or no_medium (air: 1).
inline float3 transmittance(Media media, uint which, float t) {
    if (which == serenity::contracts::no_medium) {
        return float3(1.0f);
    }
    const serenity::media::MediumRecord record = media.records[which];
    switch (record.kind) {
    case serenity::media::MediumKind::absorbing:
        return absorbing_transmittance(media.absorbing[record.index], t);
    }
    return float3(1.0f);
}

}  // namespace shaders
}  // namespace serenity
