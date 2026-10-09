#pragma once

namespace serenity::frame {

// Axis: Frame graph (the tone-map pass's settings).
//
// How a frame's linear radiance becomes what the display shows, as data:
// the settings of the tone-map pass (metal/passes/tone_map/tone_map.h), read
// from the frame graph file (graph_file.h). The mechanism is the pass's; the
// numbers are the look's, set without a code change (change-axes.md:
// mechanism is code, values are data; "a tone mapper against its
// exposure").
//
//   exposure  in stops: the radiance is scaled by 2^exposure before
//             anything else. 0 shows the scene's radiance as it is.
//   bloom     the fraction of the light spread into glare, in [0, 1): the
//             image becomes (1 - bloom) x itself + bloom x its blurred
//             self, a mean of the two, so bloom moves light and makes none
//             (Jimenez 2014). A firefly far brighter than the display
//             shows keeps its glare when its core clips: brightness reads
//             as the halo's size and strength.
//
// No automatic exposure: the exposure is the graph's, the same every frame,
// so a frame stays a function of its inputs (principle 1) and a flash shows
// as a flash rather than being adapted away.
struct ToneMap {
    float exposure = 0.0f;  // stops; finite, within [-20, 20]
    float bloom = 0.0f;     // in [0, 1)
};

}  // namespace serenity::frame
