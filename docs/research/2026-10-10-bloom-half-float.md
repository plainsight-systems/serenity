# The bloom pyramid in half floats, and the tone mapper's hue

*2026-10-10. Apple M3 Max, macOS 26.6.2, Xcode 26.2, `-std=metal4.0`.
Written with the tone-map pass (9db8877, 6a28fcb); moved here from
`src/core/passes/tone_map.h`, which keeps the rule and points here.*

## Question

The bloom pyramid's levels are half floats (`tone_map.h`, steps 2 and 3).
Step 3 sums the six levels' blurs into B_0. Summed whole, six levels at the
largest half float, 65504, would pass it and round to infinity, which the
roll-off (step 5) turns to NaN. Does dividing each level by the level count
first keep every partial sum finite, under the rounding the hardware does?

## The arithmetic

Each level holds at most bloom_ceiling / bloom_levels = 65504 / 6 =
10917.33 (step 1 clamps E to the ceiling, and each filter's weights sum to 1,
so no filter exceeds its largest input).

- Rounded to the nearest half float, a level holds at most 10920, and the
  partial sums, each rounded on its store, at most 21840, 32768, 43680,
  54592 and 65504: the last the ceiling exactly, finite.
- Rounded toward zero, each is at most its exact value, within the ceiling.

Rounding is monotone, so a field at the ceiling everywhere is the most any
level holds, and these bounds hold for every image.

## The hardware

The M3 Max's texture writes round toward zero: a level written with
65504 / 6 stores 10912, not 10920, as observed when the pass was reviewed
(6a28fcb). Either rounding keeps the sums within the ceiling, so the rule
does not depend on which the device does. The GPU test "broad light past
the half-float range: six levels at the ceiling sum within it"
(`tests/gpu/tone_map_test.cpp`) holds the device to it, whichever it does.

## The roll-off's hue

Step 5 is Khronos' PBR Neutral, chosen over ACES and AgX because it keeps
hues where they are. AgX moves brass's hue: a Blender user measured 52
degrees going to 46 through it. The brass and the fireflies are the scene's
colors, so a hue shift there is the one this scene cannot hide.
