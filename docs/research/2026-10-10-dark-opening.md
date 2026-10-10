# The dark opening: fireflies that wake, perch and descend

*2026-10-10. Design note for the marbles' opening; the headers are the
design (core/animation/flight.h, glow.h, flashes.h; core/scene/swarm.h,
scene.h). This note keeps what the headers should not: the facts behind
the design, and the alternatives weighed and set aside.*

## Question

The marbles' first seconds show no firefly: at t = 1 the frame holds the
table, the marbles and a faint sky. The art direction (2026-10-10) is the
opposite of an empty frame that fills by chance: start pitch black, with no
sky; a few fireflies wake one by one, some resting on the table and the
marbles, lighting what they rest on, then rise; the main swarm comes down
from above; and the scene fills. What must the animation do for that, and
how is it kept a closed form of t, checked at load, as everything else is?

## Facts

- The camera is at y = 0.88, looking down at the cluster, about 0.3 by
  0.2 m in frame at the focus. The marbles stand on the table top at
  0.75, their tops at most 0.775. The swarms' volume starts at 0.795.
- A swarm's start (swarm.h, step 2) must be clearance + radius + delta +
  first_drift_reach sqrt 3, some 0.19 m, from every still surface: every
  start is at y 0.97 or higher, above the camera. A firefly reaches the
  frame only when its loop brings it down, so the frame at t = 0 is empty,
  and fills at the loops' pace.
- The naive path tracer traced a shadow ray to the light it chose
  whatever its glow (metal/integrator/path.metal.h, next_event, and the
  preview's direct.metal.h: no test of the radiance before `occluded`),
  where pbrt-v4's SimplePathIntegrator, which it follows, tests `ls->L`
  first. A dark firefly cost a frame what a lit one does. The test is now
  made (below, "A sleeping light's shadow ray").
- It does change the image's noise. The path tracer chooses one light of
  616 uniformly; with a few awake, almost every choice is dark. The opening
  is the naive estimator's worst case, which is the case ReSTIR DI is for.
- Fireflies themselves (Photinus, Lloyd 1966): the males fly and flash,
  the females answer from perches in the grass; their flashing begins at
  dusk, a few, then many. A swarm that perches and one that flies, waking
  over tens of seconds, is that, at the scene's scale.

## Design, and what was set aside

| Choice | Kept | Set aside, and why |
|---|---|---|
| How a light comes on | A wake on the glow: 0 before `at`, a smoothstep up to the glow over `ramp` | A one-time flash at the wake: a flash outside the flight's schedule, with spacing rules of its own against it, for what the ramp and the next scheduled flash already show |
| Where the wake lives | An optional member of each glow kind's record, one function both call (ES.3) | On the Glower in animate.h: the Animate step would then compute brightness, which is glow.h's question; animate() stays a loop over records |
| Wake times | Drawn per firefly so the share awake grows as a power of time | Uniform times: an even rate, no "one by one, then many"; a fixed order by index: the first lights would all sit at the swarm's first-drawn starts |
| Above | Start in the volume's top, hold still until the wake, then the loop | A descent segment of its own: the loop's first transit already goes down to a circle, which the marbles' weights (circle 6, drift 1) make most first episodes |
| Perch | A point a perch_gap above an upward-facing surface, found by sphere tracing down (Hart 1996) | Perching at the loop's clearance, some 2 cm above with delta: hovering, not resting; allowing any surface: fireflies stuck to a marble's flank |
| Checking the rise | Conservative advancement (Mirtich 1996) to perch_gap / 2, exact everywhere on the path | Step 3's uniform samples at the perch's scale, perch_gap / 4 apart: some four thousand per rise, nearly all far from any surface; relaxing the clearance to 0: the body could touch |
| A perched firefly's loop start | Straight above its perch, at a drawn height | Step 2's start anywhere in the volume: a rise up to a meter across, some 24 s at the marbles' speed |
| Kinds with their own numbers | std::variant (C.181, C.182) for Prelude and SwarmStart | A kind and fields read for some kinds only, as the GPU records are: those are shared layouts, which a variant cannot be; these never leave the CPU |
| Written fireflies | May hold, perch and wake as a swarm's do | Swarm-only preludes: swarm.h's rule is that a swarm's firefly is what a written one is |
| Flashes before the loop | The flight draws its opening's flashes at its drifting rate, from keys of their own | None while waiting: a perched firefly that only glows, which Photinus females do not do |

The loop of every flight without a prelude is what it was: the same
episodes, the same flashes, from the same keys. A swarm with no start and
no wake draws its fireflies as before. tests/flight_test.cpp pins one
seed's loop, which this change must leave unmoved.

## Cost, to be measured

- Per frame: one comparison in position(), and a search of at most two
  segments for a firefly in its opening; in glow(), a test that a wake
  exists and two comparisons, and during a ramp a subtraction, a division,
  the smoothstep and a product (glow.h counts them). The first count here
  said one comparison for the wake; the design review (50bf979) showed it
  short. `animate()` took 43 µs a frame for the marbles; it is measured
  before and after.
- At load: a perch's sphere trace and checks, and its rise's advancement,
  some tens to hundreds of distances per perched firefly, beside the
  10^4 to 10^5 of its loop. The load is measured before and after.
- On the GPU: none per frame (above). The time baseline (28.44 ms,
  2026-10-10-corpus-sweep.md) is of the marbles as they were; the scene
  this change makes is re-measured at the same frames, and the baseline
  recorded again if it moves.

## Review

Codex reviewed the design (50bf979, both focuses). Its four findings were
all taken, in the headers:

- The rise's clearance was proved of the double curve, while the points
  asked about and rendered are floats: at 8192 m a float's spacing passes
  perch_gap. Every threshold, the loop's step 3 too, now carries a
  rounding allowance rho of its box, and a perch too far out for its gap
  is refused.
- The rise left the perch at the cruising speed from a standstill: it now
  leaves from rest. The hold's end, still to the drift's speed, is the one
  join that is not smooth, and the header says so.
- The wake's cost was undercounted (above).
- A wait of any length made an opening of as many flashes: a billion
  seconds, twenty million of them. A wait is now at most most_wait, an
  hour, some 300 flashes.

## Implementation

The core is implemented as the headers say (branch `dark-opening-core`),
with the header corrections listed in its commit message. Its rule-by-rule
pass over both corpora is
[2026-10-10-dark-opening/core-guidelines-pass.md](2026-10-10-dark-opening/core-guidelines-pass.md).
The load and frame measurements above are still to be made, with the
scene's art.

## A sleeping light's shadow ray

The design review of the implementation (55f968a, performance focus)
found the missing test above: next event estimation drew a direction
toward a light whose radiance was 0, then evaluated the BSDF and traced
the shadow ray for a term that is 0 whatever the ray finds. The test of
the sample's radiance now comes first, as pbrt-v4's does (its
SimplePathIntegrator: `if (ls && ls->L && ls->pdf > 0)`), in the path
kernel and in the preview's two loops. The random numbers a path draws
are the same either way, and the image the same: a sleeping light's term
was 0, and is not computed.

Measured: the path graph at 3456 x 2234 on the M3 Max, 200 frames, each
waited for, the median GPU time of frames 21 to 199; the two builds
alternated three times, each run started after the GPU had been under 5%
busy for six seconds. Each figure is the mean of the three rounds'
medians, the rounds' range beside it. The scene is the marbles with their
opening (scripts/make_marbles.py's), at three times:

| Time | Lights awake | Without the test | With it |
|---|---|---|---|
| 0.5 s | none | 32.35 ms (32.33–32.37) | 19.16 ms (19.14–19.19) |
| 8 s | a few of the perched swarm | 32.45 ms (32.45–32.45) | 24.34 ms (24.31–24.36) |
| 120 s | all | 29.89 ms (29.88–29.90) | 29.77 ms (29.76–29.78) |

The marbles without an opening, every light lit, at t = 628: 29.78 ms
without the test, 29.67 ms with it, the same means (round medians 29.48 / 29.39, 29.93 /
29.81, 29.94 / 29.82). With every light lit the test saves nothing to
trace; its 0.1 ms in every round is the compiled kernel's, not fewer
rays, and is not claimed. These frames were each waited for, so their
times are not the corpus sweep's 28.44 ms baseline, which kept two frames
in flight; that baseline is re-measured with the scene's art.

## A swarm that a firefly's bad luck refused

Loading the marbles with their opening and the perched swarm's seed
changed, 3 to 32, refused 14 of the 30 seeds (at b4063fd). Two causes,
one firefly each time:

- a rise from its perch not clear (flight.h, P2): 8 seeds. Step 2p
  places the perch and the loop's first point, but nothing redraws the
  firefly when the rise between them meets a marble;
- an episode not drawn clear, or the loop not closed: 6 seeds. Not new:
  the marbles as they were before this work (082cb0f), the same 30 seeds of
  their white swarm, refused the same 6. The marbles have loaded because
  their seeds were ones that did.

Some 0.3% of perched fireflies' draws, and 0.15% of flying ones', are
refused by chance in the draws, not by the swarm's numbers: a swarm of 480
fails for about half its seeds. A scene that loads for one seed and not
for the next is not a scene a reader can edit. Kept: a refused swarm
firefly is drawn again whole, from its seed's next draw, in rounds, up to
eight draws (swarm.h, step 5); a firefly whose first flight is made is
unchanged, so every scene that loaded loads the same. Set aside: more
rounds or backtracks inside make_flight, which shift the failure rate
without removing it and change loops that load today; checking the rise
in step 2p, which covers one cause of two.

## The time baseline, again

The corpus sweep's baseline (2026-10-10-corpus-sweep.md) was the naive path
tracer on the marbles at d1a5702: 28.44 ms a frame at 3456 x 2234. Its
harness was not kept, and the opening makes a frame's time depend on when
it is: before the first wake it is a third shorter. So the baseline is
measured again, the old code and scene beside the new, by one method,
stated here so it can be repeated:

- the path graph at 3456 x 2234, 201 frames from a named time t, each
  1/60 s on, two in flight as the window keeps them (frame i submitted,
  then frame i - 1 waited for); each frame's GPU time from its
  submission's feedback (GPU end - GPU start, metal/device/submission.h);
  the median of frames 21 to 199;
- a test case appended to tests/gpu/frame_images_test.cpp for the run and
  removed after, never committed; the release build;
- every case alternated, three rounds, each run started after the GPU had
  been under 5% busy for six seconds; no window open. M3 Max, macOS
  26.6.2.

Each figure is the mean of the three rounds' medians, the rounds' range
beside it:

| Code | Scene | t | Frame |
|---|---|---|---|
| d1a5702, the sweep's | the marbles before the opening | 0 | 28.60 ms (28.58–28.62) |
| 1a617ab, this work's | the marbles before the opening | 0 | 28.47 ms (28.44–28.50) |
| d1a5702 | the marbles before the opening | 120 | 29.99 ms (29.97–30.00) |
| 1a617ab | the marbles before the opening | 120 | 29.86 ms (29.86–29.87) |
| 1a617ab | the marbles with the opening | 0.5, none awake | 19.10 ms (19.08–19.15) |
| 1a617ab | the marbles with the opening | 8, a few awake | 24.23 ms (24.11–24.37) |
| 1a617ab | the marbles with the opening | 120, all awake | 29.90 ms (29.89–29.91) |

- The method reproduces the sweep's figure: the sweep's code and scene at
  t = 0, 28.60 ms against 28.44 ms, within 0.6%.
- This work's code costs the old scene nothing: 0.13 ms less at both
  times, in every round; not claimed as a gain (the shader change's own
  measurement, above, puts it in the compiled kernel, not in fewer rays).
- With every firefly awake, the new scene is the old one's cost: 29.90 ms
  against 29.86 at t = 120. The fireflies' positions, not the opening's
  code, set a lit frame's time.

The time baseline the later estimators are measured against is therefore
the naive path tracer at 1a617ab on the marbles with their opening, at t
= 120, every firefly awake: **29.90 ms**, by the method above. The
opening's frames, 19.1 ms with none awake and 24.2 ms with a few, are
measured beside it: the case where ReSTIR's light choice matters most.

## Open

The art itself: wake ranges, the perch box, the lingers and the ramp are
the scene's numbers, set by rendering the opening as a short clip and
looking at it.
