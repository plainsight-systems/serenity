# What a scene's load costs: making the flights

*2026-10-10. Apple M3 Max (16 cores), macOS 26.6.2, Apple clang 17.0.0
(clang-1700.6.3.2), `-O3 -DNDEBUG`, the release build's flags. CPU only.*

## Question

Loading a scene makes every firefly's flight (`src/core/animation/flight.h`):
64 episodes and their transits, each stretch sampled every centimeter and
each sample asked, through contract 11 (`src/core/contracts/obstacles.h`),
for its distance to the nearest still shape. Some 10^4 to 10^5 samples a
firefly, each a scan over every still shape. What does that cost, where does
it go, and when would a spatial index behind the obstacles pay?

## Earlier measurements

Moved here from `flight.h`'s header, where they were written as the flights
were built:

| Scene | Fireflies | Still shapes | Load |
|---|---|---|---|
| `scenes/brass_sphere_flight.toml` | 6 | 2 | 12.4 ms, 2 ms a firefly (837 segments in all); its first flight's loop 598 s |
| the marbles at a987908 | 512 | 11 (5 spheres) | 1.3 s one flight after another; 104 ms with the flights made in parallel (`make_flights`, 16 cores) |
| the marbles with fourteen marbles | 512 | 23 | 128 ms, in parallel |

## Method, today

A throwaway program (not committed) timing `scene::load` of a scene file,
nine loads a run, the minimum and median reported; three rounds, the three
builds interleaved. The marbles scene today has 616 fireflies in two
swarms among 51 still shapes (45 spheres, 6 boxes). The machine was shared:
other builds ran throughout (load averages 5 to 36), so the absolute times
spread, and the ratios within a round are the figures to read. The profile
is macOS's `sample` over a loop of loads, as the share of samples at the top
of the stack.

Every scene's animation was compared before and after the change below, as
the exact bits of every moving shape's transform and every glow at 3000
times, and every flight's loop and flash starts: equal for all four scenes
in `scenes/`.

## Results

Before: 92% of the load's samples were in the distance query: 51% in
`shapes::distance` itself, 36% in `placement()`, which checked each still
shape's transform for a rotation and widened it to double on every
question, and 5% in the scan over the still shapes' indices. Allocation,
string building and parsing together were 0.06%; libm's sines and the
closed forms most of the rest.

The change (`shapes::StillShapes`, `src/core/shapes/shapes.h`; CDSA.32,
CACHE.4): each still shape placed once, when the obstacles are made, and
kept per kind as what its test reads, a sphere's center and radius in
double, a box's world-space faces, scanned with no dispatch on the kind and
no index checked per shape. The arithmetic is the per-shape functions' own,
so every answer is the same to the bit.

| `scene::load` | before | after |
|---|---|---|
| marbles, round 1 (min / median) | 428 / 459 ms | 110 / 124 ms |
| marbles, round 2 | 461 / 475 ms | 155 / 165 ms |
| marbles, round 3 | 552 / 587 ms | 151 / 160 ms |
| brass sphere flight (min / median of 21) | 1.8 / 1.8 ms | 1.3 / 1.3 ms |

The marbles load in about a third of the time, 3 to 4 times faster in each
round. After, the scan is still 75% of the samples, now all of it the
distances themselves, and the closed forms and libm's sines 20%.

The per-frame side was timed too, with nothing changed there: `animate()`
for the 616 fireflies and their glows, 2000 frames, median 43 µs on a quiet
moment of the machine and 66 to 76 µs while loaded, under 0.5% of a 16.7 ms
frame.

## When an index would pay

The scan is linear in the still shapes per sample, so a load grows as
fireflies x still shapes. At the marbles' size, 616 fireflies among 51
still shapes, it is some 150 ms on a loaded machine, once, at start-up.
A thousand fireflies among two hundred still shapes would be some six times
that (1000 / 616 x 200 / 51), about a second: the point at which a grid or a BVH behind
contract 11 is worth building, measured then against this scan. Nothing
gates on it in the code: no scene comes near it.

## Not done

- An early-exit query, "is this point at least this far from every still
  shape", comparing squared distances for spheres: it saves a square root
  per sphere, but its rounding differs from the distance's at the
  threshold, so a sample on the edge could be accepted by one and refused
  by the other, and every firefly's loop could change. The flat scan,
  exact, is what CDSA.27 recommends at this size.
- Pools of segments and flash starts, arenas and reserved vectors for the
  load's transient allocations: allocation was 0.06% of the load.
