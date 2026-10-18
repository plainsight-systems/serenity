# Frame pacing: the window paced by a Metal display link

2026-10-18. Apple M3 Max, its built-in 120 Hz Liquid Retina XDR display,
macOS 26, the pinned toolchain (`cmake/toolchain.json`), release build.

## What was built

The window used to block on the layer's `nextDrawable()`, display sync on,
three drawables, two frames in flight, its time read from a monotonic
clock when the CPU recorded the frame. It now renders when the display
asks, as Apple paces Metal apps (`logical-overview.md`, "The frame",
"Pacing"; the files in `file-mapping.md`, "Frame pacing"):

- **A `CAMetalDisplayLink` on the window's layer** (macOS 14+,
  `src/metal/device/display_link.{h,mm}`), as Apple's sample "Achieving
  smooth frame rates with a Metal display link" makes it in its
  `GameView`: added to the main run loop in its common modes, a frame
  rate range asked for, a frame latency of two, the delegate's
  `metalDisplayLink:needsUpdate:` rendering into the update's drawable and
  presenting it with `present()` (a timed presentation asserts with a
  display link, Apple's documentation of the callback).
- **The frame's time is the update's `targetPresentationTimestamp`**,
  counted from the first frame's (`src/app/clock.h`), as the sample
  animates by it and its documentation says to: what moves, moves by the
  time between showings.
- **On the main thread**, the sample's default (`RENDER_ON_MAIN_THREAD 1`).
  SDL waits for events by running the main run loop in its default mode
  (`nextEventMatchingMask:untilDate:inMode:NSDefaultRunLoopMode`, SDL
  3.4.18's `Cocoa_PumpEventsUntilDate`), which is one of the common modes,
  so the display link calls the frame while the window waits in
  `SDL_WaitEventTimeout`. A frame that fails cannot throw through Core
  Animation's run loop; Metal's frame source keeps the failure, calls no
  frame after it, and rethrows it from its loop when the window's wait
  returns, each wait at most 100 ms. (A first version woke the wait with
  an SDL event of the window's own; the seam below removed it, since the
  window would then have known it was being woken by a frame.)
- **The window's size is followed by the frame**: SDL's Metal view sets
  the layer's drawable size to the window's full pixels on every resize
  (its event watch on `SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED`), and a frame
  can be called between that and the window's own handling, during a live
  resize in particular (the common modes include AppKit's tracking mode).
  So each frame asks the presenter for the size it renders at, which sets
  the layer's size back when it differs (set only on a change, as the
  sample's `resizeDrawable`), and a drawable made at the size before is
  skipped, not written past.
- A range of 30 to 60 frames a second, preferring 60, the project's
  target (`logical-overview.md`, "The frame"), by the user's decision
  (2026-10-18): Apple's guidance for the range is a rate the app "can
  consistently maintain", and 60 holds at about 1080p but not at the
  window's full size, where a variable-refresh display then shows a slow
  frame late rather than repeating one. First built as 60 alone, as the
  sample asks 120 for its own.

The headless renderer and the measuring program do not present and are
unchanged; every graph's headless output is the same bytes as 78f1c06's
(52 files: every graph in `graphs/`, PNG at 480 x 270 frames 40 and 41,
PFM where a graph has radiance to write, `--features` where it has the
features pass, the denoised graph also at 1920 x 1080 frames 120 and 121
and with time frozen at t = 20).

## The seam: a frame source

At the user's direction, the window's loop is not shaped by the display
link: a Vulkan backend (Linux, the RTX 3090 and Strix Halo) or a
Direct3D 12 one (Xbox Series X) paces by waiting on a swapchain image
and presenting it from its own loop, and a window inverted around the
display link's callback would have to be rewritten, not swapped. So the
core holds an interface (`src/core/presentation/frame_source.h`): a
**frame source** owns the loop and calls its **client**, the window, for
its events (waiting at most as long as the source says) and for each
frame's inputs, given the instant it will be shown; the client is told
what the frame reported once shown. Metal's source
(`src/metal/frame/display_frames.{h,cpp}`) waits in the window's events
while the display link calls frames; the drawable never leaves it. The
window's side (`src/app/window_frames.{h,cpp}`) names no backend, and
`main.cpp` only composes them; `make check` (rule 6) keeps every other
file of `src/app/` free of any backend and all of it free of the display
link.

Checked against production:

- **SDL3's main callbacks** (`SDL_AppIterate`, `SDL_AppEvent`; SDL's
  `docs/README-main-functions.md`) are the shape followed: SDL notes that
  some platforms "would rather be in charge of that `while` loop", and on
  iOS SDL drives `SDL_AppIterate` from a `CADisplayLink` with a frame rate
  range (`src/main/ios/SDL_sysmain_callbacks.m`), on the desktop from a
  plain loop; the app's code is the same either way. The frame source is
  that "in charge" side, and the client the app's callbacks.
- **Unreal** keeps the loop in its engine (it pumps the platform's
  messages, ticks, renders) and the pacing behind its RHI's viewport
  present, which waits on the swapchain: what a Vulkan or Direct3D 12
  source would run inside `run()`. Its loop shape is not taken for the
  window itself, since on Metal the display link, not a loop, decides
  when a frame comes.

## Alternatives

- **A render thread with a run loop of its own** (the sample's
  `RENDER_ON_MAIN_THREAD 0`): not taken. It needs the size handled under a
  lock against AppKit's resize on the main thread, as the sample's
  `@synchronized(_metalLayer)` does, and SDL's own size reset happens on
  the main thread outside any lock of ours; a developer forum report
  (thread 766889) says the sample's threaded path receives no updates on
  macOS. The frame's CPU part is small beside the GPU's.
- **The display link's callback handing its update to the loop**, which
  renders after the wait returns: not taken. The loop does not run while
  AppKit tracks a resize or a menu, so the window would freeze there as
  before, and each frame would be recorded a run-loop turn after the
  display asked for it, the latency the display link is for removing.
- **`CVDisplayLink`**, Apple's older display link for macOS: deprecated in
  the macOS 15 SDK (`CVDisplayLink.h`), and its callback runs on a thread
  of its own, so the main-thread arrangement needs a dispatch source as
  Apple's "Creating a custom Metal view" sample shows. `CAMetalDisplayLink`
  is the one Apple names for Metal.
- **A wait with no bound**: not taken. SDL turns an interrupt (Control-C)
  into its quit event only when it pumps events, so with no other event
  the window would not quit. The wait is bounded at 100 ms.

## Behaviour that changed

- **A window that is not visible renders nothing.** The display link stops
  calling frames while the window is occluded (fully covered, on another
  Space, minimized), as Apple's display link does for power; the old loop
  rendered at the display's full rate into drawables never shown (their
  `presentedTime` 0). Seen here: a window opened behind other windows got
  three frames, then none until it was raised; a visible one ran at 60,
  then stopped when SDL reported it occluded.
- **The title's GPU time is the frame's own time when the display paces
  below what the GPU could do**; the interval between frames is the
  display's then (`core/measurement/frame_times.h`). While the GPU is the
  limit, as the denoised graph's frame at the window's size is, it is the
  interval as before.

## Measurement

The demo graph (`graphs/restir_pt_denoised.toml` over `scenes/marbles.toml`)
in the window at `--scale 0.75`, 1920 x 1200 from the 2560 x 1600 window,
25 s each, on AC power with the GPU otherwise idle (no other serenity or
measuring process, under 5% for 6 s before each run), the window raised
and visible; release build; the first 3 s of frames dropped. Before:
78f1c06 (`nextDrawable()`); after: this change at 75dcfe5, asking 30 to
60 frames a second, preferring 60. Each drawable's `presentedTime` and
each frame's GPU instants logged by scratch instrumentation, not kept.

| | presented interval: median | p5 | p95 | std | GPU interval: median | p5 | p95 | std |
|---|---|---|---|---|---|---|---|---|
| before | 25.0 ms | 8.3 | 33.3 | 9.06 | 19.4 ms | 8.8 | 31.6 | 8.42 |
| after | 16.7 ms | 16.7 | 25.0 | 3.65 | 18.8 ms | 18.3 | 19.4 | 0.47 |

Before, the frames reached the 120 Hz display 8.3, 16.7, 25 and 33.3 ms
apart (357, 162, 409 and 165 of them): the loop ran ahead into three
drawables and the display showed what was ready. After, 16.7 ms (850)
or 25 ms (294) apart, and the GPU's interval between frames is even, at
18.8 ms (it was 19.6 on average, unevenly). The mean frame rate is about
the same (19.6 and 18.8 ms, 51 and 53 fps): the change is in evenness. The GPU's 18.8 ms is
over 60 frames a second's 16.7 at this size, so the display link shows
frames at 60 and, a quarter of the time, at 40: a rate the frame could
hold at this size would be 40 (25 ms every frame); the range 30 to 60 is
the user's.
