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
  Animation's run loop; it keeps the failure and wakes the wait with an
  event of the window's own (`SDL_RegisterEvents`, `SDL_PushEvent`, which
  posts SDL's wake-up event while a wait is under way), and the loop
  rethrows it.
- **The window's size is followed by the frame**: SDL's Metal view sets
  the layer's drawable size to the window's full pixels on every resize
  (its event watch on `SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED`), and a frame
  can be called between that and the window's own handling, during a live
  resize in particular (the common modes include AppKit's tracking mode).
  So each frame asks the presenter for the size it renders at, which sets
  the layer's size back when it differs (set only on a change, as the
  sample's `resizeDrawable`), and a drawable made at the size before is
  skipped, not written past.
- 60 frames a second, the project's target (`logical-overview.md`, "The
  frame"), asked as minimum, maximum and preferred alike, as the sample
  asks 120 for its own; Apple's guidance for the range is a rate the app
  "can consistently maintain".

The headless renderer and the measuring program do not present and are
unchanged; every graph's headless output is the same bytes as 78f1c06's
(52 files: every graph in `graphs/`, PNG at 480 x 270 frames 40 and 41,
PFM where a graph has radiance to write, `--features` where it has the
features pass, the denoised graph also at 1920 x 1080 frames 120 and 121
and with time frozen at t = 20).

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

Not taken: the machine was on battery throughout this change, and the
brief measures on AC power only. What to run, once on AC with the GPU
otherwise idle and the window visible: the demo graph
(`graphs/restir_pt_denoised.toml` over `scenes/marbles.toml`) at
`--scale 0.75` (1920 x 1200 from the 2560 x 1600 window) for 20 s with the
build before and after, each frame's presentation instant (`MTLDrawable`
`presentedTime`) and GPU instants logged by scratch instrumentation, and
the presented interval's median, p5, p95 and standard deviation compared.
