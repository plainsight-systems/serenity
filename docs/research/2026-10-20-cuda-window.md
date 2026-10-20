# The window on the RTX 3090 Ti

Status: built and run on the monitor, 2026-10-20 (results at the end). The
CUDA backend's plan (`2026-10-19-cuda-backend.md`) left the window as its
own step after Phase 6: "a thin graphics API for presentation alone (CUDA
interop)". This is that step.

## The goal

`serenity`, the window, on the 3090 as on the Mac: the demo graph rendered
live on the 3090's monitor, its frame time in the window's title, the
same program, options and keys. A tool started when wanted: nothing
graphical runs on the machine between runs.

## The machine, checked 2026-10-20

- A monitor on HDMI, 3840 x 1080; no desktop, no display manager, no
  window manager. The user installed the X server alone (Xorg 21.1.12,
  NVIDIA's X driver 595.91.07 matching the kernel driver, libinput, xinit).
- Vulkan on the RTX 3090 Ti, NVIDIA's driver: API 1.4.329. The loader and
  headers are Ubuntu's, `libvulkan-dev` 1.3.275.
- SDL3 3.4.18 is the window's library already (`CMakeLists.txt`), built
  with Vulkan on purpose: "Vulkan stays until the CUDA backend's window
  chooses the graphics API it presents through". Its X11 driver needs the
  X11 development headers to build, which the machine lacks
  (`libx11-dev`, `libxext-dev`; below, "What the user installs").

## What shipped systems do

- **Blender Cycles' viewport** (`intern/cycles/device/cuda/
  graphics_interop.cpp`, `main`, read 2026-10-20): Blender's Vulkan backend
  allocates an exportable buffer, Cycles imports it into CUDA
  (`cuImportExternalMemory`, an opaque file descriptor) and maps it once
  (`cuExternalMemoryGetMappedBuffer`), "always mapped", and CUDA writes the
  frame into it; the display copies it to the screen. Its OpenGL path
  registers a GL buffer instead (`cuGraphicsGLRegisterBuffer`).
- **NVIDIA's CUDA samples** (`simpleVulkan`, `vulkanImageCUDA`): the same
  import, the Vulkan device chosen by the CUDA device's UUID, and the
  ordering on the GPU by Vulkan semaphores exported to CUDA
  (`cuImportExternalSemaphore`, `cuSignalExternalSemaphoresAsync`,
  `cuWaitExternalSemaphoresAsync`), so neither side waits on the CPU.
- **The OptiX SDK's samples** present through GLFW and OpenGL interop.

## The design

**One window program, two platforms.** `src/app/` is the window for both
backends already (`core/presentation/frame_source.h`: the window's events
and its frames' inputs on one side, a backend's frame source on the
other). Its header names this case: a CUDA source "would run the loop an
engine's game loop runs ... the events polled without waiting, an image
acquired, the frame rendered and presented". So `src/app/` gains nothing
of CUDA or Vulkan; `main.cpp` composes the platform's backend, as the
headless program's factory does.

**The window: SDL3 on X11.** The same library, built on Linux too, with
its X11 driver alone (no Wayland, no OpenGL). On Linux the window is made
for Vulkan (`SDL_WINDOW_VULKAN`) and gives the backend what it asks for
by opaque handles, as it gives Metal its layer as a `void*` (`app/
window.h`: the app names no backend type): the instance extensions the
surface needs, and the surface made for an instance.

**Presentation: Vulkan, for presenting alone.** No Vulkan pipeline, no
shader: CUDA renders, Vulkan copies and presents.

- The instance and device: the device whose UUID is the CUDA device's,
  one queue that does transfers and presents; the swapchain at the
  window's size in pixels, FIFO (the display's rate, as the Mac's display
  link paces it), `B8G8R8A8_SRGB` where the surface offers it, else
  `B8G8R8A8_UNORM`.
- The frame target, Cycles' way: one exportable buffer a frame slot,
  RGBA8 at the render size, imported into CUDA once and kept mapped; the
  renderer's tone map writes it as it writes the offscreen target today.
- Each frame, Vulkan copies the slot's buffer into an image of the
  render size and blits that onto the acquired swapchain image: the blit
  scales the render size to the window's (`--scale`, as the Mac's layer
  stretches its drawable) and swaps RGBA to BGRA. Some 33 MB of copies at
  3840 x 1080 a frame, about 0.05 ms at the card's bandwidth.
- The order between the two, NVIDIA's samples' way: two timeline
  semaphores, exported by Vulkan and imported by CUDA. CUDA signals the
  frame's sequence on its image stream after the tone map, and Vulkan's
  copy waits for it; Vulkan signals after its copy, and CUDA's image half
  of the frame that next writes that slot waits for it. So frame N + 1's
  light half runs beside frame N's copy and present, as beside its image
  half today, and the CPU waits only where the submission already does
  (a slot two frames back).

**The frame source: `cuda::SwapchainFrames`**, Metal's DisplayFrames'
counterpart (`metal/frame/display_frames.h`): its loop, each frame: the
client's events polled; the size the client wants (none at 0); the
swapchain remade on a change of size (after the frames in flight); an
image acquired; the client's inputs, at the instant it is expected on
screen (the acquire's time and one refresh, the display's, from SDL); the
frame rendered into the slot's buffer, copied and presented; the client's
shown(), with the frame its submission settled (its GPU instants, CUDA's
timeline: `core/measurement/frame_times.h`) and the bug counters. A
frame's failure ends the run with its message, as Metal's does.

**The renderer** records into a target given by address and size
(`cuda::RenderTarget`), which the offscreen target and the slot's buffer
each give, in place of the offscreen target itself.

**Launching.** `xinit` starts an X server on the monitor with the program
as its one client, and stops it when the program exits: `make window`
on the 3090 (the Mac's `make run`, its graph and scene). Started from ssh,
so the X server's wrapper must allow it (below). Without a window manager
the window opens at the top left, undecorated; it is the size asked for
(`--size`, new, 1280 x 800 by default; `SIZE=3840x1080` for
the monitor's), closed by Escape or Ctrl-C in the
ssh session.

## What the user installs (sudo, once)

```
sudo apt install --no-install-recommends libx11-dev libxext-dev xserver-xorg-legacy
printf 'allowed_users=anybody\nneeds_root_rights=yes\n' | sudo tee /etc/X11/Xwrapper.config
```

The first two, SDL's X11 driver's headers; the third, Xorg's setuid
wrapper (`/usr/lib/xorg/Xorg.wrap`), which a `--no-install-recommends`
install of the X server leaves out: without it Xorg runs as the user and
cannot open the virtual console ("Cannot open virtual console 1
(Permission denied)", the first run's failure); the last, the wrapper
allowing a server started from an ssh session (Ubuntu allows only the
machine's own console by default).

## Dependencies (a dependency change, declared)

- SDL3 3.4.18, the pinned tarball, now built on Linux too: the X11 video
  driver and Vulkan only.
- Vulkan, Ubuntu's loader and headers (`libvulkan-dev` 1.3.275), pinned in
  `cmake/toolchain.json`'s Linux set and checked at configure as the
  others are; the driver it loads is NVIDIA's, already pinned.
- X11's headers, for SDL alone.

Risk surface: behavioral (presentation on Linux only; Metal untouched),
determinism (none: what is rendered is the renderer's, byte-identical
offscreen; a window's frames are not reproducible on either backend, as
on the Mac).

## Tests

- Units, no GPU: the swapchain's choices (format, present mode, extent,
  image count) as functions of what the surface offers; the window's
  size from `--size`.
- The presenter itself is not run by ctest: NVIDIA's Linux driver offers
  no headless surface (`VK_EXT_headless_surface`; checked 2026-10-20:
  only the X11 surfaces), so a presenter needs the X server, which a test
  run does not start. A test of the same Vulkan calls outside the
  presenter would test a copy of it, not it. So the window run on the
  monitor is the presenter's check: the demo graph shown, the frame time
  in its title, frames continuing through a resize of the render size.
- The window itself: run by the user on the monitor; the frame time in
  its title is the measurement (the demo graph at the window's sizes in
  play, 1920 x 1080 and the monitor's 3840 x 1080).

## Alternatives considered

- **OpenGL interop** (Cycles' other path; the OptiX SDK's): less code,
  no swapchain or semaphores of our own, but the GL driver orders CUDA's
  and GL's work by mapping and unmapping on the CPU's side, and SDL is
  built without OpenGL on purpose. Not taken: Vulkan is NVIDIA's current
  path for interop and the one its samples show.
- **GLFW** (the OptiX SDK's window): SDL3 is the window's library
  already, on both platforms.
- **Presenting without X** (`VK_KHR_display`, straight to the monitor) or
  **streaming to the Mac** (NVENC): offered, not chosen by the user.

## Rollback

The window on Linux is `src/cuda/present/`, `cuda::SwapchainFrames`, the
Linux branch of the window's build and `main.cpp`'s composition: removing
them returns the 3090 to the headless program, which nothing here
changes.

## Results, 2026-10-20

Run by `make window GRAPH=graphs/restir_pt_denoised.toml
SCENE=scenes/marbles.toml` on the monitor, release build, nothing else on
the GPU. The picture checked by the user on the monitor: right, as the
Mac's window shows it. The GPU frame time, the window's own line a
second (a live window's, beside the headless program's 11.2 ms at
1920 x 1080, which X's server, the copy and blit and FIFO's pacing share
the GPU with):

| Window | GPU frame time, mean (range over a second) | Frames a second |
|---|---|---|
| 1920 x 1080 | 13.8 to 14.2 ms (13.2 to 15.0) | 60, the display's rate |
| 3840 x 1080 | 17.2 to 17.3 ms (13.6 to 22.9) | 58 to 59 |

Ctrl-C in the ssh session, or Escape, ends the run; X stops with it.
