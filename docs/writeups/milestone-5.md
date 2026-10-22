# Milestone 5: the frame on an NVIDIA GPU

![The marbles twenty seconds into the opening, rendered on the RTX 3090 Ti](images/marbles-m5-20s.jpg)

*Twenty seconds into the opening, at 174 frames a second on my RTX 3090 Ti.*

## Where Milestone 4 left the frame

[Milestone 4](milestone-4.md) made the frame fast on my M3 Max: 57 frames
a second at 1920 × 1080, and about 17 at my laptop display's full size of
3456 × 2234. It ended on a question it couldn't answer on the Mac. The
reuse steps, the part of the frame that works like attention, should be
tuned as kernels, and how a kernel should be tiled and laid out depends on
the GPU it runs on.

So Serenity had to leave the Mac. Milestone 4 said a Vulkan backend would
take it to my RTX 3090 Ti, to AMD's Strix Halo and to the Xbox. I changed
that plan before starting. A portable layer would have let me tune for no
GPU in particular, and the point is to tune for each one. So each GPU gets
its vendor's own interface: CUDA with OptiX for ray tracing on NVIDIA,
HIP with HIPRT on AMD, and Direct3D 12 on the Xbox. This milestone is the
first of those, and it has two halves. First the frame is ported as it
is, and then it is designed again for what NVIDIA's hardware is good at.

## Porting the frame

Before writing any CUDA, I moved everything that wasn't Metal's out of
the Metal backend and into a shared core: how the scene is packed for the
GPU, the frame's constants, the network's plan, the rules each graph
refuses. A second backend should only have to say how its GPU does the
work, not decide again what the work is. The Mac's output stayed the same
to the bit through that move.

Then the passes were ported one at a time, each checked against the Mac.
The path tracer became an OptiX program, the way Blender's Cycles uses
OptiX: OptiX finds the hit, and the program decides what to do with it.
ReSTIR PT, MNEE and the denoiser's surface features followed, then the
network on the tensor cores and the depth of field. Each port was rendered
beside the Mac's frame, and the two agree to within a fraction of a level
of brightness, which is what two GPUs rounding differently gives.

The first full frame ran at 89 frames a second at 1920 × 1080, and 25 at
3456 × 2234. That was already faster than the Mac, and nothing in it was
yet built for NVIDIA's hardware.

The 3090 also got a window. Its machine runs Linux, so the window is SDL3
on X11. CUDA can't put an image on the screen by itself, so Vulkan does
that one job: CUDA renders into memory Vulkan can see, and the two hand
the image back and forth with semaphores, the way Cycles' viewport and
NVIDIA's own samples do. That is the only Vulkan in Serenity.

## Designing the frame for the hardware

A port runs the Mac's design on a different GPU. The second half of the
milestone asked, for each part of the frame, what this GPU has that the
Mac's doesn't, and how shipped renderers use it. Each change was measured
on its own, and every frame rate below is the fully lit scene at
1920 × 1080:

![Frames a second at 1920 × 1080 after each step: Milestone 4 on the M3 Max 57; the port 89; triangles 98; MNEE in rounds 113; a higher register limit 117; lists of pairs 130; fast math 164; each program's own register limit 169; the network on the tensor cores' own shape 173; fused shading and RTXDI's reservoir layout 174](images/m5-steps.svg)

### Every shape as triangles

The 3090's ray-tracing cores test rays against triangles in hardware.
Anything else, a sphere included, is tested by a program running on the
ordinary cores, with a round trip for every candidate. Almost everything
in Serenity is a sphere: the marbles and the fireflies.

So every sphere is now a triangle mesh, which is how NVIDIA made its own
Marbles demo. Each sphere's mesh is fine enough that the gap between it
and the true sphere stays under a thirtieth of a pixel at the largest
size I render, seen from the nearest the camera comes. Most of the
spheres need 20,480 triangles. I also tried coarser meshes, down to a pixel. No facet
showed, but the scene got up to 4% darker, because a coarse mesh tilts
the light the surface reflects. They saved at most 3%, so I kept the
fine ones. The frame went from 89 to 98 frames a second.

### MNEE's walks in rounds

MNEE finds the light the glass marbles focus onto the table by walking
toward a solution, step by step, for each candidate path. A GPU runs
threads in groups of 32 that move together, and the profiler showed what
that meant here. A fifth of the walks never converge, and they take 57% of
all the steps, so most groups spent most of their time waiting on one or
two walks while the rest sat finished.

The fix is a standard one for divergent work on a GPU: take every walk a
few steps at a time, then gather the walks still going into full groups
before the next round. Each walk takes exactly the steps it took before.
MNEE went from 2.6 to 1.0 ms, and the frame rose to 113 frames a
second.

### Registers

A thread's working values live in registers, and each of the GPU's cores
has a fixed number of them to share. A program that uses fewer registers
lets more threads run at once, but one that runs short spills its values
to memory. OptiX picks a limit of 128, and at 128 the tracing programs
were spilling. Raising the limit to 160 took the frame to 117.

Later the limit became each program's own, set by one rule taken from
the hardware: the lowest limit at which the program doesn't spill, at the
points where fewer registers actually let more threads run. The path tree
takes 200, MNEE's last step 168, and the rest 128.

### Both reuses over lists of pairs

ReSTIR PT borrows paths between neighboring pixels and from the frame
before. Each borrowed path has to be shifted, reconnected or replayed to
fit its new pixel, and in a pixel's thread those three kinds of work are
mixed, so a group's threads were often doing different things. The
ReSTIR PT Enhanced paper sorts the work first: a short pass lists every
pair that needs a reconnection and every pair that needs a replay, and
each list is then run one thread to a pair. Doing that took the frame
from 117 to 130.

### Fast math

Cycles and pbrt-v4 both compile their GPU code with fast math. It allows
the compiler slightly less precise divisions, square roots and
transcendentals, and it was the largest single step in this milestone:
130 to 164 frames a second. It does change the picture: 61% of the
pixels move by one or two levels of 255, and the frame's average
brightness by less than a fifth of a level. Every GPU test that compares
the 3090 against the CPU still passes at its old tolerance. It's a
quality trade, so I looked at the two frames side by side before taking
it.

After fast math the programs needed fewer registers, which is when each
program got its own limit: 169 frames a second.

### The network on the tensor cores' own shape

Milestone 3's network ran its convolutions on the tensor cores from the
start, but in the smallest shape they offer. The 3090's tensor cores have
a larger native shape that does twice the work per instruction. NVIDIA's
own convolution library, CUTLASS, also copies each tile into on-chip
memory without passing it through the threads' registers. I followed
CUTLASS on both. The convolutions take 14% less time, and the frame went
from 169 to 173. Adding up sixteen products at once rounds differently
from adding them eight at a time, so 0.3% of the pixels moved by a level.

### Shading where the paths are merged, and RTXDI's reservoir layout

Two last changes came from NVIDIA's RTXDI. Each pixel's light is now
shaded in the same pass that merges its borrowed paths, instead of in a
pass of its own. The reservoirs, the per-pixel records the reuse steps
read and write, are now stored as RTXDI stores them: in 16 × 16 tiles, so
the neighbors a pixel borrows from sit close together in memory, and in
only the half of the slots that a checkerboard frame traces. At
1920 × 1080 neither moved the frame measurably. At 3456 × 2234, where
those neighbors are spread over more memory, the layout took the frame
from 49.7 to 50.2.

## What didn't pay

About as many changes were measured and dropped as were kept, and two of
them taught me the most.

Cycles runs its path tracer as a *wavefront*: many small kernels, each
doing one step for every path, with the paths sorted by material in
between. It is how a production renderer handles divergence, so I rebuilt
the light's first pass that way. It lost, 141 frames a second against
164. Serenity's paths are short, and on this GPU, which can't reorder
rays in hardware, keeping a path in one program costs less than writing
its state to memory between steps. I reverted it, and kept the code on a
branch.

The frame's launches were another surprise. Serenity runs the light of
one frame and the image of the one before at the same time, on two
streams. Recording the image half as a CUDA graph, which removes the gaps
between its forty-odd small kernels, changed nothing. The light half
fills the image half's gaps already. Once both halves keep the GPU busy,
the frame costs as long as its work takes, and work removed is the only
gain.

Four more didn't pay. Reading the scene through the GPU's read-only path
made no difference, as Cycles' plain loads suggested. Putting every
tracing program in one OptiX pipeline would make each of them run at the
largest program's register count. One shift slowed by a third when I
tried it. Splitting the network's work into smaller pieces to fill the
GPU at its small levels cost more in repeated memory reads than it
gained. Launching MNEE's walks across a fixed set of threads, instead of
one thread per walk, made it more than twice as slow.

## Where it stands

The fully lit scene now runs at 174 frames a second at 1920 × 1080 on my
RTX 3090 Ti, and 50 at 3456 × 2234, twice the port's speed at both sizes.
The M3 Max runs the same frame at 57 and 17. The two GPUs are not
comparable machines, so that isn't the point. The point is how much of
the 3090's speed came from designing for it: half of it.

A few changes would trade image quality for speed, and I've left those
decisions open. One is adding up the network's products in half
precision, which this GPU does twice as fast. Another is a cap on how
long an MNEE walk may take before it's abandoned.

Next comes AMD's Strix Halo, through HIP and HIPRT, designed for its own
hardware in the same way. After that the reuse steps can finally be
tuned as kernels on each GPU, the question that started this.
