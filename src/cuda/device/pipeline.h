#pragma once

#include <cuda.h>
#include <optix_types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "core/frame/extent.h"
#include "cuda/device/buffer.h"
#include "cuda/device/optix_context.h"

namespace serenity::cuda {

// Axis: GPU backend.
//
// A pass that traces rays, as OptiX runs it (docs/research/2026-10-19-cuda-
// backend.md, "The design" and "Phase 3's design"): its programs, one `.cu`
// compiled to OptiX-IR at build time and embedded (cmake/CudaModules.cmake),
// made into a module, its program groups and a pipeline at construction,
// and launched over the frame's pixels on the frame's stream. Metal's
// compute pipeline for a kernel that traces (metal/device/library.h), and
// one of Cycles' OptiX pipelines (intern/cycles/device/optix/
// device_impl.cpp), whose programs the pipeline links.
//
//   ray generation   the pass's body: one thread a pixel, tracing by
//                    optixTraverse and reading the hit objects itself
//                    (cuda/acceleration/trace.cuh)
//   hit groups       no any-hit, no closest-hit: the shapes' meshes',
//                    OptiX's built-in triangles with no program at all;
//                    and MNEE's seed's intersection program, which walks a
//                    segment's boxes (integrator/mnee_seed.cuh)
//   miss             no program: a ray that hits nothing leaves a miss in
//                    its hit object, which the ray generation program reads
//
// The shader binding table holds a record for each, and a hit record for
// each structure and program a pass traces with, each record a hit group's
// and its data four 64-bit words the program reads
// (optixGetSbtDataPointer): for the shapes, the addresses of what it tests
// a box's shape with, and of the table that names a box's shape, or 0 for
// none (cuda/acceleration/structure_block.h, StructureRecord). The ray
// generation program picks a record by the SBT offset it traces with. Written
// once, at construction (CDSA.32: static data put once where its reader reads
// it). The triangles' record carries no data: a mesh's hit names its shape
// by its instance (cuda/acceleration/structure_block.h).
//
// What every program of the backend's is compiled for (Programs below, and
// the pipeline's compile options): one instance level over geometry
// structures as the traversable (OptiX's single-level instancing), of
// triangles and of custom primitives, no exceptions enabled (OptiX's
// default), OptiX's default optimization and debug levels, at most the
// program's own register limit (Programs, below), and the launch parameters in
// the variable the programs name. Trace depth 1, no
// callables, a traversable graph two deep, stack sizes from OptiX's own
// helpers (optix_stack_size.h) for those, as Cycles sets its own from its
// groups' sizes.
//
// Made once, outside the frame loop (MEM.9); released with this object
// (R.1), after the frames that launched it (submission.h, Lifetime), before
// the OptiX context it was made from.
//
// Throws CudaError, with OptiX's log, if the IR is empty or OptiX refuses
// the module, a program or the pipeline.
class Pipeline {
 public:
  // A hit record's data: four 64-bit words, addresses the intersection
  // program reads (above).
  using HitData = std::array<std::uint64_t, 4>;

  // A hit record: the hit group it runs, an index into the programs'
  // intersections, and its data.
  struct HitRecord {
    unsigned group = 0;
    HitData data{};
  };

  // What the programs are called, and what the pipeline gives them (I.24:
  // named, so no two strings pass in each other's places).
  struct Programs {
    std::string_view ray_generation;  // "__raygen__<pass>"
    // the hit groups' programs, one each: "__intersection__<what>", or
    // none ("") for built-in triangles
    std::span<const std::string_view> intersections;
    std::string_view launch_parameters;  // the programs' __constant__ name
    // 32-bit payload values a trace carries, as the programs use them
    unsigned payload_values = 0;
    // the hit records, one a structure and program the pass traces with
    std::span<const HitRecord> hit_records;
    // The registers a thread of the programs may hold
    // (OptixModuleCompileOptions::maxRegisterCount), in place of OptiX's
    // own choice of 128. Each pass sets its program's by one rule, the
    // hardware's and the program's (GPU.3): an SM's 65,536 registers make
    // its warps' occupancy a step function of a thread's registers
    // (allocated in eights), so a limit is a step's top, and the program's
    // is the lowest top at which OptiX's compile report (its log at level
    // 4) shows no direct spills: 128 (16 warps an SM), 136, 144, 152, 168
    // (12), 184, 200 (10), 224, 255
    // (docs/research/2026-10-21-tracing-divergence.md, section 8.6).
    int max_registers = 0;
  };

  Pipeline(const OptixContext& context, std::span<const std::byte> ir,
           const Programs& programs);

  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  Pipeline(Pipeline&&) = delete;
  Pipeline& operator=(Pipeline&&) = delete;
  ~Pipeline() = default;

  // Launches the ray generation program over one thread a pixel of `grid`
  // on `stream`, its launch parameters the `bytes` at device address
  // `parameters`, which the stream's earlier work wrote. Throws CudaError
  // if the launch is refused; a fault while it runs is reported by the
  // submission's next wait (error.h).
  void launch(CUstream stream, CUdeviceptr parameters, std::size_t bytes,
              frame::Extent grid) const;

 private:
  // Each OptiX object owned by its own handle, destroyed by OptiX's
  // function for it (R.1, R.20), so a constructor that throws part way
  // releases what it made (C.41) and the members go in reverse order: the
  // pipeline, its groups, their module.
  struct DestroyModule {
    void operator()(OptixModule module) const noexcept;
  };
  struct DestroyGroup {
    void operator()(OptixProgramGroup group) const noexcept;
  };
  struct DestroyPipeline {
    void operator()(OptixPipeline pipeline) const noexcept;
  };
  using ModuleHandle =
      std::unique_ptr<std::remove_pointer_t<OptixModule>, DestroyModule>;
  using GroupHandle =
      std::unique_ptr<std::remove_pointer_t<OptixProgramGroup>, DestroyGroup>;
  using PipelineHandle =
      std::unique_ptr<std::remove_pointer_t<OptixPipeline>, DestroyPipeline>;

  ModuleHandle module_;
  // ray generation, miss, then the hit groups
  std::vector<GroupHandle> groups_;
  PipelineHandle pipeline_;
  DeviceBuffer records_;
  OptixShaderBindingTable table_{};
};

}  // namespace serenity::cuda
