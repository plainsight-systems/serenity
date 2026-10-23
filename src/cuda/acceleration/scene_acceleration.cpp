#include "cuda/acceleration/scene_acceleration.h"

#include <optix.h>
#include <optix_stubs.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <map>
#include <string>
#include <utility>

#include "cuda/device/error.h"
#include "cuda/device/frame_sequence.h"

namespace serenity::cuda {

namespace {

// The core's box is OptiX's: two packed triples, min then max.
static_assert(sizeof(shapes::Bounds) == sizeof(OptixAabb));
static_assert(offsetof(shapes::Bounds, min) == offsetof(OptixAabb, minX));
static_assert(offsetof(shapes::Bounds, max) == offsetof(OptixAabb, maxX));
// Contract 10's matrix is an instance's (core/contracts/transform.h).
static_assert(sizeof(contracts::Transform) == sizeof(OptixInstance::transform));
// A mesh's vertex and triangle are OptiX's float3 and uint3.
static_assert(sizeof(contracts::Float3) == 3 * sizeof(float));
static_assert(sizeof(std::array<std::uint32_t, 3>) ==
              3 * sizeof(std::uint32_t));

// Any-hit off for every primitive: a triangle's hit is OptiX's, a box's
// the seed program's, and nothing else runs.
constexpr unsigned geometry_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

OptixAccelBuildOptions build_options(unsigned flags) {
  OptixAccelBuildOptions options{};
  options.buildFlags = flags;
  options.operation = OPTIX_BUILD_OPERATION_BUILD;
  return options;
}

// The meshes, still, built once: traced as fast as OptiX can make them,
// and compacted (scene_acceleration.h).
constexpr unsigned mesh_build =
    OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
// The seed's boxes and the instance level, rebuilt a frame where shapes
// move: traced every frame, built in a few microseconds either way.
constexpr unsigned frame_build = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

// OptiX's one build input for a mesh. The addresses must outlive the
// input's use (OptiX reads them from it): the caller's locals.
OptixBuildInput mesh_input(const CUdeviceptr& vertices,
                           std::uint32_t vertex_count, CUdeviceptr triangles,
                           std::uint32_t count, const unsigned& flags) {
  OptixBuildInput input{};
  input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  // The C API's descriptor is a union, filled by the member its type names,
  // as OptiX's and CUDA's documentation fill it.
  // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
  input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  input.triangleArray.vertexStrideInBytes = sizeof(contracts::Float3);
  input.triangleArray.numVertices = vertex_count;
  input.triangleArray.vertexBuffers = &vertices;
  input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  input.triangleArray.indexStrideInBytes = 3 * sizeof(std::uint32_t);
  input.triangleArray.numIndexTriplets = count;
  input.triangleArray.indexBuffer = triangles;
  input.triangleArray.flags = &flags;
  input.triangleArray.numSbtRecords = 1;
  // NOLINTEND(cppcoreguidelines-pro-type-union-access)
  return input;
}

OptixBuildInput box_input(const CUdeviceptr& boxes, std::uint32_t count,
                          const unsigned& flags) {
  OptixBuildInput input{};
  input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
  // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
  input.customPrimitiveArray.aabbBuffers = &boxes;
  input.customPrimitiveArray.numPrimitives = count;
  input.customPrimitiveArray.strideInBytes = sizeof(shapes::Bounds);
  input.customPrimitiveArray.flags = &flags;
  input.customPrimitiveArray.numSbtRecords = 1;
  // NOLINTEND(cppcoreguidelines-pro-type-union-access)
  return input;
}

OptixBuildInput instance_input(CUdeviceptr instances, std::uint32_t count) {
  OptixBuildInput input{};
  input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
  input.instanceArray.instances = instances;
  input.instanceArray.numInstances = count;
  // NOLINTEND(cppcoreguidelines-pro-type-union-access)
  return input;
}

OptixAccelBufferSizes sizes_of(OptixDeviceContext context,
                               const OptixAccelBuildOptions& options,
                               const OptixBuildInput& input) {
  OptixAccelBufferSizes sizes{};
  check_optix(
      optixAccelComputeMemoryUsage(context, &options, &input, 1, &sizes),
      "SceneAcceleration: optixAccelComputeMemoryUsage");
  return sizes;
}

// a x b, each a 3 x 4 affine matrix with the row (0, 0, 0, 1) understood.
// Contract 10's matrix is a C array (MSL has no std::array), indexed by
// these loops' bounds.
contracts::Transform compose(const contracts::Transform& a,
                             const contracts::Transform& b) {
  contracts::Transform c{};
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
  for (int r = 0; r < 3; ++r) {
    for (int k = 0; k < 4; ++k) {
      float sum = k == 3 ? a.m[r][3] : 0.0f;
      for (int j = 0; j < 3; ++j) {
        sum += a.m[r][j] * b.m[j][k];
      }
      c.m[r][k] = sum;
    }
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
  return c;
}

constexpr contracts::Transform identity{{
    {1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, 0.0f},
}};

}  // namespace

SceneAcceleration::SceneAcceleration(const OptixContext& context,
                                     Submission& submission,
                                     const shapes::Shapes& shapes, Roles roles)
    : context_(context.handle()),
      shapes_(shapes),
      moving_(roles.moving.begin(), roles.moving.end()) {
  if (const std::optional<std::string> reason =
          shapes::structure_refusal(shapes, roles.moving, roles.lights)) {
    throw CudaError("SceneAcceleration: " + *reason);
  }
  const std::size_t count = shapes.records.size();
  if (roles.levels.size() != count) {
    throw CudaError("SceneAcceleration: not one level a shape");
  }

  // The meshes: the unit box's, if a box is in the scene, and an icosphere
  // a level the spheres use; each shape's, by index.
  std::vector<shapes::TriangleMesh> meshes;
  std::map<unsigned, std::uint32_t> mesh_of_level;
  std::optional<std::uint32_t> box_mesh;
  mesh_of_.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    switch (shapes.records[i].kind) {
      case shapes::ShapeKind::sphere: {
        const unsigned level = roles.levels[i];
        if (level > shapes::finest_sphere_level) {
          throw CudaError("SceneAcceleration: shape " + std::to_string(i) +
                          ", a sphere, has no icosphere level");
        }
        const auto [at, made] = mesh_of_level.try_emplace(
            level, static_cast<std::uint32_t>(meshes.size()));
        if (made) {
          meshes.push_back(shapes::icosphere(level));
        }
        mesh_of_[i] = at->second;
        break;
      }
      case shapes::ShapeKind::box:
        if (!box_mesh) {
          box_mesh = static_cast<std::uint32_t>(meshes.size());
          meshes.push_back(shapes::unit_box());
        }
        mesh_of_[i] = *box_mesh;
        break;
    }
  }
  make_meshes(submission, meshes);

  // The seed's boxes: the occluders' (the core's plan), or every shape's in
  // a scene with no light; none with no surface.
  shapes::StructurePlan plan =
      shapes::plan_structures(shapes, roles.moving, roles.lights);
  const bool no_surfaces = !plan.occluders && plan.occluder_table.empty();
  occluder_count_ = static_cast<std::uint32_t>(plan.occluder_table.size());
  const std::span<const std::uint32_t> table = plan.occluder_table;
  // At least one entry: memory must have bytes; with no surface it is never
  // read.
  occluder_table_ = DeviceBuffer(std::max<std::size_t>(table.size_bytes(), 4));
  if (!table.empty()) {
    upload_now(occluder_table_.address(), std::as_bytes(table),
               "SceneAcceleration");
  }
  if (!no_surfaces) {
    seeds_ =
        make_seeds(submission, plan.occluders ? std::move(*plan.occluders)
                                              : std::move(plan.everything));
  }

  // The instances at rest: each shape's, then the seed's.
  const std::vector<std::uint32_t> lights(roles.lights.begin(),
                                          roles.lights.end());
  std::vector<OptixInstance> at_rest(count + (seeds_ ? 1 : 0));
  for (std::size_t i = 0; i < count; ++i) {
    OptixInstance& instance = at_rest[i];
    const auto shape = static_cast<std::uint32_t>(i);
    place(instance, shape, shapes.transforms[i]);
    instance.instanceId = shape;
    instance.sbtOffset = triangles_record;
    const bool light = std::ranges::binary_search(lights, shape);
    instance.visibilityMask = everything_bit | (light ? 0u : occluder_bit);
    instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    instance.traversableHandle = meshes_[mesh_of_[i]].handle;
  }
  if (seeds_) {
    OptixInstance& seed = at_rest.back();
    std::memcpy(&seed.transform[0], &identity, sizeof(identity));
    seed.instanceId = 0;  // a seed's hit names its shape by the table
    seed.sbtOffset = seed_record;
    seed.visibilityMask = seed_bit;
    seed.flags = OPTIX_INSTANCE_FLAG_NONE;
  }
  instance_count_ = static_cast<std::uint32_t>(at_rest.size());
  instances_ = std::make_unique<FrameUpload>(
      submission, std::as_bytes(std::span<const OptixInstance>(at_rest)));
  const OptixBuildInput input =
      instance_input(instances_->address(), instance_count_);
  const OptixAccelBufferSizes sizes =
      sizes_of(context_, build_options(frame_build), input);
  instance_structure_ = DeviceBuffer(sizes.outputSizeInBytes);
  instance_scratch_ =
      DeviceBuffer(std::max<std::size_t>(sizes.tempSizeInBytes, 1));

  // What does not move, built now, in one submission, and its scratch let
  // go: the seed's boxes where they are still, and the instance level in a
  // still scene. What moves is built by every frame's update().
  Seeds* const still_seeds =
      seeds_ && !seeds_->plan.moves() ? &*seeds_ : nullptr;
  if (moves() && still_seeds == nullptr) {
    return;
  }
  const FrameSlot built = submission.begin();
  if (still_seeds != nullptr) {
    build_seeds(*still_seeds, built.stream);
    // Its handle, known when its build is recorded, in every slot's copy
    // of the seed's instance: no frame is in flight to read them yet
    // (frame_upload.h), and a frame's update() writes it again.
    for (std::uint32_t slot = 0; slot < frames_in_flight; ++slot) {
      instances_->view<OptixInstance>(slot).back().traversableHandle =
          still_seeds->handle;
    }
  }
  if (!moves()) {
    instances_->upload(built);
    build_instances(built.stream);
  }
  submission.commit();
  // Start-up work, not a frame: waited for, so the structure is whole
  // before any frame traces it (GPU.8), and settled here, so it is never
  // measured as a frame (submission.h).
  submission.wait_until_complete(built.sequence);
  // The builds' scratch, theirs alone, and they are done (GPU.9).
  if (!moves()) {
    instance_scratch_ = DeviceBuffer();
  }
  if (still_seeds != nullptr) {
    still_seeds->scratch = DeviceBuffer();
  }
}

void SceneAcceleration::make_meshes(
    Submission& submission, std::span<const shapes::TriangleMesh> meshes) {
  struct Staged {
    DeviceBuffer vertices;
    DeviceBuffer triangles;
    DeviceBuffer scratch;
    DeviceBuffer built;
    OptixTraversableHandle handle = 0;
  };
  std::vector<Staged> staged(meshes.size());
  DeviceBuffer compacted_sizes(meshes.size() * sizeof(std::uint64_t));
  const FrameSlot built = submission.begin();
  for (std::size_t i = 0; i < meshes.size(); ++i) {
    const shapes::TriangleMesh& mesh = meshes[i];
    Staged& s = staged[i];
    s.vertices = DeviceBuffer(mesh.vertices.size() * sizeof(contracts::Float3));
    s.triangles = DeviceBuffer(mesh.triangles.size() *
                               sizeof(std::array<std::uint32_t, 3>));
    upload_now(s.vertices.address(), std::as_bytes(std::span(mesh.vertices)),
               "SceneAcceleration");
    upload_now(s.triangles.address(), std::as_bytes(std::span(mesh.triangles)),
               "SceneAcceleration");
    const CUdeviceptr vertices = s.vertices.address();
    const OptixBuildInput input = mesh_input(
        vertices, static_cast<std::uint32_t>(mesh.vertices.size()),
        s.triangles.address(),
        static_cast<std::uint32_t>(mesh.triangles.size()), geometry_flags);
    const OptixAccelBuildOptions options = build_options(mesh_build);
    const OptixAccelBufferSizes sizes = sizes_of(context_, options, input);
    s.scratch = DeviceBuffer(std::max<std::size_t>(sizes.tempSizeInBytes, 1));
    s.built = DeviceBuffer(sizes.outputSizeInBytes);
    OptixAccelEmitDesc emitted{};
    emitted.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitted.result = compacted_sizes.address() + i * sizeof(std::uint64_t);
    check_optix(optixAccelBuild(context_, built.stream, &options, &input, 1,
                                s.scratch.address(), s.scratch.bytes(),
                                s.built.address(), s.built.bytes(), &s.handle,
                                &emitted, 1),
                "SceneAcceleration: optixAccelBuild");
  }
  submission.commit();
  submission.wait_until_complete(built.sequence);

  std::vector<std::uint64_t> sizes(meshes.size());
  check(cuMemcpyDtoH(sizes.data(), compacted_sizes.address(),
                     sizes.size() * sizeof(std::uint64_t)),
        "SceneAcceleration: cuMemcpyDtoH");
  meshes_.resize(meshes.size());
  const FrameSlot compacting = submission.begin();
  for (std::size_t i = 0; i < meshes.size(); ++i) {
    meshes_[i].structure = DeviceBuffer(sizes[i]);
    check_optix(optixAccelCompact(context_, compacting.stream, staged[i].handle,
                                  meshes_[i].structure.address(), sizes[i],
                                  &meshes_[i].handle),
                "SceneAcceleration: optixAccelCompact");
  }
  submission.commit();
  // The uncompacted copies, their inputs and scratch, go when `staged`
  // does, after the compaction they feed.
  submission.wait_until_complete(compacting.sequence);
}

SceneAcceleration::Seeds SceneAcceleration::make_seeds(
    Submission& submission, shapes::TreePlan plan) const {
  Seeds seeds;
  seeds.plan = std::move(plan);
  const std::span<const shapes::Bounds> at_rest = seeds.plan.at_rest;
  seeds.boxes =
      std::make_unique<FrameUpload>(submission, std::as_bytes(at_rest));
  const CUdeviceptr boxes = seeds.boxes->address();
  const OptixBuildInput input = box_input(
      boxes, static_cast<std::uint32_t>(at_rest.size()), geometry_flags);
  const OptixAccelBufferSizes sizes =
      sizes_of(context_, build_options(frame_build), input);
  seeds.structure = DeviceBuffer(sizes.outputSizeInBytes);
  seeds.scratch = DeviceBuffer(std::max<std::size_t>(sizes.tempSizeInBytes, 1));
  return seeds;
}

void SceneAcceleration::build_seeds(Seeds& seeds, CUstream stream) {
  const CUdeviceptr boxes = seeds.boxes->address();
  const OptixBuildInput input =
      box_input(boxes, static_cast<std::uint32_t>(seeds.plan.members.size()),
                geometry_flags);
  const OptixAccelBuildOptions options = build_options(frame_build);
  check_optix(
      optixAccelBuild(context_, stream, &options, &input, 1,
                      seeds.scratch.address(), seeds.scratch.bytes(),
                      seeds.structure.address(), seeds.structure.bytes(),
                      &seeds.handle, nullptr, 0),
      "SceneAcceleration: optixAccelBuild");
}

void SceneAcceleration::build_instances(CUstream stream) {
  const OptixBuildInput input =
      instance_input(instances_->address(), instance_count_);
  const OptixAccelBuildOptions options = build_options(frame_build);
  check_optix(
      optixAccelBuild(
          context_, stream, &options, &input, 1, instance_scratch_.address(),
          instance_scratch_.bytes(), instance_structure_.address(),
          instance_structure_.bytes(), &instance_handle_, nullptr, 0),
      "SceneAcceleration: optixAccelBuild");
}

void SceneAcceleration::place(OptixInstance& instance, std::uint32_t shape,
                              const contracts::Transform& transform) const {
  contracts::Transform mesh = identity;  // a sphere's: the unit sphere's
  const shapes::ShapeRecord& record = shapes_.records[shape];
  if (record.kind == shapes::ShapeKind::box) {
    // The unit box onto the box's corners, in its own space.
    const shapes::BoxData& box = shapes_.boxes.at(record.geometry);
    mesh = contracts::Transform{{
        {box.max.x - box.min.x, 0.0f, 0.0f, box.min.x},
        {0.0f, box.max.y - box.min.y, 0.0f, box.min.y},
        {0.0f, 0.0f, box.max.z - box.min.z, box.min.z},
    }};
  }
  const contracts::Transform placed = compose(transform, mesh);
  std::memcpy(&instance.transform[0], &placed, sizeof(placed));
}

void SceneAcceleration::update(
    const FrameSlot& frame, std::span<const contracts::Transform> transforms) {
  if (!moves()) {
    throw CudaError("SceneAcceleration::update: nothing in the scene moves");
  }
  if (transforms.size() != shapes_.records.size()) {
    throw CudaError("SceneAcceleration::update: not one transform per shape");
  }
  // Step 2's first half, recorded first so its handle is known for the
  // seed's instance; its boxes are placed and uploaded before it on the
  // stream.
  if (seeds_ && seeds_->plan.moves()) {
    Seeds& seeds = *seeds_;
    const std::span<shapes::Bounds> boxes =
        seeds.boxes->view<shapes::Bounds>(frame.slot);
    shapes::place_moving(seeds.plan, transforms, boxes);
    seeds.boxes->upload(frame);
    build_seeds(seeds, frame.stream);
  }
  // Step 1: the moving shapes' instances, and the seed's handle.
  const std::span<OptixInstance> instances =
      instances_->view<OptixInstance>(frame.slot);
  for (const std::uint32_t shape : moving_) {
    place(instances[shape], shape, transforms[shape]);
  }
  if (seeds_) {
    instances.back().traversableHandle = seeds_->handle;
  }
  instances_->upload(frame);
  // Step 2's second half: the instance level over every instance.
  build_instances(frame.stream);
}

StructureBlock SceneAcceleration::block() const noexcept {
  return StructureBlock{instance_handle_, occluder_count_, 0u};
}

std::array<Pipeline::HitRecord, structure_records>
SceneAcceleration::hit_records(const gpu::SceneBlock& scene,
                               CUdeviceptr transforms) const noexcept {
  const StructureRecord seed{occluder_table_.address(), scene.shapes,
                             scene.boxes, transforms};
  Pipeline::HitData words{};
  static_assert(sizeof(seed) == sizeof(words));
  std::memcpy(words.data(), &seed, sizeof(seed));
  std::array<Pipeline::HitRecord, structure_records> records{};
  records[triangles_record] = {triangles_group, {}};
  records[seed_record] = {seed_group, words};
  return records;
}

}  // namespace serenity::cuda
