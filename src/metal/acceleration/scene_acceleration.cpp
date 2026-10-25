#include "metal/acceleration/scene_acceleration.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "core/camera/thin_lens.h"
#include "core/shapes/mesh_levels.h"
#include "core/shapes/raster_levels.h"
#include "metal/acceleration/structure_block.h"
#include "metal/device/error.h"
#include "metal/device/support.h"

namespace serenity::metal {

// A structure's resource ID is the 64 bits the block holds for it
// (structure_block.h).
// The core's bounds are Metal's bounding boxes, byte for byte, so they are
// written as they are: the same size, each corner at the same offset, and
// bytes that are their value (SL.con.4).
static_assert(sizeof(shapes::Bounds) == sizeof(MTL::AxisAlignedBoundingBox));
static_assert(offsetof(shapes::Bounds, min) ==
              offsetof(MTL::AxisAlignedBoundingBox, min));
static_assert(offsetof(shapes::Bounds, max) ==
              offsetof(MTL::AxisAlignedBoundingBox, max));
static_assert(std::is_trivially_copyable_v<shapes::Bounds>);
static_assert(sizeof(MTL::ResourceID) == sizeof(gpu::InstanceStructure));
static_assert(sizeof(MTL::ResourceID) == sizeof(gpu::BoxStructure));
static_assert(std::is_trivially_copyable_v<gpu::StructureBlock>);
// A mesh's vertex is three packed floats, MTLAttributeFormatFloat3's; its
// triangle three 32-bit indices, MTLIndexTypeUInt32's.
static_assert(sizeof(contracts::Float3) == 3 * sizeof(float));
static_assert(sizeof(std::array<std::uint32_t, 3>) ==
              3 * sizeof(std::uint32_t));

namespace {

using Instance = MTL::IndirectAccelerationStructureInstanceDescriptor;

// Contract 10's row-major 3 x 4 matrix as Metal's instances take it: four
// packed columns.
MTL::PackedFloat4x3 packed(const contracts::Transform& t) {
  return MTL::PackedFloat4x3(
      MTL::PackedFloat3(t.m[0][0], t.m[1][0], t.m[2][0]),
      MTL::PackedFloat3(t.m[0][1], t.m[1][1], t.m[2][1]),
      MTL::PackedFloat3(t.m[0][2], t.m[1][2], t.m[2][2]),
      MTL::PackedFloat3(t.m[0][3], t.m[1][3], t.m[2][3]));
}

// A structure and scratch sized for `descriptor`; throws MetalError when
// the device makes neither.
std::pair<NS::SharedPtr<MTL::AccelerationStructure>, NS::SharedPtr<MTL::Buffer>>
make_structure(MTL::Device* device,
               const MTL4::AccelerationStructureDescriptor* descriptor) {
  const MTL::AccelerationStructureSizes sizes =
      device->accelerationStructureSizes(descriptor);
  auto structure = NS::TransferPtr(
      device->newAccelerationStructure(sizes.accelerationStructureSize));
  auto scratch = NS::TransferPtr(
      device->newBuffer(std::max<NS::UInteger>(sizes.buildScratchBufferSize, 1),
                        MTL::ResourceStorageModePrivate));
  if (!structure || !scratch) {
    throw MetalError(
        "SceneAcceleration: the device made no acceleration structure or "
        "scratch buffer");
  }
  return {structure, scratch};
}

MTL4::BufferRange whole(const MTL::Buffer* buffer) {
  return MTL4::BufferRange(buffer->gpuAddress(), buffer->length());
}

// A compute encoder for a start-up submission's builds; throws MetalError
// if the command buffer makes none.
MTL4::ComputeCommandEncoder* encoder_of(const FrameSlot& slot) {
  MTL4::ComputeCommandEncoder* encoder = slot.commands->computeCommandEncoder();
  if (encoder == nullptr) {
    throw MetalError(
        "SceneAcceleration: the command buffer made no compute encoder");
  }
  return encoder;
}

// A residency set for one start-up submission's own allocations.
NS::SharedPtr<MTL::ResidencySet> residency_set(MTL::Device* device) {
  auto descriptor =
      NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
  NS::Error* error = nullptr;
  auto set = NS::TransferPtr(device->newResidencySet(descriptor.get(), &error));
  if (!set) {
    throw MetalError("SceneAcceleration: the device made no residency set: " +
                     describe(error));
  }
  return set;
}

// Point `p` placed by `t` (contract 10's row-major 3 x 4 matrix).
contracts::Float3 placed_point(const contracts::Transform& t,
                               contracts::Float3 p) {
  const auto row = [&](std::size_t r) {
    return t.m[r][0] * p.x + t.m[r][1] * p.y + t.m[r][2] * p.z + t.m[r][3];
  };
  return contracts::Float3{row(0), row(1), row(2)};
}

// A triangle's per-primitive data: its vertices from `vertices` by
// `triangle`, and `shape` (structure_block.h, MeshTriangle).
gpu::MeshTriangle mesh_triangle(std::span<const contracts::Float3> vertices,
                                const std::array<std::uint32_t, 3>& triangle,
                                std::uint32_t shape) {
  gpu::MeshTriangle data{};
  for (std::size_t c = 0; c < 3; ++c) {
    const contracts::Float3& v = vertices[triangle[c]];
    data.v[c][0] = v.x;
    data.v[c][1] = v.y;
    data.v[c][2] = v.z;
  }
  data.shape = shape;
  return data;
}

}  // namespace

const shapes::TriangleMesh& SceneAcceleration::mesh_for(
    std::uint32_t shape, std::span<const std::uint8_t> levels,
    MeshCache& cache) const {
  switch (records_[shape].kind) {
    case shapes::ShapeKind::sphere: {
      const std::uint8_t level = levels[shape];
      if (level > shapes::finest_sphere_level) {
        throw MetalError("SceneAcceleration: shape " + std::to_string(shape) +
                         ", a sphere, has no mesh level");
      }
      if (!cache.levels.at(level)) {
        cache.levels.at(level) = shapes::icosphere(level);
      }
      return *cache.levels.at(level);
    }
    case shapes::ShapeKind::box:
      if (!cache.box) {
        cache.box = shapes::unit_box();
      }
      return *cache.box;
  }
  throw MetalError("SceneAcceleration: shape " + std::to_string(shape) +
                   " is of no kind");
}

void SceneAcceleration::make_meshes(const Device& device,
                                    Submission& submission,
                                    std::span<const MeshArrays> made) {
  MTL::Device* metal_device = device.handle();

  // Each mesh's arrays and its structure, built in one submission for fast
  // intersection, then compacted in a second (Apple's build options;
  // Cycles' static builds).
  struct Building {
    NS::SharedPtr<MTL4::PrimitiveAccelerationStructureDescriptor> descriptor;
    NS::SharedPtr<MTL::AccelerationStructure> structure;
    NS::SharedPtr<MTL::Buffer> scratch;
  };
  std::vector<Building> building(made.size());
  const NS::SharedPtr<MTL::ResidencySet> build_only =
      residency_set(metal_device);
  meshes_.resize(made.size());
  for (std::size_t m = 0; m < made.size(); ++m) {
    const MeshArrays& mesh = made[m];
    const std::vector<gpu::MeshTriangle>& triangles = mesh.data;
    const std::array<std::span<const std::byte>, 3> arrays{
        std::as_bytes(std::span(mesh.vertices)),
        std::as_bytes(std::span(mesh.triangles)),
        std::as_bytes(std::span(triangles))};
    meshes_[m].data =
        std::make_unique<StaticArrays>(device, submission, arrays);
    const StaticArrays& data = *meshes_[m].data;

    auto geometry = NS::TransferPtr(
        MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init());
    geometry->setVertexBuffer(MTL4::BufferRange(
        data.address(0), mesh.vertices.size() * sizeof(contracts::Float3)));
    geometry->setVertexFormat(MTL::AttributeFormatFloat3);
    geometry->setVertexStride(sizeof(contracts::Float3));
    geometry->setIndexBuffer(MTL4::BufferRange(
        data.address(1),
        mesh.triangles.size() * sizeof(std::array<std::uint32_t, 3>)));
    geometry->setIndexType(MTL::IndexTypeUInt32);
    geometry->setTriangleCount(mesh.triangles.size());
    geometry->setOpaque(true);
    geometry->setPrimitiveDataBuffer(MTL4::BufferRange(
        data.address(2), triangles.size() * sizeof(gpu::MeshTriangle)));
    geometry->setPrimitiveDataStride(sizeof(gpu::MeshTriangle));
    geometry->setPrimitiveDataElementSize(sizeof(gpu::MeshTriangle));

    Building& b = building[m];
    b.descriptor = NS::TransferPtr(
        MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init());
    const std::array<const NS::Object*, 1> geometries{geometry.get()};
    b.descriptor->setGeometryDescriptors(
        NS::Array::array(geometries.data(), geometries.size()));
    b.descriptor->setUsage(
        MTL::AccelerationStructureUsagePreferFastIntersection);
    std::tie(b.structure, b.scratch) =
        make_structure(metal_device, b.descriptor.get());
    build_only->addAllocation(b.structure.get());
    build_only->addAllocation(b.scratch.get());
  }
  // The compacted sizes, written by the GPU: an 8-byte slot a mesh,
  // zeroed, so a 32-bit or a 64-bit count reads the same (little-endian);
  // Metal 4's command takes a range, not the count's type.
  auto sizes = NS::TransferPtr(metal_device->newBuffer(
      std::max<std::size_t>(made.size(), 1) * sizeof(std::uint64_t),
      MTL::ResourceStorageModeShared));
  if (!sizes) {
    throw MetalError("SceneAcceleration: the device made no buffer");
  }
  std::memset(sizes->contents(), 0, sizes->length());
  build_only->addAllocation(sizes.get());
  build_only->commit();
  {
    const FrameSlot build = submission.begin();
    build.commands->useResidencySet(build_only.get());
    MTL4::ComputeCommandEncoder* encoder = encoder_of(build);
    for (const Building& b : building) {
      encoder->buildAccelerationStructure(b.structure.get(), b.descriptor.get(),
                                          whole(b.scratch.get()));
    }
    encoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure,
                                       MTL::StageAccelerationStructure,
                                       MTL4::VisibilityOptionDevice);
    for (std::size_t m = 0; m < building.size(); ++m) {
      encoder->writeCompactedAccelerationStructureSize(
          building[m].structure.get(),
          MTL4::BufferRange(sizes->gpuAddress() + m * sizeof(std::uint64_t),
                            sizeof(std::uint64_t)));
    }
    encoder->endEncoding();
    submission.commit();
    // Start-up work: waited for, so the sizes are read once written
    // (GPU.8), and never measured as a frame (submission.h).
    submission.wait_until_complete(build.sequence);
  }
  // The compacted copies, and the build's memory let go (GPU.9).
  const NS::SharedPtr<MTL::ResidencySet> compact_only =
      residency_set(metal_device);
  const auto* compacted = static_cast<const std::uint64_t*>(sizes->contents());
  for (std::size_t m = 0; m < building.size(); ++m) {
    meshes_[m].structure =
        NS::TransferPtr(metal_device->newAccelerationStructure(compacted[m]));
    if (!meshes_[m].structure) {
      throw MetalError(
          "SceneAcceleration: the device made no compacted structure");
    }
    compact_only->addAllocation(building[m].structure.get());
    compact_only->addAllocation(meshes_[m].structure.get());
  }
  compact_only->commit();
  {
    const FrameSlot compact = submission.begin();
    compact.commands->useResidencySet(compact_only.get());
    MTL4::ComputeCommandEncoder* encoder = encoder_of(compact);
    for (std::size_t m = 0; m < building.size(); ++m) {
      encoder->copyAndCompactAccelerationStructure(building[m].structure.get(),
                                                   meshes_[m].structure.get());
    }
    encoder->endEncoding();
    submission.commit();
    submission.wait_until_complete(compact.sequence);
  }
  for (Mesh& mesh : meshes_) {
    // Traced every frame, through the instances: resident until this
    // object is destroyed.
    mesh.resident = submission.keep_resident(mesh.structure.get());
  }
}

void SceneAcceleration::make_seeds(const Device& device, Submission& submission,
                                   shapes::TreePlan plan,
                                   MTL::ResidencySet* build_only) {
  Seeds& seeds = seeds_.emplace();
  seeds.plan = std::move(plan);
  const bool moves = seeds.moves();
  const std::size_t members = seeds.plan.members.size();
  seeds.boxes = std::make_unique<FrameArray>(
      device, submission, std::as_bytes(std::span(seeds.plan.at_rest)),
      moves ? FrameArray::Copies::per_frame : FrameArray::Copies::one);
  // Box i's shape, its per-primitive data (structure_block.h).
  const std::array<std::span<const std::byte>, 1> shape_bytes{
      std::as_bytes(std::span(seeds.plan.members))};
  seeds.shapes =
      std::make_unique<StaticArrays>(device, submission, shape_bytes);
  MTL::Device* metal_device = device.handle();
  const std::uint32_t structures = moves ? frames_in_flight : 1u;
  for (std::uint32_t slot = 0; slot < structures; ++slot) {
    auto geometry = NS::TransferPtr(
        MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::alloc()
            ->init());
    geometry->setBoundingBoxBuffer(MTL4::BufferRange(
        seeds.boxes->address(slot), seeds.boxes->bytes(slot).size()));
    geometry->setBoundingBoxCount(members);
    geometry->setBoundingBoxStride(sizeof(shapes::Bounds));
    geometry->setIntersectionFunctionTableOffset(0);
    geometry->setPrimitiveDataBuffer(MTL4::BufferRange(
        seeds.shapes->address(0), members * sizeof(std::uint32_t)));
    geometry->setPrimitiveDataStride(sizeof(std::uint32_t));
    geometry->setPrimitiveDataElementSize(sizeof(std::uint32_t));
    auto& descriptor = seeds.descriptors.at(slot);
    descriptor = NS::TransferPtr(
        MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init());
    const std::array<const NS::Object*, 1> geometries{geometry.get()};
    descriptor->setGeometryDescriptors(
        NS::Array::array(geometries.data(), geometries.size()));
    Built& built = seeds.structures.at(slot);
    std::tie(built.structure, built.scratch) =
        make_structure(metal_device, descriptor.get());
    built.structure_resident = submission.keep_resident(built.structure.get());
    if (moves) {
      built.scratch_resident = submission.keep_resident(built.scratch.get());
    } else {
      build_only->addAllocation(built.scratch.get());
    }
  }
}

void SceneAcceleration::place(std::uint32_t instance, std::uint32_t shape,
                              const contracts::Transform& transform,
                              std::span<Instance> instances) const {
  instances[instance].transformationMatrix = packed(
      shapes::instance_transform(records_[shape], box_geometry_, transform));
}

SceneAcceleration::SceneAcceleration(const Device& device,
                                     Submission& submission,
                                     const shapes::Shapes& shapes, Roles roles)
    : records_(shapes.records),
      box_geometry_(shapes.boxes),
      moving_(roles.moving.begin(), roles.moving.end()) {
  if (const std::optional<std::string> reason =
          shapes::structure_refusal(shapes, roles.moving, roles.lights)) {
    throw MetalError("SceneAcceleration: " + *reason);
  }
  if (roles.levels.size() != records_.size()) {
    throw MetalError("SceneAcceleration: not one mesh level a shape");
  }
  moves_ = !moving_.empty();

  const auto pool = scoped_pool();
  MTL::Device* metal_device = device.handle();

  std::vector<bool> light(records_.size(), false);
  for (const std::uint32_t l : roles.lights) {
    light.at(l) = true;
  }
  std::vector<bool> moving(records_.size(), false);
  for (const std::uint32_t m : moving_) {
    moving.at(m) = true;
  }
  // Which shapes are their own instances (scene_acceleration.h): every
  // moving shape and every box; the still spheres are concatenated.
  for (std::uint32_t shape = 0; shape < records_.size(); ++shape) {
    if (moving[shape] || records_[shape].kind == shapes::ShapeKind::box) {
      instanced_.push_back(shape);
    }
  }
  // The structures' triangles: each instanced shape's mesh, each level and
  // the box once; then the still occluders' and the still lights' spheres,
  // each placed in the world by its instance transform at rest and
  // concatenated (structure_block.h, still_instance).
  MeshCache cache;
  std::vector<MeshArrays> made;
  std::array<std::optional<std::uint32_t>, shapes::finest_sphere_level + 1>
      level_mesh{};
  std::optional<std::uint32_t> box_mesh;
  shape_mesh_.assign(records_.size(), 0u);
  for (const std::uint32_t shape : instanced_) {
    std::optional<std::uint32_t>& index =
        records_[shape].kind == shapes::ShapeKind::box
            ? box_mesh
            : level_mesh.at(std::min<std::size_t>(roles.levels[shape],
                                                  shapes::finest_sphere_level));
    const shapes::TriangleMesh& mesh = mesh_for(shape, roles.levels, cache);
    if (!index) {
      index = static_cast<std::uint32_t>(made.size());
      MeshArrays arrays{mesh.vertices, mesh.triangles, {}};
      arrays.data.reserve(mesh.triangles.size());
      for (const auto& triangle : mesh.triangles) {
        arrays.data.push_back(
            mesh_triangle(mesh.vertices, triangle, gpu::still_instance));
      }
      made.push_back(std::move(arrays));
    }
    shape_mesh_[shape] = *index;
  }
  std::array<MeshArrays, 2> still{};  // the occluders', the lights'
  for (std::uint32_t shape = 0; shape < records_.size(); ++shape) {
    if (moving[shape] || records_[shape].kind == shapes::ShapeKind::box) {
      continue;
    }
    const shapes::TriangleMesh& mesh = mesh_for(shape, roles.levels, cache);
    const contracts::Transform placement = shapes::instance_transform(
        records_[shape], box_geometry_, shapes.transforms.at(shape));
    MeshArrays& into = still.at(light[shape] ? 1 : 0);
    const auto first = static_cast<std::uint32_t>(into.vertices.size());
    for (const contracts::Float3& v : mesh.vertices) {
      into.vertices.push_back(placed_point(placement, v));
    }
    const std::span<const contracts::Float3> world(into.vertices.data() + first,
                                                   mesh.vertices.size());
    for (const auto& triangle : mesh.triangles) {
      into.triangles.push_back(
          {first + triangle[0], first + triangle[1], first + triangle[2]});
      // the vertices by the mesh's own indices, into its placed copies
      into.data.push_back(mesh_triangle(world, triangle, shape));
    }
  }
  std::array<std::optional<std::uint32_t>, 2> still_mesh{};
  for (std::size_t s = 0; s < still.size(); ++s) {
    if (!still.at(s).triangles.empty()) {
      still_mesh.at(s) = static_cast<std::uint32_t>(made.size());
      made.push_back(std::move(still.at(s)));
    }
  }
  make_meshes(device, submission, made);

  // The rasterizer's sphere levels: every level up to the finest an
  // instanced sphere takes in the tracer, a mesh each, the tracer's where
  // it has one, else one drawn alone, its arrays but no structure
  // (raster()).
  std::vector<std::size_t> triangle_counts;
  triangle_counts.reserve(made.size());
  for (const MeshArrays& arrays : made) {
    triangle_counts.push_back(arrays.triangles.size());
  }
  trace_level_.assign(records_.size(), shapes::no_level);
  std::optional<unsigned> raster_finest;
  for (const std::uint32_t shape : instanced_) {
    if (records_[shape].kind == shapes::ShapeKind::sphere) {
      trace_level_[shape] = roles.levels[shape];
      raster_finest =
          std::max(raster_finest.value_or(0u), unsigned{roles.levels[shape]});
    }
  }
  if (raster_finest) {
    raster_mesh_.resize(*raster_finest + 1);
    for (unsigned level = 0; level <= *raster_finest; ++level) {
      if (level_mesh.at(level)) {
        raster_mesh_[level] = *level_mesh.at(level);
        continue;
      }
      if (!cache.levels.at(level)) {
        cache.levels.at(level) = shapes::icosphere(level);
      }
      const shapes::TriangleMesh& mesh = *cache.levels.at(level);
      std::vector<gpu::MeshTriangle> data;
      data.reserve(mesh.triangles.size());
      for (const auto& triangle : mesh.triangles) {
        data.push_back(
            mesh_triangle(mesh.vertices, triangle, gpu::still_instance));
      }
      const std::array<std::span<const std::byte>, 3> arrays{
          std::as_bytes(std::span(mesh.vertices)),
          std::as_bytes(std::span(mesh.triangles)),
          std::as_bytes(std::span(data))};
      raster_mesh_[level] = static_cast<std::uint32_t>(meshes_.size());
      Mesh& drawn = meshes_.emplace_back();
      drawn.data = std::make_unique<StaticArrays>(device, submission, arrays);
      triangle_counts.push_back(mesh.triangles.size());
    }
  }
  raster_box_ = box_mesh;

  // Each mesh's record and draw; each shape's mesh (an instanced shape's
  // the tracer's, which says only that it is instanced: the visibility
  // buffer names the mesh drawn, visibility.h).
  std::vector<gpu::MeshRecord> records(meshes_.size());
  std::vector<std::uint32_t> shape_mesh(records_.size(), 0u);
  mesh_draws_.resize(meshes_.size());
  for (std::uint32_t m = 0; m < meshes_.size(); ++m) {
    const bool in_world = still_mesh[0] == m || still_mesh[1] == m;
    const StaticArrays& data = *meshes_[m].data;
    records[m] = gpu::MeshRecord{data.address(2), in_world ? 1u : 0u, 0u};
    mesh_draws_[m] = Draw{data.address(0),
                          data.address(1),
                          data.address(2),
                          static_cast<std::uint32_t>(3 * triangle_counts[m]),
                          0u,
                          0u,
                          in_world};
  }
  for (std::uint32_t shape = 0; shape < records_.size(); ++shape) {
    const bool own =
        moving[shape] || records_[shape].kind == shapes::ShapeKind::box;
    shape_mesh[shape] =
        own ? shape_mesh_[shape] : *still_mesh.at(light[shape] ? 1 : 0);
  }
  const std::array<std::span<const std::byte>, 2> tables{
      std::as_bytes(std::span(records)), std::as_bytes(std::span(shape_mesh))};
  mesh_tables_ = std::make_unique<StaticArrays>(device, submission, tables);
  const gpu::MeshBlock block{mesh_tables_->address(0),
                             mesh_tables_->address(1)};
  const std::array<std::span<const std::byte>, 1> block_bytes{
      std::as_bytes(std::span(&block, 1))};
  mesh_block_ = std::make_unique<StaticArrays>(device, submission, block_bytes);
  // A slot's instance list and draws, and the counts, sized once (MEM.9).
  const std::vector<std::uint32_t> no_entries(
      std::max<std::size_t>(instanced_.size(), 1), 0u);
  raster_instances_ = std::make_unique<FrameArray>(
      device, submission, std::as_bytes(std::span(no_entries)),
      FrameArray::Copies::per_frame);
  for (std::vector<Draw>& draws : raster_draws_) {
    draws.reserve(meshes_.size());
  }
  mesh_counts_.resize(meshes_.size());
  mesh_cursors_.resize(meshes_.size());
  chosen_mesh_.resize(instanced_.size());

  const NS::SharedPtr<MTL::ResidencySet> build_only =
      residency_set(metal_device);

  // The seed's boxes: the occluders' own tree; everything's in a scene with
  // no light; none with no surface (the core's plan).
  shapes::StructurePlan plan =
      shapes::plan_structures(shapes, roles.moving, roles.lights);
  if (plan.occluders) {
    make_seeds(device, submission, std::move(*plan.occluders),
               build_only.get());
  } else if (!plan.occluder_table.empty()) {
    make_seeds(device, submission, std::move(plan.everything),
               build_only.get());
  }

  // The instances at rest: each still structure once, in the world, its
  // mask its shapes' questions (structure_block.h); then each moving shape,
  // its mesh, its mask the questions it answers, its id the shape's index.
  const auto instance_of = [&](std::uint32_t mesh, std::uint32_t mask,
                               std::uint32_t id) {
    Instance instance{};
    instance.options = MTL::AccelerationStructureInstanceOptionOpaque;
    instance.mask = mask;
    instance.intersectionFunctionTableOffset = 0;
    instance.userID = id;
    instance.accelerationStructureID =
        meshes_.at(mesh).structure->gpuResourceID();
    instance.transformationMatrix =
        packed(contracts::Transform{{{1.0f, 0.0f, 0.0f, 0.0f},
                                     {0.0f, 1.0f, 0.0f, 0.0f},
                                     {0.0f, 0.0f, 1.0f, 0.0f}}});
    return instance;
  };
  std::vector<Instance> at_rest;
  at_rest.reserve(2 + moving_.size());
  if (still_mesh[0]) {
    at_rest.push_back(instance_of(*still_mesh[0],
                                  gpu::everything_bit | gpu::occluder_bit,
                                  gpu::still_instance));
  }
  if (still_mesh[1]) {
    at_rest.push_back(
        instance_of(*still_mesh[1], gpu::everything_bit, gpu::still_instance));
  }
  const auto still_count = static_cast<std::uint32_t>(at_rest.size());
  for (std::size_t k = 0; k < instanced_.size(); ++k) {
    const std::uint32_t shape = instanced_[k];
    at_rest.push_back(instance_of(
        shape_mesh_[shape],
        gpu::everything_bit | (light[shape] ? 0u : gpu::occluder_bit), shape));
    const std::uint32_t instance = still_count + static_cast<std::uint32_t>(k);
    place(instance, shape, shapes.transforms.at(shape), at_rest);
    if (moving[shape]) {
      moving_instance_.push_back(instance);
    }
  }
  instances_ = std::make_unique<FrameArray>(
      device, submission, std::as_bytes(std::span(at_rest)),
      moves_ ? FrameArray::Copies::per_frame : FrameArray::Copies::one);
  const std::uint32_t tops = moves_ ? frames_in_flight : 1u;
  for (std::uint32_t slot = 0; slot < tops; ++slot) {
    auto& top = tops_descriptors_.at(slot);
    top = NS::TransferPtr(
        MTL4::InstanceAccelerationStructureDescriptor::alloc()->init());
    top->setInstanceDescriptorType(
        MTL::AccelerationStructureInstanceDescriptorTypeIndirect);
    top->setInstanceDescriptorStride(sizeof(Instance));
    top->setInstanceCount(at_rest.size());
    top->setInstanceDescriptorBuffer(MTL4::BufferRange(
        instances_->address(slot), instances_->bytes(slot).size()));
    Built& built = tops_.at(slot);
    std::tie(built.structure, built.scratch) =
        make_structure(metal_device, top.get());
    built.structure_resident = submission.keep_resident(built.structure.get());
    if (moves_) {
      built.scratch_resident = submission.keep_resident(built.scratch.get());
    } else {
      build_only->addAllocation(built.scratch.get());
    }
  }

  // The block each slot binds, written now and never again: each slot's
  // structures are the same objects every frame.
  // Every shape not a light is an occluder (contract 19).
  const auto occluders =
      static_cast<std::uint32_t>(records_.size() - roles.lights.size());
  std::array<gpu::StructureBlock, frames_in_flight> blocks{};
  for (std::uint32_t slot = 0; slot < frames_in_flight; ++slot) {
    const MTL::ResourceID scene =
        tops_.at(moves_ ? slot : 0u).structure->gpuResourceID();
    // with no surface, never traced (structure_block.h): the scene's
    const MTL::ResourceID seed =
        seeds_ ? seeds_->in(slot).structure->gpuResourceID() : scene;
    blocks.at(slot) =
        gpu::StructureBlock{scene._impl, seed._impl, occluders, 0u};
  }
  blocks_ = std::make_unique<FrameArray>(
      device, submission, std::as_bytes(std::span(blocks.data(), 1)),
      moves_ ? FrameArray::Copies::per_frame : FrameArray::Copies::one);
  if (moves_) {
    for (std::uint32_t slot = 1; slot < frames_in_flight; ++slot) {
      blocks_->view<gpu::StructureBlock>(slot)[0] = blocks.at(slot);
    }
  }

  // The still structures, built now, in one submission.
  const bool top_still = !moves_;
  const bool seeds_still = seeds_ && !seeds_->moves();
  if (!top_still && !seeds_still) {
    return;  // each is built by its slot's first update()
  }
  build_only->commit();
  const FrameSlot build = submission.begin();
  build.commands->useResidencySet(build_only.get());
  MTL4::ComputeCommandEncoder* encoder = encoder_of(build);
  if (seeds_still) {
    const Built& built = seeds_->structures[0];
    encoder->buildAccelerationStructure(built.structure.get(),
                                        seeds_->descriptors[0].get(),
                                        whole(built.scratch.get()));
  }
  if (top_still) {
    encoder->buildAccelerationStructure(tops_[0].structure.get(),
                                        tops_descriptors_[0].get(),
                                        whole(tops_[0].scratch.get()));
  }
  encoder->endEncoding();
  submission.commit();
  // Start-up work, not a frame: waited for here, so the structures are
  // whole before any frame traces them (GPU.8), and settled here, so it is
  // never measured as a frame (submission.h). The scratch was the builds'
  // alone: let go rather than held for the run (GPU.9).
  submission.wait_until_complete(build.sequence);
  if (seeds_still) {
    seeds_->structures[0].scratch.reset();
  }
  if (top_still) {
    tops_[0].scratch.reset();
  }
}

void SceneAcceleration::update(
    MTL4::ComputeCommandEncoder* encoder, std::uint32_t slot,
    std::span<const contracts::Transform> transforms) {
  if (!moves_) {
    throw MetalError("SceneAcceleration::update: nothing in the scene moves");
  }
  if (encoder == nullptr || slot >= frames_in_flight) {
    throw MetalError(
        "SceneAcceleration::update: no encoder, or no frame slot " +
        std::to_string(slot));
  }
  if (transforms.size() != records_.size()) {
    throw MetalError("SceneAcceleration::update: not one transform per shape");
  }
  // Step 1: the moving shapes' instances, and the seed's moving boxes.
  const std::span<Instance> instances = instances_->view<Instance>(slot);
  for (std::size_t k = 0; k < moving_.size(); ++k) {
    place(moving_instance_[k], moving_[k], transforms[moving_[k]], instances);
  }
  // Step 2: the seed's structure where it moves, and the instance
  // structure, each over the slot's own copy.
  if (seeds_ && seeds_->moves()) {
    shapes::place_moving(seeds_->plan, transforms,
                         seeds_->boxes->view<shapes::Bounds>(slot));
    const Built& built = seeds_->structures.at(slot);
    encoder->buildAccelerationStructure(built.structure.get(),
                                        seeds_->descriptors.at(slot).get(),
                                        whole(built.scratch.get()));
  }
  const Built& top = tops_.at(slot);
  encoder->buildAccelerationStructure(top.structure.get(),
                                      tops_descriptors_.at(slot).get(),
                                      whole(top.scratch.get()));
  // Step 3: every pass after it traces the finished structures.
  encoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure,
                                     MTL::StageDispatch,
                                     MTL4::VisibilityOptionDevice);
}

SceneAcceleration::Raster SceneAcceleration::raster(
    std::uint32_t slot, std::span<const contracts::Transform> transforms,
    const contracts::Camera& camera, frame::Extent size) {
  if (slot >= frames_in_flight || transforms.size() != records_.size()) {
    throw MetalError("SceneAcceleration::raster: no frame slot " +
                     std::to_string(slot) + ", or not one transform per shape");
  }
  const contracts::CameraData framed = camera::shader_form(camera, size);
  std::ranges::fill(mesh_counts_, 0u);
  // Each instanced shape's mesh this frame, or none out of the view.
  constexpr std::uint32_t culled = 0xffffffffu;
  for (std::size_t k = 0; k < instanced_.size(); ++k) {
    const std::uint32_t shape = instanced_[k];
    std::uint32_t& mesh = chosen_mesh_[k];
    if (records_[shape].kind == shapes::ShapeKind::box) {
      mesh = *raster_box_;  // the scenes' few boxes, always drawn
    } else {
      const contracts::Float3 center =
          contracts::translation(transforms[shape]);
      const double radius = contracts::scale(transforms[shape]);
      if (!shapes::in_view(framed, center, radius)) {
        mesh = culled;
        continue;
      }
      const unsigned traced = trace_level_[shape];
      const unsigned level = std::min(
          shapes::raster_level(radius, center, camera).value_or(traced),
          traced);
      mesh = raster_mesh_.at(level);
    }
    ++mesh_counts_[mesh];
  }
  // The entries grouped by mesh, a draw each mesh drawn.
  std::vector<Draw>& draws = raster_draws_.at(slot);
  draws.clear();
  std::uint32_t next = 0;
  for (std::uint32_t m = 0; m < meshes_.size(); ++m) {
    mesh_cursors_[m] = next;
    const Draw& built = mesh_draws_[m];
    if (built.in_world) {
      draws.push_back(built);
    } else if (mesh_counts_[m] > 0) {
      Draw drawn = built;
      drawn.first_shape = next;
      drawn.shape_count = mesh_counts_[m];
      draws.push_back(drawn);
    }
    next += mesh_counts_[m];
  }
  const std::span<std::uint32_t> entries =
      raster_instances_->view<std::uint32_t>(slot);
  for (std::size_t k = 0; k < instanced_.size(); ++k) {
    const std::uint32_t mesh = chosen_mesh_[k];
    if (mesh != culled) {
      entries[mesh_cursors_[mesh]++] =
          (mesh << gpu::instance_mesh_shift) | instanced_[k];
    }
  }
  return Raster{draws, raster_instances_->address(slot)};
}

MTL::GPUAddress SceneAcceleration::mesh_block() const {
  return mesh_block_->address(0);
}

MTL::GPUAddress SceneAcceleration::address(std::uint32_t slot) const {
  if (slot >= frames_in_flight) {
    throw MetalError("SceneAcceleration::address: no frame slot " +
                     std::to_string(slot));
  }
  return blocks_->address(slot);
}

}  // namespace serenity::metal
