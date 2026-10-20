#include "cuda/device/pipeline.h"

#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <algorithm>
#include <string>
#include <vector>

#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

// A record of the shader binding table: OptiX's header, then the record's
// data (pipeline.h), at OptiX's alignment.
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
  std::array<char, OPTIX_SBT_RECORD_HEADER_SIZE> header;
  Pipeline::HitData data;
};

// The log a call that compiles or links fills: room for OptiX's message,
// which it ends with a null and cuts to fit, and its size on the way in.
struct Log {
  static constexpr std::size_t capacity = 4096;  // a few lines of messages
  std::array<char, capacity> text{};
  std::size_t size = capacity;

  std::string_view view() const {
    return {text.data(),
            static_cast<std::size_t>(std::find(text.begin(), text.end(), '\0') -
                                     text.begin())};
  }
};

OptixPipelineCompileOptions compile_options(unsigned payload_values,
                                            const std::string& parameters) {
  OptixPipelineCompileOptions options{};
  options.usesMotionBlur = 0;
  options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  options.numPayloadValues = static_cast<int>(payload_values);
  options.numAttributeValues = 2;  // OptiX's minimum; the hits carry none
  options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  options.pipelineLaunchParamsVariableName = parameters.c_str();
  options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
  return options;
}

}  // namespace

void Pipeline::DestroyModule::operator()(OptixModule module) const noexcept {
  (void)optixModuleDestroy(module);
}

void Pipeline::DestroyGroup::operator()(
    OptixProgramGroup group) const noexcept {
  (void)optixProgramGroupDestroy(group);
}

void Pipeline::DestroyPipeline::operator()(
    OptixPipeline pipeline) const noexcept {
  (void)optixPipelineDestroy(pipeline);
}

Pipeline::Pipeline(const OptixContext& context, std::span<const std::byte> ir,
                   const Programs& programs) {
  if (ir.empty()) {
    throw CudaError("Pipeline: no programs to load");
  }
  if (programs.hit_records.empty()) {
    throw CudaError("Pipeline: no structure to trace");
  }
  // OptiX takes its names as C strings, read while it makes the objects.
  const std::string ray_generation(programs.ray_generation);
  const std::string intersection(programs.intersection);
  const std::string parameters(programs.launch_parameters);
  const OptixPipelineCompileOptions pipeline_options =
      compile_options(programs.payload_values, parameters);
  const OptixModuleCompileOptions module_options{};  // OptiX's defaults

  Log log;
  OptixModule module = nullptr;
  // OptiX takes the IR as chars: the embedded bytes, viewed as what they
  // are, the one cast this interop needs.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  const auto* input = reinterpret_cast<const char*>(ir.data());
  check_optix(
      optixModuleCreate(context.handle(), &module_options, &pipeline_options,
                        input, ir.size(), log.text.data(), &log.size, &module),
      "optixModuleCreate", log.view());
  module_.reset(module);

  std::array<OptixProgramGroupDesc, 3> descriptions{};
  // The C API's descriptor is a union, filled by the member its kind names,
  // as OptiX's documentation fills it.
  // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
  descriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  descriptions[0].raygen.module = module;
  descriptions[0].raygen.entryFunctionName = ray_generation.c_str();
  descriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  descriptions[1].hitgroup.moduleIS = module;
  descriptions[1].hitgroup.entryFunctionNameIS = intersection.c_str();
  descriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;  // no program
  // NOLINTEND(cppcoreguidelines-pro-type-union-access)
  const OptixProgramGroupOptions group_options{};
  std::array<OptixProgramGroup, 3> groups{};
  for (std::size_t i = 0; i < groups.size(); ++i) {
    log = Log{};
    check_optix(optixProgramGroupCreate(context.handle(), &descriptions.at(i),
                                        1, &group_options, log.text.data(),
                                        &log.size, &groups.at(i)),
                "optixProgramGroupCreate", log.view());
    groups_.at(i).reset(groups.at(i));
  }

  OptixPipelineLinkOptions link_options{};
  link_options.maxTraceDepth = 1;
  log = Log{};
  OptixPipeline pipeline = nullptr;
  check_optix(
      optixPipelineCreate(context.handle(), &pipeline_options, &link_options,
                          groups.data(), static_cast<unsigned>(groups.size()),
                          log.text.data(), &log.size, &pipeline),
      "optixPipelineCreate", log.view());
  pipeline_.reset(pipeline);

  OptixStackSizes sizes{};
  for (OptixProgramGroup group : groups) {
    check_optix(optixUtilAccumulateStackSizes(group, &sizes, pipeline),
                "optixUtilAccumulateStackSizes");
  }
  unsigned traversal = 0;
  unsigned state = 0;
  unsigned continuation = 0;
  check_optix(optixUtilComputeStackSizes(&sizes, link_options.maxTraceDepth, 0,
                                         0, &traversal, &state, &continuation),
              "optixUtilComputeStackSizes");
  // One level: a single geometry structure.
  check_optix(
      optixPipelineSetStackSize(pipeline, traversal, state, continuation, 1),
      "optixPipelineSetStackSize");

  // The table: ray generation, miss, then a hit record a structure.
  std::vector<Record> records(2 + programs.hit_records.size());
  check_optix(optixSbtRecordPackHeader(groups[0], &records[0]),
              "optixSbtRecordPackHeader");
  check_optix(optixSbtRecordPackHeader(groups[2], &records[1]),
              "optixSbtRecordPackHeader");
  for (std::size_t i = 0; i < programs.hit_records.size(); ++i) {
    Record& record = records.at(2 + i);
    check_optix(optixSbtRecordPackHeader(groups[1], &record),
                "optixSbtRecordPackHeader");
    record.data = programs.hit_records[i];
  }
  records_ = DeviceBuffer(records.size() * sizeof(Record));
  check(cuMemcpyHtoD(records_.address(), records.data(),
                     records.size() * sizeof(Record)),
        "Pipeline: cuMemcpyHtoD");
  table_.raygenRecord = records_.address();
  table_.missRecordBase = records_.address() + sizeof(Record);
  table_.missRecordStrideInBytes = sizeof(Record);
  table_.missRecordCount = 1;
  table_.hitgroupRecordBase = records_.address() + 2 * sizeof(Record);
  table_.hitgroupRecordStrideInBytes = sizeof(Record);
  table_.hitgroupRecordCount =
      static_cast<unsigned>(programs.hit_records.size());
}

void Pipeline::launch(CUstream stream, CUdeviceptr parameters,
                      std::size_t bytes, frame::Extent grid) const {
  check_optix(optixLaunch(pipeline_.get(), stream, parameters, bytes, &table_,
                          grid.width, grid.height, 1),
              "optixLaunch");
}

}  // namespace serenity::cuda
