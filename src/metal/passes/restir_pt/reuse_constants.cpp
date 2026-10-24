#include "metal/passes/restir_pt/reuse_constants.h"

#include <cstdint>

#include "metal/device/error.h"

namespace serenity::metal {

namespace {

namespace index = bindings::reuse_constant;

void set(MTL::FunctionConstantValues* values, const std::uint32_t& value,
         index::Index at) {
  values->setConstantValue(&value, MTL::DataTypeUInt, at);
}

void set(MTL::FunctionConstantValues* values, const float& value,
         index::Index at) {
  values->setConstantValue(&value, MTL::DataTypeFloat, at);
}

}  // namespace

NS::SharedPtr<MTL::FunctionConstantValues> reuse_constants(
    const contracts::PathReuseSettings& settings) {
  auto values = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  if (!values) {
    throw MetalError("reuse_constants: Metal made no function constant values");
  }
  set(values.get(), settings.nee_candidates, index::nee_candidates);
  set(values.get(), settings.max_surfaces, index::max_surfaces);
  set(values.get(), settings.footprint, index::footprint);
  set(values.get(), settings.roughness, index::roughness);
  set(values.get(), settings.neighbors, index::neighbors);
  set(values.get(), settings.sigma, index::sigma);
  set(values.get(), settings.confidence_cap, index::confidence_cap);
  set(values.get(), settings.cap_floor, index::cap_floor);
  set(values.get(), settings.cap_exponent, index::cap_exponent);
  set(values.get(), settings.checkerboard, index::checkerboard);
  return values;
}

}  // namespace serenity::metal
