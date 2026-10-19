#pragma once

#include <cuda.h>

#include <stdexcept>
#include <string>
#include <string_view>

namespace serenity::cuda {

// Axis: GPU backend.
//
// The CUDA backend's one error: any failure of the driver, a refusal of
// the backend's own (a pass it does not build yet, a readback the graph has
// nothing for), named. A std::exception, so the programs report it as they
// report any (I.10: a failure to do a required task is an exception; E.14:
// one type for the backend's failures).
class CudaError : public std::runtime_error {
 public:
  explicit CudaError(const std::string& what) : std::runtime_error(what) {}
};

// Throws CudaError naming `what`, and the driver's name and words for
// `result`, if `result` is not CUDA_SUCCESS. The driver's errors are sticky
// for a context once a kernel faults, so the first call after a fault
// reports it, wherever it is (the submission's waits, submission.h).
void check(CUresult result, std::string_view what);

}  // namespace serenity::cuda
