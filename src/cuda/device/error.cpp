#include "cuda/device/error.h"

namespace serenity::cuda {

void check(CUresult result, std::string_view what) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* words = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &words);
  throw CudaError(std::string(what) + ": " +
                  (name != nullptr ? name : "an unknown CUresult") + ", " +
                  (words != nullptr ? words : "no description"));
}

}  // namespace serenity::cuda
