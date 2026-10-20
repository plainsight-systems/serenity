// The MNEE pass's kernel (mnee.h) that traces nothing: after the solve, the
// requests that found no room, by which the count passes the capacity
// (contract 16), added to the slot's count of lost requests, and the count
// set back to 0 for the next frame's list. Metal's mnee_grid
// (metal/passes/mnee/mnee.metal) without its grid, which a launch here
// does not read (mnee.cu).

#include "core/contracts/mnee_request.h"
#include "cuda/frame/frame_launch.h"
#include "cuda/math/vector.cuh"

extern "C" __global__ void mnee_settle(
    const serenity::cuda::FrameLaunch* launch) {
  namespace c = serenity::contracts;
  unsigned int* const header = launch->mnee_header;
  const unsigned int written = header[c::mnee_count_word];
  const unsigned int capacity = header[c::mnee_capacity_word];
  header[c::mnee_count_word] = 0u;
  if (written > capacity) {
    atomicAdd(launch->mnee_lost, written - capacity);
  }
}
