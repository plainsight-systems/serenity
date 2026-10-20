// A delay, for a test that holds one stream's work back while the other's
// runs (tests/gpu/denoise_test.cpp, the hand-off; tests/cuda/support/
// hand_off_rig.h), Metal's (tests/gpu/kernels/delay.metal): one thread
// takes `count` steps of a linear congruential sequence, each depending on
// the last, and writes where it ended to out[0], so the steps cannot be
// left out. A test's kernel, never the renderer's. Included by probes.cu.

extern "C" __global__ void delay(unsigned* out, const unsigned* count_at) {
  unsigned x = 1u;
  for (unsigned i = 0; i < *count_at; ++i) {
    x = x * 1664525u + 1013904223u;
  }
  out[0] = x;
}
