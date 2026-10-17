// A delay, for a test that holds one queue's work back while the other's
// runs (denoise_test.cpp, the hand-off): one thread takes `count` steps of
// a linear congruential sequence, each depending on the last, and writes
// where it ended to out[0], so the steps cannot be left out. A test's
// kernel, never the renderer's.

kernel void delay(device uint* out [[buffer(0)]],
                  constant uint& count [[buffer(1)]]) {
    uint x = 1u;
    for (uint i = 0; i < count; ++i) {
        x = x * 1664525u + 1013904223u;
    }
    out[0] = x;
}
