// The smallest kernel that shows a compiled-in library runs: thread i writes
// 3i + 1 to out[i], a value no unwritten buffer holds by accident.

kernel void smoke(device uint* out [[buffer(0)]],
                  constant uint& count [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
    if (i < count) {
        out[i] = 3 * i + 1;
    }
}
