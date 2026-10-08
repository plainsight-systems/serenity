// Compile check only: every core header builds as C++20 on its own, with
// nothing included before it. The core's headers that shaders include are
// also compiled as Metal by the GPU tests' libraries.

#include "core/portable_math.h"
