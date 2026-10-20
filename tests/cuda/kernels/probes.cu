// The probe kernels, one module (tests/cuda/CMakeLists.txt), as Metal's
// probes are one library: each file runs one family's device code on its
// own for its test (tests/gpu/<family>_test.cpp, shared with Metal), its
// records in the header it shares with Metal's kernel
// (tests/gpu/kernels/<kernel>.h).

#include "bsdf_probe.cu"
#include "camera_probe.cu"
#include "delay.cu"
#include "encoding_probe.cu"
#include "film_probe.cu"
#include "light_probe.cu"
#include "mnee_probe.cu"
#include "path_probe.cu"
#include "sampler_probe.cu"
#include "scene_block_probe.cu"
#include "shape_probe.cu"
#include "texture_probe.cu"
