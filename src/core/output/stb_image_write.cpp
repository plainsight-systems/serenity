// stb_image_write's implementation: its header declares the encoder and
// defines it only in the one translation unit that asks. This is that unit,
// apart from png.cpp, its one caller, as metal_cpp.cpp is metal-cpp's: the
// encoder is third-party code, compiled as a system header, and the
// static analyzer (make tidy) checks this project's code, not its paths
// through stb's.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
