// metal-cpp's implementation: its headers declare everything and define it
// only in the one translation unit that asks. This is that unit, for the
// whole program; defining these anywhere else links every symbol twice.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
