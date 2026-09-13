#ifndef VKEXP_TRANSPARENCY_KERNEL_GLSL
#define VKEXP_TRANSPARENCY_KERNEL_GLSL

// GLSL has no constexpr, so the marker expands to nothing. `exp` is built in;
// it is reached through a name of our own only so that the shared source can
// call the same thing in both languages.
float latticeWeightExp(float value) { return exp(value); }

#define VKEXP_TRANSPARENCY_FN
#include "vkexp/graphics/TransparencyKernel.inl"

#endif
