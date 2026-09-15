#ifndef VKEXP_BRAIN_KERNEL_GLSL
#define VKEXP_BRAIN_KERNEL_GLSL

// uint is built in here, and GLSL has no constexpr, so the marker expands to
// nothing. See BrainKernel.inl for the rules the shared source follows.
#define VKEXP_BRAIN_FN
#define VKEXP_BRAIN_MATH_FN
#define VKEXP_BRAIN_INOUT inout float
#include "vkexp/neuro/BrainKernel.inl"

#endif
