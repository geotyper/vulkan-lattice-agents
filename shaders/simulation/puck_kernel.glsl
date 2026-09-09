#ifndef VKEXP_PUCK_KERNEL_GLSL
#define VKEXP_PUCK_KERNEL_GLSL

// vec2, uint and length are built in here, so the shared kernel needs no shim
// on this side. See PuckKernel.inl for the rules the shared source follows.
#define VKEXP_PUCK_FN
#include "vkexp/simulation/PuckKernel.inl"

#endif
