#ifndef VKEXP_LATTICE_KERNEL_GLSL
#define VKEXP_LATTICE_KERNEL_GLSL

// uint, clamp and sqrt are built in here, and GLSL has no constexpr, so both
// markers expand to nothing. See LatticeKernel.inl for the rules the shared
// source follows.
#define VKEXP_LATTICE_FN
#define VKEXP_LATTICE_MATH_FN
#include "vkexp/lattice/LatticeKernel.inl"

#endif
