#ifndef VKEXP_PUCK_KERNEL_GLSL
#define VKEXP_PUCK_KERNEL_GLSL

// vec2, uint and length are built in here, so the shared kernel needs no shim
// on this side. See PuckKernel.inl for the rules the shared source follows.
//
// The scenario kernel comes first because the scattered placement draws its
// numbers from scenarioRandom01, the hash that already places a relocating home:
// a second hash here would be a second answer to the same question.
#include "worlds/scenario_kernel.glsl"

#define VKEXP_PUCK_FN
#include "vkexp/simulation/PuckKernel.inl"

#endif
