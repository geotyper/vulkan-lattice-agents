#pragma once

#include <cmath>

// The C++ side of the transparency weight. `exp` is built in for GLSL and comes
// from <cmath> here, so the one name the shared source uses is defined once on
// each side; the marker expands to nothing in GLSL and makes the function
// callable from a test without a device here.
namespace vkexp::graphics::kernel {

inline float latticeWeightExp(const float value) { return std::exp(value); }

#define VKEXP_TRANSPARENCY_FN inline

#include "vkexp/graphics/TransparencyKernel.inl"

#undef VKEXP_TRANSPARENCY_FN

} // namespace vkexp::graphics::kernel
