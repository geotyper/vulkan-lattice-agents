#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vkexp::neuro::kernel {

using uint = std::uint32_t;

// GLSL-shaped names the shared preset needs; built in on the shader side.
[[nodiscard]] inline float pow(const float base, const float exponent) {
    return std::pow(base, exponent);
}
[[nodiscard]] inline float clamp(const float value, const float low, const float high) {
    return std::clamp(value, low, high);
}
[[nodiscard]] inline float exp(const float value) { return std::exp(value); }
[[nodiscard]] inline float tanh(const float value) { return std::tanh(value); }

// Layout arithmetic is pure integer maths, so the C++ side evaluates it at
// compile time and uses the results as array bounds. The sensor response model
// needs pow(), which cannot be constexpr, hence the second marker.
#define VKEXP_BRAIN_FN constexpr
#define VKEXP_BRAIN_MATH_FN inline
// GLSL's `inout float`, which C++ spells with a reference. A shared function
// that writes back through its parameters needs one word per language and no
// second implementation.
#define VKEXP_BRAIN_INOUT float&
#include "vkexp/neuro/BrainKernel.inl"
#undef VKEXP_BRAIN_INOUT
#undef VKEXP_BRAIN_MATH_FN
#undef VKEXP_BRAIN_FN

} // namespace vkexp::neuro::kernel
