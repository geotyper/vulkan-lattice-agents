#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vkexp::lattice::kernel {

using uint = std::uint32_t;

// GLSL-shaped names the shared source needs; built in on the shader side.
[[nodiscard]] inline float clamp(const float value, const float low, const float high) {
    return std::clamp(value, low, high);
}
[[nodiscard]] inline float sqrt(const float value) { return std::sqrt(value); }

// Cell addressing, neighbour numbering and the arbitration rule are integer
// maths, so the C++ side evaluates them at compile time and uses the results as
// array bounds. The parts that decide a move from a drive need float maths,
// which cannot be constexpr, hence the second marker -- the same split
// BrainKernel.hpp makes for the same reason.
#define VKEXP_LATTICE_FN constexpr
#define VKEXP_LATTICE_MATH_FN inline
#include "vkexp/lattice/LatticeKernel.inl"
#undef VKEXP_LATTICE_MATH_FN
#undef VKEXP_LATTICE_FN

} // namespace vkexp::lattice::kernel
