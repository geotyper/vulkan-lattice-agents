#pragma once

#include <cmath>
#include <cstdint>

namespace vkexp::worlds::kernel {

// The smallest GLSL-shaped surface that lets ScenarioKernel.inl compile as C++.
// Deliberately minimal: anything richer would tempt the shared kernel to use
// constructs the shader side cannot express.
using uint = std::uint32_t;

struct vec2 {
    float x{};
    float y{};

    constexpr vec2() = default;
    constexpr vec2(const float xValue, const float yValue) : x(xValue), y(yValue) {}
};

[[nodiscard]] inline float cos(const float value) { return std::cos(value); }
[[nodiscard]] inline float sin(const float value) { return std::sin(value); }
[[nodiscard]] inline float floor(const float value) { return std::floor(value); }
[[nodiscard]] inline float sqrt(const float value) { return std::sqrt(value); }
[[nodiscard]] inline float max(const float a, const float b) { return a > b ? a : b; }
[[nodiscard]] inline float length(const vec2 value) {
    return std::sqrt(value.x * value.x + value.y * value.y);
}

#define VKEXP_KERNEL_FN inline
#include "vkexp/worlds/ScenarioKernel.inl"
#undef VKEXP_KERNEL_FN

} // namespace vkexp::worlds::kernel
