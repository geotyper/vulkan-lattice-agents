#pragma once

#include "vkexp/worlds/ScenarioKernel.hpp"

#include <cmath>
#include <cstdint>

namespace vkexp::puck::kernel {

using uint = std::uint32_t;
using vec2 = worlds::kernel::vec2;

// vec2 and length() come from the scenario shim rather than being defined
// again here: vec2 is the same type on both sides of the boundary, and a second
// length() for it is not merely redundant -- it is ambiguous, because argument
// lookup finds both.
using worlds::kernel::length;

// Everything here is either integer arithmetic or one square root, so unlike
// the brain preset it needs no second marker for the parts that cannot be
// constexpr -- length() is the only call and it is not used in a constant.
#define VKEXP_PUCK_FN inline
#include "vkexp/simulation/PuckKernel.inl"
#undef VKEXP_PUCK_FN

} // namespace vkexp::puck::kernel
