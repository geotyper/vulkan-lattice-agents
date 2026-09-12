#pragma once

#include "vkexp/simulation/LatticeTypes.hpp"

#include <cstdint>

namespace vkexp {

// What the packing needs that SimulationStep does not carry: the population
// layout, all owned by whoever created the device resources.
struct StepParameterLayout {
    std::uint32_t agentCount{};
    std::uint32_t trialsPerGenome{1};
    std::uint32_t agentsPerWorld{1};
    std::uint32_t worldCount{1};
};

// The one place a GpuStepParameters is built.
//
// It used to be built twice -- once by the driver and once by the parity test --
// under a comment claiming they were the same path. They were not, and the copy
// silently dropped two fields in a row: the neuron-memory flag, and then the
// obstacle count. Both times the shader ran with the field at zero while the CPU
// reference ran with it set, so parity reported a numeric drift rather than a
// missing feature, which is a slow way to find a field you forgot to copy.
[[nodiscard]] GpuStepParameters packStepParameters(const SimulationStep& settings,
                                                   const StepParameterLayout& layout);

} // namespace vkexp
