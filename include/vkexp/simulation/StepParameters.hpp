#pragma once

#include "vkexp/simulation/AgentTypes.hpp"

#include <cstdint>

namespace vkexp {

// What the packing needs that SimulationStep does not carry: buffer dimensions
// and the population layout, all owned by whoever created the device resources.
struct StepParameterLayout {
    std::uint32_t agentCount{};
    std::uint32_t trialsPerGenome{1};
    std::uint32_t agentsPerWorld{1};
    float gridCellSize{};
    std::uint32_t gridWidth{};
    std::uint32_t gridCellsPerWorld{};
    std::uint32_t trailWidth{1};
    std::uint32_t trailCellsPerWorld{1};
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
//
// `resolved` must already have been through resolveStepSettings for the step
// being packed; the scenario is looked up from it.
[[nodiscard]] GpuStepParameters packStepParameters(const SimulationStep& resolved,
                                                   const StepParameterLayout& layout);

} // namespace vkexp
