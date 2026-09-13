#pragma once

#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/simulation/LatticeTypes.hpp"

#include <cstdint>
#include <span>

namespace vkexp {

// What one agent reads: the twenty-six cells around it, the direction to its
// beacon, and what it did last step.
//
// This is the whole of perception now. The 2D build raycast seven photoreceptors
// across a field of view, attenuated by distance, occluded by geometry, and
// summed over every beacon in the world -- about ninety lines of trigonometry
// per agent per step. A lattice answers the same question by reading twenty-six
// integers, because the index *is* the neighbourhood.
//
// `occupancy` is this agent's own world's slice, not the whole grid: a sensor
// that could address another world's cells is a sensor that will, once the
// world index is off by one somewhere.
//
// `signals` is what every agent was broadcasting when the step began, indexed by
// agent index, and not the agents themselves. That is the contract and not an
// optimisation: on the device this pass reads the input agent buffer while
// writing the output one, so a neighbour's broadcast is always last step's. A
// reference that read the live records would update agent 0 before agent 1
// sensed it, and the two would disagree by an amount that depends on the
// numbering -- which is the hardest kind of parity failure to read.
[[nodiscard]] neuro::Inputs sampleAgentInputs(const AgentState& agent,
                                              std::span<const float> signals,
                                              std::span<const std::int32_t> occupancy,
                                              const SimulationStep& settings,
                                              std::span<const std::int32_t> structures = {});

} // namespace vkexp
