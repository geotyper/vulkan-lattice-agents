#pragma once

#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/simulation/AgentTypes.hpp"

#include <span>

namespace vkexp {

void stepAgentCpu(AgentState& agent, std::span<const float> weights,
                  const SimulationStep& settings);

// Scores a finished trial with the scenario's own fitness function.
[[nodiscard]] float agentFitness(const AgentState& agent, BeaconScenario scenario,
                                 const FitnessWeights& weights = {});

} // namespace vkexp
