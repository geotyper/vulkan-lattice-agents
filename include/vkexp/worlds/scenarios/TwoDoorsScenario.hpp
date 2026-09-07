#pragma once

#include "vkexp/worlds/WorldScenario.hpp"

namespace vkexp::worlds::two_doors {

const ScenarioDefinition& definition();

ActiveBeacons beacons(const AgentState& agent, const SimulationStep& settings);
float targetDistance(const AgentState& agent, const SimulationStep& settings);

// Which gap is a dead end for the trial this agent is running.
[[nodiscard]] std::uint32_t blockedDoor(const AgentState& agent, const SimulationStep& settings);

} // namespace vkexp::worlds::two_doors
