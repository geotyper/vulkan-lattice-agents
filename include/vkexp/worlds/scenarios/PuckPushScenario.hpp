#pragma once

#include "vkexp/worlds/WorldScenario.hpp"

namespace vkexp::worlds::puck_push {

[[nodiscard]] const ScenarioDefinition& definition();
[[nodiscard]] ActiveBeacons beacons(const AgentState& agent, const SimulationStep& settings);
[[nodiscard]] float targetDistance(const AgentState& agent, const SimulationStep& settings);

} // namespace vkexp::worlds::puck_push
