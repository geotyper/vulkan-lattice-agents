#pragma once

#include "vkexp/worlds/WorldScenario.hpp"

namespace vkexp::worlds::two_gaps {

[[nodiscard]] const ScenarioDefinition& definition();
[[nodiscard]] ActiveBeacons beacons(const AgentState& agent, const SimulationStep& settings);
[[nodiscard]] float targetDistance(const AgentState& agent, const SimulationStep& settings);
[[nodiscard]] bool endsSwapped(const SimulationStep& settings);

} // namespace vkexp::worlds::two_gaps
