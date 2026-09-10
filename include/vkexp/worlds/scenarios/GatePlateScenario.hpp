#pragma once

#include "vkexp/worlds/WorldScenario.hpp"

namespace vkexp::worlds::gate_plate {

[[nodiscard]] const ScenarioDefinition& definition();
[[nodiscard]] ActiveBeacons beacons(const AgentState& agent, const SimulationStep& settings);
[[nodiscard]] float targetDistance(const AgentState& agent, const SimulationStep& settings);

// Whether this agent's world has its gate running, read from the agent record.
// The state is a fact about the world -- it depends on where every agent in it
// is standing -- and it reaches an agent the way the puck's position does, by
// being mirrored onto it during the step.
[[nodiscard]] bool gateOpen(const AgentState& agent);

} // namespace vkexp::worlds::gate_plate
