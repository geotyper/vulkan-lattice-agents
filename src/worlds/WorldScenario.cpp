#include "vkexp/worlds/WorldScenario.hpp"

#include "vkexp/worlds/scenarios/AlternatingScenario.hpp"
#include "vkexp/worlds/scenarios/ForageHomeScenario.hpp"
#include "vkexp/worlds/scenarios/RandomMovementScenario.hpp"
#include "vkexp/worlds/scenarios/RotatingScenario.hpp"
#include "vkexp/worlds/scenarios/ScentRelayScenario.hpp"
#include "vkexp/worlds/scenarios/ShuttleScenario.hpp"
#include "vkexp/worlds/scenarios/StationaryScenario.hpp"
#include "vkexp/worlds/scenarios/TwoDoorsScenario.hpp"
#include "vkexp/worlds/scenarios/GatePlateScenario.hpp"
#include "vkexp/worlds/scenarios/PuckPushScenario.hpp"
#include "vkexp/worlds/scenarios/TwoGapsScenario.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>

namespace vkexp {
namespace {

// The one place that knows the full set of scenarios. Adding one means adding a
// source file, its shader pair and a line here.
const std::array<const ScenarioDefinition*, beaconScenarioCount>& registry() {
    static const std::array<const ScenarioDefinition*, beaconScenarioCount> definitions = [] {
        const std::array<const ScenarioDefinition*, beaconScenarioCount> entries{
            &worlds::stationary::definition(), &worlds::alternating::definition(),
            &worlds::rotating::definition(), &worlds::random_movement::definition(),
            &worlds::forage_home::definition(), &worlds::scent_relay::definition(),
            &worlds::two_doors::definition(), &worlds::shuttle::definition(),
            &worlds::two_gaps::definition(), &worlds::puck_push::definition(),
            &worlds::gate_plate::definition()};
        // A registry out of order would silently run the wrong world rules, so
        // the mismatch has to be fatal rather than merely wrong.
        for (std::size_t index = 0; index < entries.size(); ++index) {
            if (entries[index] == nullptr ||
                entries[index]->id != static_cast<BeaconScenario>(index)) {
                throw std::logic_error("Scenario registry is not in BeaconScenario order");
            }
            if (entries[index]->key == nullptr || entries[index]->beacons == nullptr ||
                entries[index]->fitness == nullptr ||
                entries[index]->achievedObjectives == nullptr ||
                entries[index]->gpuParameters == nullptr ||
                entries[index]->objectivesPerAgent == 0 || entries[index]->beaconCount == 0 ||
                entries[index]->beaconCount > std::tuple_size_v<decltype(ActiveBeacons::values)> ||
                (entries[index]->obstacleCount > 0) != (entries[index]->obstacle != nullptr) ||
                // A world scores from something. Either it has a step hook that
                // notices arrivals, or it has a puck, whose level is latched by
                // the puck pass and mirrored onto every agent in the world.
                // Requiring an afterStep of every scenario was right until one
                // of them kept its score somewhere no per-agent hook could see.
                (entries[index]->afterStep == nullptr && !entries[index]->puck)) {
                throw std::logic_error(std::string{"Scenario '"} + entries[index]->name +
                                       "' does not implement the full scenario contract");
            }
        }
        return entries;
    }();
    return definitions;
}

} // namespace

std::span<const ScenarioDefinition* const> scenarioRegistry() { return registry(); }

bool sightBlocked(const AgentState& agent, const SimulationStep& settings, const float fromX,
                  const float fromY, const float toX, const float toY) {
    const ScenarioDefinition& scenario = scenarioDefinition(settings.beaconScenario);
    for (std::uint32_t index = 0; index < scenario.obstacleCount; ++index) {
        const ObstacleBox box = scenario.obstacle(index, agent, settings);
        if (worlds::kernel::segmentHitsBox({fromX, fromY}, {toX, toY},
                                           {box.centre.x, box.centre.y},
                                           {box.halfExtent.x, box.halfExtent.y})) {
            return true;
        }
    }
    return false;
}

const ScenarioDefinition& scenarioDefinition(const BeaconScenario scenario) {
    const auto index = static_cast<std::size_t>(scenario);
    const auto& definitions = registry();
    return *definitions[index < definitions.size() ? index : 0];
}

Float4 stationaryBeaconPosition(const std::uint32_t trial, const float worldRadius) {
    return worlds::stationary::beaconPosition(trial, worldRadius);
}

Float4 homeBeaconPosition(const AgentState& agent, const SimulationStep& settings) {
    return worlds::forage_home::homePosition(agent, settings);
}

bool homeBeaconRelocated(const SimulationStep& settings) {
    return worlds::forage_home::homeRelocated(settings);
}

ActiveBeacons activeBeacons(const AgentState& agent, const SimulationStep& settings) {
    ActiveBeacons beacons = scenarioDefinition(settings.beaconScenario).beacons(agent, settings);
    // Applied here rather than in each scenario: this is the one place a CPU
    // caller gets a beacon colour, so the ablation cannot be on for the sensors
    // and off for anything else that asks. Mirrored by scenarioBeaconColorCue in
    // shaders/worlds/world_scenarios.glsl.
    if (settings.uniformBeaconColor && beacons.count >= 2) {
        const Float4 first = beacons.values[0].color;
        const Float4 second = beacons.values[1].color;
        const Float4 shared{worlds::kernel::scenarioColorCueRemoved(first.x, second.x),
                            worlds::kernel::scenarioColorCueRemoved(first.y, second.y),
                            worlds::kernel::scenarioColorCueRemoved(first.z, second.z), 0.0F};
        for (std::size_t index = 0; index < beacons.count; ++index) {
            beacons.values[index].color = shared;
        }
    }
    return beacons;
}

float nearestBeaconDistance(const AgentState& agent, const SimulationStep& settings) {
    const ScenarioDefinition& scenario = scenarioDefinition(settings.beaconScenario);
    if (scenario.targetDistance != nullptr) {
        return scenario.targetDistance(agent, settings);
    }
    const ActiveBeacons beacons = scenario.beacons(agent, settings);
    float nearest = settings.worldRadius * 4.0F;
    for (std::size_t index = 0; index < beacons.count; ++index) {
        const float dx = beacons.values[index].position.x - agent.pose.x;
        const float dy = beacons.values[index].position.y - agent.pose.y;
        nearest = std::min(nearest, std::sqrt(dx * dx + dy * dy));
    }
    return nearest;
}

std::uint32_t completedBeaconPhases(const AgentState& agent) {
    const auto mask = static_cast<std::uint32_t>(std::max(agent.target.w, 0.0F) + 0.5F);
    return std::popcount(mask);
}

std::uint32_t completedForageCycles(const AgentState& agent) {
    return static_cast<std::uint32_t>(std::max(agent.target.w, 0.0F) + 0.5F);
}

std::uint32_t beaconPhaseForStep(const BeaconScenario scenario, const std::uint32_t step,
                                 const std::uint32_t stepsPerGeneration) {
    const ScenarioPhaseForStep phaseForStep = scenarioDefinition(scenario).phaseForStep;
    return phaseForStep != nullptr ? phaseForStep(step, stepsPerGeneration) : 0U;
}

float beaconRotationAngleForStep(const float angularSpeed, const float deltaTime,
                                 const std::uint32_t step) {
    return worlds::rotating::angleForStep(angularSpeed, deltaTime, step);
}

SimulationStep resolveStepSettings(const SimulationStep& base, const std::uint32_t step,
                                   const std::uint32_t stepsPerGeneration) {
    SimulationStep resolved = base;
    resolved.beaconPhase = beaconPhaseForStep(base.beaconScenario, step, stepsPerGeneration);
    // Derived rather than special-cased, so a future multi-phase scenario gets
    // its transitions reported without touching this function.
    resolved.beaconPhaseChanged =
        step > 0 && beaconPhaseForStep(base.beaconScenario, step - 1, stepsPerGeneration) !=
                        resolved.beaconPhase;
    resolved.beaconRotationAngle =
        beaconRotationAngleForStep(base.beaconAngularSpeed, base.deltaTime, step);
    resolved.beaconMotionTime = base.deltaTime * static_cast<float>(step);
    return resolved;
}

} // namespace vkexp
