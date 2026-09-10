#include "vkexp/worlds/scenarios/StationaryScenario.hpp"

#include "vkexp/worlds/ScenarioMath.hpp"

#include <algorithm>
#include <array>

namespace vkexp::worlds::stationary {
namespace {

float fitness(const AgentState& agent, const FitnessWeights& weights) {
    return objectiveFitness(agent, completedBeaconPhases(agent), weights);
}

std::uint32_t achievedObjectives(const AgentState& agent) { return completedBeaconPhases(agent); }

// Per-trial beacons live in agent.target, so nothing has to reach the GPU.
ScenarioParameterBlock gpuParameters(const SimulationStep&) { return {}; }

constexpr neuro::BrainShape brain{neuro::Topology::inputCount - neuro::Topology::taskInputCount -
                                      neuro::Topology::recurrentMemoryCount,
                                  neuro::Topology::defaultHiddenCount,
                                  neuro::Topology::actuatorOutputCount};
static_assert(brain.fitsCapacity());

} // namespace

const ScenarioDefinition& definition() {
    static constexpr ScenarioDefinition value{
        .name = "Stationary",
        .key = "stationary",
        .id = BeaconScenario::Stationary,
        .brain = brain,
        .tunables = {},
        .beacons = beacons,
        .beaconCount = 1,
        .targetDistance = nullptr,
        .phaseForStep = nullptr,
        .fitness = fitness,
        .achievedObjectives = achievedObjectives,
        .objectivesPerAgent = 1,
        .beforeStep = nullptr,
        .afterStep = recordPhaseArrival,
        .gpuParameters = gpuParameters,
    };
    return value;
}

Float4 beaconPosition(const std::uint32_t trial, const float worldRadius) {
    constexpr std::array<Float4, 4> normalizedPositions{
        Float4{1.28F / smallWorldRadius, 0.96F / smallWorldRadius, 0.0F, 0.0F},
        Float4{-1.40F / smallWorldRadius, 0.70F / smallWorldRadius, 0.0F, 0.0F},
        Float4{-0.96F / smallWorldRadius, -1.32F / smallWorldRadius, 0.0F, 0.0F},
        Float4{1.36F / smallWorldRadius, -0.92F / smallWorldRadius, 0.0F, 0.0F}};
    const Float4 normalized = normalizedPositions[trial % normalizedPositions.size()];
    return {normalized.x * worldRadius, normalized.y * worldRadius, 0.0F, 0.0F};
}

ActiveBeacons beacons(const AgentState& agent, const SimulationStep&) {
    const auto trial = static_cast<std::uint32_t>(std::max(agent.target.z, 0.0F));
    return {{{Beacon{{agent.target.x, agent.target.y, 0.0F, 0.0F},
                     trialColors[trial % trialColors.size()]},
              {}}},
            1};
}

} // namespace vkexp::worlds::stationary
