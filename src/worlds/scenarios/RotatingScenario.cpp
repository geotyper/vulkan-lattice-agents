#include "vkexp/worlds/scenarios/RotatingScenario.hpp"

#include "vkexp/worlds/ScenarioMath.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp::worlds::rotating {
namespace {

float fitness(const AgentState& agent, const FitnessWeights& weights) {
    return objectiveFitness(agent, completedBeaconPhases(agent), weights);
}

std::uint32_t achievedObjectives(const AgentState& agent) { return completedBeaconPhases(agent); }

// The beacon never stops, so arrivals alone would leave the gradient flat.
void afterStep(AgentState& agent, const SimulationStep& settings, const float distance) {
    rewardVisibleTracking(agent, settings, distance);
    recordPhaseArrival(agent, settings, distance);
}

// floats0 = {rotation angle, orbit radius ratio, unused, unused}.
// Unpacked by rotatingScenarioPosition in shaders/worlds/rotating.glsl.
ScenarioParameterBlock gpuParameters(const SimulationStep& settings) {
    return {{settings.beaconRotationAngle, settings.beaconRadiusRatio, 0.0F, 0.0F}, {}, {}};
}

constexpr neuro::BrainShape brain{neuro::Topology::inputCount - neuro::Topology::taskInputCount -
                                      neuro::Topology::recurrentMemoryCount,
                                  neuro::Topology::defaultHiddenCount,
                                  neuro::Topology::actuatorOutputCount};
static_assert(brain.fitsCapacity());

} // namespace

const ScenarioDefinition& definition() {
    static constexpr ScenarioDefinition value{
        .name = "Rotating",
        .key = "rotating",
        .id = BeaconScenario::Rotating,
        .brain = brain,
        .tunables = {.beaconRadiusRatio = true, .beaconAngularSpeed = true},
        .beacons = beacons,
        .beaconCount = 1,
        .targetDistance = nullptr,
        .phaseForStep = nullptr,
        .fitness = fitness,
        .achievedObjectives = achievedObjectives,
        .objectivesPerAgent = 1,
        .beforeStep = nullptr,
        .afterStep = afterStep,
        .gpuParameters = gpuParameters,
    };
    return value;
}

Float4 beaconPosition(const AgentState& agent, const SimulationStep& settings) {
    const float orbitRadius = settings.worldRadius * settings.beaconRadiusRatio;
    const float targetLength = std::hypot(agent.target.x, agent.target.y);
    const kernel::vec2 baseDirection =
        targetLength > 0.000001F
            ? kernel::vec2{agent.target.x / targetLength, agent.target.y / targetLength}
            : kernel::vec2{1.0F, 0.0F};
    return scaledOffset(kernel::rotatingOrbitOffset(baseDirection, settings.beaconRotationAngle),
                        orbitRadius);
}

ActiveBeacons beacons(const AgentState& agent, const SimulationStep& settings) {
    const auto trial = static_cast<std::uint32_t>(std::max(agent.target.z, 0.0F));
    return {
        {{Beacon{beaconPosition(agent, settings), trialColors[trial % trialColors.size()]}, {}}},
        1};
}

float angleForStep(const float angularSpeed, const float deltaTime, const std::uint32_t step) {
    return std::fmod(angularSpeed * deltaTime * static_cast<float>(step), tau);
}

} // namespace vkexp::worlds::rotating
