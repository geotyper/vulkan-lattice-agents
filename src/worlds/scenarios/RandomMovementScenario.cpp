#include "vkexp/worlds/scenarios/RandomMovementScenario.hpp"

#include "vkexp/worlds/ScenarioMath.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp::worlds::random_movement {
namespace {

float fitness(const AgentState& agent, const FitnessWeights& weights) {
    return objectiveFitness(agent, completedBeaconPhases(agent), weights);
}

std::uint32_t achievedObjectives(const AgentState& agent) { return completedBeaconPhases(agent); }

// A wandering beacon is only worth chasing if closing on it pays continuously.
void afterStep(AgentState& agent, const SimulationStep& settings, const float distance) {
    rewardVisibleTracking(agent, settings, distance);
    recordPhaseArrival(agent, settings, distance);
}

// floats0 = {wander speed, roam radius ratio, motion time, teleport probability},
// integers[0] = motion seed. Unpacked by shaders/worlds/random_movement.glsl.
ScenarioParameterBlock gpuParameters(const SimulationStep& settings) {
    return {{settings.beaconRandomSpeed, settings.beaconRadiusRatio, settings.beaconMotionTime,
             settings.beaconTeleportProbability},
            {},
            {settings.beaconMotionSeed, 0U, 0U, 0U}};
}

constexpr neuro::BrainShape brain{neuro::Topology::inputCount - neuro::Topology::taskInputCount -
                                      neuro::Topology::recurrentMemoryCount,
                                  neuro::Topology::defaultHiddenCount,
                                  neuro::Topology::actuatorOutputCount};
static_assert(brain.fitsCapacity());

std::uint32_t latestWanderEpoch(const std::uint32_t segment, const std::uint32_t trial,
                                const SimulationStep& settings) {
    for (std::uint32_t candidate = segment; candidate > 0; --candidate) {
        if (kernel::randomTeleportSegment(settings.beaconMotionSeed, trial, candidate,
                                          settings.beaconTeleportProbability)) {
            return candidate;
        }
    }
    return 0;
}

Float4 beaconPosition(const std::uint32_t trial, const SimulationStep& settings) {
    const float motionTime = std::max(settings.beaconMotionTime, 0.0F);
    const std::uint32_t segment = kernel::randomMotionSegment(motionTime);
    const std::uint32_t epoch = latestWanderEpoch(segment, trial, settings);
    const float localTime =
        motionTime - static_cast<float>(epoch) * kernel::RandomMotionSegmentSeconds;
    const float roamRadius = settings.worldRadius * settings.beaconRadiusRatio;
    const float scaledTime = localTime * settings.beaconRandomSpeed / std::max(roamRadius, 0.001F);
    const std::uint32_t key = kernel::randomWanderKey(settings.beaconMotionSeed, trial, epoch);
    return scaledOffset(kernel::randomWanderOffset(key, scaledTime), roamRadius);
}

} // namespace

const ScenarioDefinition& definition() {
    static constexpr ScenarioDefinition value{
        .name = "Random movement",
        .key = "random",
        .id = BeaconScenario::RandomMovement,
        .brain = brain,
        .tunables = {.beaconRadiusRatio = true, .beaconRandomMotion = true},
        .objectiveLabel = "Beacon objectives",
        .radiusLabel = "Roam radius",
        .description = "Teleport check every 3.0 s",
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

ActiveBeacons beacons(const AgentState& agent, const SimulationStep& settings) {
    const auto trial = static_cast<std::uint32_t>(std::max(agent.target.z, 0.0F));
    return {
        {{Beacon{beaconPosition(trial, settings), trialColors[trial % trialColors.size()]}, {}}},
        1};
}

} // namespace vkexp::worlds::random_movement
