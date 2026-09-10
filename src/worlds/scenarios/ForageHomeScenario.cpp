#include "vkexp/worlds/scenarios/ForageHomeScenario.hpp"

#include "vkexp/worlds/ScenarioMath.hpp"
#include "vkexp/worlds/scenarios/RotatingScenario.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp::worlds::forage_home {
namespace {

constexpr Float4 resourceColor{1.00F, 0.55F, 0.08F, 0.0F};
constexpr Float4 homeColor{0.12F, 0.72F, 1.00F, 0.0F};

float fitness(const AgentState& agent, const FitnessWeights& weights) {
    return objectiveFitness(agent, completedForageCycles(agent), weights);
}

// One completed round trip is enough to count as a solved task; the fitness
// function still rewards every further cycle.
std::uint32_t achievedObjectives(const AgentState& agent) {
    return completedForageCycles(agent) > 0 ? 1U : 0U;
}

// Home jumps on an epoch boundary. A carrying agent must not lose the progress
// it already made toward the old home.
void beforeStep(AgentState& agent, const SimulationStep& settings) {
    if (agent.internal.y >= 0.5F && homeRelocated(settings)) {
        bankObjectiveProgress(agent, settings, true);
    }
}

void afterStep(AgentState& agent, const SimulationStep& settings, const float distance) {
    deliveryCycleAfterStep(agent, settings, distance);
}

// floats0 = {resource rotation angle, orbit radius ratio, motion time, cargo decay rate},
// floats1 = {pickup reward, delivery reward, unused, unused},
// integers[0] = motion seed. Unpacked by shaders/worlds/forage_home.glsl and
// shaders/worlds/steps/forage_home.glsl.
ScenarioParameterBlock gpuParameters(const SimulationStep& settings) {
    return {{settings.beaconRotationAngle, settings.beaconRadiusRatio, settings.beaconMotionTime,
             settings.forageCargoDecayRate},
            {settings.foragePickupReward, settings.forageDeliveryReward, 0.0F, 0.0F},
            {settings.beaconMotionSeed, 0U, 0U, 0U}};
}

constexpr neuro::BrainShape brain = neuro::defaultBrainShape;
static_assert(brain.fitsCapacity());

} // namespace

const ScenarioDefinition& definition() {
    static constexpr ScenarioDefinition value{
        .name = "Forage + home",
        .key = "forage",
        .id = BeaconScenario::ForageHome,
        .brain = brain,
        .tunables = {.beaconRadiusRatio = true,
                     .beaconAngularSpeed = true,
                     .beaconRandomMotion = false,
                     .forageCargoDecay = true},
        .objectiveLabel = "Completed a forage cycle",
        .radiusLabel = "Orbit radius",
        .description = "Orange resource -> blue home -> repeat",
        .beacons = beacons,
        .beaconCount = 2,
        .targetDistance = targetDistance,
        .phaseForStep = nullptr,
        .fitness = fitness,
        .achievedObjectives = achievedObjectives,
        .objectivesPerAgent = 1,
        .beforeStep = beforeStep,
        .afterStep = afterStep,
        .gpuParameters = gpuParameters,
    };
    return value;
}

Float4 homePosition(const AgentState& agent, const SimulationStep& settings) {
    const auto trial = static_cast<std::uint32_t>(std::max(agent.target.z, 0.0F));
    const std::uint32_t key = kernel::forageHomeKey(
        settings.beaconMotionSeed, trial, kernel::forageHomeEpoch(settings.beaconMotionTime));
    return scaledOffset(kernel::forageHomeOffset(key), settings.worldRadius);
}

bool homeRelocated(const SimulationStep& settings) {
    return kernel::forageHomeRelocated(settings.beaconMotionTime, settings.deltaTime);
}

ActiveBeacons beacons(const AgentState& agent, const SimulationStep& settings) {
    return {{{Beacon{rotating::beaconPosition(agent, settings), resourceColor},
              Beacon{homePosition(agent, settings), homeColor}}},
            2};
}

float targetDistance(const AgentState& agent, const SimulationStep& settings) {
    const ActiveBeacons active = beacons(agent, settings);
    const std::size_t targetIndex = agent.internal.y >= 0.5F ? 1 : 0;
    const float dx = active.values[targetIndex].position.x - agent.pose.x;
    const float dy = active.values[targetIndex].position.y - agent.pose.y;
    return std::sqrt(dx * dx + dy * dy);
}

} // namespace vkexp::worlds::forage_home
