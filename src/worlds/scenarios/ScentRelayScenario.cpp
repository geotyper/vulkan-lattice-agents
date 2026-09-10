#include "vkexp/worlds/scenarios/ScentRelayScenario.hpp"

#include "vkexp/worlds/ScenarioMath.hpp"
#include "vkexp/worlds/scenarios/RotatingScenario.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp::worlds::scent_relay {
namespace {

constexpr Float4 resourceColor{1.00F, 0.82F, 0.20F, 0.0F};

// Home emits nothing, and that is the entire scenario. The outbound leg is
// ordinary phototaxis; the return leg has no light to follow, so it has to come
// from somewhere else -- the agent's recurrent state, or the ground it marked on
// the way out. A black beacon also deposits nothing, so the only scent in the
// world is the one the agents put there themselves.
constexpr Float4 homeColor{0.0F, 0.0F, 0.0F, 0.0F};

float fitness(const AgentState& agent, const FitnessWeights& weights) {
    return objectiveFitness(agent, completedForageCycles(agent), weights);
}

std::uint32_t achievedObjectives(const AgentState& agent) {
    return completedForageCycles(agent) > 0 ? 1U : 0U;
}

void afterStep(AgentState& agent, const SimulationStep& settings, const float distance) {
    deliveryCycleAfterStep(agent, settings, distance);
}

// floats0 = {resource rotation angle, orbit radius ratio, unused, cargo decay rate},
// floats1 = {pickup reward, delivery reward, unused, unused}.
// Unpacked by shaders/worlds/scent_relay.glsl and its steps file. Home needs no
// parameters: it is derived from the trial's own base beacon.
ScenarioParameterBlock gpuParameters(const SimulationStep& settings) {
    return {{settings.beaconRotationAngle, settings.beaconRadiusRatio, 0.0F,
             settings.forageCargoDecayRate},
            {settings.foragePickupReward, settings.forageDeliveryReward, 0.0F, 0.0F},
            {}};
}

constexpr neuro::BrainShape brain = neuro::maximumBrainShape;
static_assert(brain.fitsCapacity());

} // namespace

const ScenarioDefinition& definition() {
    static constexpr ScenarioDefinition value{
        .name = "Scent relay",
        .key = "scent",
        .id = BeaconScenario::ScentRelay,
        .brain = brain,
        .tunables = {.beaconRadiusRatio = true,
                     .beaconAngularSpeed = true,
                     .beaconRandomMotion = false,
                     .forageCargoDecay = true,
                     .needsTrail = true},
        .objectiveLabel = "Returned home unseen",
        .radiusLabel = "Orbit radius",
        .description = "Lit resource -> unlit home: the way back is not visible",
        .beacons = beacons,
        .beaconCount = 2,
        .targetDistance = targetDistance,
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

// Home sits opposite the trial's base beacon: fixed for the whole trial, distinct
// per trial, and derivable on both sides from state the agent already carries, so
// it costs no parameter slot.
Float4 homePosition(const AgentState& agent) {
    return {-agent.target.x, -agent.target.y, 0.0F, 0.0F};
}

ActiveBeacons beacons(const AgentState& agent, const SimulationStep& settings) {
    return {{{Beacon{rotating::beaconPosition(agent, settings), resourceColor},
              Beacon{homePosition(agent), homeColor}}},
            2};
}

float targetDistance(const AgentState& agent, const SimulationStep& settings) {
    const ActiveBeacons active = beacons(agent, settings);
    const std::size_t targetIndex = agent.internal.y >= 0.5F ? 1 : 0;
    const float dx = active.values[targetIndex].position.x - agent.pose.x;
    const float dy = active.values[targetIndex].position.y - agent.pose.y;
    return std::sqrt(dx * dx + dy * dy);
}

} // namespace vkexp::worlds::scent_relay
