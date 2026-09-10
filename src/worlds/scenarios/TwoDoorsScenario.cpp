#include "vkexp/worlds/scenarios/TwoDoorsScenario.hpp"

#include "vkexp/worlds/ScenarioMath.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp::worlds::two_doors {
namespace {

namespace kernel = worlds::kernel;

constexpr Float4 resourceColor{1.00F, 0.82F, 0.20F, 0.0F};
// Home is lit here, unlike the scent relay. The question this world asks is
// which gap leads through, and an invisible home would stack a second, already
// answered question on top of it and make the answer to neither legible.
constexpr Float4 homeColor{0.20F, 0.55F, 1.00F, 0.0F};

float fitness(const AgentState& agent, const FitnessWeights& weights) {
    return objectiveFitness(agent, completedForageCycles(agent), weights);
}

std::uint32_t achievedObjectives(const AgentState& agent) {
    return completedForageCycles(agent) > 0 ? 1U : 0U;
}

void afterStep(AgentState& agent, const SimulationStep& settings, const float distance) {
    deliveryCycleAfterStep(agent, settings, distance);
}

ObstacleBox obstacle(const std::uint32_t index, const AgentState& agent,
                     const SimulationStep& settings) {
    const std::uint32_t door = blockedDoor(agent, settings);
    const kernel::vec2 centre = kernel::twoDoorsBoxCentre(index, settings.worldRadius, door);
    const kernel::vec2 halfExtent = kernel::twoDoorsBoxHalfExtent(index, settings.worldRadius);
    return {{centre.x, centre.y, 0.0F, 0.0F}, {halfExtent.x, halfExtent.y, 0.0F, 0.0F}};
}

// The driver's default spiral covers the whole arena, which here would start
// half the population already past the wall with nothing left to solve. The
// spiral's spread is worth keeping, so it is compressed and moved onto the home
// side rather than replaced with a grid.
void spawn(AgentState& agent, const SimulationStep& settings) {
    const kernel::vec2 home = kernel::twoDoorsHomePosition(settings.worldRadius);
    agent.pose.x = agent.pose.x * 0.45F + home.x;
    agent.pose.y = agent.pose.y * 0.30F + home.y;
}

// floats0 = {unused, unused, unused, cargo decay rate},
// floats1 = {pickup reward, delivery reward, unused, unused},
// integers = {generation, layout keyed to the generation, unused, unused}.
//
// The geometry still comes from the arena radius on both sides. What has to be
// sent is the clock the dead end runs on, and it goes unresolved -- the
// generation and the flag, not the answer -- so both sides ask
// twoDoorsBlockedDoor and the rule is the shared thing.
ScenarioParameterBlock gpuParameters(const SimulationStep& settings) {
    return {{0.0F, 0.0F, 0.0F, settings.forageCargoDecayRate},
            {settings.foragePickupReward, settings.forageDeliveryReward, 0.0F, 0.0F},
            {settings.beaconMotionSeed, settings.blockedDoorPerGeneration ? 1U : 0U, 0U, 0U}};
}

constexpr neuro::BrainShape brain = neuro::defaultBrainShape;
static_assert(brain.fitsCapacity());

} // namespace

// The generation arrives as the beacon motion seed, which is what the driver
// sets it from, so keying the layout to it needs no new field of its own.
std::uint32_t blockedDoor(const AgentState& agent, const SimulationStep& settings) {
    return kernel::twoDoorsBlockedDoor(static_cast<std::uint32_t>(std::max(agent.target.z, 0.0F)),
                                       settings.beaconMotionSeed,
                                       settings.blockedDoorPerGeneration);
}

const ScenarioDefinition& definition() {
    static constexpr ScenarioDefinition value{
        .name = "Two doors",
        .key = "doors",
        .id = BeaconScenario::TwoDoors,
        .brain = brain,
        .tunables = {.beaconRadiusRatio = false,
                     .beaconAngularSpeed = false,
                     .beaconRandomMotion = false,
                     .forageCargoDecay = true,
                     .blockedDoorPerGeneration = true},
        .objectiveLabel = "Delivered through a door",
        .radiusLabel = "Orbit radius",
        .description = "A wall with two gaps; one is a dead end, and it swaps every trial",
        .beacons = beacons,
        .beaconCount = 2,
        .targetDistance = targetDistance,
        .phaseForStep = nullptr,
        .fitness = fitness,
        .achievedObjectives = achievedObjectives,
        .objectivesPerAgent = 1,
        .beforeStep = nullptr,
        .afterStep = afterStep,
        .obstacleCount = kernel::TwoDoorsBoxCount,
        .obstacle = obstacle,
        .spawn = spawn,
        .gpuParameters = gpuParameters,
    };
    return value;
}

ActiveBeacons beacons(const AgentState&, const SimulationStep& settings) {
    const kernel::vec2 resource = kernel::twoDoorsResourcePosition(settings.worldRadius);
    const kernel::vec2 home = kernel::twoDoorsHomePosition(settings.worldRadius);
    return {{{Beacon{{resource.x, resource.y, 0.0F, 0.0F}, resourceColor},
              Beacon{{home.x, home.y, 0.0F, 0.0F}, homeColor}}},
            2};
}

float targetDistance(const AgentState& agent, const SimulationStep& settings) {
    const ActiveBeacons active = beacons(agent, settings);
    const std::size_t targetIndex = agent.internal.y >= 0.5F ? 1 : 0;
    const float dx = active.values[targetIndex].position.x - agent.pose.x;
    const float dy = active.values[targetIndex].position.y - agent.pose.y;
    return std::sqrt(dx * dx + dy * dy);
}

} // namespace vkexp::worlds::two_doors
