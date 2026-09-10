#include "vkexp/worlds/scenarios/ShuttleScenario.hpp"

#include "vkexp/worlds/ScenarioMath.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp::worlds::shuttle {
namespace {

namespace kernel = worlds::kernel;

constexpr Float4 resourceColor{1.00F, 0.82F, 0.20F, 0.0F};
constexpr Float4 homeColor{0.20F, 0.55F, 1.00F, 0.0F};

// How many round trips the default trial has room for, at a pace an agent that
// has learned the detour can actually hold. It is what the reported completion
// ratio is measured against; the fitness is not capped, so a faster agent still
// scores for every extra trip.
constexpr std::uint32_t nominalRoundTrips = 2;

float fitness(const AgentState& agent, const FitnessWeights& weights) {
    return objectiveFitness(agent, completedForageCycles(agent), weights);
}

// Unlike the other cycle scenarios this counts trips rather than asking whether
// there was one: shuttling until the time runs out is the task, so a run that
// manages it twice has to be distinguishable from one that manages it once.
std::uint32_t achievedObjectives(const AgentState& agent) {
    return std::min(completedForageCycles(agent), nominalRoundTrips);
}

void afterStep(AgentState& agent, const SimulationStep& settings, const float distance) {
    deliveryCycleAfterStep(agent, settings, distance);
}

ObstacleBox obstacle(std::uint32_t, const AgentState&, const SimulationStep& settings) {
    const kernel::vec2 centre = kernel::shuttleBoxCentre();
    const kernel::vec2 halfExtent = kernel::shuttleBoxHalfExtent(settings.worldRadius);
    return {{centre.x, centre.y, 0.0F, 0.0F}, {halfExtent.x, halfExtent.y, 0.0F, 0.0F}};
}

// Both beacons are fixed and the wall does not divide the arena, so a spawn
// anywhere would still be a valid start -- but starting on the home side makes
// the first leg the outbound one for everyone, which is what makes a run legible
// to watch.
void spawn(AgentState& agent, const SimulationStep& settings) {
    const kernel::vec2 home = kernel::shuttleHomePosition(settings.worldRadius);
    agent.pose.x = agent.pose.x * 0.45F + home.x;
    agent.pose.y = agent.pose.y * 0.22F + home.y;
}

// floats0 = {unused, unused, unused, cargo decay rate},
// floats1 = {pickup reward, delivery reward, unused, unused} -- the slots
// scenarioDeliveryCycleAfterStep reads. The geometry needs none: it comes from
// the arena radius through the shared kernel.
ScenarioParameterBlock gpuParameters(const SimulationStep& settings) {
    return {{0.0F, 0.0F, 0.0F, settings.forageCargoDecayRate},
            {settings.foragePickupReward, settings.forageDeliveryReward, 0.0F, 0.0F},
            {}};
}

constexpr neuro::BrainShape brain = neuro::defaultBrainShape;
static_assert(brain.fitsCapacity());

} // namespace

const ScenarioDefinition& definition() {
    static constexpr ScenarioDefinition value{
        .name = "Shuttle",
        .key = "shuttle",
        .id = BeaconScenario::Shuttle,
        .brain = brain,
        .tunables = {.beaconRadiusRatio = false,
                     .beaconAngularSpeed = false,
                     .beaconRandomMotion = false,
                     .forageCargoDecay = true},
        .objectiveLabel = "Round trips",
        .radiusLabel = "Orbit radius",
        .description = "Fetch and carry back, over and over, around a wall in the way",
        .beacons = beacons,
        .beaconCount = 2,
        .targetDistance = targetDistance,
        .phaseForStep = nullptr,
        .fitness = fitness,
        .achievedObjectives = achievedObjectives,
        .objectivesPerAgent = nominalRoundTrips,
        .beforeStep = nullptr,
        .afterStep = afterStep,
        .obstacleCount = kernel::ShuttleBoxCount,
        .obstacle = obstacle,
        .spawn = spawn,
        .gpuParameters = gpuParameters,
    };
    return value;
}

ActiveBeacons beacons(const AgentState&, const SimulationStep& settings) {
    const kernel::vec2 resource = kernel::shuttleResourcePosition(settings.worldRadius);
    const kernel::vec2 home = kernel::shuttleHomePosition(settings.worldRadius);
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

} // namespace vkexp::worlds::shuttle
