#include "vkexp/worlds/scenarios/TwoGapsScenario.hpp"

#include "vkexp/worlds/ScenarioMath.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp::worlds::two_gaps {
namespace {

namespace kernel = worlds::kernel;

// The two ends are told apart by colour and by nothing else, which is the whole
// point of the swap: with the ends trading places every generation, a heading is
// worthless and the hue is the only thing that says which beacon is which.
constexpr Float4 resourceColor{1.00F, 0.82F, 0.20F, 0.0F};
constexpr Float4 homeColor{0.20F, 0.55F, 1.00F, 0.0F};

// What the trial has room for at a pace an agent that has found a gap can hold.
// Reported completion is measured against it; fitness is not capped, so a
// quicker agent still scores for the extra trips.
constexpr std::uint32_t nominalRoundTrips = 2;

float fitness(const AgentState& agent, const FitnessWeights& weights) {
    return objectiveFitness(agent, completedForageCycles(agent), weights);
}

std::uint32_t achievedObjectives(const AgentState& agent) {
    return std::min(completedForageCycles(agent), nominalRoundTrips);
}

void afterStep(AgentState& agent, const SimulationStep& settings, const float distance) {
    deliveryCycleAfterStep(agent, settings, distance);
}

ObstacleBox obstacle(const std::uint32_t index, const AgentState&, const SimulationStep& settings) {
    const kernel::vec2 centre = kernel::twoGapsBoxCentre(index, settings.worldRadius);
    const kernel::vec2 halfExtent = kernel::twoGapsBoxHalfExtent(index, settings.worldRadius);
    return {{centre.x, centre.y, 0.0F, 0.0F}, {halfExtent.x, halfExtent.y, 0.0F, 0.0F}};
}

// The wall divides the arena, so the driver's default spiral would start half
// the population already on the far side. Spawning everyone on the home side
// makes the first leg outbound for all of them, which is what makes a run
// legible to watch -- and when the ends swap, "the home side" swaps with them,
// so the start stays the same task rather than becoming the mirror one.
void spawn(AgentState& agent, const SimulationStep& settings) {
    const kernel::vec2 home = kernel::twoGapsHomePosition(settings.worldRadius, endsSwapped(settings));
    agent.pose.x = agent.pose.x * 0.45F + home.x;
    agent.pose.y = agent.pose.y * 0.22F + home.y;
}

// floats0 = {unused, unused, unused, cargo decay rate},
// floats1 = {pickup reward, delivery reward, unused, unused},
// integers = {generation, swap enabled, unused, unused}.
//
// The swap is sent unresolved -- the generation and the flag, not the answer --
// so that both sides resolve it through twoGapsEndsSwapped in the shared kernel.
// The rule is then the shared thing rather than one side's reading of it.
ScenarioParameterBlock gpuParameters(const SimulationStep& settings) {
    return {{0.0F, 0.0F, 0.0F, settings.forageCargoDecayRate},
            {settings.foragePickupReward, settings.forageDeliveryReward, 0.0F, 0.0F},
            {settings.beaconMotionSeed, settings.swapDeliveryEnds ? 1U : 0U, 0U, 0U}};
}

constexpr neuro::BrainShape brain = neuro::maximumBrainShape;
static_assert(brain.fitsCapacity());

} // namespace

// The generation number arrives as the beacon motion seed, which is what the
// driver sets it from; no new field has to reach the GPU for the swap to work.
bool endsSwapped(const SimulationStep& settings) {
    return kernel::twoGapsEndsSwapped(settings.beaconMotionSeed, settings.swapDeliveryEnds);
}

const ScenarioDefinition& definition() {
    static constexpr ScenarioDefinition value{
        .name = "Two gaps",
        .key = "gaps",
        .id = BeaconScenario::TwoGaps,
        .brain = brain,
        .tunables = {.beaconRadiusRatio = false,
                     .beaconAngularSpeed = false,
                     .beaconRandomMotion = false,
                     .forageCargoDecay = true,
                     .swapDeliveryEnds = true},
        .objectiveLabel = "Round trips",
        .radiusLabel = "Orbit radius",
        .description = "A wall with two ways through; the ends can trade places each generation",
        .beacons = beacons,
        .beaconCount = 2,
        .targetDistance = targetDistance,
        .phaseForStep = nullptr,
        .fitness = fitness,
        .achievedObjectives = achievedObjectives,
        .objectivesPerAgent = nominalRoundTrips,
        .beforeStep = nullptr,
        .afterStep = afterStep,
        .obstacleCount = kernel::TwoGapsBoxCount,
        .obstacle = obstacle,
        .spawn = spawn,
        .gpuParameters = gpuParameters,
    };
    return value;
}

ActiveBeacons beacons(const AgentState&, const SimulationStep& settings) {
    const bool swapped = endsSwapped(settings);
    const kernel::vec2 resource = kernel::twoGapsResourcePosition(settings.worldRadius, swapped);
    const kernel::vec2 home = kernel::twoGapsHomePosition(settings.worldRadius, swapped);
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

} // namespace vkexp::worlds::two_gaps
