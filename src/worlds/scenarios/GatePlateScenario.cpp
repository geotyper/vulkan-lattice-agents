#include "vkexp/worlds/scenarios/GatePlateScenario.hpp"

#include "vkexp/worlds/ScenarioMath.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp::worlds::gate_plate {
namespace {

namespace kernel = worlds::kernel;

// The plate is lit, and it has to be. The receptors see beacons and other
// agents' light and nothing else, so a plate that was neither could only be
// found by walking over it -- the same fault the puck world was built with and
// had to be repaired of. It is a beacon, and reaching it is phototaxis.
//
// The resource is lit too, and while the gate is shut the wall hides it: the
// gate leaf blocks the opening, and the opening is the only line of sight to the
// far side. So the world tells an agent what it has done. Press the plate and
// the far light appears; step off and, after the latch runs down, it goes out
// again. That is the perceptual handle the task hangs on.
constexpr Float4 plateColor{0.95F, 0.55F, 0.15F, 0.0F};
constexpr Float4 resourceColor{0.35F, 1.00F, 0.55F, 0.0F};

// A round trip, not a crossing. Getting through the gate was the first version
// of this world and it is only half a task: an agent that is through is done,
// the plate behind it stops mattering to it, and the door being held is worth
// something exactly once. Coming back makes the gate a thing that has to be open
// twice, so whoever is holding it is worth something for as long as anybody is
// still out.
//
// The plate is also home, which is what makes the cycle close on itself rather
// than needing a fourth landmark: pressing it on the way back is the same act as
// pressing it on the way out, and re-opens the gate for the next trip.
constexpr std::uint32_t nominalRoundTrips = 2;

std::uint32_t completedRoundTrips(const AgentState& agent) {
    return static_cast<std::uint32_t>(std::max(agent.target.w, 0.0F));
}

std::uint32_t achievedObjectives(const AgentState& agent) {
    return std::min(completedRoundTrips(agent), nominalRoundTrips);
}

// Uncapped in the score and capped in the report, the way every repeating world
// here does it: a quicker agent still gains from the extra trips, and the
// published ratio stays a fraction of what the trial has room for.
float fitness(const AgentState& agent, const FitnessWeights& weights) {
    return objectiveFitness(agent, completedRoundTrips(agent), weights);
}

// Which leg the agent is on. Two things decide it, and they are different kinds
// of thing: whether the agent is carrying the resource, which is private to it,
// and whether the gate is running, which is a fact about the room. An agent whose
// neighbour opened the gate is on the outbound leg too.
//
// Both reasons to head for the plate -- "I have to open it" and "I am coming
// home" -- point at the same place, which is why the cycle needs only the two
// beacons it has.
void afterStep(AgentState& agent, const SimulationStep& settings, const float distance) {
    const bool open = gateOpen(agent);
    // internal.y is "the target is the second beacon", the same meaning the
    // delivery worlds give it, so the shared target-distance dispatcher needs no
    // case of its own: the second beacon here is the plate. internal.x is the
    // carrying flag, the slot the delivery worlds use for cargo.
    const bool wasSeekingPlate = agent.internal.y >= 0.5F;
    const bool wasCarrying = agent.internal.x >= 0.5F;

    // Arrivals, measured against whatever the agent was actually heading for.
    // Reaching the plate while pressing it is not an arrival: pressing is
    // positional and happens by standing there, so only a carrying agent closes
    // a trip.
    if (distance < beaconArrivalRadius(settings)) {
        if (!wasSeekingPlate && !wasCarrying) {
            agent.internal.x = 1.0F;
        } else if (wasSeekingPlate && wasCarrying) {
            agent.internal.x = 0.0F;
            agent.target.w = std::floor(std::max(agent.target.w, 0.0F) + 0.5F) + 1.0F;
        }
    }

    const bool seekPlate = agent.internal.x >= 0.5F || !open;
    if (seekPlate == wasSeekingPlate) {
        return;
    }
    // The leg changed, so the distance progress is banked against has to change
    // with it, exactly as the delivery cycle rebanks at a pickup. Without this a
    // gate opening reads as a metre and a half of free progress.
    //
    // The leg that ended is weighted by what it was. Walking to the plate to
    // press it is the leg with no other signal -- the resource is behind a shut
    // gate and invisible -- so it is worth more than a leg walked toward
    // something the agent can see.
    const float weight = wasCarrying || !wasSeekingPlate ? 1.0F : kernel::GatePlateProgressReward;
    agent.metrics.w += std::max(agent.metrics.x - agent.metrics.y, 0.0F) * weight;
    agent.internal.y = seekPlate ? 1.0F : 0.0F;
    const float rebanked = targetDistance(agent, settings);
    agent.metrics.x = rebanked;
    agent.metrics.y = rebanked;
}

// Two: the plate in front of the wall and the resource behind it. The gate leaf
// hides the second one whenever it is shut, which is what makes pressing
// visible as a change in the world rather than only as a change in the score.
ActiveBeacons activeBeacons(const SimulationStep& settings) {
    const kernel::vec2 plate = kernel::gatePlatePosition(settings.worldRadius);
    const kernel::vec2 resource = kernel::gateResourcePosition(settings.worldRadius);
    return {{{Beacon{{resource.x, resource.y, 0.0F, 0.0F}, resourceColor},
              Beacon{{plate.x, plate.y, 0.0F, 0.0F}, plateColor}}},
            2};
}

ObstacleBox obstacle(const std::uint32_t index, const AgentState& agent,
                     const SimulationStep& settings) {
    const kernel::vec2 centre =
        kernel::gateBoxCentre(index, settings.worldRadius, gateOpen(agent));
    const kernel::vec2 halfExtent = kernel::gateBoxHalfExtent(index, settings.worldRadius);
    return {{centre.x, centre.y, 0.0F, 0.0F}, {halfExtent.x, halfExtent.y, 0.0F, 0.0F}};
}

// Everyone starts on the near side, so the first leg is the same task for all of
// them. The driver's spiral would otherwise put half the population behind a
// wall it never had to open.
void spawn(AgentState& agent, const SimulationStep& settings) {
    agent.pose.x *= 0.75F;
    agent.pose.y = -std::abs(agent.pose.y) * 0.45F - settings.worldRadius * 0.45F;
    agent.internal.x = 0.0F; // carrying nothing
    agent.internal.y = 1.0F; // the plate first, because the gate starts shut
    agent.target.x = 0.0F;
}

// floats0 = {latch seconds, unused, unused, unused}. The latch is sent as the
// setting and not as an already-resolved "is it open", so both sides run
// gateRemaining from the shared kernel and the rule is the shared thing.
ScenarioParameterBlock gpuParameters(const SimulationStep& settings) {
    return {{settings.gateLatchSeconds, 0.0F, 0.0F, 0.0F}, {}, {}};
}

constexpr neuro::BrainShape brain = neuro::maximumBrainShape;
static_assert(brain.fitsCapacity());

} // namespace

// target.x carries how long the gate still has to run. It is written by the
// agent step, which is the only place that can see every agent in a world at
// once; nothing in this scenario uses the base-beacon slot it occupies.
bool gateOpen(const AgentState& agent) { return kernel::gateIsOpen(agent.target.x); }

const ScenarioDefinition& definition() {
    static constexpr ScenarioDefinition value{
        .name = "Gate and plate",
        .key = "gate",
        .id = BeaconScenario::GatePlate,
        .brain = brain,
        .tunables = {.beaconRadiusRatio = false,
                     .beaconAngularSpeed = false,
                     .beaconRandomMotion = false,
                     .forageCargoDecay = false,
                     .swapDeliveryEnds = false,
                     .blockedDoorPerGeneration = false,
                     .gateLatch = true},
        .objectiveLabel = "Round trips",
        .radiusLabel = "Orbit radius",
        .description = "Press the plate, cross the gate, bring it back -- and the gate has to be "
                       "open both ways",
        .beacons = beacons,
        .beaconCount = 2,
        .targetDistance = targetDistance,
        .phaseForStep = nullptr,
        .fitness = fitness,
        .achievedObjectives = achievedObjectives,
        .objectivesPerAgent = nominalRoundTrips,
        // A leg is 2.1 m and a round trip about 840 steps at the speed limit, so
        // two of them want roughly 1800 -- twice what every other world here
        // needs. The unit test asserts both that the nominal fits in this and
        // that it does not fit in the default.
        .nominalStepsPerGeneration = 1800,
        .beforeStep = nullptr,
        .afterStep = afterStep,
        .obstacleCount = kernel::GatePlateBoxCount,
        .obstacle = obstacle,
        .spawn = spawn,
        .gpuParameters = gpuParameters,
    };
    return value;
}

ActiveBeacons beacons(const AgentState&, const SimulationStep& settings) {
    return activeBeacons(settings);
}

// The leg the agent is currently on: the plate while the gate is shut or while it
// is carrying, the resource otherwise. This is the whole of what the world tells
// an agent about the order of the legs -- it shapes the one that is the task now
// and says nothing about how to do it.
float targetDistance(const AgentState& agent, const SimulationStep& settings) {
    const ActiveBeacons active = beacons(agent, settings);
    // internal.y and not the expression that derives it. The shared GLSL
    // dispatcher reads this slot, so deriving the leg here instead would give the
    // two sides different answers for the part of a step between the gate
    // changing and the hook noticing -- which is exactly the step whose distance
    // decides whether an arrival counts.
    const std::size_t targetIndex = agent.internal.y >= 0.5F ? 1 : 0;
    const float dx = active.values[targetIndex].position.x - agent.pose.x;
    const float dy = active.values[targetIndex].position.y - agent.pose.y;
    return std::sqrt(dx * dx + dy * dy);
}

} // namespace vkexp::worlds::gate_plate
