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

// One objective: get to the far side. Read against the whole world this is the
// share of a world's agents that made it through, which is the number worth
// watching -- with the latch at zero someone has to stay on the plate, so a
// world that solves the task perfectly reports every agent but one.
constexpr std::uint32_t objectivesPerAgent = 1;

std::uint32_t achievedObjectives(const AgentState& agent) {
    return std::min(static_cast<std::uint32_t>(std::max(agent.target.w, 0.0F)),
                    objectivesPerAgent);
}

float fitness(const AgentState& agent, const FitnessWeights& weights) {
    return objectiveFitness(agent, achievedObjectives(agent), weights);
}

// Which leg the agent is on, and it is the world that says so rather than the
// agent: while the gate is shut the thing to do is press the plate, and while it
// runs the thing to do is go through. An agent whose neighbour opened the gate
// is on the second leg too, which is correct -- the door being open is a fact
// about the room, not a private state.
void afterStep(AgentState& agent, const SimulationStep& settings, const float distance) {
    const bool open = gateOpen(agent);
    // internal.y is "the target is the second beacon", the same meaning the
    // delivery worlds give it, so the shared target-distance dispatcher needs no
    // case of its own: shut gate, second beacon, which here is the plate.
    const bool wasOpen = agent.internal.y < 0.5F;
    if (open != wasOpen) {
        // The leg changed under the agent, so the distance progress is banked
        // against has to change with it, exactly as the delivery cycle rebanks
        // at a pickup. Without this a gate opening would read as a jump of a
        // metre and a half of free progress.
        agent.metrics.w += std::max(agent.metrics.x - agent.metrics.y, 0.0F) *
                           (wasOpen ? 1.0F : kernel::GatePlateProgressReward);
        agent.internal.y = open ? 0.0F : 1.0F;
        const float rebanked = targetDistance(agent, settings);
        agent.metrics.x = rebanked;
        agent.metrics.y = rebanked;
        return;
    }
    if (!open || distance >= beaconArrivalRadius(settings)) {
        return;
    }
    // Through, and scored once. Latched in target.w rather than counted, because
    // an agent that arrives and lingers has not arrived twice.
    agent.target.w = 1.0F;
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
        .objectiveLabel = "Through the gate",
        .radiusLabel = "Orbit radius",
        .description = "A gate that opens only while an agent stands on the plate in front of it",
        .beacons = beacons,
        .beaconCount = 2,
        .targetDistance = targetDistance,
        .phaseForStep = nullptr,
        .fitness = fitness,
        .achievedObjectives = achievedObjectives,
        .objectivesPerAgent = objectivesPerAgent,
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

// The plate while the gate is shut, the resource while it runs. This is the
// whole of what the world tells an agent about the order of the two legs: it
// shapes the leg that is currently the task and says nothing about how to do it.
float targetDistance(const AgentState& agent, const SimulationStep& settings) {
    const ActiveBeacons active = beacons(agent, settings);
    const std::size_t targetIndex = gateOpen(agent) ? 0 : 1;
    const float dx = active.values[targetIndex].position.x - agent.pose.x;
    const float dy = active.values[targetIndex].position.y - agent.pose.y;
    return std::sqrt(dx * dx + dy * dy);
}

} // namespace vkexp::worlds::gate_plate
