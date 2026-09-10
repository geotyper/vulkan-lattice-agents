#pragma once

#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/simulation/AgentTypes.hpp"
#include "vkexp/worlds/ScenarioKernel.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexp {

// Single source of truth shared with the shaders.
inline const float forageHomeRelocationSeconds = worlds::kernel::ForageHomeRelocationSeconds;

struct Beacon {
    Float4 position;
    Float4 color;
};

// One static, axis-aligned obstacle. Boxes and not segments because the contact
// test is then a clamp and a subtraction, identical in both languages, and a
// wall of any shape this experiment needs is a handful of boxes.
struct ObstacleBox {
    Float4 centre;     // xy
    Float4 halfExtent; // xy
};

struct ActiveBeacons {
    std::array<Beacon, 2> values{};
    std::size_t count{};
};

// Beacons visible to the agent this step.
using ScenarioBeacons = ActiveBeacons (*)(const AgentState& agent, const SimulationStep& settings);
// Distance the scenario scores progress against. Null means "nearest beacon".
using ScenarioTargetDistance = float (*)(const AgentState& agent, const SimulationStep& settings);
// Active objective index for a step. Null means the scenario has a single phase.
using ScenarioPhaseForStep = std::uint32_t (*)(std::uint32_t step,
                                               std::uint32_t stepsPerGeneration);
// Final score for one finished trial, under the current shaping weights.
using ScenarioFitness = float (*)(const AgentState& agent, const FitnessWeights& weights);
// Objectives this agent actually completed, for the published arrival ratio.
using ScenarioObjectives = std::uint32_t (*)(const AgentState& agent);
// Runs before perception. Null means the scenario needs no world transition.
using ScenarioBeforeStep = void (*)(AgentState& agent, const SimulationStep& settings);
// Runs after physics with the distance to the current target.
using ScenarioAfterStep = void (*)(AgentState& agent, const SimulationStep& settings,
                                   float distance);
// Packs the scenario's own GPU parameters. `settings` must already be resolved
// for the step being packed (see resolveStepSettings).
using ScenarioGpuPacker = ScenarioParameterBlock (*)(const SimulationStep& settings);
// One of the scenario's obstacle boxes. Which boxes exist may depend on the
// agent, because the trial decides which gap is a dead end.
using ScenarioObstacle = ObstacleBox (*)(std::uint32_t index, const AgentState& agent,
                                         const SimulationStep& settings);
// Where an agent starts. Null keeps the driver's default spiral; a scenario with
// a divided arena needs to say which side of it a trial begins on.
using ScenarioSpawn = void (*)(AgentState& agent, const SimulationStep& settings);

// Which shared tunables this scenario actually reads. The UI shows controls from
// these flags instead of switching on the scenario identity.
struct ScenarioTunables {
    bool beaconRadiusRatio{};
    bool beaconAngularSpeed{};
    bool beaconRandomMotion{};
    bool forageCargoDecay{};
    // Whether this scenario does anything with SimulationStep::swapDeliveryEnds,
    // so the control appears where it has an effect instead of everywhere.
    bool swapDeliveryEnds{};
    // Whether this scenario reads SimulationStep::blockedDoorPerGeneration.
    bool blockedDoorPerGeneration{};
    // Whether this scenario reads SimulationStep::gateLatchSeconds.
    bool gateLatch{};
    // Whether the trail is the point of this world rather than decoration. The
    // window warns when such a scenario is run with the antennae switched off,
    // because the result is then a world with no route to its objective and
    // nothing about the numbers says so.
    bool needsTrail{};
};

// The complete contract of one experiment. Everything the simulation, the
// scoring, the UI and the GPU need is reachable from here, so adding a scenario
// means adding one source file, one shader pair and one registry entry rather
// than extending switches across the code base.
struct ScenarioDefinition {
    const char* name{};
    // Stable short identifier for command lines and archives; unlike `name` it
    // must not change once experiments have been recorded against it.
    const char* key{};
    BeaconScenario id{};
    neuro::BrainShape brain{};
    ScenarioTunables tunables{};
    // What this scenario calls its own concepts, so the UI never has to know
    // which scenario is selected in order to label a control or a statistic.
    const char* objectiveLabel{"Beacon objectives"};
    const char* radiusLabel{"Orbit radius"};
    const char* description{};

    ScenarioBeacons beacons{};
    // How many beacons `beacons` always reports. Drawn instance count on the CPU
    // side and the loop bound on the GPU side; asserted against `beacons` in the
    // unit tests.
    std::uint32_t beaconCount{1};
    ScenarioTargetDistance targetDistance{};
    ScenarioPhaseForStep phaseForStep{};

    ScenarioFitness fitness{};
    ScenarioObjectives achievedObjectives{};
    std::uint32_t objectivesPerAgent{1};
    // How long a trial has to be for `objectivesPerAgent` to be reachable at all.
    // A world whose nominal does not fit in the trial reports a ratio that
    // flattens below one with nothing looking wrong, which is a slow thing to
    // notice, so the scenario says the number and the UI and the headless runner
    // both act on it instead of it living in a comment.
    std::uint32_t nominalStepsPerGeneration{900};

    // Mirrored by shaders/worlds/steps/<scenario>.glsl.
    ScenarioBeforeStep beforeStep{};
    ScenarioAfterStep afterStep{};

    // Static geometry, the same way beacons work: a count the step loop and the
    // renderer both read, and one accessor. Zero means an empty arena, which is
    // every scenario but one.
    std::uint32_t obstacleCount{};
    // Whether this world has a puck. Decides whether the driver runs the puck
    // pass at all, and whether the target-radius slider is offered.
    bool puck{};
    ScenarioObstacle obstacle{};

    ScenarioSpawn spawn{};

    ScenarioGpuPacker gpuParameters{};
};

// Registry in BeaconScenario order; validated once on first use.
[[nodiscard]] std::span<const ScenarioDefinition* const> scenarioRegistry();
[[nodiscard]] const ScenarioDefinition& scenarioDefinition(BeaconScenario scenario);

// Resolves the step-dependent fields (beacon phase, rotation angle, motion time)
// for `step` within a generation. Simulation and visualization share this so the
// rendered beacon cannot drift away from the simulated one.
[[nodiscard]] SimulationStep resolveStepSettings(const SimulationStep& base, std::uint32_t step,
                                                 std::uint32_t stepsPerGeneration);

// True when the scenario's static geometry stands between the two points.
// Scenarios without obstacles answer false without looking at anything.
[[nodiscard]] bool sightBlocked(const AgentState& agent, const SimulationStep& settings,
                                float fromX, float fromY, float toX, float toY);

[[nodiscard]] Float4 stationaryBeaconPosition(std::uint32_t trial, float worldRadius);
[[nodiscard]] Float4 homeBeaconPosition(const AgentState& agent, const SimulationStep& settings);
[[nodiscard]] bool homeBeaconRelocated(const SimulationStep& settings);
[[nodiscard]] ActiveBeacons activeBeacons(const AgentState& agent, const SimulationStep& settings);
[[nodiscard]] float nearestBeaconDistance(const AgentState& agent, const SimulationStep& settings);
[[nodiscard]] std::uint32_t completedBeaconPhases(const AgentState& agent);
[[nodiscard]] std::uint32_t completedForageCycles(const AgentState& agent);
[[nodiscard]] std::uint32_t beaconPhaseForStep(BeaconScenario scenario, std::uint32_t step,
                                               std::uint32_t stepsPerGeneration);
[[nodiscard]] float beaconRotationAngleForStep(float angularSpeed, float deltaTime,
                                               std::uint32_t step);

} // namespace vkexp
