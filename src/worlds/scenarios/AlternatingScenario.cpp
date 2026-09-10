#include "vkexp/worlds/scenarios/AlternatingScenario.hpp"

#include "vkexp/worlds/ScenarioMath.hpp"

namespace vkexp::worlds::alternating {
namespace {

float fitness(const AgentState& agent, const FitnessWeights& weights) {
    return objectiveFitness(agent, completedBeaconPhases(agent), weights);
}

std::uint32_t achievedObjectives(const AgentState& agent) { return completedBeaconPhases(agent); }

// The diagonal swaps mid-generation, so progress toward the old pair has to be
// banked before the new distance becomes the baseline.
void beforeStep(AgentState& agent, const SimulationStep& settings) {
    if (settings.beaconPhaseChanged) {
        bankObjectiveProgress(agent, settings, false);
    }
}

// Diagonal positions derive from the world radius and the shared beacon phase.
ScenarioParameterBlock gpuParameters(const SimulationStep&) { return {}; }

constexpr neuro::BrainShape brain{neuro::Topology::inputCount - neuro::Topology::taskInputCount -
                                      neuro::Topology::recurrentMemoryCount,
                                  neuro::Topology::defaultHiddenCount,
                                  neuro::Topology::actuatorOutputCount};
static_assert(brain.fitsCapacity());

} // namespace

const ScenarioDefinition& definition() {
    static constexpr ScenarioDefinition value{
        .name = "Alternating diagonals",
        .key = "alternating",
        .id = BeaconScenario::AlternatingDiagonals,
        .brain = brain,
        .tunables = {},
        .beacons = beacons,
        .beaconCount = 2,
        .targetDistance = nullptr,
        .phaseForStep = phaseForStep,
        .fitness = fitness,
        .achievedObjectives = achievedObjectives,
        .objectivesPerAgent = 2,
        .beforeStep = beforeStep,
        .afterStep = recordPhaseArrival,
        .gpuParameters = gpuParameters,
    };
    return value;
}

ActiveBeacons beacons(const AgentState&, const SimulationStep& settings) {
    const std::size_t colorOffset = settings.beaconPhase != 0 ? 2 : 0;
    const auto position = [&](const std::uint32_t index) {
        return scaledOffset(kernel::alternatingDiagonalOffset(index, settings.beaconPhase),
                            settings.worldRadius);
    };
    return {{{Beacon{position(0), trialColors[colorOffset]},
              Beacon{position(1), trialColors[colorOffset + 1]}}},
            2};
}

std::uint32_t phaseForStep(const std::uint32_t step, const std::uint32_t stepsPerGeneration) {
    if (stepsPerGeneration == 0) {
        return 0;
    }
    return step >= stepsPerGeneration / 2 ? 1U : 0U;
}

} // namespace vkexp::worlds::alternating
