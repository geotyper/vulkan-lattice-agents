// Replay and sweep coverage for SimulationDriver.
//
// Both features are defined by what they do NOT do, which is why they need a
// device to check. Replay must score and report a generation and then select
// nothing, so the weights that were loaded are the weights that keep running.
// A sweep must restart evolution between stages, so a stage measures its own
// setting rather than the setting before it applied to an already-evolved
// population. Neither claim is visible in the fitness numbers -- a replayed run
// and a training run both produce plausible ones -- so they are asserted here
// against the population and the generation counter directly.

#include "vkexp/compute/HeadlessComputeContext.hpp"
#include "vkexp/simulation/SimulationDriver.hpp"
#include "vkexp/simulation/SimulationState.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int skipExitCode = 77;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool samePopulation(const std::vector<vkexp::Genome>& left,
                    const std::vector<vkexp::Genome>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index].weights != right[index].weights) {
            return false;
        }
    }
    return true;
}

int run() {
    vkexp::HeadlessComputeContext context{
        vkexp::HeadlessComputeConfig{.applicationName = "vklat replay smoke"}};

    vkexp::SimulationState state{};
    state.controls.stepsPerGeneration = 24;
    state.worlds.requestedAgentsPerWorld = 16;
    // A small lattice, so the sixteen agents sharing one actually meet: a replay
    // that repeats exactly in a world nobody contests would be a weaker claim
    // than one that repeats with the arbitration firing every step.
    state.settings.latticeWidth = 12;
    state.settings.latticeHeight = 12;
    state.settings.latticeDepth = 8;

    vkexp::SimulationDriver driver{state, vkexp::EvolutionSettings{.populationSize = 64}, {}};
    driver.createResources(context.physicalDevice(), context.device());

    const auto stepGeneration = [&] {
        while (!driver.generationComplete()) {
            context.immediate().execute(
                [&](const VkCommandBuffer commands) { driver.recordSteps(commands, 64); });
        }
    };

    // A training generation is the baseline: it must move both the population and
    // the counter, or the replay assertions below would pass on a driver that
    // never evolves anything.
    const std::vector<vkexp::Genome> initial = driver.evolution().population();
    stepGeneration();
    const vkexp::GenerationSummary trained = driver.finishGeneration();
    require(std::isfinite(trained.bestFitness), "training produced finite fitness");
    require(driver.evolution().generation() == 1, "training advances the generation");
    require(!samePopulation(driver.evolution().population(), initial),
            "training changes the population");

    state.controls.replay = true;
    const std::vector<vkexp::Genome> watched = driver.evolution().population();
    const std::uint64_t frozen = driver.evolution().generation();

    stepGeneration();
    // Read the agents before finishing: finishGeneration respawns them, so the
    // end of the generation is only observable up to this point.
    const std::vector<vkexp::AgentState> firstRun = driver.snapshot().agents;
    const vkexp::GenerationSummary replayed = driver.finishGeneration();
    require(std::isfinite(replayed.bestFitness), "replay still scores the generation");
    require(driver.evolution().generation() == frozen, "replay does not advance the generation");
    require(samePopulation(driver.evolution().population(), watched),
            "replay selects and mutates nothing");

    // The strong claim: because nothing was selected and the beacon seed is
    // derived from the generation number, the next generation is the same one
    // again rather than a similar one. Anything that leaked evolution or advanced the seed
    // would show up here as a different ending, not as a worse number.
    stepGeneration();
    const std::vector<vkexp::AgentState> secondRun = driver.snapshot().agents;
    driver.finishGeneration();
    require(firstRun.size() == secondRun.size() && !firstRun.empty(),
            "both replayed generations produced agents");
    require(std::memcmp(firstRun.data(), secondRun.data(),
                        firstRun.size() * sizeof(vkexp::AgentState)) == 0,
            "a replayed generation repeats exactly");

    state.controls.replay = true;
    state.sweep.values = {0.25F, 0.75F};
    state.sweep.generationsPerStage = 2;
    driver.beginSweep();
    require(!state.controls.replay, "starting a sweep leaves replay mode");
    require(state.sweep.running && state.sweep.stages.size() == 1, "a sweep arms its first stage");
    require(std::abs(state.settings.fitness.groupSharing - 0.25F) < 1.0e-6F,
            "a sweep applies its first value");
    require(driver.evolution().generation() == 0, "a sweep restarts evolution");

    for (int generation = 0; generation < 2; ++generation) {
        stepGeneration();
        driver.finishGeneration();
    }
    require(std::abs(state.settings.fitness.groupSharing - 0.75F) < 1.0e-6F,
            "a filled stage applies the next value");
    require(driver.evolution().generation() == 0,
            "each stage begins from the seeded population, not the previous stage's");
    require(state.sweep.stages.size() == 2 && state.sweep.stages.front().arrivalRatio.size() == 2,
            "a finished stage keeps exactly its own generations");

    for (int generation = 0; generation < 2; ++generation) {
        stepGeneration();
        driver.finishGeneration();
    }
    require(!state.sweep.running, "the sweep stops after its last stage");
    require(state.sweep.stages.size() == 2 && state.sweep.stages.back().arrivalRatio.size() == 2,
            "the last stage is kept whole");

    // The last stage has nothing to hand over to, so it is not restarted: the
    // run simply carries on from where the sweep left it, at the last setting,
    // and the stages stay put for reading. A further generation is therefore an
    // ordinary one that neither records nor steers.
    require(driver.evolution().generation() == 2,
            "the final stage is left running rather than restarted");
    stepGeneration();
    driver.finishGeneration();
    require(state.sweep.stages.back().arrivalRatio.size() == 2,
            "a stopped sweep records nothing further");
    require(std::abs(state.settings.fitness.groupSharing - 0.75F) < 1.0e-6F,
            "a stopped sweep leaves the last setting in place");
    require(driver.evolution().generation() == 3, "evolution continues after the sweep ends");

    driver.destroyResources();
    std::cout << "Replay and sweep smoke passed on " << context.deviceName() << '\n';
    return EXIT_SUCCESS;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const vkexp::HeadlessComputeUnavailable& unavailable) {
        std::cout << "Skipping replay smoke: " << unavailable.what() << '\n';
        return skipExitCode;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
