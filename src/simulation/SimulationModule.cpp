#include "vkexp/simulation/SimulationModule.hpp"

#include "vkexp/core/VulkanContext.hpp"
#include "vkexp/evolution/GenomeArchive.hpp"
#include "vkexp/neuro/BrainDescription.hpp"
#include "vkexp/profiling/Profiler.hpp"
#include "vkexp/worlds/WorldScenario.hpp"

#include <exception>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace vkexp {
namespace {

// The short form, for files rather than for reading; the headless runner writes
// the same three words.
[[nodiscard]] const char* neuronModelKey(const NeuronModel model) {
    switch (model) {
    case NeuronModel::Reactive:
        return "reactive";
    case NeuronModel::TimeConstant:
        return "time";
    case NeuronModel::Gated:
        return "gated";
    }
    return "time";
}

} // namespace


// The driver writes per-step parameters into host-visible memory during
// onRender. That is only safe because VulkanContext waits on the frame fence in
// beginFrame, so no earlier submission can still be reading them.
static_assert(VulkanContext::framesInFlight == 1,
              "SimulationDriver's host-visible step parameters need per-frame slots "
              "once more than one frame is in flight");

SimulationModule::SimulationModule(SimulationState& state, Profiler& profiler,
                                   EvolutionSettings evolution, SimulationDriverConfig config)
    : state_(state), driver_(state, evolution, config),
      metric_(profiler.registerMetric("Agent simulation")) {}

void SimulationModule::onAttach(AppContext& context) {
    driver_.createResources(context.vulkan.physicalDevice(), context.vulkan.device());
}

void SimulationModule::onUpdate(AppContext& context, const FrameInfo&) {
    // Before the reset and step branches: a snapshot is about the run as it
    // stands, so it must not race a restart or a pending generation boundary.
    if (state_.controls.saveRequested || state_.controls.loadRequested) {
        const bool saving = state_.controls.saveRequested;
        state_.controls.saveRequested = false;
        state_.controls.loadRequested = false;
        context.vulkan.waitIdle();
        try {
            if (saving) {
                saveWorldSnapshot(state_.controls.snapshotPath, driver_.snapshot());
                state_.controls.snapshotStatus = "Saved " + state_.controls.snapshotPath;
            } else {
                driver_.restoreSnapshot(loadWorldSnapshot(state_.controls.snapshotPath));
                state_.controls.snapshotStatus = "Loaded " + state_.controls.snapshotPath;
                finishPending_ = false;
            }
        } catch (const std::exception& error) {
            state_.controls.snapshotStatus = error.what();
        }
        return;
    }
    // Saving the champion, which for a long time the window could not do at all:
    // it could read a .vkng archive and never write one, so the only file an
    // interactive run produced was a world snapshot -- the whole population plus
    // the agents and the pucks, resumable only into a run of the same size. The
    // champion was in there, and there was no way to get it out.
    if (state_.controls.saveGenomesRequested) {
        state_.controls.saveGenomesRequested = false;
        try {
            const std::span<const Genome> population = driver_.evolution().population();
            if (population.empty()) {
                throw std::runtime_error("There is no population to save yet");
            }
            // The first entry: evolve() writes the ranked survivors to the front
            // of the next population, champion first, which is the same genome
            // the headless runner's --save-champion writes.
            const std::size_t saved = state_.controls.saveWholePopulation ? population.size() : 1;
            saveGenomeArchive(
                state_.controls.genomePath, population.first(saved),
                genomeArchiveMetadata(state_, driver_,
                                      scenarioDefinition(state_.physics.beaconScenario).brain));
            state_.controls.snapshotStatus =
                "Saved " + std::to_string(saved) + " genome(s) from generation " +
                std::to_string(driver_.evolution().generation()) + " to " +
                state_.controls.genomePath;
        } catch (const std::exception& error) {
            state_.controls.snapshotStatus = error.what();
        }
        return;
    }
    // The structure the weights are laid out under, on its own. An archive
    // already carries it, but a file nobody can open is a poor way to answer
    // "which input is the left antenna" while looking at a champion.
    if (state_.controls.saveBrainStructureRequested) {
        state_.controls.saveBrainStructureRequested = false;
        try {
            std::filesystem::path path{state_.controls.genomePath};
            path.replace_extension(".json");
            std::ofstream stream{path, std::ios::trunc};
            if (!stream) {
                throw std::runtime_error("Unable to write " + path.string());
            }
            stream << neuro::brainDescriptionToJson(neuro::describeBrain(
                scenarioDefinition(state_.physics.beaconScenario).brain,
                neuronModelKey(state_.physics.neuronModel)));
            if (!stream) {
                throw std::runtime_error("Failed while writing " + path.string());
            }
            state_.controls.snapshotStatus = "Wrote the network's structure to " + path.string();
        } catch (const std::exception& error) {
            state_.controls.snapshotStatus = error.what();
        }
        return;
    }
    if (state_.controls.loadGenomesRequested) {
        state_.controls.loadGenomesRequested = false;
        context.vulkan.waitIdle();
        try {
            const GenomeArchive archive = loadGenomeArchive(state_.controls.genomePath);
            // An archive is usually one champion or a handful of elites, while a
            // run has a population size fixed when its buffers were made. The
            // archive is repeated to fill it, so every agent runs the loaded
            // weights -- which is what makes a champion watchable rather than
            // one agent among five hundred strangers. The status line says the
            // repetition happened instead of leaving it to be inferred from the
            // picture.
            const std::size_t populationSize = driver_.evolution().population().size();
            std::vector<Genome> filled(populationSize);
            for (std::size_t index = 0; index < populationSize; ++index) {
                filled[index] = archive.genomes[index % archive.genomes.size()];
            }
            driver_.loadPopulation(filled, archive.metadata.generation);
            state_.controls.snapshotStatus =
                "Loaded " + std::to_string(archive.genomes.size()) + " genome(s) from " +
                state_.controls.genomePath + ", repeated across " + std::to_string(populationSize);
        } catch (const std::exception& error) {
            state_.controls.snapshotStatus = error.what();
        }
        finishPending_ = false;
        return;
    }
    if (state_.controls.sweepStartRequested || state_.controls.sweepStopRequested) {
        const bool starting = state_.controls.sweepStartRequested;
        state_.controls.sweepStartRequested = false;
        state_.controls.sweepStopRequested = false;
        // Starting restarts evolution, so it needs the same idle device a reset
        // does; stopping only clears a flag, but goes through the same branch to
        // keep one place where a sweep changes state.
        context.vulkan.waitIdle();
        if (starting) {
            driver_.beginSweep();
        } else {
            driver_.endSweep();
        }
        finishPending_ = false;
        return;
    }
    if (state_.controls.resetRequested) {
        context.vulkan.waitIdle();
        driver_.restart();
        state_.controls.resetRequested = false;
        finishPending_ = false;
    } else if (finishPending_) {
        context.vulkan.waitIdle();
        driver_.finishGeneration();
        finishPending_ = false;
    }
}

void SimulationModule::onRender(AppContext& context, const FrameInfo&) {
    if (state_.controls.paused || finishPending_) {
        return;
    }
    auto cpuScope = context.profiler.cpu().scope(metric_);
    auto gpuScope = context.profiler.gpu().scope(context.vulkan.commandBuffer(), metric_);
    if (driver_.recordSteps(context.vulkan.commandBuffer(), state_.controls.stepsPerFrame) == 0) {
        finishPending_ = true;
        return;
    }
    finishPending_ = driver_.generationComplete();
}

void SimulationModule::onDetach(AppContext&) { driver_.destroyResources(); }

} // namespace vkexp
