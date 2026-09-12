#include "vkexp/ui/SimulationUiModule.hpp"

#include "vkexp/lattice/LatticeKernel.hpp"
#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/profiling/Profiler.hpp"
#include "vkexp/simulation/Units.hpp"
#include "vkexp/ui/ImGuiModule.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace vkexp {
namespace {

void plotHistory(const char* label, const std::vector<float>& values, const float minimum = FLT_MAX,
                 const float maximum = FLT_MAX) {
    if (values.empty()) {
        ImGui::TextDisabled("%s: waiting for the first completed generation", label);
        return;
    }
    ImGui::PushID(label);
    ImGui::PlotLines("##history", values.data(), static_cast<int>(values.size()), 0, nullptr,
                     minimum, maximum, ImVec2(-1.0F, 62.0F));
    ImGui::TextDisabled("%s", label);
    ImGui::PopID();
}

// Sweep stages are only a comparison if they are drawn against one axis.
// Separately autoscaled plots make a flat run and a climbing one look alike,
// and ImGui draws one series per plot, so the shared range is what turns a
// stack of plots into an answer.
std::pair<float, float> stackedRange(const std::vector<SweepStage>& stages,
                                     std::vector<float> SweepStage::*series) {
    float minimum = FLT_MAX;
    float maximum = -FLT_MAX;
    for (const SweepStage& stage : stages) {
        for (const float value : stage.*series) {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
    }
    if (minimum > maximum) {
        return {0.0F, 1.0F};
    }
    // A stage that never moved would otherwise be drawn as a full-height line.
    if (maximum - minimum < 1.0e-6F) {
        return {minimum - 0.5F, maximum + 0.5F};
    }
    return {minimum, maximum};
}

// One slider per axis, and a reset on any change: the lattice is a buffer
// dimension, so a new box is a new run rather than a live adjustment. Returns
// whether the extent moved, so the caller raises the reset once for all three.
bool latticeExtentSlider(const char* label, std::uint32_t& extent) {
    int value = static_cast<int>(extent);
    if (!ImGui::SliderInt(label, &value, static_cast<int>(latticeMinimumExtent),
                          static_cast<int>(latticeMaximumExtent), "%d",
                          ImGuiSliderFlags_Logarithmic)) {
        return false;
    }
    const std::uint32_t clamped = clampLatticeExtent(static_cast<std::uint32_t>(value));
    if (clamped == extent) {
        return false;
    }
    extent = clamped;
    return true;
}

} // namespace

SimulationUiModule::SimulationUiModule(SimulationState& state, ImGuiModule& imgui,
                                       Profiler& profiler)
    : state_(state), imgui_(imgui), metric_(profiler.registerMetric("Simulation UI")) {}

void SimulationUiModule::onAttach(AppContext&) { syncTexture(); }

void SimulationUiModule::syncTexture() {
    if (viewportGeneration_ == state_.viewport.generation) {
        return;
    }
    imgui_.removeTexture(viewportDescriptor_);
    viewportDescriptor_ = imgui_.addTexture(state_.viewport.sampler, state_.viewport.imageView,
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    viewportGeneration_ = state_.viewport.generation;
}

void SimulationUiModule::onUpdate(AppContext& context, const FrameInfo& frame) {
    auto scope = context.profiler.cpu().scope(metric_);
    syncTexture();
    (void)frame;

    ImGui::SetNextWindowPos(ImVec2(16.0F, 16.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330.0F, 470.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Simulation");

    ImGui::Checkbox("Paused", &state_.controls.paused);
    int stepsPerFrame = static_cast<int>(state_.controls.stepsPerFrame);
    if (ImGui::SliderInt("Steps / frame", &stepsPerFrame, 1, 64)) {
        state_.controls.stepsPerFrame = static_cast<std::uint32_t>(stepsPerFrame);
    }
    ImGui::SetItemTooltip("How much simulation one displayed frame advances. It changes how fast "
                          "a run goes, never what it computes: a step is the unit of "
                          "reproducibility and nothing here is a function of wall-clock time.");
    int stepsPerGeneration = static_cast<int>(state_.controls.stepsPerGeneration);
    if (ImGui::SliderInt("Steps / generation", &stepsPerGeneration, 60, 3000, "%d",
                         ImGuiSliderFlags_Logarithmic)) {
        state_.controls.stepsPerGeneration = static_cast<std::uint32_t>(stepsPerGeneration);
    }
    ImGui::TextDisabled(
        "trial %.1f s at %.0f Hz (%.1f ms per step)",
        static_cast<double>(units::secondsForSteps(state_.controls.stepsPerGeneration,
                                                   state_.settings.deltaTime)),
        static_cast<double>(1.0F / state_.settings.deltaTime),
        static_cast<double>(state_.settings.deltaTime * 1000.0F));

    ImGui::SeparatorText("The lattice");
    // Every extent is a buffer dimension, so a change here can only take effect
    // on a reset. They are sliders and not fixed because what the plan leaves
    // open is exactly how much room the neighbourhood work needs -- and the
    // answer is different for a chain than for a formation.
    bool latticeChanged = latticeExtentSlider("Width", state_.settings.latticeWidth);
    latticeChanged |= latticeExtentSlider("Height", state_.settings.latticeHeight);
    latticeChanged |= latticeExtentSlider("Depth", state_.settings.latticeDepth);
    if (latticeChanged) {
        state_.controls.resetRequested = true;
    }
    // Cells per world times worlds is what the fixed allocation has to hold, and
    // the driver shrinks the box rather than growing the buffer -- so the numbers
    // here are what is running, not what was asked for.
    const std::uint32_t cells = latticeCellsPerWorld(state_.settings);
    const double gridBytes = static_cast<double>(cells) * state_.worlds.worldCount *
                             sizeof(std::int32_t);
    ImGui::TextDisabled("%u cells per world, %u worlds, %.1f MB of occupancy", cells,
                        state_.worlds.worldCount, gridBytes / (1024.0 * 1024.0));
    if (state_.worlds.agentsPerWorld > 0 && cells > 0) {
        ImGui::TextDisabled("density %.2f%% -- agents meet below about 1%%",
                            100.0 * static_cast<double>(state_.worlds.agentsPerWorld) /
                                static_cast<double>(cells));
    }

    int neighborhood = static_cast<int>(state_.settings.neighborhood);
    constexpr const char* neighborhoods[] = {"Faces only (6)", "Moore (26)"};
    static_assert(std::size(neighborhoods) == neighborhoodCount);
    if (ImGui::Combo("Neighbourhood", &neighborhood, neighborhoods,
                     static_cast<int>(neighborhoodCount))) {
        state_.settings.neighborhood = static_cast<Neighborhood>(neighborhood);
        state_.controls.resetRequested = true;
    }
    ImGui::SetItemTooltip("Whether a step may be diagonal. The input vector is 26 cells wide "
                          "under both -- how many directions can be seen and how many can be "
                          "walked are separate questions -- so a population trained under one "
                          "setting still loads under the other. It also changes what distance "
                          "means: Chebyshev under Moore, Manhattan under faces.");
    ImGui::TextDisabled("longest journey %u moves", latticeMaximumDistance(state_.settings));

    ImGui::SliderFloat("Move threshold", &state_.settings.moveThreshold, 0.0F, 0.95F, "%.2f");
    ImGui::SetItemTooltip("How sure a drive has to be before it becomes a step. This is the whole "
                          "of the decision to stand still: at zero an agent moves every step "
                          "whatever it thinks, and near one it has to commit.");

    int contactRadius = static_cast<int>(state_.settings.beaconContactRadius);
    if (ImGui::SliderInt("Contact radius", &contactRadius, 0, 6)) {
        state_.settings.beaconContactRadius = static_cast<std::uint32_t>(contactRadius);
    }
    ImGui::SetItemTooltip("How near the beacon counts as having reached it. A cell holds one "
                          "agent, so at zero eleven of twelve lose the objective however well "
                          "they steered -- which measures the arbitration rule rather than the "
                          "policy. Widen it to let a group crowd the beacon.");
    if (state_.settings.beaconContactRadius == 0) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4{0.95F, 0.75F, 0.25F, 1.0F}, "one winner per world");
    }
    ImGui::TextDisabled("beacon seed %u, redrawn every generation",
                        state_.settings.beaconSeed);

    ImGui::SeparatorText("Fitness shaping");
    ImGui::SliderFloat("Tracking reward", &state_.settings.fitness.trackingReward, 0.0F, 4.0F,
                       "%.2f");
    ImGui::SetItemTooltip("What closing the distance is worth, scored once at the end against "
                          "the nearest the agent ever got. Shaping and not the objective: an "
                          "agent pushed off the beacon keeps this.");
    ImGui::SliderFloat("Objective bonus", &state_.settings.fitness.objectiveBonus, 0.0F, 0.5F,
                       "%.3f");
    ImGui::SetItemTooltip("Score per step spent within the contact radius. Per step rather than "
                          "per arrival, for the reason the contact radius exists.");
    ImGui::SliderFloat("Motor cost", &state_.settings.fitness.motorCostWeight, 0.0F, 0.05F,
                       "%.4f");
    ImGui::SliderFloat("Refusal penalty", &state_.settings.fitness.refusalPenalty, 0.0F, 0.1F,
                       "%.4f");
    ImGui::SetItemTooltip("Charged per move that could not happen -- into a wall, into a "
                          "neighbour, or lost to a lower-numbered agent. This is the whole of "
                          "the pressure toward not crowding, so it wants to be larger than the "
                          "cost of a move that worked.");
    if (state_.settings.fitness.refusalPenalty < state_.settings.fitness.motorCostWeight) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4{0.95F, 0.75F, 0.25F, 1.0F}, "cheaper than moving");
    }
    ImGui::SliderFloat("Signal cost", &state_.settings.fitness.signalCostFactor, 0.0F, 2.0F,
                       "%.2f");
    ImGui::SetItemTooltip("What broadcasting costs, relative to moving. Signalling is free to a "
                          "sender otherwise, and a channel nobody pays for is one every genome "
                          "saturates.");
    if (ImGui::SliderFloat("Group fitness sharing", &state_.settings.fitness.groupSharing, 0.0F,
                           1.0F, "%.2f")) {
        state_.controls.resetRequested = true;
    }
    ImGui::SetItemTooltip("0 is pure individual selection; 1 gives every genome sharing a lattice "
                          "the same score, so selection acts on the group and a broadcast that "
                          "only helps a neighbour finally pays its sender back. The plotted "
                          "fitness stays individual either way, so runs at different settings "
                          "stay comparable.");

    // A snapshot is the fast way back to a run worth looking at, so it sits with
    // the run controls rather than in an export menu. Everything except the
    // occupancy grid is stored; the grid is a function of where everybody stands
    // and is rebuilt on load.
    ImGui::SeparatorText("Snapshot");
    std::array<char, 256> snapshotPath{};
    const std::size_t pathLength =
        std::min(state_.controls.snapshotPath.size(), snapshotPath.size() - 1);
    std::copy_n(state_.controls.snapshotPath.begin(), pathLength, snapshotPath.begin());
    if (ImGui::InputText("File", snapshotPath.data(), snapshotPath.size())) {
        state_.controls.snapshotPath = snapshotPath.data();
    }
    ImGui::BeginDisabled(state_.controls.snapshotPath.empty());
    if (ImGui::Button("Save run")) {
        state_.controls.saveRequested = true;
    }
    ImGui::SetItemTooltip("Population, every agent's cell, generation, step and every setting. "
                          "Not the occupancy grid, which follows from the agents.");
    ImGui::SameLine();
    if (ImGui::Button("Load run")) {
        state_.controls.loadRequested = true;
    }
    ImGui::SetItemTooltip("Resumes on the step it was saved on. Needs the same population size "
                          "and trial count as this run, and a lattice this run can allocate.");
    ImGui::EndDisabled();

    // Watching trained weights is a different job from training them, and the
    // difference is one flag: everything is scored and reported as usual, and
    // nothing is selected. It sits beside the loaders because that is the order
    // it is used in -- load a champion, then watch it.
    ImGui::SeparatorText("Replay");
    ImGui::BeginDisabled(state_.sweep.running);
    ImGui::Checkbox("Replay only (no evolution)", &state_.controls.replay);
    ImGui::SetItemTooltip("Scores and reports every generation as usual but selects and mutates "
                          "nothing, so the same population respawns and the run repeats instead "
                          "of evolving away from the weights you loaded.");
    std::array<char, 256> genomePath{};
    const std::size_t genomeLength =
        std::min(state_.controls.genomePath.size(), genomePath.size() - 1);
    std::copy_n(state_.controls.genomePath.begin(), genomeLength, genomePath.begin());
    if (ImGui::InputText("Genome file", genomePath.data(), genomePath.size())) {
        state_.controls.genomePath = genomePath.data();
    }
    ImGui::BeginDisabled(state_.controls.genomePath.empty());
    ImGui::Checkbox("Whole population", &state_.controls.saveWholePopulation);
    ImGui::SetItemTooltip("Off, the button writes the best genome alone -- a small file to load "
                          "back and watch. On, it writes every genome, which is what a run is "
                          "resumed from without carrying the agents and the world with it.");
    if (ImGui::Button(state_.controls.saveWholePopulation ? "Save population" : "Save champion")) {
        state_.controls.saveGenomesRequested = true;
    }
    ImGui::SetItemTooltip("Writes the best genome of the last evaluated generation to a .vkng "
                          "archive.");
    ImGui::SameLine();
    if (ImGui::Button("Save structure")) {
        state_.controls.saveBrainStructureRequested = true;
    }
    ImGui::SetItemTooltip("Writes what the weights *mean*, as JSON next to the archive: which "
                          "slot of the input vector is which cell of the neighbourhood, which "
                          "span of the genome is which weight block, and what connects to what. "
                          "An archive already carries this and refuses to load into a build whose "
                          "network differs, naming the block that moved -- this is the same "
                          "document on its own, to read.");
    ImGui::SameLine();
    if (ImGui::Button("Load genomes")) {
        state_.controls.loadGenomesRequested = true;
    }
    ImGui::SetItemTooltip("Weights only, from a .vkng archive. A champion or a few elites are "
                          "repeated across the whole population, so every agent runs the loaded "
                          "brain. The lattice stays as it is set up here.");
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    if (!state_.controls.snapshotStatus.empty()) {
        ImGui::TextWrapped("%s", state_.controls.snapshotStatus.c_str());
    }

    ImGui::SeparatorText("Show");
    ImGui::Checkbox("Agents", &state_.display.agents);
    ImGui::SameLine();
    ImGui::Checkbox("Beacons", &state_.display.beacons);
    ImGui::SameLine();
    ImGui::Checkbox("Bounds", &state_.display.bounds);
    ImGui::SliderFloat("Background", &state_.display.backgroundBrightness, 0.0F, 1.0F, "%.2f");

    ImGui::SeparatorText("Brain contract");
    // One integrator, four sources for the rate it runs at, and the same genome
    // under all of them -- so this is a live ablation rather than a choice
    // between networks, and a population stays meaningful across a switch.
    int neuronModel = static_cast<int>(state_.settings.neuronModel);
    constexpr const char* neuronModels[] = {"Reactive", "Time constant", "Gated", "Spiking (LIF)"};
    static_assert(std::size(neuronModels) == neuronModelCount);
    if (ImGui::Combo("Neuron model", &neuronModel, neuronModels,
                     static_cast<int>(neuronModelCount))) {
        state_.settings.neuronModel = static_cast<NeuronModel>(neuronModel);
        state_.controls.resetRequested = true;
    }
    switch (state_.settings.neuronModel) {
    case NeuronModel::Reactive:
        ImGui::SetItemTooltip("No state at all: every time constant is pinned to one step, so a "
                              "neuron is its input. This is the network from before time "
                              "constants existed, reached by the same arithmetic.");
        break;
    case NeuronModel::TimeConstant:
        ImGui::SetItemTooltip("Each neuron holds its own state and an evolved time constant, so "
                              "fast neurons are reflexes and slow ones hold a fact across "
                              "seconds. The rate is fixed for the neuron's life.");
        break;
    case NeuronModel::Gated:
        ImGui::SetItemTooltip("The time constant is recomputed every step from the inputs, so a "
                              "neuron can hold a value and then let go of it when something "
                              "tells it to. Feed it a constant and this is the row above, "
                              "exactly.");
        break;
    case NeuronModel::Spiking:
        ImGui::SetItemTooltip("Leaky Integrate-and-Fire: the state accumulates input and decays, "
                              "and a neuron emits a discrete pulse when it crosses threshold.");
        break;
    }
    const neuro::BrainShape brain = resolvedBrain(state_.settings);
    std::string layerText;
    for (std::size_t layer = 0; layer < brain.hiddenLayerCount(); ++layer) {
        layerText += layerText.empty() ? "" : " -> ";
        layerText += std::to_string(brain.hiddenLayer(layer));
    }
    ImGui::Text("%zu inputs -> %s tanh -> %zu outputs", brain.inputCount, layerText.c_str(),
                brain.outputCount);
    ImGui::TextDisabled("%zu weights per genome", brain.weightCount());
    if (state_.settings.neuronModel != NeuronModel::Reactive) {
        ImGui::TextDisabled("gate block %zu of %zu genes",
                            brain.hiddenTotal() * (brain.inputCount + 1), brain.weightCount());
    }
    ImGui::TextDisabled("time constants %.0f ms .. %.1f s",
                        static_cast<double>(neuro::kernel::BrainTimeConstantMinimum * 1000.0F),
                        static_cast<double>(neuro::kernel::BrainTimeConstantMaximum));
    ImGui::TextDisabled("%u cells x %u channels, beacon, heading and memory",
                        neuro::kernel::BrainNeighborCount, neuro::kernel::BrainNeighborChannels);
    ImGui::TextDisabled("three move drives, a broadcast and memory updates");
    ImGui::End();

    drawBrainWindow(brain);

    ImGui::SetNextWindowPos(ImVec2(16.0F, 500.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330.0F, 380.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Genetic Algorithm");
    if (ImGui::Button("Reset evolution")) {
        state_.controls.resetRequested = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("completed: %llu",
                        static_cast<unsigned long long>(state_.statistics.evaluatedGenerations));
    ImGui::SeparatorText("Last evaluated generation");
    ImGui::Text("Best fitness:   %.4f", state_.statistics.bestFitness);
    ImGui::Text("Median fitness: %.4f", state_.statistics.medianFitness);
    ImGui::Text("Mean fitness:   %.4f", state_.statistics.meanFitness);
    ImGui::Text("Reached the beacon: %.1f%%", state_.statistics.arrivalRatio * 100.0F);
    ImGui::SeparatorText("Fitness history");
    // Say so when the plots are a window onto a longer run, rather than letting
    // a curve that has stopped extending read as a run that has stopped.
    if (state_.history.bestFitness.size() < state_.statistics.evaluatedGenerations) {
        ImGui::TextDisabled("last %zu generations of %llu", state_.history.bestFitness.size(),
                            static_cast<unsigned long long>(
                                state_.statistics.evaluatedGenerations));
    }
    plotHistory("Best", state_.history.bestFitness);
    plotHistory("Median", state_.history.medianFitness);
    plotHistory("Mean", state_.history.meanFitness);
    plotHistory("Reached the beacon", state_.history.arrivalRatio, 0.0F, 1.0F);
    ImGui::SeparatorText("Evolution parameters");
    ImGui::Text("Population: %zu", state_.evolution.populationSize);
    ImGui::Text("Elites: %zu   Tournament: %zu", state_.evolution.eliteCount,
                state_.evolution.tournamentSize);
    ImGui::Text("Crossover: %.1f%%", state_.evolution.crossoverProbability * 100.0F);
    ImGui::Text("Mutation: %.1f%%  strength %.3f", state_.evolution.mutationProbability * 100.0F,
                state_.evolution.mutationStrength);

    // Whether group fitness sharing helps is an empirical question, and one run
    // cannot answer it. A sweep runs the same experiment once per setting from
    // the same seed, restarting evolution between stages, so the curves below
    // differ in the swept value and in nothing else.
    ImGui::SeparatorText("Sharing sweep");
    ImGui::BeginDisabled(state_.sweep.running);
    for (std::size_t index = 0; index < state_.sweep.values.size(); ++index) {
        ImGui::PushID(static_cast<int>(index));
        ImGui::SetNextItemWidth(72.0F);
        ImGui::SliderFloat("##value", &state_.sweep.values[index], 0.0F, 1.0F, "%.2f");
        ImGui::PopID();
        if (index + 1 < state_.sweep.values.size()) {
            ImGui::SameLine();
        }
    }
    ImGui::BeginDisabled(state_.sweep.values.size() >= maximumSweepStages);
    if (ImGui::Button("Add stage")) {
        state_.sweep.values.push_back(state_.sweep.values.empty() ? 0.0F
                                                                  : state_.sweep.values.back());
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(state_.sweep.values.empty());
    if (ImGui::Button("Remove stage")) {
        state_.sweep.values.pop_back();
    }
    ImGui::EndDisabled();
    int generationsPerStage = static_cast<int>(state_.sweep.generationsPerStage);
    if (ImGui::SliderInt("Generations / stage", &generationsPerStage, 5, 1000, "%d",
                         ImGuiSliderFlags_Logarithmic)) {
        state_.sweep.generationsPerStage = static_cast<std::uint32_t>(generationsPerStage);
    }
    ImGui::EndDisabled();

    if (state_.sweep.running) {
        if (ImGui::Button("Stop sweep")) {
            state_.controls.sweepStopRequested = true;
        }
        ImGui::SameLine();
        ImGui::Text("Stage %zu/%zu at %.2f, generation %u/%u", state_.sweep.stage + 1,
                    state_.sweep.values.size(), static_cast<double>(sweepValue(state_.sweep)),
                    state_.sweep.generationsInStage, state_.sweep.generationsPerStage);
    } else {
        ImGui::BeginDisabled(state_.sweep.values.empty() || state_.controls.replay);
        if (ImGui::Button("Start sweep")) {
            state_.controls.sweepStartRequested = true;
        }
        ImGui::SetItemTooltip("Restarts evolution at the first value and moves on when a stage "
                              "fills up. Overwrites any previous sweep result.");
        ImGui::EndDisabled();
        if (state_.controls.replay) {
            ImGui::SameLine();
            ImGui::TextDisabled("replay is on");
        }
    }

    if (!state_.sweep.stages.empty()) {
        // Arrival first, and on a fixed 0..1 axis: it is the one number sharing
        // does not touch arithmetically, so it is the honest comparison between
        // settings. Fitness follows on a shared axis.
        const std::pair<float, float> fitnessRange =
            stackedRange(state_.sweep.stages, &SweepStage::medianFitness);
        for (const SweepStage& stage : state_.sweep.stages) {
            char label[64];
            std::snprintf(label, sizeof(label), "Reached at sharing %.2f",
                          static_cast<double>(stage.groupSharing));
            plotHistory(label, stage.arrivalRatio, 0.0F, 1.0F);
        }
        for (const SweepStage& stage : state_.sweep.stages) {
            char label[64];
            std::snprintf(label, sizeof(label), "Median fitness at sharing %.2f",
                          static_cast<double>(stage.groupSharing));
            plotHistory(label, stage.medianFitness, fitnessRange.first, fitnessRange.second);
        }
    }
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(360.0F, 16.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(1040.0F, 820.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Lattice");
    if (state_.worlds.worldCount > 0) {
        int visibleWorld = static_cast<int>(state_.worlds.selectedWorld + 1);
        if (ImGui::SliderInt("Visible world", &visibleWorld, 1,
                             static_cast<int>(state_.worlds.worldCount))) {
            state_.worlds.selectedWorld = static_cast<std::uint32_t>(visibleWorld - 1);
        }
        const std::uint32_t visibleGroup =
            state_.worlds.selectedWorld / state_.agents.trialsPerGenome;
        const std::uint32_t visibleTrial =
            state_.worlds.selectedWorld % state_.agents.trialsPerGenome;
        const std::uint32_t visibleAgentCount =
            agentsInLogicalWorld(state_.agents.genomeCount, state_.worlds.agentsPerWorld,
                                 state_.agents.trialsPerGenome, state_.worlds.selectedWorld);
        ImGui::TextDisabled("Group %u / %u, trial %u / %u, %u agents", visibleGroup + 1,
                            state_.worlds.groupCount, visibleTrial + 1,
                            state_.agents.trialsPerGenome, visibleAgentCount);
    }
    // The 3D view is step 5 of the plan and deliberately last: until it exists,
    // everything a run needs is in the panels and the headless runner, and a
    // placeholder that says so beats a black rectangle that looks broken.
    if (state_.viewport.imageView == VK_NULL_HANDLE) {
        ImGui::TextDisabled("No view of the lattice yet.");
        ImGui::TextWrapped("The 3D renderer is the last piece of the lattice conversion: the "
                           "2D one drew normalised coordinates with no camera at all, so there "
                           "was nothing to carry over. Until then the run is readable through "
                           "the panels here and through the headless runner, which is what the "
                           "statistics and the sweeps were built for.");
    } else {
        const ImVec2 available = ImGui::GetContentRegionAvail();
        if (available.x >= 64.0F && available.y >= 64.0F) {
            state_.viewport.requestedWidth = static_cast<std::uint32_t>(std::floor(available.x));
            state_.viewport.requestedHeight = static_cast<std::uint32_t>(std::floor(available.y));
            const ImTextureID texture =
                static_cast<ImTextureID>(reinterpret_cast<std::uintptr_t>(viewportDescriptor_));
            ImGui::Image(texture, available);
        }
    }
    ImGui::End();
    profilerPanel_.draw(context.profiler);
}

void SimulationUiModule::drawBrainWindow(const neuro::BrainShape& brain) {
    ImGui::SetNextWindowPos(ImVec2(360.0F, 16.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340.0F, 330.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Brain");

    // Only the hidden layers are editable, and that is the point rather than a
    // limitation: how many cells surround one and how many drives a move needs
    // are statements about the lattice. How much brain to spend on it is the
    // question worth asking, and it is the one this window asks.
    ImGui::TextDisabled("%zu sensor inputs and %zu outputs, set by the lattice", brain.inputCount,
                        brain.outputCount);

    // Edited as a draft and applied on a button, not live: a plan is a different
    // genome layout, so it can only take effect on a reset, and a slider that
    // silently did nothing until later would be worse than one that says so.
    auto& draft = state_.controls.hiddenLayerDraft;
    const bool defaulted = state_.settings.hiddenLayers[0] == 0;
    if (draft[0] == 0) {
        for (std::size_t layer = 0; layer < draft.size(); ++layer) {
            draft[layer] = static_cast<int>(brain.hiddenLayer(layer));
        }
    }
    int layerCount = 0;
    for (std::size_t layer = 0; layer < draft.size(); ++layer) {
        if (draft[layer] > 0) {
            layerCount = static_cast<int>(layer) + 1;
        }
    }
    layerCount = std::max(layerCount, 1);
    if (ImGui::SliderInt("Hidden layers", &layerCount, 1,
                         static_cast<int>(neuro::Topology::hiddenLayerCount))) {
        for (std::size_t layer = 0; layer < draft.size(); ++layer) {
            const bool wanted = static_cast<int>(layer) < layerCount;
            // A layer switched on starts as wide as the one before it rather
            // than at zero, so the plan is always one the capacity accepts and
            // the button below is never disabled for a reason nobody chose.
            draft[layer] = wanted ? std::max(draft[layer], layer == 0 ? 8 : draft[layer - 1]) : 0;
        }
    }
    ImGui::SetItemTooltip("Layers are dense from the front. More layers with the same total is a "
                          "narrower path with more turns in it: the same neurons composed rather "
                          "than laid side by side.");

    const auto capacity = static_cast<int>(neuro::Topology::hiddenNeuronCapacity);
    int total = 0;
    for (int layer = 0; layer < layerCount; ++layer) {
        const auto slot = static_cast<std::size_t>(layer);
        ImGui::PushID(layer);
        char label[24];
        std::snprintf(label, sizeof(label), "Layer %d", layer + 1);
        // Each layer is bounded by what the others have not already spent, so a
        // plan under construction is always one that fits.
        int spentElsewhere = 0;
        for (int other = 0; other < layerCount; ++other) {
            if (other != layer) {
                spentElsewhere += draft[static_cast<std::size_t>(other)];
            }
        }
        ImGui::SliderInt(label, &draft[slot], 1, std::max(capacity - spentElsewhere, 1));
        ImGui::PopID();
        total += draft[slot];
    }
    for (std::size_t layer = static_cast<std::size_t>(layerCount); layer < draft.size(); ++layer) {
        draft[layer] = 0;
    }

    neuro::BrainShape planned = brain;
    planned.hiddenCount = static_cast<std::size_t>(draft[0]);
    planned.secondHiddenCount = static_cast<std::size_t>(draft[1]);
    planned.thirdHiddenCount = static_cast<std::size_t>(draft[2]);
    const bool fits = planned.fitsCapacity();
    // The genome is exactly as long as the plan needs, so this number is what a
    // run actually costs and what a file of it will hold -- not a share of some
    // fixed capacity. A deeper plan is usually cheaper than it looks: the first
    // matrix dominates, and a narrow first layer shrinks it.
    ImGui::TextDisabled("%d of %d neurons, %zu weights per genome", total, capacity,
                        fits ? planned.weightCount() : 0);
    if (fits && planned.weightCount() != brain.weightCount()) {
        ImGui::TextDisabled("currently %zu", brain.weightCount());
    }

    const bool changed = planned.hiddenCount != brain.hiddenCount ||
                         planned.secondHiddenCount != brain.secondHiddenCount ||
                         planned.thirdHiddenCount != brain.thirdHiddenCount;
    ImGui::BeginDisabled(!fits || !changed);
    if (ImGui::Button("Apply and reset")) {
        state_.settings.hiddenLayers = {static_cast<std::uint32_t>(draft[0]),
                                        static_cast<std::uint32_t>(draft[1]),
                                        static_cast<std::uint32_t>(draft[2])};
        state_.controls.resetRequested = true;
    }
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("A plan is a different layout of the same genome, so it can only take "
                          "effect on a reset -- the population that was evolving under the old "
                          "one does not carry over meaningfully.");
    ImGui::SameLine();
    ImGui::BeginDisabled(defaulted);
    if (ImGui::Button("Back to the default")) {
        state_.settings.hiddenLayers = {};
        draft = {};
        state_.controls.resetRequested = true;
    }
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("One hidden layer of twenty, which is what a fresh run uses and what "
                          "every measurement so far was taken on.");

    if (changed) {
        ImGui::TextColored(ImVec4{0.95F, 0.75F, 0.25F, 1.0F}, "not applied yet");
    } else if (!defaulted) {
        ImGui::TextDisabled("running a plan of your own, not the default");
    }
    ImGui::End();
}

void SimulationUiModule::onDetach(AppContext&) {
    imgui_.removeTexture(viewportDescriptor_);
    viewportDescriptor_ = VK_NULL_HANDLE;
}

} // namespace vkexp
