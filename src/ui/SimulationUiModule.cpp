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

    ImGui::Text("Generation %llu", static_cast<unsigned long long>(state_.statistics.generation));
    const std::uint32_t generationSteps = std::max(state_.controls.stepsPerGeneration, 1U);
    const float generationProgress =
        std::clamp(static_cast<float>(state_.statistics.step) / static_cast<float>(generationSteps),
                   0.0F, 1.0F);
    char tickProgress[48];
    std::snprintf(tickProgress, sizeof(tickProgress), "%u / %u ticks", state_.statistics.step,
                  state_.controls.stepsPerGeneration);
    ImGui::ProgressBar(generationProgress, ImVec2(-1.0F, 0.0F), tickProgress);

    ImGui::Checkbox("Paused", &state_.controls.paused);
    int stepsPerFrame = static_cast<int>(state_.controls.stepsPerFrame);
    if (ImGui::SliderInt("Simulation steps / frame", &stepsPerFrame, 1, 64)) {
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
    ImGui::TextDisabled("trial %.1f s at %.0f Hz (%.1f ms per step)",
                        static_cast<double>(units::secondsForSteps(
                            state_.controls.stepsPerGeneration, state_.settings.deltaTime)),
                        static_cast<double>(1.0F / state_.settings.deltaTime),
                        static_cast<double>(state_.settings.deltaTime * 1000.0F));

    ImGui::SeparatorText("The lattice");
    int worldMode = static_cast<int>(state_.settings.worldMode);
    constexpr const char* worldModes[] = {"Beacon", "Construction", "Harvest", "Chasm"};
    static_assert(std::size(worldModes) == worldModeCount);
    if (ImGui::Combo("World", &worldMode, worldModes, static_cast<int>(worldModeCount))) {
        state_.settings.worldMode = static_cast<WorldMode>(worldMode);
        applyWorldDefaults(state_.settings);
        state_.controls.resetRequested = true;
    }
    ImGui::SetItemTooltip("Beacon is the navigation task. The other three build: every agent "
                          "starts on the bedrock course, climbs, falls and places supported "
                          "blocks. Construction scores height alone; harvest scores loads fetched "
                          "from a hanging resource; the chasm takes half the floor away, so the "
                          "resource can only be reached across something the group builds.");

    int requestedAgentsPerWorld = static_cast<int>(state_.worlds.requestedAgentsPerWorld);
    // The ceiling is whichever runs out first: genomes, or cells to stand them
    // in. Offering more than the lattice can place would put the surplus agents
    // inside each other at the origin rather than refuse anything.
    const int populationSize = static_cast<int>(
        std::max(1U, std::min(state_.agents.genomeCount, latticeSpawnCapacity(state_.settings))));
    requestedAgentsPerWorld = std::clamp(requestedAgentsPerWorld, 1, populationSize);
    if (ImGui::SliderInt("Agents / world", &requestedAgentsPerWorld, 1, populationSize)) {
        state_.worlds.requestedAgentsPerWorld = static_cast<std::uint32_t>(requestedAgentsPerWorld);
        state_.controls.resetRequested = true;
    }
    ImGui::SetItemTooltip("How many genomes share one logical lattice. One gives every agent its "
                          "own world; the maximum puts the whole population together, or as much "
                          "of it as the lattice has cells to stand. Changing it restarts the run "
                          "and may shrink the lattice to stay in its fixed GPU memory budget.");
    if (ImGui::SmallButton("1 agent")) {
        state_.worlds.requestedAgentsPerWorld = 1;
        state_.controls.resetRequested = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("12 agents")) {
        state_.worlds.requestedAgentsPerWorld = std::min(12U, state_.agents.genomeCount);
        state_.controls.resetRequested = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("All agents")) {
        state_.worlds.requestedAgentsPerWorld =
            std::min(state_.agents.genomeCount, latticeSpawnCapacity(state_.settings));
        state_.controls.resetRequested = true;
    }
    ImGui::TextDisabled("%u groups x %u trials = %u worlds", state_.worlds.groupCount,
                        state_.agents.trialsPerGenome, state_.worlds.worldCount);

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
    const double gridBytes =
        static_cast<double>(cells) * state_.worlds.worldCount * sizeof(std::int32_t);
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

    ImGui::SliderFloat("Turn threshold", &state_.settings.turnThreshold, 0.0F, 0.95F, "%.2f");
    ImGui::SetItemTooltip("How sure both turn outputs have to be before the agent pivots. They "
                          "have to agree: both over it turns one way, both under the negative "
                          "one turns the other, and a disagreement is no turn. A turn costs the "
                          "whole tick.");

    if (worldBuilds(state_.settings.worldMode)) {
        if (worldHarvests(state_.settings.worldMode)) {
            const int ceiling = std::max(static_cast<int>(state_.settings.latticeHeight), 2) - 1;
            std::array<int, 2> band{static_cast<int>(state_.settings.resourceHeightLow),
                                    static_cast<int>(state_.settings.resourceHeightHigh)};
            if (ImGui::SliderInt2("Resource band", band.data(), 1, ceiling, "%d levels")) {
                state_.settings.resourceHeightLow =
                    static_cast<std::uint32_t>(std::clamp(band[0], 1, ceiling));
                state_.settings.resourceHeightHigh = static_cast<std::uint32_t>(
                    std::clamp(band[1], static_cast<int>(state_.settings.resourceHeightLow),
                               ceiling));
            }
            ImGui::SetItemTooltip("The band of heights the resource hangs in, drawn per world. A "
                                  "band and not a height: a fixed height is a number a genome can "
                                  "learn to count to rather than a place it has to find.");
        }
        if (state_.settings.worldMode == WorldMode::Chasm) {
            int ground = static_cast<int>(latticeGroundWidth(state_.settings));
            if (ImGui::SliderInt("Ground columns", &ground, 1,
                                 static_cast<int>(state_.settings.latticeWidth) - 1, "%d of %d")) {
                state_.settings.chasmGroundWidth = static_cast<std::uint32_t>(ground);
            }
            ImGui::SetItemTooltip("How much of the floor is solid, counted along x from the near "
                                  "edge. Everything beyond is open air all the way down, and the "
                                  "resource hangs over it -- so the only route is one the group "
                                  "builds out from the edge.");
            ImGui::TextDisabled("Side support is forced on: without a cantilever the far half "
                                "cannot be reached at all.");
        }
        int buildInterval = static_cast<int>(state_.settings.buildIntervalTicks);
        if (ImGui::SliderInt("Build interval", &buildInterval, 1, 120, "%d ticks")) {
            state_.settings.buildIntervalTicks = static_cast<std::uint32_t>(buildInterval);
        }
        ImGui::SliderFloat("Build threshold", &state_.settings.buildThreshold, 0.0F, 0.95F, "%.2f");

        // The foundation rule. Off in the chasm, and not as a default the user
        // may override: a cantilever has nothing beneath it, so this test would
        // refuse every block of a bridge and leave that world unsolvable.
        ImGui::BeginDisabled(!lattice::kernel::latticeWorldFrontier(
            static_cast<std::uint32_t>(state_.settings.worldMode)));
        float courseFillPercent = state_.settings.constructionCourseFill * 100.0F;
        if (ImGui::SliderFloat("Course fill", &courseFillPercent, 5.0F, 100.0F, "%.0f%%",
                               ImGuiSliderFlags_AlwaysClamp)) {
            state_.settings.constructionCourseFill = courseFillPercent * 0.01F;
        }
        ImGui::SetItemTooltip("How full a level must be, around a build site, before it counts "
                              "as something to stand on.");
        int supportRadius = static_cast<int>(state_.settings.constructionSupportRadius);
        if (ImGui::SliderInt("Support radius", &supportRadius, 0, 16, "%d cells")) {
            state_.settings.constructionSupportRadius = static_cast<std::uint32_t>(supportRadius);
        }
        ImGui::SetItemTooltip("How wide the fill question is asked. Zero asks only about the "
                              "column itself; a radius that spans the floor asks about the whole "
                              "world, which is the old global course frontier. In between, one "
                              "corner of a world may run ahead of another.");
        int heightLead = static_cast<int>(state_.settings.constructionHeightLead);
        if (ImGui::SliderInt("Height above foundation", &heightLead, 1, 16, "%d levels")) {
            state_.settings.constructionHeightLead = static_cast<std::uint32_t>(heightLead);
        }
        ImGui::SetItemTooltip("A block cannot be placed more than this many levels above the "
                              "nearest level below it that is filled enough to stand on.");
        ImGui::EndDisabled();
        if (state_.settings.worldMode == WorldMode::Chasm) {
            ImGui::TextDisabled("The foundation rule is off here: a bridge block has nothing "
                                "under it, so the test would refuse every one of them.");
        }
        bool allowSideSupport = state_.settings.allowSideSupportedBlocks != 0U ||
                                state_.settings.worldMode == WorldMode::Chasm;
        ImGui::BeginDisabled(state_.settings.worldMode == WorldMode::Chasm);
        if (ImGui::Checkbox("Side-supported bridges", &allowSideSupport)) {
            state_.settings.allowSideSupportedBlocks = allowSideSupport ? 1U : 0U;
        }
        ImGui::SetItemTooltip("Allow a block to hang from a cardinal x/z face. Diagonal edge or "
                              "corner contact never supports it.");
        ImGui::EndDisabled();
        ImGui::SliderFloat("Boundary penalty", &state_.settings.fitness.boundaryPenalty, 0.0F,
                           0.05F, "%.4f");
        ImGui::SetItemTooltip("Group charge per agent and tick spent on the x/z perimeter. The "
                              "height ceiling is not penalised.");
        if (worldHarvests(state_.settings.worldMode)) {
            ImGui::TextWrapped(
                "Fitness is loads delivered, plus how near anyone got to the resource. Blocks "
                "score nothing: a block is time spent, and spending it well is the problem. A "
                "load is picked up at the resource and scored on the floor, so a route that can "
                "be used twice is worth more than one lucky scramble.");
        } else {
            ImGui::TextWrapped(
                "Fitness is the sum of block levels minus boundary dwell. Every genome in the "
                "world receives the same total.");
        }
        ImGui::TextDisabled(state_.settings.allowSideSupportedBlocks != 0U
                                ? "Blocks may use a floor, lower block or cardinal side face."
                                : "Blocks need floor or a block directly below; walls climb.");
    } else {
        int contactRadius = static_cast<int>(state_.settings.beaconContactRadius);
        if (ImGui::SliderInt("Contact radius", &contactRadius, 0, 6)) {
            state_.settings.beaconContactRadius = static_cast<std::uint32_t>(contactRadius);
        }
        ImGui::TextDisabled("beacon seed %u, redrawn every generation", state_.settings.beaconSeed);

        ImGui::SeparatorText("Fitness shaping");
        ImGui::SliderFloat("Tracking reward", &state_.settings.fitness.trackingReward, 0.0F, 4.0F,
                           "%.2f");
        ImGui::SliderFloat("Objective bonus", &state_.settings.fitness.objectiveBonus, 0.0F, 0.5F,
                           "%.3f");
        ImGui::SliderFloat("Motor cost", &state_.settings.fitness.motorCostWeight, 0.0F, 0.05F,
                           "%.4f");
        ImGui::SliderFloat("Refusal penalty", &state_.settings.fitness.refusalPenalty, 0.0F, 0.1F,
                           "%.4f");
        ImGui::SliderFloat("Signal cost", &state_.settings.fitness.signalCostFactor, 0.0F, 2.0F,
                           "%.2f");
        if (ImGui::SliderFloat("Group fitness sharing", &state_.settings.fitness.groupSharing, 0.0F,
                               1.0F, "%.2f")) {
            state_.controls.resetRequested = true;
        }
    }

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
    // No squash named here: it is per layer now, and a summary that says "tanh"
    // whatever the plan carries is a line that contradicts the one below it.
    ImGui::Text("%zu inputs -> %s -> %zu outputs", brain.inputCount, layerText.c_str(),
                brain.outputCount);
    ImGui::TextDisabled("%zu weights per genome", brain.weightCount());
    {
        std::string squashes;
        for (std::size_t layer = 0; layer < brain.hiddenLayerCount(); ++layer) {
            squashes += squashes.empty() ? "" : "+";
            squashes += brain.hiddenActivation[layer] == neuro::kernel::BrainActivationSine
                            ? "sin"
                            : "tanh";
        }
        ImGui::TextDisabled("hidden squash %s, outputs tanh", squashes.c_str());
        ImGui::SetItemTooltip("Outputs are always tanh: every threshold in the rules reads one "
                              "as how far and which way. A spiking run ignores this -- it writes "
                              "1 or 0 and never reaches a squash.");
    }
    if (state_.settings.neuronModel != NeuronModel::Reactive) {
        ImGui::TextDisabled("gate block %zu of %zu genes",
                            brain.hiddenTotal() * (brain.inputCount + 1), brain.weightCount());
    }
    ImGui::TextDisabled("time constants %.0f ms .. %.1f s",
                        static_cast<double>(neuro::kernel::BrainTimeConstantMinimum * 1000.0F),
                        static_cast<double>(neuro::kernel::BrainTimeConstantMaximum));
    ImGui::TextDisabled("%u cells x %u channels in the body frame, task state and memory",
                        neuro::kernel::BrainNeighborCount, neuro::kernel::BrainNeighborChannels);
    ImGui::TextDisabled("two turn votes, walk or build, broadcast and memory updates");
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
    drawBestWorld();
    if (worldHarvests(state_.settings.worldMode)) {
        ImGui::Text("Delivered a load: %.1f%% of agents", state_.statistics.arrivalRatio * 100.0F);
        drawStructureShapes();
        drawBuildOutcomes();
    } else if (state_.settings.worldMode == WorldMode::Construction) {
        ImGui::Text("Mean weighted fill: %.2f%%", state_.statistics.arrivalRatio * 100.0F);
        drawStructureShapes();
        drawBuildOutcomes();
    } else {
        ImGui::Text("Reached the beacon: %.1f%%", state_.statistics.arrivalRatio * 100.0F);
    }
    ImGui::SeparatorText("Fitness history");
    // Say so when the plots are a window onto a longer run, rather than letting
    // a curve that has stopped extending read as a run that has stopped.
    if (state_.history.bestFitness.size() < state_.statistics.evaluatedGenerations) {
        ImGui::TextDisabled(
            "last %zu generations of %llu", state_.history.bestFitness.size(),
            static_cast<unsigned long long>(state_.statistics.evaluatedGenerations));
    }
    plotHistory("Best", state_.history.bestFitness);
    plotHistory("Median", state_.history.medianFitness);
    plotHistory("Mean", state_.history.meanFitness);
    // Floored at zero and scaled to the data above it. A fixed 0..1 axis is
    // what a fraction deserves in principle and unreadable in practice: a run
    // filling three per cent of its lattice draws as a flat line on the bottom
    // edge whatever it is doing, and the headline number above says the level
    // anyway.
    plotHistory(state_.settings.worldMode == WorldMode::Beacon ? "Reached the beacon"
                                                               : "Weighted block fill",
                state_.history.arrivalRatio, 0.0F);
    ImGui::SeparatorText("Evolution parameters");
    ImGui::Text("Population: %zu", state_.evolution.populationSize);
    ImGui::Text("Elites: %zu   Tournament: %zu", state_.evolution.eliteCount,
                state_.evolution.tournamentSize);
    ImGui::Text("Crossover: %.1f%%", state_.evolution.crossoverProbability * 100.0F);
    ImGui::Text("Mutation: %.1f%%  strength %.3f", state_.evolution.mutationProbability * 100.0F,
                state_.evolution.mutationStrength);
    ImGui::Text("Fresh genomes: %s", state_.evolution.weightInit == WeightInit::FanIn
                                         ? "drawn per block by fan-in"
                                         : "drawn at one width, outputs saturate");
    ImGui::SetItemTooltip("Set at the start of a run. Fan-in leaves outputs unsaturated, which "
                          "needs the turn and build thresholds scaled down with it -- at the "
                          "shipped ones nothing turns and nothing builds.");

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

    ImGui::SetNextWindowPos(ImVec2(710.0F, 16.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330.0F, 330.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("View settings");
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
    drawViewControls();
    ImGui::End();

    // The view is deliberately only a picture. Selection, projection and all
    // other controls live in View settings, leaving every pixel here available
    // to the lattice and making this window suitable for moving to a second
    // monitor without carrying a strip of controls with it.
    ImGui::SetNextWindowPos(ImVec2(360.0F, 360.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(1040.0F, 520.0F), ImGuiCond_FirstUseEver);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Lattice view", nullptr,
                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    if (state_.viewport.imageView == VK_NULL_HANDLE) {
        ImGui::TextDisabled("The view has not published an image yet.");
    } else {
        const ImVec2 available = ImGui::GetContentRegionAvail();
        if (available.x >= 64.0F && available.y >= 64.0F) {
            state_.viewport.requestedWidth = static_cast<std::uint32_t>(std::floor(available.x));
            state_.viewport.requestedHeight = static_cast<std::uint32_t>(std::floor(available.y));
            const ImTextureID texture =
                static_cast<ImTextureID>(reinterpret_cast<std::uintptr_t>(viewportDescriptor_));
            const ImVec2 origin = ImGui::GetCursorScreenPos();
            ImGui::Image(texture, available);
            // The camera is dragged on the picture rather than typed into
            // sliders: a box is a thing one turns over, and three numbers are
            // how that gets stored, not how it gets done. The button sits on top
            // of the image because an ImGui::Image is not something that can be
            // held.
            ImGui::SetCursorScreenPos(origin);
            ImGui::InvisibleButton("##orbit", available,
                                   ImGuiButtonFlags_MouseButtonLeft |
                                       ImGuiButtonFlags_MouseButtonRight |
                                       ImGuiButtonFlags_MouseButtonMiddle);
            LatticeCamera& camera = state_.display.camera;
            if (ImGui::IsItemActive()) {
                const ImVec2 drag = ImGui::GetIO().MouseDelta;
                const bool sliding = ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
                                     ImGui::IsMouseDown(ImGuiMouseButton_Middle);
                if (sliding) {
                    // One dragged pixel moves the box by one pixel's worth of
                    // the world, so a slide feels the same however far away the
                    // camera is and whichever projection is on: both frame the
                    // same height at the same distance, which is what makes the
                    // two comparable at all.
                    const float halfDiagonal =
                        0.5F * std::sqrt(
                                   static_cast<float>(state_.settings.latticeWidth) *
                                       static_cast<float>(state_.settings.latticeWidth) +
                                   static_cast<float>(state_.settings.latticeHeight) *
                                       static_cast<float>(state_.settings.latticeHeight) +
                                   static_cast<float>(state_.settings.latticeDepth) *
                                       static_cast<float>(state_.settings.latticeDepth));
                    const float radius = std::max(camera.distance * halfDiagonal, 0.2F);
                    const float perPixel =
                        2.0F * radius * std::tan(latticeCameraFieldOfView * 0.5F) /
                        std::max(available.y, 1.0F);
                    // Dragging right carries the box right, so the camera goes
                    // left. A slide that moved the box the other way would be a
                    // camera control rather than a handle on the thing itself.
                    camera.slide(-drag.x * perPixel, drag.y * perPixel);
                } else {
                    camera.yaw -= drag.x * 0.008F;
                    camera.pitch = std::clamp(camera.pitch + drag.y * 0.008F, -1.53F, 1.53F);
                }
            }
            if (ImGui::IsItemHovered()) {
                const float wheel = ImGui::GetIO().MouseWheel;
                if (wheel != 0.0F) {
                    camera.distance =
                        std::clamp(camera.distance * std::exp(-wheel * 0.12F), 0.35F, 12.0F);
                }
            }
        }
    }
    ImGui::End();
    profilerPanel_.draw(context.profiler);
}

// What the last generation actually built, as opposed to what it scored. The
// score cannot tell a slab from a spire; these can. The second column is the
// champion's world, so the two columns answer "what am I looking at" and "what
// did the winner do". Nothing here is fed back into fitness -- see
// StructureShape.hpp.
void SimulationUiModule::drawStructureShapes() {
    if (state_.statistics.worldShapes.empty()) {
        return;
    }
    const std::size_t worlds = state_.statistics.worldShapes.size();
    const std::size_t visible = std::min<std::size_t>(state_.worlds.selectedWorld, worlds - 1);
    const std::size_t best = std::min<std::size_t>(state_.statistics.bestWorld, worlds - 1);

    ImGui::SeparatorText("Shape of the last building");
    if (!ImGui::BeginTable("structure shape", 3,
                           ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) {
        return;
    }
    ImGui::TableSetupColumn("");
    ImGui::TableSetupColumn("visible");
    ImGui::TableSetupColumn("champion");
    ImGui::TableHeadersRow();
    const auto row = [&](const char* label, const char* tooltip, const char* format,
                         const auto visibleValue, const auto bestValue) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(label);
        ImGui::SetItemTooltip("%s", tooltip);
        ImGui::TableNextColumn();
        ImGui::Text(format, visibleValue);
        ImGui::TableNextColumn();
        ImGui::Text(format, bestValue);
    };
    const StructureShape& here = state_.statistics.worldShapes[visible];
    const StructureShape& top = state_.statistics.worldShapes[best];
    row("Blocks", "Blocks placed in the world.", "%u", here.blocks, top.blocks);
    row("Footprint", "Floor squares carrying at least one block.", "%u", here.footprint,
        top.footprint);
    row("Peak", "Height of the tallest column, in cells.", "%u", here.peak, top.peak);
    row("Mean height", "Mean height over the columns that carry anything. A slab and a spire of "
                       "equal mass differ here first.",
        "%.2f", here.meanHeight, top.meanHeight);
    row("Height spread", "Standard deviation of those heights. Zero means every column is the "
                         "same height, which is what a flat course looks like.",
        "%.2f", here.heightSpread, top.heightSpread);
    row("Compactness", "Occupied squares as a fraction of their own bounding rectangle. One is a "
                       "solid plan; lower is a ring, a cross or scattered piers.",
        "%.2f", here.compactness, top.compactness);
    row("Overhangs", "Blocks standing on empty space. Only reachable with side support enabled.",
        "%u", here.overhangs, top.overhangs);
    row("Roofed", "Empty cells with a block above them in the same column. A tower of solid "
                  "courses has none.",
        "%u", here.enclosed, top.enclosed);
    ImGui::EndTable();
}

// What every tick was spent on. Exactly one entry is recorded per agent per
// step, so these sum to agents times steps: the first three say how the walking
// went and the rest read as a funnel through the build rules -- everything that
// did not become a block was stopped somewhere, and this says where.
void SimulationUiModule::drawBuildOutcomes() {
    const auto count = static_cast<std::size_t>(vkexp::lattice::kernel::LatticeBuildOutcomeCount);
    if (state_.statistics.buildOutcomes.size() < count) {
        return;
    }
    const std::size_t worlds = state_.statistics.buildOutcomes.size() / count;
    const std::size_t visible = std::min<std::size_t>(state_.worlds.selectedWorld, worlds - 1);

    ImGui::SeparatorText("What the ticks went on");
    static constexpr std::array<const char*, vkexp::lattice::kernel::LatticeBuildOutcomeCount>
        names{"Turning",         "Walking",    "Edge of the world", "Over a chasm",
              "Under a ceiling",  "Crowded",    "Cooling",           "Off the lattice",
              "Blocked",          "No support", "Above frontier",    "In the way",
              "Placed",           "Lost the cell"};
    std::array<std::uint64_t, names.size()> total{};
    std::uint64_t attempts = 0;
    for (std::size_t world = 0; world < worlds; ++world) {
        for (std::size_t reason = 0; reason < count; ++reason) {
            total[reason] += state_.statistics.buildOutcomes[world * count + reason];
        }
    }
    for (const std::uint64_t reason : total) {
        attempts += reason;
    }
    if (attempts == 0) {
        ImGui::TextDisabled("no ticks recorded");
        return;
    }

    if (!ImGui::BeginTable("build outcomes", 3,
                           ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) {
        return;
    }
    ImGui::TableSetupColumn("");
    ImGui::TableSetupColumn("visible");
    ImGui::TableSetupColumn("all worlds");
    ImGui::TableHeadersRow();
    for (std::size_t reason = 0; reason < count; ++reason) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(names[reason]);
        ImGui::TableNextColumn();
        ImGui::Text("%u", state_.statistics.buildOutcomes[visible * count + reason]);
        ImGui::TableNextColumn();
        ImGui::Text("%llu  %.1f%%", static_cast<unsigned long long>(total[reason]),
                    100.0 * static_cast<double>(total[reason]) / static_cast<double>(attempts));
    }
    ImGui::EndTable();
    ImGui::TextDisabled("one reason per agent and tick, %llu in all",
                        static_cast<unsigned long long>(attempts));
}

// Where the champion of the last generation ran, in every world mode. Offered
// rather than imposed, with a switch for a run being watched rather than read.
void SimulationUiModule::drawBestWorld() {
    if (state_.worlds.worldCount == 0) {
        return;
    }
    const std::uint32_t best =
        std::min(state_.statistics.bestWorld, state_.worlds.worldCount - 1);
    ImGui::Text("Champion ran in world %u", best + 1);
    ImGui::SameLine();
    if (ImGui::SmallButton("Look at it")) {
        state_.worlds.selectedWorld = best;
    }
    ImGui::Checkbox("Follow the champion", &state_.display.followBestWorld);
    ImGui::SetItemTooltip("Move the visible world to the champion's at the end of every "
                          "generation. Off while reading one world carefully; on while watching "
                          "a run. The first world is not the champion's by default -- that it "
                          "often is comes from the elite count and the group size both being 12.");
}

void SimulationUiModule::drawViewControls() {
    SimulationDisplay& display = state_.display;
    if (!ImGui::CollapsingHeader("View", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    int style = static_cast<int>(display.voxelStyle);
    constexpr const char* styles[] = {"Solid", "See-through"};
    if (ImGui::Combo("Voxels", &style, styles, static_cast<int>(std::size(styles)))) {
        display.voxelStyle = static_cast<VoxelStyle>(style);
    }
    ImGui::SetItemTooltip("Solid voxels hide the ones behind them, which is what makes a "
                          "crowd readable. See-through ones let the whole box be read at "
                          "once and resolve without sorting anything.");

    ImGui::BeginDisabled(display.voxelStyle != VoxelStyle::Transparent);
    ImGui::SliderFloat("Opacity", &display.voxelOpacity, 0.02F, 1.0F, "%.2f");
    ImGui::EndDisabled();
    ImGui::SliderFloat("Cell fill", &display.voxelScale, 0.10F, 1.0F, "%.2f");
    ImGui::SetItemTooltip("How much of its cell a voxel fills. At 1.0 two neighbours are one "
                          "block, which is the honest picture of a lattice and a poor picture "
                          "of two agents.");

    ImGui::Checkbox("Agents", &display.agents);
    ImGui::SameLine();
    ImGui::Checkbox("Trails", &display.trails);
    ImGui::SameLine();
    if (state_.settings.worldMode == WorldMode::Construction) {
        ImGui::Checkbox("Blocks", &display.structures);
    } else {
        ImGui::Checkbox("Beacon", &display.beacons);
    }
    ImGui::SameLine();
    ImGui::Checkbox("Box", &display.bounds);
    ImGui::SliderFloat("Background", &display.backgroundBrightness, 0.0F, 1.0F, "%.2f");

    ImGui::BeginDisabled(!display.trails);
    int trailLength = static_cast<int>(display.trailLength);
    const int maximumTrailLength = static_cast<int>(std::max(state_.trails.capacity, 1U));
    if (ImGui::SliderInt("Trail ticks", &trailLength, 1, maximumTrailLength)) {
        display.trailLength = static_cast<std::uint32_t>(trailLength);
    }
    ImGui::SliderFloat("Trail opacity", &display.trailOpacity, 0.01F, 0.80F, "%.2f");
    ImGui::SliderFloat("Trail size", &display.trailScale, 0.08F, 0.90F, "%.2f");
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Each agent leaves a coloured voxel breadcrumb after a resolved tick. "
                          "The history is display-only: agents cannot sense it and it does not "
                          "change movement or fitness.");

    // A slab of the box, which is the other way of seeing inside one: solid
    // voxels and a thin slice answer "who is next to whom", see-through ones
    // answer "where is everybody".
    int axis = static_cast<int>(std::min(display.sliceAxis, 2U));
    constexpr const char* axes[] = {"x", "y", "z"};
    const std::array<std::uint32_t, 3> extents{
        state_.settings.latticeWidth, state_.settings.latticeHeight, state_.settings.latticeDepth};
    if (ImGui::Combo("Slice axis", &axis, axes, static_cast<int>(std::size(axes)))) {
        display.sliceAxis = static_cast<std::uint32_t>(axis);
        display.sliceLow = 0;
        display.sliceHigh = latticeMaximumExtent;
    }
    const auto extent = static_cast<int>(extents[static_cast<std::size_t>(axis)]);
    int low = std::clamp(static_cast<int>(display.sliceLow), 0, extent - 1);
    int high = std::clamp(static_cast<int>(display.sliceHigh), low, extent - 1);
    // Clamped for the slider, never written back. The renderer clamps the slab
    // for itself, so persisting it here bought nothing and cost the one state
    // worth keeping: "show all of it". Writing the clamp back turned an open
    // slab into a fixed number the moment any lattice was smaller, and then a
    // world that grew -- picking the chasm, which is a 32-cube -- kept showing
    // the old half. Half a lattice looks exactly like a lattice, which is what
    // makes this worth a comment rather than a clamp.
    if (ImGui::DragIntRange2("Slice", &low, &high, 0.25F, 0, extent - 1, "%d", "%d")) {
        display.sliceLow = static_cast<std::uint32_t>(low);
        display.sliceHigh = static_cast<std::uint32_t>(high);
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("All")) {
        display.sliceLow = 0;
        // Not extent - 1: the point of this button is a slab that stays open
        // when the lattice changes under it.
        display.sliceHigh = latticeMaximumExtent;
    }

    LatticeCamera& camera = display.camera;
    int projection = static_cast<int>(camera.projection);
    constexpr const char* projections[] = {"Perspective", "Orthographic"};
    if (ImGui::Combo("Projection", &projection, projections,
                     static_cast<int>(std::size(projections)))) {
        camera.projection = static_cast<CameraProjection>(projection);
    }
    int spin = static_cast<int>(camera.spin);
    constexpr const char* spins[] = {"Still", "Orbit", "Turntable"};
    if (ImGui::Combo("Turn", &spin, spins, static_cast<int>(std::size(spins)))) {
        camera.spin = static_cast<CameraSpin>(spin);
    }
    ImGui::SetItemTooltip("Orbit circles the eye around whatever is being looked at. Turntable "
                          "turns the box on its own axis and leaves it where it is in the frame, "
                          "which is what a recording wants once the view has been dragged off "
                          "centre. With the view centred the two are the same motion.");
    ImGui::BeginDisabled(camera.spin == CameraSpin::Off);
    // Seconds for one full turn rather than radians per second: the useful
    // question when filming is how long the clip has to be, not what the
    // angular rate is.
    constexpr float tau = 6.283185307F;
    float secondsPerTurn = tau / std::max(camera.spinRate, 1.0e-3F);
    if (ImGui::SliderFloat("Seconds per turn", &secondsPerTurn, 4.0F, 300.0F, "%.0f s",
                           ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp)) {
        camera.spinRate = tau / std::max(secondsPerTurn, 1.0F);
    }
    ImGui::EndDisabled();
    if (ImGui::SmallButton("Reset view")) {
        camera = latticeHomeCamera(state_.settings);
    }
    ImGui::SameLine();
    // Undoes a slide without undoing the angle it was made from, which is
    // usually what is wanted: the view was turned to something deliberately and
    // then pushed off centre while looking at it.
    if (ImGui::SmallButton("Centre")) {
        camera.targetX = 0.0F;
        camera.targetY = 0.0F;
        camera.targetZ = 0.0F;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("drag to orbit, right-drag to slide, wheel to zoom");
    // The colours carry the three things a still frame cannot say by itself.
    if (state_.settings.worldMode == WorldMode::Construction) {
        ImGui::TextDisabled("blocks: clay low / sunlit high; trails: colour per genome");
    } else {
        ImGui::TextDisabled(
            "agents: blue far / warm near / red refused; trails: colour per genome");
    }
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
    auto& squashDraft = state_.controls.hiddenSquashDraft;
    const bool defaulted = state_.settings.hiddenLayers[0] == 0 &&
                           state_.settings.hiddenActivation ==
                               neuro::defaultBrainShape.hiddenActivation;
    if (draft[0] == 0) {
        for (std::size_t layer = 0; layer < draft.size(); ++layer) {
            draft[layer] = static_cast<int>(brain.hiddenLayer(layer));
            squashDraft[layer] = static_cast<int>(brain.hiddenActivation[layer]);
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
        // The squash beside the width, because they are one decision about one
        // layer. Sine is the default on the first and it is not a conclusion: it
        // measured three times the blocks of tanh over four generations of
        // construction, and four generations is where a run starts, not where it
        // gets to.
        static constexpr std::array<const char*, 4> squashNames{"tanh", "sin", "tanh / sqrt(n)",
                                                                "softsign"};
        ImGui::SetNextItemWidth(ImGui::CalcItemWidth() * 0.6F);
        ImGui::Combo("##squash", &squashDraft[slot], squashNames.data(),
                     static_cast<int>(squashNames.size()));
        ImGui::SetItemTooltip("What squashes this layer. Outputs are always tanh: every threshold "
                              "in the rules reads one as how far and which way. A spiking run "
                              "ignores this -- it writes 1 or 0 and never reaches a squash.");
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
    for (std::size_t layer = 0; layer < squashDraft.size(); ++layer) {
        planned.hiddenActivation[layer] =
            static_cast<std::uint32_t>(std::max(squashDraft[layer], 0));
    }
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
                         planned.thirdHiddenCount != brain.thirdHiddenCount ||
                         planned.hiddenActivation != brain.hiddenActivation;
    ImGui::BeginDisabled(!fits || !changed);
    if (ImGui::Button("Apply and reset")) {
        state_.settings.hiddenLayers = {static_cast<std::uint32_t>(draft[0]),
                                        static_cast<std::uint32_t>(draft[1]),
                                        static_cast<std::uint32_t>(draft[2])};
        state_.settings.hiddenActivation = planned.hiddenActivation;
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
        state_.settings.hiddenActivation = neuro::defaultBrainShape.hiddenActivation;
        draft = {};
        squashDraft = {-1, -1, -1};
        state_.controls.resetRequested = true;
    }
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("One hidden layer of 35, squashed by sine -- what a fresh run uses. "
                          "The squash is what the measurements in BrainKernel.inl chose; the "
                          "single layer is a trade against the genome length.");

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
