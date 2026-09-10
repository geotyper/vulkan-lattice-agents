#pragma once

#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/simulation/AgentTypes.hpp"
#include "vkexp/simulation/ExperimentSweep.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vkexp {

struct SimulationControls {
    bool paused{};
    bool resetRequested{};
    std::uint32_t stepsPerFrame{4};
    std::uint32_t stepsPerGeneration{900};

    // Snapshot requests, handled the same way as resetRequested: the UI raises a
    // flag and SimulationModule acts on it between frames, where the device can
    // be made idle. `snapshotStatus` is what the last attempt did, shown back in
    // the panel so a failed load is visible rather than silent.
    std::string snapshotPath{"world.vknw"};
    std::string snapshotStatus;
    bool saveRequested{};
    bool loadRequested{};

    // A genome archive carries weights and nothing else, which is exactly what
    // replaying a champion needs: the world is whatever is set up here, and the
    // brain is the one that was trained elsewhere.
    std::string genomePath{"champion.vkng"};
    bool loadGenomesRequested{};
    bool saveGenomesRequested{};
    // Whether that save is the whole population or the champion alone.
    bool saveWholePopulation{};
    // Writes what the weights *mean* rather than the weights: which slot of the
    // input vector is which sensor, which span of the genome is which weight
    // block. Goes next to the archive with a .json extension.
    bool saveBrainStructureRequested{};

    // The hidden-layer plan being edited in the Brain window, before it is
    // applied. Kept beside the other controls rather than in the physics block
    // because it is a draft: what the run is actually using lives in
    // SimulationStep, and these two differing is exactly what "not applied yet"
    // means. All zero means "not started editing".
    std::array<int, 3> hiddenLayerDraft{};

    // Watching rather than training. The generation is still scored and
    // reported -- that is how loaded weights get judged -- but nothing is
    // selected or mutated, so the same population respawns and the run repeats
    // instead of drifting away from the weights that were loaded.
    bool replay{};

    // Raised by the UI and acted on between frames, like the flags above.
    bool sweepStartRequested{};
    bool sweepStopRequested{};
};

struct SimulationStatistics {
    std::uint64_t generation{};
    // Generations actually evaluated since the run started. Not the same as
    // `generation`, which does not advance under replay, and emphatically not
    // the same as the length of the history below, which stops at
    // maximumSamples and would otherwise report a run as having frozen the
    // moment the plots filled up.
    std::uint64_t evaluatedGenerations{};
    std::uint32_t step{};
    float bestFitness{};
    float meanFitness{};
    float medianFitness{};
    float arrivalRatio{};
};

struct EvolutionHistory {
    std::vector<float> bestFitness;
    std::vector<float> medianFitness;
    std::vector<float> meanFitness;
    std::vector<float> arrivalRatio;
    // The plots keep a window, not the whole run. Anything reading a length
    // here is reading how much is plotted; how far the run has got is
    // SimulationStatistics::evaluatedGenerations.
    std::size_t maximumSamples{256};
};

// What the viewport draws. None of this reaches the simulation -- turning the
// agents off to watch the bare trail field changes the picture, not the run.
struct SimulationDisplay {
    bool agents{true};
    bool beacons{true};
    bool trail{true};
    // Discs read as a track, squares tile the cell exactly and show the grid the
    // field really is. Both are worth having in front of you, so this stays a
    // switch rather than a decision.
    bool roundTrailMarks{true};
    // Multiplies the arena fill and the ground outside it, and nothing else: a
    // dark background is what makes a faint trail and a low-intensity signal
    // readable, and both are already near the bottom of the range. 1 is the
    // palette as designed and 0 is black, so the control only ever darkens.
    float backgroundBrightness{1.0F};
};

struct SimulationWorlds {
    std::uint32_t requestedAgentsPerWorld{12};
    std::uint32_t agentsPerWorld{12};
    std::uint32_t groupCount{};
    std::uint32_t worldCount{};
    std::uint32_t selectedWorld{};
};

struct AgentBufferView {
    std::array<VkBuffer, 2> buffers{};
    VkDeviceSize size{};
    std::uint32_t currentIndex{};
    std::uint32_t agentCount{};
    std::uint32_t genomeCount{};
    std::uint32_t trialsPerGenome{};
    std::uint64_t generation{};

    [[nodiscard]] VkBuffer currentBuffer() const { return buffers[currentIndex]; }
};

// Published so the renderer can draw the field it never writes. Sized once for
// the largest arena and the most logical worlds the population can be split
// into, so the buffer never reallocates and this handle never goes stale.
struct TrailBufferView {
    VkBuffer buffer{};
    VkDeviceSize size{};
    std::uint32_t width{};
    std::uint32_t cellsPerWorld{};
};

// The puck buffer, published so the renderer can draw what the simulation is
// pushing around rather than a copy of it.
struct PuckBufferView {
    VkBuffer buffer{};
    VkDeviceSize size{};
};

struct SimulationViewport {
    VkImageView imageView{};
    VkSampler sampler{};
    VkExtent2D extent{960, 720};
    std::uint32_t requestedWidth{960};
    std::uint32_t requestedHeight{720};
    std::uint64_t generation{};
};

struct SimulationState {
    SimulationControls controls;
    SimulationStatistics statistics;
    EvolutionSettings evolution;
    EvolutionHistory history;
    SimulationWorlds worlds;
    SimulationStep physics;
    SweepState sweep;
    SimulationDisplay display;
    AgentBufferView agents;
    TrailBufferView trail;
    PuckBufferView puck;
    SimulationViewport viewport;
};

} // namespace vkexp
