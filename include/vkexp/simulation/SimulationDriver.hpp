#pragma once

#include "vkexp/compute/ComputeResources.hpp"
#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/simulation/SimulationState.hpp"
#include "vkexp/simulation/WorldSnapshot.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace vkexp {

struct SimulationDriverConfig {
    std::uint32_t trialsPerGenome{4};
    float gridCellSize{0.12F};
    // Upper bound on the steps a single recordSteps() call may batch. It sizes
    // the per-step parameter buffer.
    std::uint32_t maximumStepsPerBatch{128};
};

// Owns the GPU population, the genetic algorithm and the per-step dispatch
// recording for one experiment.
//
// Deliberately free of Module, Window and swapchain concepts: the interactive
// application drives it from a frame loop, the headless runner drives it from
// an ImmediateContext, and both get identical results. Everything the UI needs
// is published through SimulationState.
class SimulationDriver {
public:
    explicit SimulationDriver(SimulationState& state, EvolutionSettings evolution = {},
                              SimulationDriverConfig config = {});

    SimulationDriver(const SimulationDriver&) = delete;
    SimulationDriver& operator=(const SimulationDriver&) = delete;

    void createResources(VkPhysicalDevice physicalDevice, VkDevice device);
    void destroyResources();

    // Records up to `maximumSteps` simulation steps into an already-begun
    // command buffer and returns how many were recorded; 0 means the generation
    // is exhausted and finishGeneration() should run next.
    //
    // The per-step parameters are written straight into host-visible memory, so
    // the caller must guarantee that any earlier submission reading them has
    // completed. The interactive path gets this from VulkanContext::beginFrame
    // waiting on the frame fence; the headless path from ImmediateContext.
    std::uint32_t recordSteps(VkCommandBuffer commands, std::uint32_t maximumSteps);

    [[nodiscard]] bool generationComplete() const;

    // Scores the finished generation, evolves, and uploads the next one.
    // Requires all recorded work to have completed on the device.
    GenerationSummary finishGeneration();

    // Restarts evolution from the configured seed and clears published history.
    void restart();

    // Replaces the population, e.g. when resuming from a genome archive.
    void loadPopulation(std::span<const Genome> genomes, std::uint64_t generation);

    // Runs the plan in SimulationState::sweep: one clean run per swept value,
    // restarting evolution between stages so a stage is never handed the
    // population the previous setting produced. The plan and the results are
    // published state, so the UI reads them without reaching into the driver.
    void beginSweep();
    void endSweep();

    // Freezes and restores a whole experiment rather than just its weights.
    //
    // Both require the device to be idle: snapshot() reads the agent buffer back
    // to the host, and restoreSnapshot() overwrites it. Unlike loadPopulation
    // these do not restart the generation -- an experiment resumes on the step
    // it was saved on, with the agents where they stood.
    [[nodiscard]] WorldSnapshot snapshot();
    void restoreSnapshot(const WorldSnapshot& snapshot);

    void updateWorldLayout();
    void refreshGridForWorldSize();

    [[nodiscard]] const GeneticAlgorithm& evolution() const { return evolution_; }
    [[nodiscard]] std::span<const AgentState> agents() const { return agents_; }
    [[nodiscard]] const SimulationDriverConfig& config() const { return config_; }

private:
    void createStepResources();
    void resetGeneration();
    void uploadPopulation();
    void ensureGridCapacity();
    void updateGridDescriptors();
    [[nodiscard]] float gridCellSize() const;
    void updateGridDimensions();
    [[nodiscard]] std::uint64_t trailFieldBudget() const;
    [[nodiscard]] std::uint64_t trailFieldBytes(float cellSize) const;
    void updateTrailDimensions();
    [[nodiscard]] GpuStepParameters stepParameters(std::uint32_t generationStep) const;
    [[nodiscard]] std::vector<AgentState> makeInitialAgents() const;
    [[nodiscard]] std::vector<PuckState> makeInitialPucks() const;

    SimulationState& state_;
    GeneticAlgorithm evolution_;
    SimulationDriverConfig config_;

    VkPhysicalDevice physicalDevice_{};
    VkDevice device_{};

    PingPongBuffer agentBuffers_;
    BufferResource genomeBuffer_;
    BufferResource stepParameterBuffer_;
    BufferResource gridHeads_;
    BufferResource gridNext_;
    BufferResource trailField_;
    BufferResource puckField_;
    UniqueDescriptorSetLayout stepDescriptorSetLayout_;
    UniqueDescriptorSetLayout gridClearDescriptorSetLayout_;
    UniqueDescriptorSetLayout gridBuildDescriptorSetLayout_;
    UniqueDescriptorSetLayout trailDecayDescriptorSetLayout_;
    UniqueDescriptorSetLayout trailDepositDescriptorSetLayout_;
    UniqueDescriptorSetLayout puckStepDescriptorSetLayout_;
    DescriptorAllocator descriptorAllocator_;
    std::array<VkDescriptorSet, 2> stepDescriptorSets_{};
    VkDescriptorSet gridClearDescriptorSet_{};
    std::array<VkDescriptorSet, 2> gridBuildDescriptorSets_{};
    VkDescriptorSet trailDecayDescriptorSet_{};
    std::array<VkDescriptorSet, 2> trailDepositDescriptorSets_{};
    std::array<VkDescriptorSet, 2> puckStepDescriptorSets_{};
    ComputePipeline stepPipeline_;
    ComputePipeline gridClearPipeline_;
    ComputePipeline gridBuildPipeline_;
    ComputePipeline trailDecayPipeline_;
    ComputePipeline trailDepositPipeline_;
    ComputePipeline puckStepPipeline_;
    std::vector<AgentState> agents_;
    std::vector<PuckState> pucks_;
    std::vector<GpuStepParameters> stepParameterStaging_;
    std::uint32_t gridWidth_{};
    std::uint32_t gridCellsPerWorld_{};
    std::uint32_t trailWidth_{};
    std::uint32_t trailCellsPerWorld_{};
    VkDeviceSize trailActiveBytes_{};
    bool hostUploadPending_{};
    bool trailClearPending_{true};
};

} // namespace vkexp
