#pragma once

#include "vkexp/compute/ComputeResources.hpp"
#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/evolution/GenomeArchive.hpp"
#include "vkexp/lattice/LatticeWorld.hpp"
#include "vkexp/simulation/RunSnapshot.hpp"
#include "vkexp/simulation/SimulationState.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace vkexp {

struct SimulationDriverConfig {
    std::uint32_t trialsPerGenome{4};
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
    [[nodiscard]] RunSnapshot snapshot();
    void restoreSnapshot(const RunSnapshot& snapshot);

    void updateWorldLayout();
    // The two above settle against each other. How many agents share a world
    // decides how many worlds there are, which decides how much grid the budget
    // has to hold, which can shrink the lattice -- and a smaller lattice holds
    // fewer agents. Both quantities only ever fall, and both have a floor, so
    // repeating the pair until the group size stops moving terminates.
    void settleLayout();
    void readStructureDiagnosis();
    // Re-reads the lattice extents from the settings, clamping them to what the
    // occupancy allocation can hold, and republishes the view. Called whenever a
    // slider moves the box.
    void refreshLattice();

    [[nodiscard]] const GeneticAlgorithm& evolution() const { return evolution_; }
    [[nodiscard]] std::span<const AgentState> agents() const { return agents_; }
    [[nodiscard]] const SimulationDriverConfig& config() const { return config_; }
    [[nodiscard]] lattice::PopulationLayout populationLayout() const;
    // How many times the fixed step resources have been built. Exposed for
    // reconfiguration_smoke, which needs to assert that the answer stays one.
    //
    // Comparing published handles across a reconfiguration does not settle it:
    // freeing a buffer and immediately allocating one of the same size usually
    // hands back the same VkBuffer, so a test written that way passes while the
    // buffers are being destroyed under live descriptors. A count of builds is
    // the invariant itself rather than a proxy for it.
    [[nodiscard]] std::uint32_t stepResourceBuilds() const { return stepResourceBuilds_; }

private:
    void adoptBrainPlan();
    void createStepResources();
    // The one step resource whose size follows the brain plan rather than the
    // launch configuration. Split out of createStepResources so that adopting a
    // plan resizes this buffer alone: see the note there for why remaking the
    // rest is not merely wasteful but unsound.
    [[nodiscard]] VkDeviceSize genomeBufferBytes() const;
    void resizeGenomeBuffer();
    void resetGeneration();
    void uploadPopulation(bool preserveStructures = false);
    [[nodiscard]] std::uint64_t latticeBudget() const;
    [[nodiscard]] std::uint64_t latticeBytes(const SimulationStep& settings) const;
    [[nodiscard]] GpuStepParameters stepParameters() const;

    SimulationState& state_;
    GeneticAlgorithm evolution_;
    SimulationDriverConfig config_;

    VkPhysicalDevice physicalDevice_{};
    VkDevice device_{};
    std::uint32_t stepResourceBuilds_{};

    PingPongBuffer agentBuffers_;
    BufferResource genomeBuffer_;
    BufferResource stepParameterBuffer_;
    // Who stands in which cell, and who has bid for which cell. Both one int per
    // cell per world. The occupancy is host-visible because a reset and a
    // snapshot restore both write it from the host; the bids are device-local,
    // because nothing outside a step ever looks at them.
    BufferResource occupancy_;
    BufferResource claims_;
    BufferResource structures_;
    // One counter per world and per reason a build attempt can end, cleared at
    // the top of a generation and read back at the end of it. Nothing in the
    // simulation reads it: it exists because the block count says only "few"
    // and never says what refused.
    BufferResource buildOutcomes_;
    // Display-only breadcrumb rings. Captured after resolve so a marker is the
    // cell the agent actually won, never the cell it merely requested.
    BufferResource trailHistory_;
    UniqueDescriptorSetLayout stepDescriptorSetLayout_;
    UniqueDescriptorSetLayout resolveDescriptorSetLayout_;
    UniqueDescriptorSetLayout clearDescriptorSetLayout_;
    UniqueDescriptorSetLayout trailCaptureDescriptorSetLayout_;
    DescriptorAllocator descriptorAllocator_;
    std::array<VkDescriptorSet, 2> stepDescriptorSets_{};
    std::array<VkDescriptorSet, 2> resolveDescriptorSets_{};
    std::array<VkDescriptorSet, 2> trailCaptureDescriptorSets_{};
    VkDescriptorSet clearDescriptorSet_{};
    ComputePipeline stepPipeline_;
    ComputePipeline resolvePipeline_;
    ComputePipeline clearPipeline_;
    ComputePipeline trailCapturePipeline_;
    std::vector<AgentState> agents_;
    std::vector<std::int32_t> occupancyStaging_;
    std::vector<std::int32_t> structureStaging_;
    std::vector<std::uint32_t> buildOutcomeStaging_;
    std::vector<GpuStepParameters> stepParameterStaging_;
    bool hostUploadPending_{};
    bool trailClearPending_{};
};

// Provenance for a genome archive, built from the run that produced it. Here
// rather than in GenomeArchive.hpp because it reads a driver and a state, and
// the evolution layer knows about neither; here rather than in each caller
// because the interactive save and the headless one have to stamp their files
// the same way, or two archives cannot be told apart after the fact.
[[nodiscard]] GenomeArchiveMetadata genomeArchiveMetadata(const SimulationState& state,
                                                          const SimulationDriver& driver,
                                                          const neuro::BrainShape& brain);

} // namespace vkexp
