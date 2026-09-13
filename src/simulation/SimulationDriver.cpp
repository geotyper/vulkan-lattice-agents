#include "vkexp/simulation/SimulationDriver.hpp"

#include "vkexp/simulation/CpuLattice.hpp"
#include "vkexp/simulation/LatticeBindings.hpp"
#include "vkexp/simulation/StepParameters.hpp"
#include "vkexp/simulation/StructureShape.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>

namespace vkexp {
namespace {

constexpr VkMemoryPropertyFlags hostMemory =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

struct TrailCaptureParameters {
    std::uint32_t agentCount{};
    std::uint32_t cursor{};
    std::uint32_t capacity{};
    std::uint32_t reserved{};
};

static_assert(sizeof(TrailCaptureParameters) == 16);

void createStorageLayout(const VkDevice device, const std::uint32_t bindingCount,
                         UniqueDescriptorSetLayout& layout) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount);
    for (std::uint32_t binding = 0; binding < bindingCount; ++binding) {
        bindings[binding].binding = binding;
        bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[binding].descriptorCount = 1;
        bindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount = bindingCount;
    info.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &info, nullptr, layout.put(device)) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create lattice compute descriptor layout");
    }
}

} // namespace

SimulationDriver::SimulationDriver(SimulationState& state, EvolutionSettings evolution,
                                   SimulationDriverConfig config)
    : state_(state), evolution_(evolution), config_(config) {
    if (config_.trialsPerGenome == 0 || config_.maximumStepsPerBatch == 0) {
        throw std::invalid_argument("Invalid simulation driver configuration");
    }
    adoptBrainPlan();
    state_.evolution = evolution_.settings();
}

lattice::PopulationLayout SimulationDriver::populationLayout() const {
    return {static_cast<std::uint32_t>(evolution_.population().size()),
            state_.worlds.agentsPerWorld, config_.trialsPerGenome};
}

// The genome is as long as the plan reading it, so the plan settles the length
// once and everything after -- the population, the GPU buffer, the files --
// reads it from the settings rather than from a compiled-in constant.
void SimulationDriver::adoptBrainPlan() {
    const std::size_t weights = resolvedBrain(state_.settings).weightCount();
    if (evolution_.settings().weightCount == weights) {
        return;
    }
    EvolutionSettings settings = evolution_.settings();
    settings.weightCount = weights;
    evolution_ = GeneticAlgorithm{settings};
    state_.evolution = evolution_.settings();
}

void SimulationDriver::createResources(const VkPhysicalDevice physicalDevice,
                                       const VkDevice device) {
    physicalDevice_ = physicalDevice;
    device_ = device;
    createStepResources();
    resetGeneration();
}

// Everything a step needs on the device, made once and only once.
//
// "Once" is load-bearing rather than tidy. Every buffer here but one is
// dimensioned by the population size and the trial count, and both of those are
// fixed when the driver is constructed -- so nothing that happens afterwards can
// change their sizes, and every descriptor naming them stays correct for the
// life of the driver. The exception is the genome buffer, which follows the
// brain plan; resizeGenomeBuffer is how it changes, and it deliberately leaves
// the others alone.
//
// The full-cell lattice fields are allocated at the budget and never
// reallocated, which
// is the lesson the trail field taught in the 2D build: every resize there was a
// lifetime bug rather than a capacity one -- a descriptor somewhere still named
// the buffer that had just been freed. A fixed allocation makes the whole class
// of bug unreachable instead of patching each holder of a handle, and the
// lattice extents give way instead. See refreshLattice.
void SimulationDriver::createStepResources() {
    ++stepResourceBuilds_;
    const auto genomeCount = static_cast<std::uint32_t>(evolution_.population().size());
    const std::uint32_t agentCount = genomeCount * config_.trialsPerGenome;
    const VkDeviceSize agentBytes = sizeof(AgentState) * agentCount;
    const VkDeviceSize stepParameterBytes =
        sizeof(GpuStepParameters) * config_.maximumStepsPerBatch;
    agentBuffers_.create(physicalDevice_, device_,
                         {agentBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    genomeBuffer_.create(physicalDevice_, device_,
                         {genomeBufferBytes(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    stepParameterBuffer_.create(
        physicalDevice_, device_,
        {stepParameterBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    stepParameterStaging_.resize(config_.maximumStepsPerBatch);

    const VkDeviceSize budget = latticeBudget();
    // Host-visible, unlike the bids beside it: a reset and a snapshot restore
    // both write the whole grid from the host, and the grid is derived state
    // that is cheaper to rebuild and upload than to reconstruct on the device.
    occupancy_.create(physicalDevice_, device_,
                      {budget, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    claims_.create(physicalDevice_, device_, {budget, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});
    structures_.create(physicalDevice_, device_,
                       {budget, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    // One row per agent is far more rows than there are worlds, and sizing it
    // by the population is what lets the world layout be renegotiated without
    // reallocating this buffer beside it.
    const VkDeviceSize buildOutcomeBytes = static_cast<VkDeviceSize>(agentCount) *
                                           lattice::kernel::LatticeBuildOutcomeCount *
                                           sizeof(std::uint32_t);
    buildOutcomes_.create(physicalDevice_, device_,
                          {buildOutcomeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    const VkDeviceSize trailHistoryBytes =
        static_cast<VkDeviceSize>(agentCount) * trailHistoryCapacity * sizeof(Int4);
    trailHistory_.create(
        physicalDevice_, device_,
        {trailHistoryBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT});

    settleLayout();

    createStorageLayout(device_, latticeStepBindings, stepDescriptorSetLayout_);
    createStorageLayout(device_, latticeResolveBindings, resolveDescriptorSetLayout_);
    createStorageLayout(device_, latticeClearBindings, clearDescriptorSetLayout_);
    createStorageLayout(device_, latticeTrailCaptureBindings, trailCaptureDescriptorSetLayout_);
    descriptorAllocator_.create(device_, {10, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 40}}});

    for (std::uint32_t readIndex = 0; readIndex < 2; ++readIndex) {
        const VkBuffer readBuffer =
            readIndex == 0 ? agentBuffers_.read().buffer() : agentBuffers_.write().buffer();
        const VkBuffer writeBuffer =
            readIndex == 0 ? agentBuffers_.write().buffer() : agentBuffers_.read().buffer();
        stepDescriptorSets_[readIndex] =
            descriptorAllocator_.allocate(stepDescriptorSetLayout_.get());
        DescriptorSetWriter{}
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, readBuffer, 0, agentBytes)
            .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, writeBuffer, 0, agentBytes)
            .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, genomeBuffer_.buffer(), 0,
                         genomeBuffer_.size())
            .writeBuffer(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, occupancy_.buffer(), 0,
                         occupancy_.size())
            .writeBuffer(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, claims_.buffer(), 0, claims_.size())
            .writeBuffer(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stepParameterBuffer_.buffer(), 0,
                         stepParameterBuffer_.size())
            .writeBuffer(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, structures_.buffer(), 0,
                         structures_.size())
            .writeBuffer(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buildOutcomes_.buffer(), 0,
                         buildOutcomes_.size())
            .update(device_, stepDescriptorSets_[readIndex]);

        // Reads and writes the record the step pass has just produced: this pass
        // adds to an agent rather than producing a new one.
        resolveDescriptorSets_[readIndex] =
            descriptorAllocator_.allocate(resolveDescriptorSetLayout_.get());
        DescriptorSetWriter{}
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, writeBuffer, 0, agentBytes)
            .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, occupancy_.buffer(), 0,
                         occupancy_.size())
            .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, claims_.buffer(), 0, claims_.size())
            .writeBuffer(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stepParameterBuffer_.buffer(), 0,
                         stepParameterBuffer_.size())
            .writeBuffer(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, structures_.buffer(), 0,
                         structures_.size())
            .update(device_, resolveDescriptorSets_[readIndex]);

        // `writeBuffer` is the resolved buffer for this read index. The agent
        // ping-pong swaps immediately before capture, making that same handle
        // the current read buffer and therefore the state actually displayed.
        trailCaptureDescriptorSets_[readIndex] =
            descriptorAllocator_.allocate(trailCaptureDescriptorSetLayout_.get());
        DescriptorSetWriter{}
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, writeBuffer, 0, agentBytes)
            .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, trailHistory_.buffer(), 0,
                         trailHistory_.size())
            .update(device_, trailCaptureDescriptorSets_[readIndex]);
    }
    clearDescriptorSet_ = descriptorAllocator_.allocate(clearDescriptorSetLayout_.get());
    DescriptorSetWriter{}
        .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, claims_.buffer(), 0, claims_.size())
        .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stepParameterBuffer_.buffer(), 0,
                     stepParameterBuffer_.size())
        .update(device_, clearDescriptorSet_);

    stepPipeline_ = ComputePipelineBuilder{physicalDevice_, device_}
                        .shader(VKEXP_SHADER_DIR "/lattice_step.comp.spv")
                        .addDescriptorSetLayout(stepDescriptorSetLayout_.get())
                        .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(std::uint32_t))
                        .build();
    resolvePipeline_ = ComputePipelineBuilder{physicalDevice_, device_}
                           .shader(VKEXP_SHADER_DIR "/lattice_resolve.comp.spv")
                           .addDescriptorSetLayout(resolveDescriptorSetLayout_.get())
                           .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(std::uint32_t))
                           .build();
    clearPipeline_ = ComputePipelineBuilder{physicalDevice_, device_}
                         .shader(VKEXP_SHADER_DIR "/lattice_clear.comp.spv")
                         .addDescriptorSetLayout(clearDescriptorSetLayout_.get())
                         .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(std::uint32_t))
                         .build();
    trailCapturePipeline_ =
        ComputePipelineBuilder{physicalDevice_, device_}
            .shader(VKEXP_SHADER_DIR "/trail_capture.comp.spv")
            .addDescriptorSetLayout(trailCaptureDescriptorSetLayout_.get())
            .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(TrailCaptureParameters))
            .build();

    state_.agents = {{agentBuffers_.read().buffer(), agentBuffers_.write().buffer()},
                     agentBytes,
                     agentBuffers_.readIndex(),
                     agentCount,
                     genomeCount,
                     config_.trialsPerGenome,
                     0};
    state_.trails = {trailHistory_.buffer(), trailHistory_.size(), trailHistoryCapacity};
    state_.structures = {structures_.buffer(), structures_.size()};
}

VkDeviceSize SimulationDriver::genomeBufferBytes() const {
    return sizeof(float) * evolution_.settings().weightCount * evolution_.population().size();
}

// Adopting a brain plan changes how long a genome is, and nothing else. So this
// remakes the one buffer whose size followed, and rewrites the one descriptor
// that names it -- rather than calling createStepResources again, which is what
// it used to do.
//
// The difference is not the wasted allocation. It is that createStepResources
// also freed the agent and lattice buffers, while other descriptor sets went on
// naming them: the next frame read memory the driver had already handed back,
// and the GPU hung. Resizing only what changed size makes those handles stable
// again, which is what every holder of them was written to assume.
//
// Callers wait for the device to go idle first -- SimulationModule does it
// around restart, and headless restores before its loop begins -- so the sets
// rewritten here are not in use.
void SimulationDriver::resizeGenomeBuffer() {
    // The population is rebuilt around the new length but never resized: the
    // count comes from EvolutionSettings, which adoptBrainPlan carries across.
    // Were that to stop holding, the agent buffers would be wrong too and this
    // narrow path would be the wrong one -- so it is checked rather than assumed.
    const auto genomeCount = static_cast<std::uint32_t>(evolution_.population().size());
    if (genomeCount != state_.agents.genomeCount) {
        throw std::logic_error("Brain plan changed the population size, which is fixed at launch");
    }
    genomeBuffer_.create(physicalDevice_, device_,
                         {genomeBufferBytes(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    for (const VkDescriptorSet set : stepDescriptorSets_) {
        DescriptorSetWriter{}
            .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, genomeBuffer_.buffer(), 0,
                         genomeBuffer_.size())
            .update(device_, set);
    }
}

void SimulationDriver::destroyResources() {
    state_.agents = {};
    state_.lattice = {};
    state_.structures = {};
    state_.trails = {};
    trailCapturePipeline_ = {};
    clearPipeline_ = {};
    resolvePipeline_ = {};
    stepPipeline_ = {};
    descriptorAllocator_.reset();
    trailCaptureDescriptorSetLayout_.reset();
    clearDescriptorSetLayout_.reset();
    resolveDescriptorSetLayout_.reset();
    stepDescriptorSetLayout_.reset();
    clearDescriptorSet_ = VK_NULL_HANDLE;
    trailCaptureDescriptorSets_ = {};
    resolveDescriptorSets_ = {};
    stepDescriptorSets_ = {};
    trailHistory_.reset();
    buildOutcomes_.reset();
    structures_.reset();
    claims_.reset();
    occupancy_.reset();
    stepParameterBuffer_.reset();
    genomeBuffer_.reset();
    agentBuffers_.reset();
    device_ = VK_NULL_HANDLE;
    physicalDevice_ = VK_NULL_HANDLE;
}

void SimulationDriver::uploadPopulation(const bool preserveStructures) {
    std::vector<float> flattened;
    flattened.reserve(evolution_.population().size() * evolution_.settings().weightCount);
    for (const Genome& genome : evolution_.population()) {
        flattened.insert(flattened.end(), genome.weights.begin(), genome.weights.end());
    }
    genomeBuffer_.write(flattened.data(), flattened.size() * sizeof(float));
    agentBuffers_.read().write(agents_.data(), agents_.size() * sizeof(AgentState));
    agentBuffers_.write().write(agents_.data(), agents_.size() * sizeof(AgentState));

    // Derived from the agents rather than carried beside them, which is what
    // lets a snapshot leave it out: two agents can never be in one cell, so the
    // grid is a function of where everybody stands.
    occupancyStaging_.assign(static_cast<std::size_t>(state_.lattice.cellsPerWorld) *
                                 std::max(state_.worlds.worldCount, 1U),
                             lattice::kernel::LatticeNoOccupant);
    lattice::buildOccupancy(agents_, state_.settings, populationLayout(), occupancyStaging_);
    occupancy_.write(occupancyStaging_.data(), occupancyStaging_.size() * sizeof(std::int32_t));
    if (!preserveStructures) {
        structureStaging_.assign(occupancyStaging_.size(), lattice::kernel::LatticeNoStructure);
    }
    if (!structureStaging_.empty()) {
        structures_.write(structureStaging_.data(),
                          structureStaging_.size() * sizeof(std::int32_t));
    }
    // A fresh count for a fresh generation: what the previous population was
    // refused for says nothing about this one.
    buildOutcomeStaging_.assign(static_cast<std::size_t>(state_.worlds.worldCount) *
                                    lattice::kernel::LatticeBuildOutcomeCount,
                                0U);
    buildOutcomes_.write(buildOutcomeStaging_.data(),
                         buildOutcomeStaging_.size() * sizeof(std::uint32_t));
    hostUploadPending_ = true;
    // Histories are derived pictures, not snapshot state. A fresh generation or
    // a restored population starts with no breadcrumbs and fills them from its
    // next resolved step onward.
    trailClearPending_ = true;
    state_.trails.recordedTicks = 0;
    state_.trails.newest = 0;
}

void SimulationDriver::resetGeneration() {
    // The beacon seed is a property of the generation, so a genome is scored
    // against a lattice that moves under it rather than one it can memorise --
    // and publishing it here keeps the simulation and any view of it on the same
    // world.
    state_.settings.beaconSeed = static_cast<std::uint32_t>(evolution_.generation()) * 2654435761U +
                                 evolution_.settings().seed;
    agents_ = lattice::makeInitialAgents(state_.settings, populationLayout());
    uploadPopulation();
    state_.statistics.step = 0;
    state_.statistics.generation = evolution_.generation();
    state_.agents.generation = evolution_.generation();
    state_.agents.currentIndex = agentBuffers_.readIndex();
}

void SimulationDriver::restart() {
    // A restart is where a new brain plan takes hold: it changes how long a
    // genome is, so the population and the buffer holding it are both remade --
    // that buffer and no other, because nothing else here is sized by the plan.
    const std::size_t previous = evolution_.settings().weightCount;
    adoptBrainPlan();
    evolution_.reset();
    if (evolution_.settings().weightCount != previous && device_ != VK_NULL_HANDLE) {
        resizeGenomeBuffer();
    }
    state_.statistics = {};
    state_.history.bestFitness.clear();
    state_.history.medianFitness.clear();
    state_.history.meanFitness.clear();
    state_.history.arrivalRatio.clear();
    settleLayout();
    resetGeneration();
}

void SimulationDriver::loadPopulation(const std::span<const Genome> genomes,
                                      const std::uint64_t generation) {
    evolution_.setPopulation(genomes, generation);
    state_.statistics.generation = generation;
    resetGeneration();
}

RunSnapshot SimulationDriver::snapshot() {
    agentBuffers_.read().read(agents_.data(), agents_.size() * sizeof(AgentState));
    RunSnapshot result;
    result.settings = state_.settings;
    result.genomes.assign(evolution_.population().begin(), evolution_.population().end());
    result.agents = agents_;
    const std::size_t activeCells =
        static_cast<std::size_t>(state_.lattice.cellsPerWorld) * state_.worlds.worldCount;
    result.structures.resize(activeCells);
    structures_.read(result.structures.data(), result.structures.size() * sizeof(std::int32_t));
    result.generation = evolution_.generation();
    result.step = state_.statistics.step;
    result.stepsPerGeneration = state_.controls.stepsPerGeneration;
    result.requestedAgentsPerWorld = state_.worlds.requestedAgentsPerWorld;
    result.trialsPerGenome = config_.trialsPerGenome;
    result.seed = evolution_.settings().seed;
    return result;
}

void SimulationDriver::restoreSnapshot(const RunSnapshot& snapshot) {
    // Population size and trial count are buffer dimensions fixed when the
    // device resources were created, not settings. A snapshot that disagrees
    // cannot be resumed into this driver, and saying so beats writing past the
    // end of the agent buffer.
    if (snapshot.genomes.size() != evolution_.population().size()) {
        throw RunSnapshotError("Snapshot holds " + std::to_string(snapshot.genomes.size()) +
                               " genomes but this run was launched with " +
                               std::to_string(evolution_.population().size()));
    }
    if (snapshot.trialsPerGenome != config_.trialsPerGenome) {
        throw RunSnapshotError("Snapshot ran " + std::to_string(snapshot.trialsPerGenome) +
                               " trials per genome but this run was launched with " +
                               std::to_string(config_.trialsPerGenome));
    }
    if (snapshot.agents.size() != evolution_.population().size() * config_.trialsPerGenome) {
        throw RunSnapshotError("Snapshot agent count does not match its own genome count");
    }

    // Order matters. The world layout depends on the group size, and the lattice
    // dimensions depend on the layout, so the settings go in before anything is
    // sized. The agents go in last, after uploadPopulation would otherwise have
    // replaced them with fresh ones.
    state_.settings = snapshot.settings;
    state_.controls.stepsPerGeneration = snapshot.stepsPerGeneration;
    state_.worlds.requestedAgentsPerWorld = snapshot.requestedAgentsPerWorld;

    // The settings that just went in decide how long a genome is, so the run
    // adopts that before the population is handed over.
    const std::size_t previousWeights = evolution_.settings().weightCount;
    adoptBrainPlan();
    if (evolution_.settings().weightCount != previousWeights && device_ != VK_NULL_HANDLE) {
        resizeGenomeBuffer();
    }
    evolution_.setPopulation(snapshot.genomes, snapshot.generation);
    updateWorldLayout();
    const std::uint32_t requestedCells = latticeCellsPerWorld(snapshot.settings);
    refreshLattice();
    // A lattice the budget will not hold is refused rather than silently
    // shrunk: every agent's recorded cell is an index into the box that was
    // saved, so resuming into a smaller one would put the whole population
    // somewhere else and report it as the same run.
    if (state_.lattice.cellsPerWorld != requestedCells) {
        throw RunSnapshotError("Snapshot needs a lattice this run cannot allocate at its "
                               "population size");
    }

    agents_ = snapshot.agents;
    const std::size_t activeCells =
        static_cast<std::size_t>(state_.lattice.cellsPerWorld) * state_.worlds.worldCount;
    if (snapshot.structures.size() != activeCells) {
        throw RunSnapshotError("Snapshot construction field does not match its lattice layout");
    }
    structureStaging_ = snapshot.structures;
    state_.statistics.step = std::min(snapshot.step, snapshot.stepsPerGeneration);
    uploadPopulation(true);
    state_.statistics.generation = snapshot.generation;
    state_.agents.generation = snapshot.generation;
    state_.agents.currentIndex = agentBuffers_.readIndex();
}

bool SimulationDriver::generationComplete() const {
    return state_.statistics.step >= state_.controls.stepsPerGeneration;
}

// The block field and why the builders were refused, read back together
// because they answer one question: what happened to all that effort. Neither
// reaches fitness in any world.
void SimulationDriver::readStructureDiagnosis() {
    const std::size_t activeCells =
        static_cast<std::size_t>(state_.lattice.cellsPerWorld) * state_.worlds.worldCount;
    structureStaging_.resize(activeCells);
    structures_.read(structureStaging_.data(), structureStaging_.size() * sizeof(std::int32_t));

    buildOutcomeStaging_.assign(static_cast<std::size_t>(state_.worlds.worldCount) *
                                    lattice::kernel::LatticeBuildOutcomeCount,
                                0U);
    buildOutcomes_.read(buildOutcomeStaging_.data(),
                        buildOutcomeStaging_.size() * sizeof(std::uint32_t));
    state_.statistics.buildOutcomes = buildOutcomeStaging_;

    state_.statistics.worldShapes.assign(state_.worlds.worldCount, StructureShape{});
    state_.statistics.bestWorld = 0;
    for (std::uint32_t world = 0; world < state_.worlds.worldCount; ++world) {
        const std::size_t begin = static_cast<std::size_t>(world) * state_.lattice.cellsPerWorld;
        state_.statistics.worldShapes[world] = measureStructureShape(
            std::span<const std::int32_t>{structureStaging_}.subspan(
                begin, state_.lattice.cellsPerWorld),
            state_.settings);
    }
}

GenerationSummary SimulationDriver::finishGeneration() {
    agentBuffers_.read().read(agents_.data(), agents_.size() * sizeof(AgentState));
    std::vector<float> fitness(evolution_.population().size());
    std::size_t arrived = 0;
    float objectiveRatio = 0.0F;
    // Read back before scoring, and for every world that builds rather than for
    // the one that is scored on building. The harvest world is where what was
    // built matters most and is counted least: the fitness below does not look
    // at a single block, so the only way to see whether a group found a cheaper
    // route or a bigger pile is to measure the shape.
    if (worldBuilds(state_.settings.worldMode)) {
        readStructureDiagnosis();
    } else {
        state_.statistics.worldShapes.clear();
        state_.statistics.buildOutcomes.clear();
        state_.statistics.bestWorld = 0;
    }
    if (state_.settings.worldMode == WorldMode::Harvest) {
        // Deliveries, and the best anyone got to the resource. Blocks score
        // nothing at all here: a block is time spent, and whether it was spent
        // well is exactly the question the world asks. The nearness term is
        // shaping and not the objective -- without it no early population has a
        // gradient, because nobody reaches height four by accident.
        std::vector<float> deliveries(state_.worlds.worldCount, 0.0F);
        std::vector<float> reach(state_.worlds.worldCount, 0.0F);
        for (std::size_t agent = 0; agent < agents_.size(); ++agent) {
            const std::uint32_t world =
                logicalWorldForAgent(static_cast<std::uint32_t>(agent),
                                     state_.worlds.agentsPerWorld, config_.trialsPerGenome);
            deliveries[world] += agents_[agent].metrics.w;
            reach[world] = std::max(reach[world], agents_[agent].metrics.x);
        }
        for (std::size_t genome = 0; genome < fitness.size(); ++genome) {
            const std::uint32_t group =
                static_cast<std::uint32_t>(genome) / state_.worlds.agentsPerWorld;
            for (std::uint32_t trial = 0; trial < config_.trialsPerGenome; ++trial) {
                const std::uint32_t world = group * config_.trialsPerGenome + trial;
                fitness[genome] +=
                    state_.settings.fitness.objectiveBonus * deliveries[world] +
                    state_.settings.fitness.trackingReward * reach[world];
            }
            fitness[genome] /= static_cast<float>(config_.trialsPerGenome);
        }
        // Which world the viewer is offered: the one that delivered most, and
        // among worlds that delivered nothing the one that got nearest. Early on
        // every world delivers nothing, and "world 1" would then be an answer
        // about the population layout rather than about the run.
        for (std::uint32_t world = 0; world < state_.worlds.worldCount; ++world) {
            const std::uint32_t best = state_.statistics.bestWorld;
            const bool better = deliveries[world] > deliveries[best] ||
                                (deliveries[world] == deliveries[best] && reach[world] > reach[best]);
            if (better) {
                state_.statistics.bestWorld = world;
            }
        }
        arrived = static_cast<std::size_t>(
            std::count_if(agents_.begin(), agents_.end(),
                          [](const AgentState& agent) { return agent.metrics.w > 0.0F; }));
        objectiveRatio =
            static_cast<float>(arrived) / static_cast<float>(std::max<std::size_t>(agents_.size(), 1));
    } else if (state_.settings.worldMode == WorldMode::Construction) {
        std::vector<float> weightedBlocks(state_.worlds.worldCount, 0.0F);
        std::vector<float> perimeterTicks(state_.worlds.worldCount, 0.0F);
        const std::uint32_t width = state_.settings.latticeWidth;
        const std::uint32_t latticeHeight = state_.settings.latticeHeight;
        for (std::size_t absolute = 0; absolute < structureStaging_.size(); ++absolute) {
            if (structureStaging_[absolute] == lattice::kernel::LatticeNoStructure) {
                continue;
            }
            const std::uint32_t world =
                static_cast<std::uint32_t>(absolute / state_.lattice.cellsPerWorld);
            const std::uint32_t local =
                static_cast<std::uint32_t>(absolute % state_.lattice.cellsPerWorld);
            const std::uint32_t y = (local / width) % latticeHeight;
            // Every block matters, while upper courses matter more. Dividing by
            // block count would make a useful foundation lower the mean and
            // restore the one-thin-column loophole of tallest-only fitness.
            weightedBlocks[world] += static_cast<float>(y + 1U);
        }
        for (std::size_t agent = 0; agent < agents_.size(); ++agent) {
            const std::uint32_t world =
                logicalWorldForAgent(static_cast<std::uint32_t>(agent),
                                     state_.worlds.agentsPerWorld, config_.trialsPerGenome);
            perimeterTicks[world] += agents_[agent].metrics.w;
        }
        for (std::uint32_t world = 0; world < state_.worlds.worldCount; ++world) {
            if (weightedBlocks[world] > weightedBlocks[state_.statistics.bestWorld]) {
                state_.statistics.bestWorld = world;
            }
        }

        for (std::size_t genome = 0; genome < fitness.size(); ++genome) {
            const std::uint32_t group =
                static_cast<std::uint32_t>(genome) / state_.worlds.agentsPerWorld;
            for (std::uint32_t trial = 0; trial < config_.trialsPerGenome; ++trial) {
                const std::uint32_t world = group * config_.trialsPerGenome + trial;
                fitness[genome] += weightedBlocks[world] -
                                   state_.settings.fitness.boundaryPenalty * perimeterTicks[world];
            }
            fitness[genome] /= static_cast<float>(config_.trialsPerGenome);
        }
        const float totalWeighted =
            std::accumulate(weightedBlocks.begin(), weightedBlocks.end(), 0.0F);
        const float maximumPerWorld =
            static_cast<float>(state_.settings.latticeWidth * state_.settings.latticeDepth) *
            (static_cast<float>(latticeHeight) * static_cast<float>(latticeHeight + 1U) * 0.5F);
        objectiveRatio =
            totalWeighted / (static_cast<float>(std::max<std::size_t>(weightedBlocks.size(), 1)) *
                             std::max(maximumPerWorld, 1.0F));
    } else {
        state_.statistics.worldShapes.clear();
        state_.statistics.buildOutcomes.clear();
        state_.statistics.bestWorld = 0;
        for (std::size_t genome = 0; genome < fitness.size(); ++genome) {
            for (std::size_t trial = 0; trial < config_.trialsPerGenome; ++trial) {
                const AgentState& agent = agents_[genome * config_.trialsPerGenome + trial];
                fitness[genome] += agentFitness(agent, state_.settings.fitness);
                arrived += agent.metrics.y > 0.0F ? 1 : 0;
            }
            fitness[genome] /= static_cast<float>(config_.trialsPerGenome);
        }
        objectiveRatio = static_cast<float>(arrived) /
                         static_cast<float>(std::max<std::size_t>(agents_.size(), 1));
    }
    // Selection may see a different number than the run reports. Sharing blends
    // each genome's score with its world's average, which is the point -- it
    // makes helping a neighbour pay -- but it also compresses the spread, so a
    // shared run and an unshared one could not be compared on their own headline
    // numbers. What is published stays individual, and is therefore the same
    // quantity at every sharing level; what selects is the shared vector.
    const std::vector<float> selectionFitness = shareFitnessWithinGroups(
        fitness, state_.worlds.agentsPerWorld, state_.settings.fitness.groupSharing);
    // Replay scores and reports a generation exactly as training does -- that is
    // how loaded weights get judged -- and then selects nothing. The population
    // is left untouched, so resetGeneration below respawns the same genomes and
    // the generation counter does not move. Since the beacon seed is derived
    // from the generation number, the next run is the same run again rather than
    // a similar one, which is what makes a behaviour watchable more than once.
    GenerationSummary summary{};
    if (state_.controls.replay) {
        summary.generation = evolution_.generation();
        summary.championIndex = static_cast<std::size_t>(
            std::distance(fitness.begin(), std::max_element(fitness.begin(), fitness.end())));
    } else {
        summary = evolution_.evolve(selectionFitness);
    }
    summary.bestFitness = *std::max_element(fitness.begin(), fitness.end());
    summary.meanFitness =
        std::accumulate(fitness.begin(), fitness.end(), 0.0F) / static_cast<float>(fitness.size());
    {
        std::vector<float> sorted = fitness;
        std::sort(sorted.begin(), sorted.end());
        summary.medianFitness = sorted[sorted.size() / 2];
    }
    state_.statistics.bestFitness = summary.bestFitness;
    state_.statistics.meanFitness = summary.meanFitness;
    state_.statistics.medianFitness = summary.medianFitness;
    state_.statistics.arrivalRatio = objectiveRatio;
    // Counted here rather than taken from the history length, which stops
    // growing at maximumSamples, and rather than from the generation number,
    // which stands still under replay. A sweep stage restarts the run, and
    // restart() zeroes the statistics, so a stage counts its own generations.
    ++state_.statistics.evaluatedGenerations;
    const auto appendHistory = [&](std::vector<float>& history, const float value) {
        history.push_back(value);
        if (history.size() > state_.history.maximumSamples) {
            history.erase(history.begin(),
                          history.begin() + static_cast<std::ptrdiff_t>(
                                                history.size() - state_.history.maximumSamples));
        }
    };
    appendHistory(state_.history.bestFitness, summary.bestFitness);
    appendHistory(state_.history.medianFitness, summary.medianFitness);
    appendHistory(state_.history.meanFitness, summary.meanFitness);
    appendHistory(state_.history.arrivalRatio, state_.statistics.arrivalRatio);

    // A stage boundary is a generation boundary, where the population was going
    // to be respawned anyway, so a sweep costs nothing but the restart it asks
    // for. Replay never advances a sweep: there is nothing to compare between
    // settings when no setting is selecting anything.
    if (state_.sweep.running && !state_.controls.replay) {
        if (recordSweepGeneration(state_.sweep, summary.bestFitness, summary.medianFitness,
                                  state_.statistics.arrivalRatio)) {
            state_.settings.fitness.groupSharing = sweepValue(state_.sweep);
            // restart() and not resetGeneration(): a stage has to begin from the
            // seeded initial population, or it would measure its setting applied
            // to whatever the previous setting had already evolved.
            restart();
            return summary;
        }
    }
    resetGeneration();
    return summary;
}

void SimulationDriver::beginSweep() {
    startSweep(state_.sweep);
    if (!state_.sweep.running) {
        return;
    }
    state_.controls.replay = false;
    state_.settings.fitness.groupSharing = sweepValue(state_.sweep);
    restart();
}

void SimulationDriver::endSweep() { stopSweep(state_.sweep); }

// The budget the full-cell grids must each fit, capped against what the device has.
std::uint64_t SimulationDriver::latticeBudget() const {
    std::uint64_t deviceLocalBytes = 0;
    if (physicalDevice_ != VK_NULL_HANDLE) {
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memory);
        for (std::uint32_t index = 0; index < memory.memoryHeapCount; ++index) {
            if ((memory.memoryHeaps[index].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                deviceLocalBytes = std::max(
                    deviceLocalBytes, static_cast<std::uint64_t>(memory.memoryHeaps[index].size));
            }
        }
    }
    if (deviceLocalBytes == 0) {
        return latticeFieldByteBudget;
    }
    return std::min(latticeFieldByteBudget, deviceLocalBytes / latticeFieldHeapFraction);
}

std::uint64_t SimulationDriver::latticeBytes(const SimulationStep& settings) const {
    return static_cast<std::uint64_t>(latticeCellsPerWorld(settings)) * sizeof(std::int32_t) *
           std::max(state_.worlds.worldCount, 1U);
}

void SimulationDriver::refreshLattice() {
    SimulationStep& settings = state_.settings;
    settings.latticeWidth = clampLatticeExtent(settings.latticeWidth);
    settings.latticeHeight = clampLatticeExtent(settings.latticeHeight);
    settings.latticeDepth = clampLatticeExtent(settings.latticeDepth);

    // Shrink until the grids fit, halving whichever extent is currently the
    // longest so the box stays roughly the shape it was asked for. The
    // allocation is fixed, so the lattice is what gives way -- and the clamped
    // extents are written back into the settings, so the sliders show the box
    // that is actually running rather than the one that was requested.
    const std::uint64_t budget = occupancy_.size() != 0 ? occupancy_.size() : latticeBudget();
    while (latticeBytes(settings) > budget) {
        std::uint32_t* longest = &settings.latticeWidth;
        if (settings.latticeHeight > *longest) {
            longest = &settings.latticeHeight;
        }
        if (settings.latticeDepth > *longest) {
            longest = &settings.latticeDepth;
        }
        if (*longest <= latticeMinimumExtent) {
            throw std::runtime_error("The lattice does not fit even at its smallest extent");
        }
        *longest = std::max(*longest / 2, latticeMinimumExtent);
    }

    state_.lattice = {occupancy_.buffer(), occupancy_.size(), latticeCellsPerWorld(settings)};
}

void SimulationDriver::settleLayout() {
    // Four passes is generous: the first shrink is the one that can move the
    // group size, and the loop exists so that a lattice shrunk twice cannot
    // leave more agents in a world than the box can stand them in. The bound is
    // here so that a future rule which does not converge fails as a wrong number
    // rather than as a hang.
    for (int pass = 0; pass < 4; ++pass) {
        const std::uint32_t before = state_.worlds.agentsPerWorld;
        updateWorldLayout();
        refreshLattice();
        if (state_.worlds.agentsPerWorld == before) {
            return;
        }
    }
    updateWorldLayout();
}

void SimulationDriver::updateWorldLayout() {
    const auto genomeCount = static_cast<std::uint32_t>(evolution_.population().size());
    const std::uint32_t requested = std::min(state_.worlds.requestedAgentsPerWorld,
                                             latticeSpawnCapacity(state_.settings));
    state_.worlds.agentsPerWorld = clampAgentsPerWorld(genomeCount, requested);
    state_.worlds.requestedAgentsPerWorld = state_.worlds.agentsPerWorld;
    state_.worlds.groupCount = worldGroupCount(genomeCount, state_.worlds.agentsPerWorld);
    state_.worlds.worldCount =
        logicalWorldCount(genomeCount, state_.worlds.agentsPerWorld, config_.trialsPerGenome);
    if (state_.worlds.worldCount == 0) {
        state_.worlds.selectedWorld = 0;
    } else {
        state_.worlds.selectedWorld =
            std::min(state_.worlds.selectedWorld, state_.worlds.worldCount - 1);
    }
}

GpuStepParameters SimulationDriver::stepParameters() const {
    // Every field named. Positionally this was eight initialisers for a nine
    // field aggregate once, and the ninth kept its default, so a whole pass ran
    // against a world count of one. That is the third time a field has been
    // silently dropped on the way to this struct; see the note on
    // packStepParameters.
    return packStepParameters(state_.settings,
                              StepParameterLayout{.agentCount = state_.agents.agentCount,
                                                  .trialsPerGenome = config_.trialsPerGenome,
                                                  .agentsPerWorld = state_.worlds.agentsPerWorld,
                                                  .worldCount = state_.worlds.worldCount});
}

std::uint32_t SimulationDriver::recordSteps(const VkCommandBuffer commands,
                                            const std::uint32_t maximumSteps) {
    const std::uint32_t remaining =
        state_.controls.stepsPerGeneration -
        std::min(state_.statistics.step, state_.controls.stepsPerGeneration);
    const std::uint32_t stepCount =
        std::min({maximumSteps, remaining, config_.maximumStepsPerBatch});
    if (stepCount == 0) {
        return 0;
    }

    // Nothing in the block varies with the step any more. The 2D build moved a
    // beacon and faded a trail, so every step got its own copy; a lattice's
    // settings are constant across a generation, and the buffer is still indexed
    // by step only so that a future per-step quantity has somewhere to go.
    const GpuStepParameters parameters = stepParameters();
    for (std::uint32_t step = 0; step < stepCount; ++step) {
        stepParameterStaging_[step] = parameters;
    }
    stepParameterBuffer_.write(stepParameterStaging_.data(), sizeof(GpuStepParameters) * stepCount);
    cmdBufferBarrier(commands, stepParameterBuffer_.buffer(), VK_PIPELINE_STAGE_2_HOST_BIT,
                     VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    if (hostUploadPending_) {
        for (const VkBuffer buffer : state_.agents.buffers) {
            cmdBufferBarrier(commands, buffer, VK_PIPELINE_STAGE_2_HOST_BIT,
                             VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
        cmdBufferBarrier(commands, genomeBuffer_.buffer(), VK_PIPELINE_STAGE_2_HOST_BIT,
                         VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        cmdBufferBarrier(commands, occupancy_.buffer(), VK_PIPELINE_STAGE_2_HOST_BIT,
                         VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        cmdBufferBarrier(commands, structures_.buffer(), VK_PIPELINE_STAGE_2_HOST_BIT,
                         VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        cmdBufferBarrier(commands, buildOutcomes_.buffer(), VK_PIPELINE_STAGE_2_HOST_BIT,
                         VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        hostUploadPending_ = false;
    }
    if (trailClearPending_) {
        vkCmdFillBuffer(commands, trailHistory_.buffer(), 0, trailHistory_.size(), 0);
        cmdBufferBarrier(commands, trailHistory_.buffer(), VK_PIPELINE_STAGE_2_CLEAR_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        trailClearPending_ = false;
    }

    const std::uint32_t cellCount = state_.lattice.cellsPerWorld * state_.worlds.worldCount;
    const std::array<VkDeviceSize, 2> clearRanges{claims_.size(), stepParameterBuffer_.size()};
    const DispatchSize clearGroups = checkedDispatchSize(
        physicalDevice_, {{cellCount, 1, 1}, {256, 1, 1}, sizeof(std::uint32_t), clearRanges});
    const std::array<VkDeviceSize, 8> stepRanges{agentBuffers_.read().size(),
                                                 agentBuffers_.write().size(),
                                                 genomeBuffer_.size(),
                                                 occupancy_.size(),
                                                 claims_.size(),
                                                 stepParameterBuffer_.size(),
                                                 structures_.size(),
                                                 buildOutcomes_.size()};
    const DispatchSize stepGroups = checkedDispatchSize(
        physicalDevice_,
        {{state_.agents.agentCount, 1, 1}, {64, 1, 1}, sizeof(std::uint32_t), stepRanges});
    const std::array<VkDeviceSize, 5> resolveRanges{agentBuffers_.write().size(),
                                                    occupancy_.size(), claims_.size(),
                                                    stepParameterBuffer_.size(),
                                                    structures_.size()};
    const DispatchSize resolveGroups = checkedDispatchSize(
        physicalDevice_,
        {{state_.agents.agentCount, 1, 1}, {64, 1, 1}, sizeof(std::uint32_t), resolveRanges});
    const std::array<VkDeviceSize, 2> trailCaptureRanges{agentBuffers_.read().size(),
                                                         trailHistory_.size()};
    const DispatchSize trailCaptureGroups =
        checkedDispatchSize(physicalDevice_, {{state_.agents.agentCount, 1, 1},
                                              {64, 1, 1},
                                              sizeof(TrailCaptureParameters),
                                              trailCaptureRanges});

    for (std::uint32_t step = 0; step < stepCount; ++step) {
        const std::uint32_t readIndex = agentBuffers_.readIndex();

        // Empty every bid before anybody bids. A barrier and not a loop order:
        // the clear and the bids are different dispatches precisely because a
        // cell has to be empty for every agent, not just for the ones that
        // happened to run after it was cleared.
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, clearPipeline_.pipeline());
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, clearPipeline_.layout(),
                                0, 1, &clearDescriptorSet_, 0, nullptr);
        vkCmdPushConstants(commands, clearPipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(step), &step);
        vkCmdDispatch(commands, clearGroups.x, 1, 1);
        cmdComputeWriteToComputeRead(commands, claims_.buffer());

        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, stepPipeline_.pipeline());
        const VkDescriptorSet stepSet = stepDescriptorSets_[readIndex];
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, stepPipeline_.layout(), 0,
                                1, &stepSet, 0, nullptr);
        vkCmdPushConstants(commands, stepPipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(step), &step);
        vkCmdDispatch(commands, stepGroups.x, 1, 1);
        // Every bid has to have landed before any of them is read back, and the
        // records the step wrote have to be visible to the pass that adds to
        // them.
        cmdComputeWriteToComputeRead(commands, claims_.buffer());
        cmdBufferBarrier(
            commands, agentBuffers_.write().buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, resolvePipeline_.pipeline());
        const VkDescriptorSet resolveSet = resolveDescriptorSets_[readIndex];
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, resolvePipeline_.layout(),
                                0, 1, &resolveSet, 0, nullptr);
        vkCmdPushConstants(commands, resolvePipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(step), &step);
        vkCmdDispatch(commands, resolveGroups.x, 1, 1);
        cmdComputeWriteToComputeRead(commands, agentBuffers_.write().buffer());
        cmdComputeWriteToComputeRead(commands, occupancy_.buffer());
        cmdComputeWriteToComputeRead(commands, structures_.buffer());

        agentBuffers_.swap();
        state_.agents.currentIndex = agentBuffers_.readIndex();

        const TrailCaptureParameters trailCapture{
            state_.agents.agentCount,
            (state_.statistics.step + step) % trailHistoryCapacity,
            trailHistoryCapacity,
            0,
        };
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          trailCapturePipeline_.pipeline());
        const VkDescriptorSet trailCaptureSet = trailCaptureDescriptorSets_[readIndex];
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                trailCapturePipeline_.layout(), 0, 1, &trailCaptureSet, 0, nullptr);
        vkCmdPushConstants(commands, trailCapturePipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(trailCapture), &trailCapture);
        vkCmdDispatch(commands, trailCaptureGroups.x, 1, 1);
        state_.trails.newest = trailCapture.cursor;
        state_.trails.recordedTicks =
            std::min(state_.trails.recordedTicks + 1U, state_.trails.capacity);
        if (step + 1U < stepCount) {
            // A batch may be configured longer than the ring. Serialize captures
            // so a wrapped slot always contains the newest step, independent of
            // how workgroups from consecutive dispatches overlap.
            cmdBufferBarrier(
                commands, trailHistory_.buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
    }

    cmdBufferBarrier(commands, agentBuffers_.read().buffer(),
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_HOST_READ_BIT);
    cmdBufferBarrier(commands, occupancy_.buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_HOST_READ_BIT);
    cmdBufferBarrier(commands, structures_.buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_HOST_READ_BIT);
    cmdBufferBarrier(commands, trailHistory_.buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    state_.statistics.step += stepCount;
    return stepCount;
}

GenomeArchiveMetadata genomeArchiveMetadata(const SimulationState& state,
                                            const SimulationDriver& driver,
                                            const neuro::BrainShape& brain) {
    return {driver.evolution().generation(),
            state.settings.beaconSeed,
            driver.evolution().settings().seed,
            state.statistics.bestFitness,
            state.statistics.meanFitness,
            static_cast<std::uint32_t>(brain.inputCount),
            static_cast<std::uint32_t>(brain.hiddenTotal()),
            static_cast<std::uint32_t>(brain.outputCount),
            brain.packedLayers()};
}

} // namespace vkexp
