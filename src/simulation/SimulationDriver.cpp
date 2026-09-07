#include "vkexp/simulation/SimulationDriver.hpp"

#include "vkexp/simulation/CpuSimulation.hpp"
#include "vkexp/simulation/StepParameters.hpp"
#include "vkexp/worlds/WorldScenario.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>

namespace vkexp {
namespace {

constexpr VkMemoryPropertyFlags hostMemory =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

struct alignas(16) GridBuildParameters {
    float worldRadius{};
    float cellSize{};
    std::uint32_t agentCount{};
    std::uint32_t worldCount{};
    std::uint32_t gridWidth{};
    std::uint32_t gridCellsPerWorld{};
    std::uint32_t reserved0{};
    std::uint32_t reserved1{};
};
static_assert(sizeof(GridBuildParameters) == 32);

struct TrailDecayParameters {
    std::uint32_t valueCount;
    float survival;
};

static_assert(sizeof(TrailDecayParameters) == 8);

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
        throw std::runtime_error("Unable to create agent compute descriptor layout");
    }
}

} // namespace

SimulationDriver::SimulationDriver(SimulationState& state, EvolutionSettings evolution,
                                   SimulationDriverConfig config)
    : state_(state), evolution_(evolution), config_(config) {
    if (config_.trialsPerGenome == 0 || config_.maximumStepsPerBatch == 0 ||
        config_.gridCellSize <= 0.0F) {
        throw std::invalid_argument("Invalid simulation driver configuration");
    }
    state_.evolution = evolution_.settings();
}

void SimulationDriver::createResources(const VkPhysicalDevice physicalDevice,
                                       const VkDevice device) {
    physicalDevice_ = physicalDevice;
    device_ = device;
    createStepResources();
    resetGeneration();
}

void SimulationDriver::createStepResources() {
    const auto genomeCount = static_cast<std::uint32_t>(evolution_.population().size());
    const std::uint32_t agentCount = genomeCount * config_.trialsPerGenome;
    const VkDeviceSize agentBytes = sizeof(AgentState) * agentCount;
    const VkDeviceSize genomeBytes = sizeof(float) * neuro::Topology::weightCount * genomeCount;
    const VkDeviceSize stepParameterBytes =
        sizeof(GpuStepParameters) * config_.maximumStepsPerBatch;
    agentBuffers_.create(physicalDevice_, device_,
                         {agentBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    genomeBuffer_.create(physicalDevice_, device_,
                         {genomeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    stepParameterBuffer_.create(
        physicalDevice_, device_,
        {stepParameterBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    stepParameterStaging_.resize(config_.maximumStepsPerBatch);

    // The field is allocated once, at the budget, and never reallocated. Every
    // resize so far has been a lifetime bug rather than a capacity one -- a
    // descriptor somewhere kept pointing at the buffer that had just been freed --
    // and the numbers say resizing buys nothing: the coarsest grid on the largest
    // arena across every world the population can be split into is 150 MB against
    // a 256 MiB budget. A fixed allocation makes the whole class of bug
    // unreachable instead of patching each holder of a handle.
    trailField_.create(physicalDevice_, device_,
                       {trailFieldBudget(),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT});
    updateWorldLayout();
    updateGridDimensions();
    ensureGridCapacity();
    updateTrailDimensions();
    const VkDeviceSize gridLinkBytes = sizeof(std::int32_t) * agentCount;
    gridNext_.create(physicalDevice_, device_, {gridLinkBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});

    createStorageLayout(device_, 7, stepDescriptorSetLayout_);
    createStorageLayout(device_, 1, gridClearDescriptorSetLayout_);
    createStorageLayout(device_, 3, gridBuildDescriptorSetLayout_);
    createStorageLayout(device_, 1, trailDecayDescriptorSetLayout_);
    createStorageLayout(device_, 3, trailDepositDescriptorSetLayout_);
    descriptorAllocator_.create(device_, {8, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 29}}});

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
            .writeBuffer(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads_.buffer(), 0,
                         gridHeads_.size())
            .writeBuffer(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridNext_.buffer(), 0,
                         gridNext_.size())
            .writeBuffer(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stepParameterBuffer_.buffer(), 0,
                         stepParameterBuffer_.size())
            .writeBuffer(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, trailField_.buffer(), 0,
                         trailField_.size())
            .update(device_, stepDescriptorSets_[readIndex]);

        trailDepositDescriptorSets_[readIndex] =
            descriptorAllocator_.allocate(trailDepositDescriptorSetLayout_.get());
        DescriptorSetWriter{}
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, readBuffer, 0, agentBytes)
            .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, trailField_.buffer(), 0,
                         trailField_.size())
            .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stepParameterBuffer_.buffer(), 0,
                         stepParameterBuffer_.size())
            .update(device_, trailDepositDescriptorSets_[readIndex]);

        gridBuildDescriptorSets_[readIndex] =
            descriptorAllocator_.allocate(gridBuildDescriptorSetLayout_.get());
        DescriptorSetWriter{}
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, readBuffer, 0, agentBytes)
            .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads_.buffer(), 0,
                         gridHeads_.size())
            .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridNext_.buffer(), 0,
                         gridNext_.size())
            .update(device_, gridBuildDescriptorSets_[readIndex]);
    }
    trailDecayDescriptorSet_ = descriptorAllocator_.allocate(trailDecayDescriptorSetLayout_.get());
    DescriptorSetWriter{}
        .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, trailField_.buffer(), 0,
                     trailField_.size())
        .update(device_, trailDecayDescriptorSet_);
    gridClearDescriptorSet_ = descriptorAllocator_.allocate(gridClearDescriptorSetLayout_.get());
    DescriptorSetWriter{}
        .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads_.buffer(), 0,
                     gridHeads_.size())
        .update(device_, gridClearDescriptorSet_);

    stepPipeline_ = ComputePipelineBuilder{physicalDevice_, device_}
                        .shader(VKEXP_SHADER_DIR "/agent_step.comp.spv")
                        .addDescriptorSetLayout(stepDescriptorSetLayout_.get())
                        .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(std::uint32_t))
                        .build();
    gridClearPipeline_ =
        ComputePipelineBuilder{physicalDevice_, device_}
            .shader(VKEXP_SHADER_DIR "/agent_grid_clear.comp.spv")
            .addDescriptorSetLayout(gridClearDescriptorSetLayout_.get())
            .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(std::uint32_t))
            .build();
    gridBuildPipeline_ =
        ComputePipelineBuilder{physicalDevice_, device_}
            .shader(VKEXP_SHADER_DIR "/agent_grid_build.comp.spv")
            .addDescriptorSetLayout(gridBuildDescriptorSetLayout_.get())
            .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(GridBuildParameters))
            .build();
    trailDecayPipeline_ =
        ComputePipelineBuilder{physicalDevice_, device_}
            .shader(VKEXP_SHADER_DIR "/trail_decay.comp.spv")
            .addDescriptorSetLayout(trailDecayDescriptorSetLayout_.get())
            .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(TrailDecayParameters))
            .build();
    trailDepositPipeline_ =
        ComputePipelineBuilder{physicalDevice_, device_}
            .shader(VKEXP_SHADER_DIR "/trail_deposit.comp.spv")
            .addDescriptorSetLayout(trailDepositDescriptorSetLayout_.get())
            .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(std::uint32_t))
            .build();

    state_.agents = {{agentBuffers_.read().buffer(), agentBuffers_.write().buffer()},
                     agentBytes,
                     agentBuffers_.readIndex(),
                     agentCount,
                     genomeCount,
                     config_.trialsPerGenome,
                     0};
}

void SimulationDriver::destroyResources() {
    state_.agents = {};
    state_.trail = {};
    gridBuildPipeline_ = {};
    gridClearPipeline_ = {};
    stepPipeline_ = {};
    descriptorAllocator_.reset();
    gridBuildDescriptorSetLayout_.reset();
    gridClearDescriptorSetLayout_.reset();
    stepDescriptorSetLayout_.reset();
    gridBuildDescriptorSets_ = {};
    gridClearDescriptorSet_ = VK_NULL_HANDLE;
    stepDescriptorSets_ = {};
    gridNext_.reset();
    gridHeads_.reset();
    stepParameterBuffer_.reset();
    genomeBuffer_.reset();
    agentBuffers_.reset();
    device_ = VK_NULL_HANDLE;
    physicalDevice_ = VK_NULL_HANDLE;
}

std::vector<AgentState> SimulationDriver::makeInitialAgents() const {
    const std::size_t genomeCount = evolution_.population().size();
    std::vector<AgentState> result(genomeCount * config_.trialsPerGenome);
    constexpr float goldenAngle = 2.39996323F;
    const SimulationStep initialSettings = resolveStepSettings(state_.physics, 0, 0);
    for (std::size_t genome = 0; genome < genomeCount; ++genome) {
        const std::size_t group = genome / state_.worlds.agentsPerWorld;
        const std::size_t firstGenome = group * state_.worlds.agentsPerWorld;
        const std::size_t agentsInGroup =
            std::min<std::size_t>(state_.worlds.agentsPerWorld, genomeCount - firstGenome);
        const std::size_t agentInGroup = genome - firstGenome;
        const float normalizedRadius = std::sqrt((static_cast<float>(agentInGroup) + 0.5F) /
                                                 static_cast<float>(agentsInGroup));
        for (std::size_t trial = 0; trial < config_.trialsPerGenome; ++trial) {
            const std::size_t index = genome * config_.trialsPerGenome + trial;
            const float positionAngle =
                goldenAngle * static_cast<float>(agentInGroup) + static_cast<float>(trial) * 0.43F;
            const float spawnRadius = normalizedRadius * state_.physics.worldRadius * 0.70F;
            const float heading =
                std::fmod(positionAngle * 1.73F + 0.37F, 2.0F * std::numbers::pi_v<float>);
            AgentState agent{};
            agent.pose = {std::cos(positionAngle) * spawnRadius,
                          std::sin(positionAngle) * spawnRadius, heading, agentBodyRadius};
            agent.motion.w = 1.0F;
            agent.signal = {0.15F, 0.45F, 0.85F, 0.0F};
            const Float4 target = stationaryBeaconPosition(static_cast<std::uint32_t>(trial),
                                                           state_.physics.worldRadius);
            agent.target = {target.x, target.y, static_cast<float>(trial), 0.0F};
            const float distance = nearestBeaconDistance(agent, initialSettings);
            agent.metrics = {distance, distance, 0.0F, 0.0F};
            agent.penalties.w = static_cast<float>(
                logicalWorldForAgent(static_cast<std::uint32_t>(index),
                                     state_.worlds.agentsPerWorld, config_.trialsPerGenome));
            // After the metrics, because a scenario that moves the spawn moves
            // the distance the trial is scored against with it.
            const ScenarioDefinition& scenario =
                scenarioDefinition(state_.physics.beaconScenario);
            if (scenario.spawn != nullptr) {
                scenario.spawn(agent, initialSettings);
                const float spawned = scenario.targetDistance != nullptr
                                          ? scenario.targetDistance(agent, initialSettings)
                                          : nearestBeaconDistance(agent, initialSettings);
                agent.metrics = {spawned, spawned, 0.0F, 0.0F};
            }
            result[index] = agent;
        }
    }
    return result;
}

void SimulationDriver::uploadPopulation() {
    std::vector<float> flattened;
    flattened.reserve(evolution_.population().size() * neuro::Topology::weightCount);
    for (const Genome& genome : evolution_.population()) {
        flattened.insert(flattened.end(), genome.weights.begin(), genome.weights.end());
    }
    genomeBuffer_.write(flattened.data(), flattened.size() * sizeof(float));
    agentBuffers_.read().write(agents_.data(), agents_.size() * sizeof(AgentState));
    agentBuffers_.write().write(agents_.data(), agents_.size() * sizeof(AgentState));
    hostUploadPending_ = true;
    trailClearPending_ = true;
}

void SimulationDriver::resetGeneration() {
    // The beacon motion seed is a property of the generation, so publishing it
    // here keeps simulation and visualization on the same random world.
    state_.physics.beaconMotionSeed = static_cast<std::uint32_t>(evolution_.generation());
    agents_ = makeInitialAgents();
    uploadPopulation();
    state_.statistics.step = 0;
    state_.statistics.generation = evolution_.generation();
    state_.agents.generation = evolution_.generation();
    state_.agents.currentIndex = agentBuffers_.readIndex();
}

void SimulationDriver::restart() {
    evolution_.reset();
    state_.statistics = {};
    state_.history.bestFitness.clear();
    state_.history.medianFitness.clear();
    state_.history.meanFitness.clear();
    state_.history.arrivalRatio.clear();
    refreshGridForWorldSize();
    resetGeneration();
}

void SimulationDriver::loadPopulation(const std::span<const Genome> genomes,
                                      const std::uint64_t generation) {
    evolution_.setPopulation(genomes, generation);
    state_.statistics.generation = generation;
    resetGeneration();
}

WorldSnapshot SimulationDriver::snapshot() {
    agentBuffers_.read().read(agents_.data(), agents_.size() * sizeof(AgentState));
    WorldSnapshot result;
    result.physics = state_.physics;
    result.genomes.assign(evolution_.population().begin(), evolution_.population().end());
    result.agents = agents_;
    result.generation = evolution_.generation();
    result.step = state_.statistics.step;
    result.stepsPerGeneration = state_.controls.stepsPerGeneration;
    result.requestedAgentsPerWorld = state_.worlds.requestedAgentsPerWorld;
    result.trialsPerGenome = config_.trialsPerGenome;
    result.seed = evolution_.settings().seed;
    return result;
}

void SimulationDriver::restoreSnapshot(const WorldSnapshot& snapshot) {
    // Order matters. The world layout depends on the physics settings and the
    // group size, and the trail and grid dimensions depend on the layout, so the
    // settings go in before anything is sized. The agents go in last, after
    // uploadPopulation would otherwise have replaced them with fresh ones.
    // Population size and trial count are buffer dimensions fixed when the
    // device resources were created, not settings. A snapshot that disagrees
    // cannot be resumed into this driver, and saying so beats writing past the
    // end of the agent buffer.
    if (snapshot.genomes.size() != evolution_.population().size()) {
        throw WorldSnapshotError("Snapshot holds " + std::to_string(snapshot.genomes.size()) +
                                 " genomes but this run was launched with " +
                                 std::to_string(evolution_.population().size()));
    }
    if (snapshot.trialsPerGenome != config_.trialsPerGenome) {
        throw WorldSnapshotError("Snapshot ran " + std::to_string(snapshot.trialsPerGenome) +
                                 " trials per genome but this run was launched with " +
                                 std::to_string(config_.trialsPerGenome));
    }
    if (snapshot.agents.size() != evolution_.population().size() * config_.trialsPerGenome) {
        throw WorldSnapshotError("Snapshot agent count does not match its own genome count");
    }

    state_.physics = snapshot.physics;
    state_.controls.stepsPerGeneration = snapshot.stepsPerGeneration;
    state_.worlds.requestedAgentsPerWorld = snapshot.requestedAgentsPerWorld;

    evolution_.setPopulation(snapshot.genomes, snapshot.generation);
    updateWorldLayout();
    refreshGridForWorldSize();

    // After uploadPopulation, which would otherwise send freshly spawned agents.
    agents_ = snapshot.agents;
    state_.statistics.step = std::min(snapshot.step, snapshot.stepsPerGeneration);

    uploadPopulation();
    state_.statistics.generation = snapshot.generation;
    state_.agents.generation = snapshot.generation;
    state_.agents.currentIndex = agentBuffers_.readIndex();
}

bool SimulationDriver::generationComplete() const {
    return state_.statistics.step >= state_.controls.stepsPerGeneration;
}

GenerationSummary SimulationDriver::finishGeneration() {
    agentBuffers_.read().read(agents_.data(), agents_.size() * sizeof(AgentState));
    const ScenarioDefinition& scenario = scenarioDefinition(state_.physics.beaconScenario);
    std::vector<float> fitness(evolution_.population().size());
    std::size_t achievedObjectives = 0;
    for (std::size_t genome = 0; genome < fitness.size(); ++genome) {
        for (std::size_t trial = 0; trial < config_.trialsPerGenome; ++trial) {
            const AgentState& agent = agents_[genome * config_.trialsPerGenome + trial];
            fitness[genome] += scenario.fitness(agent, state_.physics.fitness);
            achievedObjectives += scenario.achievedObjectives(agent);
        }
        fitness[genome] /= static_cast<float>(config_.trialsPerGenome);
    }
    // Selection may see a different number than the run reports. Sharing blends
    // each genome's score with its world's average, which is the point -- it
    // makes helping a neighbour pay -- but it also compresses the spread, so a
    // shared run and an unshared one could not be compared on their own headline
    // numbers. What is published stays individual, and is therefore the same
    // quantity at every sharing level; what selects is the shared vector.
    const std::vector<float> selectionFitness = shareFitnessWithinGroups(
        fitness, state_.worlds.agentsPerWorld, state_.physics.fitness.groupSharing);
    // Replay scores and reports a generation exactly as training does -- that is
    // how loaded weights get judged -- and then selects nothing. The population
    // is left untouched, so resetGeneration below respawns the same genomes and
    // the generation counter does not move. Since the beacon motion seed is the
    // generation number, the next run is the same run again rather than a
    // similar one, which is what makes a behaviour watchable more than once.
    GenerationSummary summary{};
    if (state_.controls.replay) {
        summary.generation = evolution_.generation();
        summary.championIndex = static_cast<std::size_t>(
            std::distance(fitness.begin(), std::max_element(fitness.begin(), fitness.end())));
    } else {
        summary = evolution_.evolve(selectionFitness);
    }
    summary.bestFitness = *std::max_element(fitness.begin(), fitness.end());
    summary.meanFitness = std::accumulate(fitness.begin(), fitness.end(), 0.0F) /
                          static_cast<float>(fitness.size());
    {
        std::vector<float> sorted = fitness;
        std::sort(sorted.begin(), sorted.end());
        summary.medianFitness = sorted[sorted.size() / 2];
    }
    state_.statistics.bestFitness = summary.bestFitness;
    state_.statistics.meanFitness = summary.meanFitness;
    state_.statistics.medianFitness = summary.medianFitness;
    state_.statistics.arrivalRatio =
        static_cast<float>(achievedObjectives) /
        static_cast<float>(agents_.size() * scenario.objectivesPerAgent);
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
            state_.physics.fitness.groupSharing = sweepValue(state_.sweep);
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
    state_.physics.fitness.groupSharing = sweepValue(state_.sweep);
    restart();
}

void SimulationDriver::endSweep() { stopSweep(state_.sweep); }

float SimulationDriver::gridCellSize() const {
    return gridCellSizeForWorld(config_.gridCellSize, state_.physics.worldRadius);
}

// The budget the field must fit, capped against what the device actually has.
std::uint64_t SimulationDriver::trailFieldBudget() const {
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
        return trailFieldByteBudget;
    }
    return std::min(trailFieldByteBudget, deviceLocalBytes / trailFieldHeapFraction);
}

// Bytes the field needs for the arena and world count actually in use.
std::uint64_t SimulationDriver::trailFieldBytes(const float cellSize) const {
    const auto width =
        static_cast<std::uint64_t>(trailWidthForWorld(state_.physics.worldRadius, cellSize));
    return width * width * trail::kernel::TrailChannels * sizeof(std::uint32_t) *
           state_.worlds.worldCount;
}

void SimulationDriver::updateTrailDimensions() {
    // Coarsen until the field fits the budget. A fine grid over a large arena and
    // two hundred worlds asks for gigabytes, and the first version both allocated
    // and cleared the worst case -- largest arena, most worlds -- whatever was
    // actually running, which is what made a resolution change look like a hang.
    // The capacity is fixed, so the resolution is what gives way, never the
    // allocation. Walk up the offered resolutions rather than doubling the cell,
    // so whatever survives is a setting the UI can name back to the user.
    std::size_t choice = 0;
    while (choice + 1 < trailCellFractions.size() &&
           trailCellSizeForBodyFraction(trailCellFractions[choice]) <
               state_.physics.trailCellSize - 0.0005F) {
        ++choice;
    }
    while (choice + 1 < trailCellFractions.size() &&
           trailFieldBytes(trailCellSizeForBodyFraction(trailCellFractions[choice])) >
               trailField_.size()) {
        ++choice;
    }
    const float cellSize = trailCellSizeForBodyFraction(trailCellFractions[choice]);
    state_.physics.trailCellSize = cellSize;

    trailWidth_ = trailWidthForWorld(state_.physics.worldRadius, cellSize);
    trailCellsPerWorld_ = trailWidth_ * trailWidth_;
    trailActiveBytes_ = static_cast<VkDeviceSize>(trailFieldBytes(cellSize));
    if (trailActiveBytes_ > trailField_.size()) {
        throw std::runtime_error("Trail field does not fit even at the coarsest resolution");
    }
    state_.trail = {trailField_.buffer(), trailField_.size(), trailWidth_, trailCellsPerWorld_};
}

void SimulationDriver::updateWorldLayout() {
    const auto genomeCount = static_cast<std::uint32_t>(evolution_.population().size());
    state_.worlds.agentsPerWorld =
        clampAgentsPerWorld(genomeCount, state_.worlds.requestedAgentsPerWorld);
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

void SimulationDriver::updateGridDimensions() {
    gridWidth_ =
        static_cast<std::uint32_t>(std::ceil((state_.physics.worldRadius * 2.0F) / gridCellSize()));
    gridCellsPerWorld_ = gridWidth_ * gridWidth_;
}

void SimulationDriver::ensureGridCapacity() {
    // The scaled cell size makes this the same width in every world; it stays a
    // Large-world calculation so a future unscaled cell size cannot under-allocate.
    const float largestRadius = worldRadiusForSize(WorldSize::Large);
    const auto maximumGridWidth = static_cast<std::uint32_t>(std::ceil(
        (largestRadius * 2.0F) / gridCellSizeForWorld(config_.gridCellSize, largestRadius)));
    const VkDeviceSize requiredBytes =
        sizeof(std::int32_t) * maximumGridWidth * maximumGridWidth * state_.worlds.worldCount;
    if (gridHeads_.size() >= requiredBytes) {
        return;
    }
    gridHeads_.reset();
    gridHeads_.create(physicalDevice_, device_,
                      {requiredBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});
    updateGridDescriptors();
}

void SimulationDriver::refreshGridForWorldSize() {
    updateWorldLayout();
    ensureGridCapacity();
    updateGridDimensions();
    updateTrailDimensions();
}

void SimulationDriver::updateGridDescriptors() {
    if (gridClearDescriptorSet_ != VK_NULL_HANDLE) {
        DescriptorSetWriter{}
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads_.buffer(), 0,
                         gridHeads_.size())
            .update(device_, gridClearDescriptorSet_);
    }
    for (std::size_t index = 0; index < stepDescriptorSets_.size(); ++index) {
        if (stepDescriptorSets_[index] != VK_NULL_HANDLE) {
            DescriptorSetWriter{}
                .writeBuffer(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads_.buffer(), 0,
                             gridHeads_.size())
                .update(device_, stepDescriptorSets_[index]);
        }
        if (gridBuildDescriptorSets_[index] != VK_NULL_HANDLE) {
            DescriptorSetWriter{}
                .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads_.buffer(), 0,
                             gridHeads_.size())
                .update(device_, gridBuildDescriptorSets_[index]);
        }
    }
}

GpuStepParameters SimulationDriver::stepParameters(const std::uint32_t generationStep) const {
    const SimulationStep settings =
        resolveStepSettings(state_.physics, generationStep, state_.controls.stepsPerGeneration);
    return packStepParameters(settings, StepParameterLayout{state_.agents.agentCount,
                                                            config_.trialsPerGenome,
                                                            state_.worlds.agentsPerWorld,
                                                            gridCellSize(),
                                                            gridWidth_,
                                                            gridCellsPerWorld_,
                                                            trailWidth_,
                                                            trailCellsPerWorld_});
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

    for (std::uint32_t step = 0; step < stepCount; ++step) {
        stepParameterStaging_[step] = stepParameters(state_.statistics.step + step);
    }
    stepParameterBuffer_.write(stepParameterStaging_.data(), sizeof(GpuStepParameters) * stepCount);
    cmdBufferBarrier(commands, stepParameterBuffer_.buffer(), VK_PIPELINE_STAGE_2_HOST_BIT,
                     VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    if (trailClearPending_) {
        vkCmdFillBuffer(commands, trailField_.buffer(), 0, trailActiveBytes_, 0);
        cmdBufferBarrier(commands, trailField_.buffer(), VK_PIPELINE_STAGE_2_CLEAR_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        trailClearPending_ = false;
    }

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
        hostUploadPending_ = false;
    }

    const GridBuildParameters gridParameters{state_.physics.worldRadius,
                                             gridCellSize(),
                                             state_.agents.agentCount,
                                             state_.worlds.worldCount,
                                             gridWidth_,
                                             gridCellsPerWorld_,
                                             0,
                                             0};
    const std::uint32_t totalGridCells = gridCellsPerWorld_ * state_.worlds.worldCount;
    const std::array<VkDeviceSize, 1> clearRanges{gridHeads_.size()};
    const DispatchSize gridGroups = checkedDispatchSize(
        physicalDevice_, {{totalGridCells, 1, 1}, {64, 1, 1}, sizeof(totalGridCells), clearRanges});
    const std::array<VkDeviceSize, 3> buildRanges{agentBuffers_.read().size(), gridHeads_.size(),
                                                  gridNext_.size()};
    const DispatchSize buildGroups = checkedDispatchSize(
        physicalDevice_,
        {{state_.agents.agentCount, 1, 1}, {64, 1, 1}, sizeof(gridParameters), buildRanges});
    const std::array<VkDeviceSize, 7> stepRanges{agentBuffers_.read().size(),
                                                 agentBuffers_.write().size(),
                                                 genomeBuffer_.size(),
                                                 gridHeads_.size(),
                                                 gridNext_.size(),
                                                 stepParameterBuffer_.size(),
                                                 trailField_.size()};
    const std::uint32_t trailValueCount =
        trailCellsPerWorld_ * state_.worlds.worldCount * trail::kernel::TrailChannels;
    const std::array<VkDeviceSize, 1> decayRanges{trailField_.size()};
    const DispatchSize trailDecayGroups = checkedDispatchSize(
        physicalDevice_,
        {{trailValueCount, 1, 1}, {64, 1, 1}, sizeof(TrailDecayParameters), decayRanges});
    const std::array<VkDeviceSize, 3> depositRanges{agentBuffers_.read().size(), trailField_.size(),
                                                    stepParameterBuffer_.size()};
    const DispatchSize trailDepositGroups = checkedDispatchSize(
        physicalDevice_,
        {{state_.agents.agentCount, 1, 1}, {64, 1, 1}, sizeof(std::uint32_t), depositRanges});
    const DispatchSize stepGroups = checkedDispatchSize(
        physicalDevice_,
        {{state_.agents.agentCount, 1, 1}, {64, 1, 1}, sizeof(std::uint32_t), stepRanges});

    for (std::uint32_t step = 0; step < stepCount; ++step) {
        cmdBufferBarrier(commands, gridHeads_.buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        cmdBufferBarrier(commands, gridNext_.buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, gridClearPipeline_.pipeline());
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                gridClearPipeline_.layout(), 0, 1, &gridClearDescriptorSet_, 0,
                                nullptr);
        vkCmdPushConstants(commands, gridClearPipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(totalGridCells), &totalGridCells);
        vkCmdDispatch(commands, gridGroups.x, 1, 1);
        cmdBufferBarrier(
            commands, gridHeads_.buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        const std::uint32_t readIndex = agentBuffers_.readIndex();
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, gridBuildPipeline_.pipeline());
        const VkDescriptorSet buildSet = gridBuildDescriptorSets_[readIndex];
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                gridBuildPipeline_.layout(), 0, 1, &buildSet, 0, nullptr);
        vkCmdPushConstants(commands, gridBuildPipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(gridParameters), &gridParameters);
        vkCmdDispatch(commands, buildGroups.x, 1, 1);
        cmdComputeWriteToComputeRead(commands, gridHeads_.buffer());
        cmdComputeWriteToComputeRead(commands, gridNext_.buffer());

        // Trail: fade the whole field, then let every agent add to it, then run
        // the step so all agents read a field nobody is still writing. Splitting
        // deposit from the step is what keeps the reading order-independent --
        // depositing inside agent_step would let some agents read a cell another
        // agent had already marked this tick.
        if (state_.physics.trailEnabled) {
            const TrailDecayParameters decayParameters{trailValueCount,
                                                       stepParameterStaging_[step].trailSurvival};
            vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                              trailDecayPipeline_.pipeline());
            vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    trailDecayPipeline_.layout(), 0, 1, &trailDecayDescriptorSet_,
                                    0, nullptr);
            vkCmdPushConstants(commands, trailDecayPipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(decayParameters), &decayParameters);
            vkCmdDispatch(commands, trailDecayGroups.x, 1, 1);
            cmdComputeWriteToComputeRead(commands, trailField_.buffer());

            vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                              trailDepositPipeline_.pipeline());
            const VkDescriptorSet depositSet = trailDepositDescriptorSets_[readIndex];
            vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    trailDepositPipeline_.layout(), 0, 1, &depositSet, 0, nullptr);
            vkCmdPushConstants(commands, trailDepositPipeline_.layout(),
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(step), &step);
            vkCmdDispatch(commands, trailDepositGroups.x, 1, 1);
            cmdComputeWriteToComputeRead(commands, trailField_.buffer());
        }

        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, stepPipeline_.pipeline());
        const VkDescriptorSet stepSet = stepDescriptorSets_[readIndex];
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, stepPipeline_.layout(), 0,
                                1, &stepSet, 0, nullptr);
        vkCmdPushConstants(commands, stepPipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(step), &step);
        vkCmdDispatch(commands, stepGroups.x, 1, 1);
        cmdBufferBarrier(
            commands, agentBuffers_.read().buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        cmdComputeWriteToComputeRead(commands, agentBuffers_.write().buffer());
        agentBuffers_.swap();
        state_.agents.currentIndex = agentBuffers_.readIndex();
    }

    cmdBufferBarrier(commands, agentBuffers_.read().buffer(),
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_HOST_READ_BIT);
    // The renderer reads the field from the vertex stage, so the deposit pass's
    // writes need to be made visible there too, not only to the next compute read.
    cmdBufferBarrier(commands, trailField_.buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    state_.statistics.step += stepCount;
    return stepCount;
}

} // namespace vkexp
