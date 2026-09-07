#include "vkexp/compute/HeadlessComputeContext.hpp"
#include "vkexp/neuro/BrainKernel.hpp"
#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/simulation/AgentTypes.hpp"
#include "vkexp/simulation/CpuSimulation.hpp"
#include "vkexp/simulation/StepParameters.hpp"
#include "vkexp/worlds/WorldScenario.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct GridSize {
    std::uint32_t width;
    std::uint32_t height;
};

void runGameOfLife(vkexp::HeadlessComputeContext& context) {
    constexpr GridSize grid{16, 16};
    constexpr std::size_t cellCount = grid.width * grid.height;
    constexpr VkDeviceSize byteSize = cellCount * sizeof(std::uint32_t);
    std::array<std::uint32_t, cellCount> initial{};
    const std::size_t center = (grid.height / 2) * grid.width + grid.width / 2;
    initial[center - 1] = 1;
    initial[center] = 1;
    initial[center + 1] = 1;

    vkexp::PingPongBuffer state;
    state.create(context.physicalDevice(), context.device(),
                 {byteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT});
    context.immediate().uploadBuffer(state.read(), initial.data(), byteSize);

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    for (std::uint32_t index = 0; index < bindings.size(); ++index) {
        bindings[index].binding = index;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    vkexp::UniqueDescriptorSetLayout setLayout;
    if (vkCreateDescriptorSetLayout(context.device(), &layoutInfo, nullptr,
                                    setLayout.put(context.device())) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create smoke descriptor set layout");
    }

    vkexp::DescriptorAllocator descriptors{context.device(),
                                           {2, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}}}};
    vkexp::PingPongDescriptorSets descriptorSets;
    descriptorSets.createStorageBuffers(context.physicalDevice(), context.device(), descriptors,
                                        setLayout.get(), state);

    constexpr std::uint32_t aliveValue = 1;
    const vkexp::ComputePipeline pipeline =
        vkexp::ComputePipelineBuilder{context.physicalDevice(), context.device()}
            .shader(VKEXP_SHADER_DIR "/game_of_life.comp.spv")
            .addDescriptorSetLayout(setLayout.get())
            .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(GridSize))
            .specializationConstant(0, aliveValue)
            .build();

    context.immediate().execute([&](const VkCommandBuffer commands) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline());
        const VkPipelineLayout pipelineLayout = pipeline.layout();
        vkCmdPushConstants(commands, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(GridSize), &grid);
        for (int step = 0; step < 2; ++step) {
            const VkDescriptorSet descriptorSet = descriptorSets.current(state);
            vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                    &descriptorSet, 0, nullptr);
            const std::array<VkDeviceSize, 2> storageRanges{state.read().size(),
                                                            state.write().size()};
            const vkexp::DispatchSize groups = vkexp::checkedDispatchSize(
                context.physicalDevice(),
                {{grid.width, grid.height, 1}, {8, 8, 1}, sizeof(GridSize), storageRanges});
            vkCmdDispatch(commands, groups.x, groups.y, groups.z);
            vkexp::cmdComputePingPongBarrier(commands, state);
            state.swap();
        }
    });

    std::array<std::uint32_t, cellCount> result{};
    context.immediate().readbackBuffer(state.read(), result.data(), byteSize);
    if (result != initial) {
        throw std::runtime_error("Two Game of Life GPU steps did not restore the blinker");
    }
}

void runImageRoundTrip(vkexp::HeadlessComputeContext& context) {
    constexpr VkExtent2D extent{4, 4};
    constexpr std::size_t byteCount = extent.width * extent.height * 4;
    std::array<std::uint8_t, byteCount> pixels{};
    for (std::size_t index = 0; index < pixels.size(); ++index) {
        pixels[index] = static_cast<std::uint8_t>(index * 3);
    }

    vkexp::PingPongImage images;
    images.create(context.physicalDevice(), context.device(),
                  {extent, VK_FORMAT_R8G8B8A8_UNORM,
                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT});
    context.immediate().uploadImage(images.read(), pixels.data(), pixels.size());

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    for (std::uint32_t index = 0; index < bindings.size(); ++index) {
        bindings[index].binding = index;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    vkexp::UniqueDescriptorSetLayout setLayout;
    if (vkCreateDescriptorSetLayout(context.device(), &layoutInfo, nullptr,
                                    setLayout.put(context.device())) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create image descriptor set layout");
    }
    vkexp::DescriptorAllocator descriptors{context.device(),
                                           {2, {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4}}}};
    vkexp::PingPongDescriptorSets descriptorSets;
    descriptorSets.createStorageImages(context.device(), descriptors, setLayout.get(), images);
    const VkDescriptorSet firstSet = descriptorSets.current(images);
    images.swap();
    const VkDescriptorSet secondSet = descriptorSets.current(images);
    if (firstSet == secondSet) {
        throw std::runtime_error("Image ping-pong descriptors did not switch sets");
    }
    images.swap();

    std::array<std::uint8_t, byteCount> downloaded{};
    context.immediate().readbackImage(images.read(), downloaded.data(), downloaded.size());
    if (downloaded != pixels) {
        throw std::runtime_error("GPU image upload/readback did not preserve RGBA8 data");
    }
}

constexpr VkMemoryPropertyFlags hostMemory =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr float parityGridCellSize = 0.12F;
// Measured noise on a 540-step run is 5e-5 to 1e-4 per scenario, from float
// rounding differences between libm and the GPU. This budget leaves an order of
// magnitude of headroom for other devices, and a systematic term difference --
// a changed shader constant, say -- overshoots it by another order.
constexpr double accumulatedDriftBudget = 1.0e-3;

// The driver's own packing, called rather than copied: this used to be a private
// mirror of it and twice quietly ran the shader with a field the CPU reference
// had set.
vkexp::GpuStepParameters
makeStepParameters(const vkexp::SimulationStep& settings, const std::uint32_t agentCount,
                   const std::uint32_t trialsPerGenome, const std::uint32_t gridWidth,
                   const std::uint32_t gridCellsPerWorld, const std::uint32_t trailWidth = 1,
                   const std::uint32_t trailCellsPerWorld = 1,
                   const std::uint32_t agentsPerWorld = 1,
                   // The trail field is GPU-only state with no CPU counterpart,
                   // so parity runs with it off and runTrailFieldProbe covers it
                   // directly.
                   const bool trailEnabled = false) {
    vkexp::SimulationStep resolved = settings;
    resolved.trailEnabled = trailEnabled;
    return vkexp::packStepParameters(
        resolved, vkexp::StepParameterLayout{agentCount, trialsPerGenome, agentsPerWorld,
                                             parityGridCellSize, gridWidth, gridCellsPerWorld,
                                             trailWidth, trailCellsPerWorld});
}



// The agent_step pipeline wired up exactly like SimulationDriver does, with
// host-visible buffers so a test can drive many steps without a staging copy
// per step.
class StepHarness {
public:
    StepHarness(vkexp::HeadlessComputeContext& context, const std::uint32_t agentCount,
                const std::uint32_t genomeCount, const std::uint32_t worldCount,
                const std::uint32_t maximumSteps, const float worldRadius)
        : context_(context), agentCount_(agentCount),
          gridWidth_(
              static_cast<std::uint32_t>(std::ceil(worldRadius * 2.0F / parityGridCellSize))),
          gridCellsPerWorld_(gridWidth_ * gridWidth_), worldCount_(worldCount) {
        const VkDeviceSize agentBytes = sizeof(vkexp::AgentState) * agentCount;
        const auto storage = static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        inputAgents.create(context.physicalDevice(), context.device(),
                           {agentBytes, storage, hostMemory});
        outputAgents.create(context.physicalDevice(), context.device(),
                            {agentBytes, storage, hostMemory});
        genomes.create(context.physicalDevice(), context.device(),
                       {sizeof(float) * vkexp::neuro::Topology::weightCount * genomeCount, storage,
                        hostMemory});
        gridHeads.create(
            context.physicalDevice(), context.device(),
            {sizeof(std::int32_t) * gridCellsPerWorld_ * worldCount, storage, hostMemory});
        gridNext.create(context.physicalDevice(), context.device(),
                        {sizeof(std::int32_t) * agentCount, storage, hostMemory});
        stepParameters.create(
            context.physicalDevice(), context.device(),
            {sizeof(vkexp::GpuStepParameters) * maximumSteps, storage, hostMemory});
        // The trail field is GPU-only state, so the parity cases run with it off
        // and this buffer stays zero; the harness still has to bind it, and
        // runTrailFieldProbe drives it for real.
        trailWidth_ = vkexp::trailWidthForWorld(worldRadius, vkexp::SimulationStep{}.trailCellSize);
        trailCellsPerWorld_ = trailWidth_ * trailWidth_;
        trail.create(context.physicalDevice(), context.device(),
                     {sizeof(std::uint32_t) * trailCellsPerWorld_ * worldCount *
                          vkexp::trail::kernel::TrailChannels,
                      storage, hostMemory});
        std::vector<std::uint32_t> emptyField(
            trailCellsPerWorld_ * worldCount * vkexp::trail::kernel::TrailChannels, 0U);
        trail.write(emptyField.data(), sizeof(std::uint32_t) * emptyField.size());

        std::array<VkDescriptorSetLayoutBinding, 7> bindings{};
        for (std::uint32_t binding = 0; binding < bindings.size(); ++binding) {
            bindings[binding].binding = binding;
            bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[binding].descriptorCount = 1;
            bindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layoutInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(context.device(), &layoutInfo, nullptr,
                                        layout_.put(context.device())) != VK_SUCCESS) {
            throw std::runtime_error("Unable to create agent step descriptor layout");
        }
        descriptors_.create(context.device(), {1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7}}});
        set_ = descriptors_.allocate(layout_.get());
        vkexp::DescriptorSetWriter{}
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, inputAgents.buffer(), 0,
                         inputAgents.size())
            .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, outputAgents.buffer(), 0,
                         outputAgents.size())
            .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, genomes.buffer(), 0, genomes.size())
            .writeBuffer(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridHeads.buffer(), 0,
                         gridHeads.size())
            .writeBuffer(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gridNext.buffer(), 0,
                         gridNext.size())
            .writeBuffer(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stepParameters.buffer(), 0,
                         stepParameters.size())
            .writeBuffer(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, trail.buffer(), 0, trail.size())
            .update(context.device(), set_);
        pipeline_ = vkexp::ComputePipelineBuilder{context.physicalDevice(), context.device()}
                        .shader(VKEXP_SHADER_DIR "/agent_step.comp.spv")
                        .addDescriptorSetLayout(layout_.get())
                        .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(std::uint32_t))
                        .build();
    }

    // Rebuilds the per-world grid on the host. The GPU grid passes have their
    // own coverage; here the point is to feed the step shader a valid grid.
    void buildGrid(const std::span<const vkexp::AgentState> agents, const float worldRadius) {
        std::vector<std::int32_t> heads(gridCellsPerWorld_ * worldCount_, -1);
        std::vector<std::int32_t> links(agents.size(), -1);
        const auto cell = [&](const float position) {
            return static_cast<std::uint32_t>(
                std::clamp(std::floor((position + worldRadius) / parityGridCellSize), 0.0F,
                           static_cast<float>(gridWidth_ - 1)));
        };
        for (std::size_t index = 0; index < agents.size(); ++index) {
            const auto world =
                static_cast<std::uint32_t>(std::max(agents[index].penalties.w, 0.0F)) % worldCount_;
            const std::uint32_t cellIndex = world * gridCellsPerWorld_ +
                                            cell(agents[index].pose.y) * gridWidth_ +
                                            cell(agents[index].pose.x);
            links[index] = heads[cellIndex];
            heads[cellIndex] = static_cast<std::int32_t>(index);
        }
        gridHeads.write(heads.data(), heads.size() * sizeof(std::int32_t));
        gridNext.write(links.data(), links.size() * sizeof(std::int32_t));
    }

    void dispatch(const std::uint32_t stepIndex) {
        context_.immediate().execute([&](const VkCommandBuffer commands) {
            vkexp::cmdBufferBarrier(commands, stepParameters.buffer(), VK_PIPELINE_STAGE_2_HOST_BIT,
                                    VK_ACCESS_2_HOST_WRITE_BIT,
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.pipeline());
            vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.layout(), 0,
                                    1, &set_, 0, nullptr);
            vkCmdPushConstants(commands, pipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(stepIndex), &stepIndex);
            vkCmdDispatch(commands, vkexp::divideRoundUp(agentCount_, 64), 1, 1);
            vkexp::cmdBufferBarrier(commands, outputAgents.buffer(),
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                    VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        });
    }

    [[nodiscard]] std::uint32_t gridWidth() const { return gridWidth_; }
    [[nodiscard]] std::uint32_t gridCellsPerWorld() const { return gridCellsPerWorld_; }
    [[nodiscard]] std::uint32_t trailWidth() const { return trailWidth_; }
    [[nodiscard]] std::uint32_t trailCellsPerWorld() const { return trailCellsPerWorld_; }

    // Writes one cell of the field directly. The deposit shader has its own
    // addressing; this drives the read path from a known ground truth so a
    // failure points at the antennae rather than at whoever wrote the field.
    void markTrailCell(const std::uint32_t world, const std::uint32_t cell, const float red,
                       const float green, const float blue) {
        std::vector<std::uint32_t> field(
            trailCellsPerWorld_ * worldCount_ * vkexp::trail::kernel::TrailChannels, 0U);
        const float channels[] = {red, green, blue};
        for (std::uint32_t channel = 0; channel < vkexp::trail::kernel::TrailChannels; ++channel) {
            field[vkexp::trail::kernel::trailValueIndex(world, trailCellsPerWorld_, cell,
                                                        channel)] =
                static_cast<std::uint32_t>(channels[channel] *
                                           vkexp::trail::kernel::TrailFixedPointScale);
        }
        trail.write(field.data(), sizeof(std::uint32_t) * field.size());
    }

    vkexp::BufferResource inputAgents;
    vkexp::BufferResource outputAgents;
    vkexp::BufferResource genomes;
    vkexp::BufferResource gridHeads;
    vkexp::BufferResource gridNext;
    vkexp::BufferResource stepParameters;
    vkexp::BufferResource trail;

private:
    vkexp::HeadlessComputeContext& context_;
    std::uint32_t agentCount_{};
    std::uint32_t gridWidth_{};
    std::uint32_t trailWidth_{};
    std::uint32_t trailCellsPerWorld_{};
    std::uint32_t gridCellsPerWorld_{};
    std::uint32_t worldCount_{};
    vkexp::UniqueDescriptorSetLayout layout_;
    vkexp::DescriptorAllocator descriptors_;
    VkDescriptorSet set_{};
    vkexp::ComputePipeline pipeline_;
};

constexpr std::size_t agentFloatCount = sizeof(vkexp::AgentState) / sizeof(float);
using AgentDrift = std::array<double, agentFloatCount>;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void compareAgents(const vkexp::AgentState& expected, const vkexp::AgentState& actual,
                   const float tolerance, const std::string& context,
                   AgentDrift* accumulated = nullptr) {
    const auto* expectedValues = reinterpret_cast<const float*>(&expected);
    const auto* actualValues = reinterpret_cast<const float*>(&actual);
    for (std::size_t index = 0; index < agentFloatCount; ++index) {
        const float difference = expectedValues[index] - actualValues[index];
        if (accumulated != nullptr) {
            // Signed, so that a systematic bias adds up while symmetric rounding
            // noise cancels instead of masquerading as one.
            (*accumulated)[index] += static_cast<double>(difference);
        }
        if (std::abs(difference) >= tolerance) {
            throw std::runtime_error(context + ": mismatch at float " + std::to_string(index) +
                                     ": expected " + std::to_string(expectedValues[index]) +
                                     ", got " + std::to_string(actualValues[index]));
        }
    }
}

vkexp::neuro::Weights makeTestWeights(const float scale = 0.31F) {
    vkexp::neuro::Weights weights{};
    for (std::size_t index = 0; index < weights.size(); ++index) {
        weights[index] = std::sin(static_cast<float>(index) * 0.37F) * scale;
    }
    return weights;
}

const char* scenarioName(const vkexp::BeaconScenario scenario) {
    return vkexp::scenarioDefinition(scenario).name;
}

// Deterministic multi-step CPU/GPU regression.
//
// Lockstep on purpose: each step feeds the CPU reference state to the GPU,
// compares one step of both, then continues from the CPU state. That exercises
// the shader at hundreds of genuinely reachable states -- including the
// alternating phase flip and the forage home relocation epochs -- while keeping
// the comparison free of the chaotic drift a free-running trajectory would
// accumulate through tanh feedback.
void runTrajectoryParity(vkexp::HeadlessComputeContext& context,
                         const vkexp::BeaconScenario scenario, const std::uint32_t steps,
                         const vkexp::NeuronModel neuronModel = vkexp::NeuronModel::TimeConstant) {
    const vkexp::neuro::Weights weights = makeTestWeights();
    vkexp::SimulationStep base{};
    base.beaconScenario = scenario;
    base.neuronModel = neuronModel;
    base.beaconMotionSeed = 0x5eed1234U;

    StepHarness harness{context, 1, 1, 1, 1, base.worldRadius};
    harness.genomes.write(weights.data(), sizeof(weights));

    vkexp::AgentState agent{};
    agent.pose = {0.42F, -0.31F, 0.7F, vkexp::agentBodyRadius};
    agent.motion = {0.05F, 0.02F, 0.0F, 1.0F};
    agent.signal = {0.15F, 0.45F, 0.85F, 0.0F};
    const vkexp::Float4 target = vkexp::stationaryBeaconPosition(0, base.worldRadius);
    agent.target = {target.x, target.y, 0.0F, 0.0F};
    const vkexp::SimulationStep initial = vkexp::resolveStepSettings(base, 0, steps);
    const float distance = vkexp::nearestBeaconDistance(agent, initial);
    agent.metrics = {distance, distance, 0.0F, 0.0F};

    AgentDrift drift{};
    std::uint32_t phaseTransitions = 0;
    std::uint32_t stepsWithVisibleBeacon = 0;
    for (std::uint32_t step = 0; step < steps; ++step) {
        const vkexp::SimulationStep settings = vkexp::resolveStepSettings(base, step, steps);
        if (settings.beaconPhaseChanged) {
            ++phaseTransitions;
        }
        const std::array<vkexp::AgentState, 1> agents{agent};
        harness.buildGrid(agents, settings.worldRadius);
        harness.inputAgents.write(&agent, sizeof(agent));
        const vkexp::GpuStepParameters parameters =
            makeStepParameters(settings, 1, 1, harness.gridWidth(), harness.gridCellsPerWorld());
        harness.stepParameters.write(&parameters, sizeof(parameters));
        harness.dispatch(0);

        vkexp::AgentState actual{};
        harness.outputAgents.read(&actual, sizeof(actual));
        if (vkexp::nearestBeaconDistance(agent, settings) < settings.lightSensorRange) {
            ++stepsWithVisibleBeacon;
        }
        vkexp::stepAgentCpu(agent, weights, settings);
        compareAgents(agent, actual, 0.0002F,
                      std::string{"Trajectory parity ["} + scenarioName(scenario) + "] step " +
                          std::to_string(step),
                      &drift);

        const float centerDistance = std::hypot(agent.pose.x, agent.pose.y);
        if (!std::isfinite(centerDistance) ||
            centerDistance > settings.worldRadius - agent.pose.w + 0.001F) {
            throw std::runtime_error(std::string{"Trajectory parity ["} + scenarioName(scenario) +
                                     "] left the world at step " + std::to_string(step));
        }
    }
    if (scenario == vkexp::BeaconScenario::AlternatingDiagonals && phaseTransitions != 1) {
        throw std::runtime_error("Alternating trajectory did not cross exactly one phase change");
    }
    // Lockstep alone cannot see a systematic bias smaller than the per-step
    // tolerance, because resetting to the CPU state every step stops it from
    // accumulating. Summing the signed per-step differences restores that:
    // symmetric rounding noise cancels, a constant offset in a shader term does
    // not.
    const double worstDrift =
        *std::max_element(drift.begin(), drift.end(),
                          [](const double a, const double b) { return std::abs(a) < std::abs(b); });
    const char* modelName = neuronModel == vkexp::NeuronModel::Reactive      ? "reactive"
                            : neuronModel == vkexp::NeuronModel::Gated       ? "gated"
                                                                            : "time constant";
    std::cout << "  drift[" << scenarioName(scenario) << ", " << modelName << "] = " << worstDrift
              << '\n';
    if (std::abs(worstDrift) > accumulatedDriftBudget) {
        throw std::runtime_error(std::string{"Trajectory parity ["} + scenarioName(scenario) +
                                 "] accumulated a systematic CPU/GPU drift of " +
                                 std::to_string(worstDrift) + " over " + std::to_string(steps) +
                                 " steps (budget " + std::to_string(accumulatedDriftBudget) + ")");
    }
    // A trajectory that never sees a beacon would silently skip the distance
    // shaping and arrival branches, making the whole run vacuous.
    if (stepsWithVisibleBeacon * 4 < steps) {
        throw std::runtime_error(std::string{"Trajectory parity ["} + scenarioName(scenario) +
                                 "] kept the beacon out of sensor range for most of the run; "
                                 "it would not cover the shaping and arrival branches");
    }
}

// Genome addressing probe.
//
// This is the class of bug the CPU mirror used to guard: a wrong genomeStride or
// base offset makes an agent read another genome's weights, which a parity test
// on a single agent cannot see. Each genome gets a distinctive motor bias, and
// every agent must drive exactly as its own genome says.
// The trail field is GPU-only state, so the parity cases cannot reach it. What
// has to be checked instead is the coupling: a mark under one antenna tip must
// reach that antenna's inputs and nobody else's, and must change what the agent
// does. A wrong cell index, a wrong channel or a wrong antenna angle all show up
// here as the wrong agent turning.
void runTrailFieldProbe(vkexp::HeadlessComputeContext& context) {
    constexpr float worldRadius = vkexp::smallWorldRadius;
    StepHarness harness{context, 1, 1, 1, 1, worldRadius};

    vkexp::SimulationStep settings{};
    settings.worldRadius = worldRadius;
    settings.lightSensorRange = vkexp::lightRangeForWorld(settings);

    // Left and right antennae are wired to opposite motors, so the sign of the
    // turn says which tip did the smelling. Reading one antenna would not be
    // enough: a shader that fed every tip the same cell would still turn.
    const vkexp::neuro::BrainShape brain = vkexp::scenarioDefinition(settings.beaconScenario).brain;
    namespace bk = vkexp::neuro::kernel;
    const auto inputCount = static_cast<bk::uint>(brain.inputCount);
    const auto hiddenCount = static_cast<bk::uint>(brain.hiddenCount);
    std::vector<float> genome(vkexp::neuro::Topology::weightCount, 0.0F);
    const bk::uint leftGreen = bk::brainAntennaChannelIndex(0u, 1u);
    const bk::uint rightGreen = bk::brainAntennaChannelIndex(bk::BrainAntennaCount - 1u, 1u);
    genome[bk::brainHiddenWeightIndex(0, inputCount, 0u, leftGreen)] = 4.0F;
    genome[bk::brainHiddenWeightIndex(0, inputCount, 1u, rightGreen)] = 4.0F;
    genome[bk::brainOutputWeightIndex(0, inputCount, hiddenCount, bk::BrainMotorRightOutput, 0u)] =
        4.0F;
    genome[bk::brainOutputWeightIndex(0, inputCount, hiddenCount, bk::BrainMotorLeftOutput, 1u)] =
        4.0F;
    harness.genomes.write(genome.data(), sizeof(float) * genome.size());

    vkexp::AgentState agent{};
    agent.pose = {0.0F, 0.0F, 0.0F, vkexp::agentBodyRadius};
    agent.motion.w = 1.0F;
    agent.target = {worldRadius * 0.6F, 0.0F, 0.0F, 0.0F};
    agent.metrics = {worldRadius, worldRadius, 0.0F, 0.0F};

    const auto stepWithField = [&](const bool trailEnabled, const bool markLeft) {
        // The cell the chosen antenna tip actually lands in, derived the same way
        // the shader derives it rather than guessed.
        const float tipAngle =
            agent.pose.z + bk::brainAntennaAngle(markLeft ? 0u : bk::BrainAntennaCount - 1u);
        const vkexp::trail::kernel::vec2 tip{
            agent.pose.x + std::cos(tipAngle) * bk::BrainAntennaLength,
            agent.pose.y + std::sin(tipAngle) * bk::BrainAntennaLength};
        const std::uint32_t cell = vkexp::trail::kernel::trailCellIndex(
            tip, worldRadius, settings.trailCellSize, harness.trailWidth());
        harness.markTrailCell(0, cell, 0.0F, 4.0F, 0.0F);

        harness.inputAgents.write(&agent, sizeof(agent));
        harness.buildGrid(std::span{&agent, 1}, worldRadius);
        const vkexp::GpuStepParameters parameters =
            makeStepParameters(settings, 1, 1, harness.gridWidth(), harness.gridCellsPerWorld(),
                               harness.trailWidth(), harness.trailCellsPerWorld(), 1, trailEnabled);
        harness.stepParameters.write(&parameters, sizeof(parameters));
        harness.dispatch(0);
        vkexp::AgentState stepped{};
        harness.outputAgents.read(&stepped, sizeof(stepped));
        return stepped;
    };

    const vkexp::AgentState blind = stepWithField(false, true);
    const vkexp::AgentState leftScent = stepWithField(true, true);
    const vkexp::AgentState rightScent = stepWithField(true, false);

    require(std::abs(blind.motion.z) < 1.0e-6F,
            "With the trail off the antenna inputs are zero and the agent does not turn");
    require(leftScent.motion.z > 1.0e-3F,
            "A mark under the left antenna drives the right motor and turns one way");
    require(rightScent.motion.z < -1.0e-3F,
            "The same mark under the right antenna turns the other way");

    // Decay is a per-second half-life, not a per-tick factor, so the same
    // simulated second has to fade the same amount at any step rate.
    const float rate = vkexp::trail::kernel::trailDecayRateForHalfLife(2.0F);
    float slow = 1.0F;
    for (int step = 0; step < 60; ++step) {
        slow *= vkexp::trail::kernel::trailSurvival(rate, 1.0F / 60.0F);
    }
    float fast = 1.0F;
    for (int step = 0; step < 240; ++step) {
        fast *= vkexp::trail::kernel::trailSurvival(rate, 1.0F / 240.0F);
    }
    require(std::abs(slow - fast) < 1.0e-4F, "Trail decay over one second ignores the step rate");
    require(std::abs(slow - 0.70710678F) < 1.0e-4F,
            "A two-second half-life leaves 1/sqrt(2) after one second");
}

void runGenomeAddressingProbe(vkexp::HeadlessComputeContext& context) {
    constexpr std::uint32_t genomeCount = 6;
    constexpr std::uint32_t trialsPerGenome = 2;
    constexpr std::uint32_t agentCount = genomeCount * trialsPerGenome;
    namespace kernel = vkexp::neuro::kernel;

    const vkexp::SimulationStep settings{};
    const vkexp::neuro::BrainShape brain = vkexp::scenarioDefinition(settings.beaconScenario).brain;
    const auto inputCount = static_cast<kernel::uint>(brain.inputCount);
    const auto hiddenCount = static_cast<kernel::uint>(brain.hiddenCount);
    const auto outputCount = static_cast<kernel::uint>(brain.outputCount);

    // Genome g biases both motors so that tanh(bias) is a value unique to g.
    std::vector<vkexp::neuro::Weights> genomes(genomeCount);
    std::array<float, genomeCount> expectedDrive{};
    for (std::uint32_t genome = 0; genome < genomeCount; ++genome) {
        const float bias = -1.0F + 0.4F * static_cast<float>(genome);
        for (const kernel::uint motor :
             {kernel::BrainMotorLeftOutput, kernel::BrainMotorRightOutput}) {
            genomes[genome][kernel::brainOutputBiasIndex(0u, inputCount, hiddenCount, outputCount,
                                                         motor)] = bias;
        }
        expectedDrive[genome] = std::tanh(bias);
    }

    StepHarness harness{context, agentCount, genomeCount, 1, 1, settings.worldRadius};
    harness.genomes.write(genomes.data(), genomes.size() * sizeof(vkexp::neuro::Weights));

    std::vector<vkexp::AgentState> agents(agentCount);
    for (std::uint32_t index = 0; index < agentCount; ++index) {
        agents[index].pose = {0.0F, 0.0F, 0.0F, vkexp::agentBodyRadius};
        agents[index].motion.w = 1.0F;
        agents[index].target = {settings.worldRadius * 0.7F, 0.0F,
                                static_cast<float>(index % trialsPerGenome), 0.0F};
        agents[index].metrics = {1.0F, 1.0F, 0.0F, 0.0F};
    }
    harness.buildGrid(agents, settings.worldRadius);
    harness.inputAgents.write(agents.data(), agents.size() * sizeof(vkexp::AgentState));
    vkexp::GpuStepParameters parameters = makeStepParameters(
        settings, agentCount, trialsPerGenome, harness.gridWidth(), harness.gridCellsPerWorld());
    parameters.agentCollisionsEnabled = 0;
    parameters.agentLightEnabled = 0;
    harness.stepParameters.write(&parameters, sizeof(parameters));
    harness.dispatch(0);
    harness.outputAgents.read(agents.data(), agents.size() * sizeof(vkexp::AgentState));

    for (std::uint32_t index = 0; index < agentCount; ++index) {
        const std::uint32_t genome = index / trialsPerGenome;
        // Straight-line drive for one step, before drag and the speed clamp:
        // both motors share the bias, so forward speed follows tanh(bias).
        const float drive = expectedDrive[genome] * settings.thrust * settings.deltaTime *
                            std::exp(-settings.linearDrag * settings.deltaTime);
        if (std::abs(agents[index].motion.x - drive) > 0.001F) {
            throw std::runtime_error("Genome addressing probe: agent " + std::to_string(index) +
                                     " should follow genome " + std::to_string(genome) +
                                     " (expected drive " + std::to_string(drive) + ", got " +
                                     std::to_string(agents[index].motion.x) +
                                     "); genomeStride or base offset is wrong");
        }
    }
}

// Many agents in one world exercise the shared spatial grid, where a race or an
// uninitialised read would show up as run-to-run variation rather than as a
// CPU/GPU mismatch.
void runMultiAgentDeterminism(vkexp::HeadlessComputeContext& context) {
    constexpr std::uint32_t agentCount = 192;
    constexpr std::uint32_t steps = 24;
    const vkexp::neuro::Weights weights = makeTestWeights(0.44F);
    vkexp::SimulationStep base{};
    base.beaconScenario = vkexp::BeaconScenario::Rotating;

    StepHarness harness{context, agentCount, 1, 1, 1, base.worldRadius};
    harness.genomes.write(weights.data(), sizeof(weights));

    const auto makeAgents = [&] {
        std::vector<vkexp::AgentState> agents(agentCount);
        for (std::uint32_t index = 0; index < agentCount; ++index) {
            const float angle = 2.39996323F * static_cast<float>(index);
            const float radius =
                base.worldRadius * 0.55F *
                std::sqrt((static_cast<float>(index) + 0.5F) / static_cast<float>(agentCount));
            agents[index].pose = {std::cos(angle) * radius, std::sin(angle) * radius, angle,
                                  vkexp::agentBodyRadius};
            agents[index].motion.w = 1.0F;
            agents[index].signal = {0.5F, 0.4F, 0.9F, 0.6F};
            agents[index].target = {base.worldRadius * 0.7F, 0.0F, 0.0F, 0.0F};
            agents[index].metrics = {1.0F, 1.0F, 0.0F, 0.0F};
        }
        return agents;
    };

    const auto runOnce = [&] {
        std::vector<vkexp::AgentState> agents = makeAgents();
        for (std::uint32_t step = 0; step < steps; ++step) {
            const vkexp::SimulationStep settings = vkexp::resolveStepSettings(base, step, steps);
            harness.buildGrid(agents, settings.worldRadius);
            harness.inputAgents.write(agents.data(), agents.size() * sizeof(vkexp::AgentState));
            const vkexp::GpuStepParameters parameters = makeStepParameters(
                settings, agentCount, 1, harness.gridWidth(), harness.gridCellsPerWorld());
            harness.stepParameters.write(&parameters, sizeof(parameters));
            harness.dispatch(0);
            harness.outputAgents.read(agents.data(), agents.size() * sizeof(vkexp::AgentState));
        }
        return agents;
    };

    const std::vector<vkexp::AgentState> first = runOnce();
    const std::vector<vkexp::AgentState> second = runOnce();
    for (std::size_t index = 0; index < first.size(); ++index) {
        if (std::memcmp(&first[index], &second[index], sizeof(vkexp::AgentState)) != 0) {
            throw std::runtime_error("Repeated GPU run diverged for agent " +
                                     std::to_string(index) +
                                     "; the spatial grid or step shader is not deterministic");
        }
    }
    const bool anyMoved = std::any_of(first.begin(), first.end(), [](const vkexp::AgentState& a) {
        return std::abs(a.motion.x) > 0.0F || std::abs(a.motion.y) > 0.0F;
    });
    if (!anyMoved) {
        throw std::runtime_error("Determinism run produced motionless agents; it proves nothing");
    }
}

// Single-step parity from a deliberately hostile state: at the world edge, over
// the speed limit, and mid phase change.
void runNeuralStepParity(
    vkexp::HeadlessComputeContext& context, const vkexp::WorldShape worldShape,
    const vkexp::BeaconScenario beaconScenario = vkexp::BeaconScenario::Stationary) {
    const vkexp::neuro::Weights weights = makeTestWeights();
    vkexp::AgentState initial{};
    initial.pose = {1.827F, 0.03F, 0.0F, 0.012F};
    initial.motion = {2.0F, -1.0F, 9.0F, 1.0F};
    initial.signal = {0.1F, 0.2F, 0.3F, 0.0F};
    initial.target = {0.62F, 0.41F, 0.0F, 0.0F};
    vkexp::SimulationStep settings{};
    settings.worldShape = worldShape;
    settings.beaconScenario = beaconScenario;
    if (beaconScenario == vkexp::BeaconScenario::Rotating) {
        settings.beaconRotationAngle = 0.73F;
    } else if (beaconScenario == vkexp::BeaconScenario::RandomMovement) {
        settings.beaconMotionTime = 4.2F;
        settings.beaconMotionSeed = 73U;
    } else if (beaconScenario == vkexp::BeaconScenario::ForageHome) {
        settings.beaconRotationAngle = 0.73F;
        settings.beaconMotionTime = vkexp::forageHomeRelocationSeconds;
        initial.internal = {0.8F, 1.0F, 0.2F, -0.3F};
        const vkexp::Float4 home = vkexp::homeBeaconPosition(initial, settings);
        initial.pose.x = home.x;
        initial.pose.y = home.y;
    }
    vkexp::SimulationStep initialSettings = settings;
    initialSettings.beaconPhase = 0;
    initialSettings.beaconRotationAngle = 0.0F;
    initialSettings.beaconMotionTime = 0.0F;
    const float initialDistance = vkexp::nearestBeaconDistance(initial, initialSettings);
    initial.metrics = {initialDistance, initialDistance, 0.0F, 0.0F};
    if (beaconScenario == vkexp::BeaconScenario::AlternatingDiagonals) {
        settings.beaconPhase = 1;
        settings.beaconPhaseChanged = true;
    }
    vkexp::AgentState expected = initial;
    vkexp::stepAgentCpu(expected, weights, settings);
    const float expectedSpeed =
        std::sqrt(expected.motion.x * expected.motion.x + expected.motion.y * expected.motion.y);
    if (expectedSpeed > settings.maximumSpeed + 0.0001F ||
        std::abs(expected.motion.z) > settings.maximumAngularSpeed + 0.0001F) {
        throw std::runtime_error("CPU agent speed limits were not applied");
    }
    const float maximumCenterDistance = settings.worldRadius - expected.pose.w + 0.0001F;
    const bool insideWorld =
        worldShape == vkexp::WorldShape::Circle
            ? std::hypot(expected.pose.x, expected.pose.y) <= maximumCenterDistance
            : std::abs(expected.pose.x) <= maximumCenterDistance &&
                  std::abs(expected.pose.y) <= maximumCenterDistance;
    if (!insideWorld) {
        throw std::runtime_error("CPU agent escaped the selected world boundary");
    }

    StepHarness harness{context, 1, 1, 1, 1, settings.worldRadius};
    harness.genomes.write(weights.data(), sizeof(weights));
    harness.inputAgents.write(&initial, sizeof(initial));
    const std::array<vkexp::AgentState, 1> agents{initial};
    harness.buildGrid(agents, settings.worldRadius);
    const vkexp::GpuStepParameters parameters =
        makeStepParameters(settings, 1, 1, harness.gridWidth(), harness.gridCellsPerWorld());
    harness.stepParameters.write(&parameters, sizeof(parameters));
    harness.dispatch(0);

    vkexp::AgentState actual{};
    harness.outputAgents.read(&actual, sizeof(actual));
    compareAgents(expected, actual, 0.0002F,
                  std::string{"Single-step parity ["} + scenarioName(beaconScenario) + "]");
}

// Two neighbouring agents: they must collide, feel each other, and see each
// other's light -- unless they belong to different logical worlds.
void runAgentInteractionTest(vkexp::HeadlessComputeContext& context, const bool isolatedWorlds) {
    constexpr std::uint32_t agentCount = 2;
    std::array<vkexp::AgentState, agentCount> initial{};
    initial[0].pose = {-0.01F, 0.0F, 0.0F, vkexp::agentBodyRadius};
    initial[1].pose = {0.01F, 0.0F, 3.14159265F, vkexp::agentBodyRadius};
    for (std::size_t index = 0; index < agentCount; ++index) {
        initial[index].motion.w = 1.0F;
        initial[index].target = {-1.0F, 0.0F, 0.0F, static_cast<float>(index)};
        initial[index].metrics = {1.0F, 1.0F, 0.0F, 0.0F};
        initial[index].penalties.w = isolatedWorlds ? static_cast<float>(index) : 0.0F;
    }
    initial[1].signal = {1.0F, 0.0F, 0.0F, 1.0F};

    // Sparse probe: wire the forward-facing red receptor straight through one
    // hidden neuron to the red emission output. Indices come from the shared
    // preset, so this keeps working when the sensor suite changes.
    namespace kernel = vkexp::neuro::kernel;
    vkexp::SimulationStep settings{};
    // What is under test is the sensor path -- does one agent's light reach
    // another's receptor and come back out of the network -- and it is read
    // after a single step. A hidden neuron with a time constant deliberately
    // cannot answer within one step, so memory is off here: this asks whether
    // the wiring carries the signal, not how fast a neuron follows it.
    settings.neuronModel = vkexp::NeuronModel::Reactive;
    const vkexp::neuro::BrainShape brain = vkexp::scenarioDefinition(settings.beaconScenario).brain;
    const auto inputCount = static_cast<kernel::uint>(brain.inputCount);
    const auto hiddenCount = static_cast<kernel::uint>(brain.hiddenCount);
    const kernel::uint centerReceptor = kernel::BrainLightReceptorCount / 2u;
    const kernel::uint centerRedInput = kernel::brainLightChannelIndex(centerReceptor, 0u);

    std::array<vkexp::neuro::Weights, agentCount> genomes{};
    genomes[0][kernel::brainHiddenWeightIndex(0u, inputCount, 0u, centerRedInput)] = 4.0F;
    genomes[0][kernel::brainOutputWeightIndex(0u, inputCount, hiddenCount,
                                              kernel::BrainSignalColorOutput, 0u)] = 4.0F;

    const std::uint32_t worldCount = isolatedWorlds ? 2U : 1U;
    StepHarness harness{context, agentCount, agentCount, worldCount, 1, settings.worldRadius};
    harness.genomes.write(genomes.data(), sizeof(genomes));
    harness.inputAgents.write(initial.data(), sizeof(initial));
    harness.buildGrid(initial, settings.worldRadius);
    const vkexp::GpuStepParameters parameters = makeStepParameters(
        settings, agentCount, 1, harness.gridWidth(), harness.gridCellsPerWorld());
    harness.stepParameters.write(&parameters, sizeof(parameters));
    harness.dispatch(0);

    std::array<vkexp::AgentState, agentCount> result{};
    harness.outputAgents.read(result.data(), sizeof(result));
    const float initialDistance = initial[1].pose.x - initial[0].pose.x;
    const float resultDistance = result[1].pose.x - result[0].pose.x;
    const float firstTouch =
        std::max({result[0].agentTouch0.x, result[0].agentTouch0.y, result[0].agentTouch0.z,
                  result[0].agentTouch0.w, result[0].agentTouch1.x, result[0].agentTouch1.y,
                  result[0].agentTouch1.z, result[0].agentTouch1.w});
    if (isolatedWorlds) {
        if (std::abs(resultDistance - initialDistance) > 0.0001F || firstTouch > 0.0F) {
            throw std::runtime_error("GPU agents interacted across logical world boundaries");
        }
        if (result[0].signal.x > 0.55F) {
            throw std::runtime_error("GPU photoreceptor observed light from another world");
        }
    } else {
        if (resultDistance <= initialDistance || firstTouch <= 0.0F) {
            throw std::runtime_error("GPU agents did not separate and report tactile contact");
        }
        if (result[0].signal.x <= 0.55F) {
            throw std::runtime_error("GPU photoreceptor did not observe another agent's red light");
        }
    }
}

int run() {
    vkexp::HeadlessComputeContext context{{"vkexp compute smoke"}};
    runGameOfLife(context);
    runImageRoundTrip(context);
    runNeuralStepParity(context, vkexp::WorldShape::Circle);
    runNeuralStepParity(context, vkexp::WorldShape::Square);
    runNeuralStepParity(context, vkexp::WorldShape::Circle,
                        vkexp::BeaconScenario::AlternatingDiagonals);
    runNeuralStepParity(context, vkexp::WorldShape::Circle, vkexp::BeaconScenario::Rotating);
    runNeuralStepParity(context, vkexp::WorldShape::Circle, vkexp::BeaconScenario::RandomMovement);
    runNeuralStepParity(context, vkexp::WorldShape::Circle, vkexp::BeaconScenario::ForageHome);
    runNeuralStepParity(context, vkexp::WorldShape::Circle, vkexp::BeaconScenario::ScentRelay);
    // The one scenario with static geometry, so the one where the contact
    // response exists twice -- in stepAgentCpu and in agent_step.comp. That is
    // exactly the duplication parity is for.
    runNeuralStepParity(context, vkexp::WorldShape::Circle, vkexp::BeaconScenario::TwoDoors);
    runNeuralStepParity(context, vkexp::WorldShape::Circle, vkexp::BeaconScenario::Shuttle);
    runAgentInteractionTest(context, false);
    runAgentInteractionTest(context, true);
    // 540 steps at the default 1/60 s covers 9 simulated seconds, so the forage
    // home relocation at 8 s and the alternating phase flip both fall inside.
    runTrajectoryParity(context, vkexp::BeaconScenario::Stationary, 540);
    runTrajectoryParity(context, vkexp::BeaconScenario::AlternatingDiagonals, 540);
    runTrajectoryParity(context, vkexp::BeaconScenario::Rotating, 540);
    runTrajectoryParity(context, vkexp::BeaconScenario::RandomMovement, 540);
    runTrajectoryParity(context, vkexp::BeaconScenario::ForageHome, 540);
    runTrajectoryParity(context, vkexp::BeaconScenario::ScentRelay, 540);
    runTrajectoryParity(context, vkexp::BeaconScenario::TwoDoors, 540);
    runTrajectoryParity(context, vkexp::BeaconScenario::Shuttle, 540);
    // Every neuron model over a long run, because the model decides the hidden
    // state and the hidden state is the thing that accumulates: a gate loop that
    // disagrees between the two languages shows up as drift and nowhere else.
    // The gated one matters most -- it is the newest arithmetic written twice.
    runTrajectoryParity(context, vkexp::BeaconScenario::Shuttle, 540,
                        vkexp::NeuronModel::Reactive);
    runTrajectoryParity(context, vkexp::BeaconScenario::Shuttle, 540, vkexp::NeuronModel::Gated);
    runTrajectoryParity(context, vkexp::BeaconScenario::TwoDoors, 540, vkexp::NeuronModel::Gated);
    runGenomeAddressingProbe(context);
    runTrailFieldProbe(context);
    runMultiAgentDeterminism(context);
    std::cout << "Headless compute, CPU/GPU parity and determinism tests passed on "
              << context.deviceName() << '\n';
    return 0;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const vkexp::HeadlessComputeUnavailable& error) {
        std::cout << "SKIPPED: " << error.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
