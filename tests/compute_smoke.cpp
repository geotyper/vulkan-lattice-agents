#include "vkexp/compute/HeadlessComputeContext.hpp"
#include "vkexp/neuro/BrainKernel.hpp"
#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/lattice/LatticeKernel.hpp"
#include "vkexp/simulation/LatticeBindings.hpp"
#include "vkexp/lattice/LatticeWorld.hpp"
#include "vkexp/simulation/CpuLattice.hpp"
#include "vkexp/simulation/LatticeTypes.hpp"
#include "vkexp/simulation/StepParameters.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <bit>
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

// --- lattice parity ----------------------------------------------------------
//
// The whole reason LatticeKernel.inl is written in a two-language common subset:
// the movement rule, the neighbour numbering and the arbitration are compiled
// once and run twice, and this is what proves the two runs agree.
//
// The 2D build's parity tests compared one agent against one wall, and could,
// because a step was a per-agent function. A lattice step is not: which cell an
// agent ends up in depends on who else asked for it. So every case here runs a
// whole population, and the interesting ones are deliberately crowded -- parity
// on a lattice nobody contests would prove only that tanh is deterministic.

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] std::vector<float> makeWeights(const vkexp::neuro::BrainShape& brain,
                                             const std::uint32_t genomeCount,
                                             const std::uint32_t seed) {
    // A fixed generator rather than std::mt19937, so the numbers are the same on
    // every standard library. Range is deliberately wide: saturated tanh outputs
    // are where a drive most often sits exactly on the dead zone, and a
    // threshold comparison that disagreed by one ulp would show up there first.
    std::vector<float> weights(static_cast<std::size_t>(brain.weightCount()) * genomeCount);
    std::uint32_t state = seed | 1U;
    for (float& weight : weights) {
        state = state * 1664525U + 1013904223U;
        weight = (static_cast<float>(state >> 8U) / static_cast<float>(1U << 24U)) * 4.0F - 2.0F;
    }
    return weights;
}

// One step of the lattice on the device, wired up exactly as SimulationDriver
// does: the same three pipelines, the same descriptor layouts, the same order
// and the same barriers. Anything this harness does differently is a way for the
// test to pass while the driver is broken.
class LatticeHarness {
public:
    // See the note beside the buffer: more than one, so the step index the
    // shaders are pushed is not the one index a layout mistake survives.
    static constexpr std::uint32_t parameterSlots = 2;

    LatticeHarness(vkexp::HeadlessComputeContext& context, const vkexp::SimulationStep& settings,
                   const vkexp::lattice::PopulationLayout& layout,
                   const std::span<const float> weights)
        : context_(context), settings_(settings), layout_(layout) {
        const auto agentCount = layout.agentCount();
        const VkDeviceSize agentBytes = sizeof(vkexp::AgentState) * agentCount;
        cellCount_ = vkexp::latticeCellsPerWorld(settings) * layout.worldCount();
        const VkDeviceSize gridBytes = sizeof(std::int32_t) * cellCount_;

        agents_.create(context.physicalDevice(), context.device(),
                       {agentBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
        genome_.create(context.physicalDevice(), context.device(),
                       {weights.size_bytes(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
        occupancy_.create(context.physicalDevice(), context.device(),
                          {gridBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
        claims_.create(context.physicalDevice(), context.device(),
                       {gridBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
        structures_.create(context.physicalDevice(), context.device(),
                           {gridBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
        const VkDeviceSize outcomeBytes =
            static_cast<VkDeviceSize>(layout.worldCount()) *
            vkexp::lattice::kernel::LatticeBuildOutcomeCount * sizeof(std::uint32_t);
        buildOutcomes_.create(context.physicalDevice(), context.device(),
                              {outcomeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
        // Two slots, and every dispatch below reads the second one. A driver
        // batches many steps into one submission and indexes this buffer by the
        // step, so slot zero is the only slot a harness that records one step
        // would ever touch -- and slot zero is exactly where a stride that
        // disagrees between C++ and std430 still very nearly works. Reading
        // slot one makes the disagreement a failed parity case instead of a
        // run that quietly gets better the fewer steps it batches.
        parameters_.create(context.physicalDevice(), context.device(),
                           {sizeof(vkexp::GpuStepParameters) * parameterSlots,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
        genome_.write(weights.data(), weights.size_bytes());

        const vkexp::GpuStepParameters packed = vkexp::packStepParameters(
            settings, {.agentCount = agentCount,
                       .trialsPerGenome = layout.trialsPerGenome,
                       .agentsPerWorld = layout.groupSize(),
                       .worldCount = layout.worldCount()});
        const std::array<vkexp::GpuStepParameters, parameterSlots> slots{packed, packed};
        parameters_.write(slots.data(), sizeof(slots));

        createLayout(vkexp::latticeStepBindings, stepLayout_);
        createLayout(vkexp::latticeResolveBindings, resolveLayout_);
        createLayout(vkexp::latticeClearBindings, clearLayout_);
        descriptors_.create(context.device(), {6, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 36}}});

        for (std::uint32_t readIndex = 0; readIndex < 2; ++readIndex) {
            const VkBuffer readBuffer =
                readIndex == 0 ? agents_.read().buffer() : agents_.write().buffer();
            const VkBuffer writeBuffer =
                readIndex == 0 ? agents_.write().buffer() : agents_.read().buffer();
            stepSets_[readIndex] = descriptors_.allocate(stepLayout_.get());
            vkexp::DescriptorSetWriter{}
                .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, readBuffer, 0, agentBytes)
                .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, writeBuffer, 0, agentBytes)
                .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, genome_.buffer(), 0,
                             genome_.size())
                .writeBuffer(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, occupancy_.buffer(), 0,
                             occupancy_.size())
                .writeBuffer(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, claims_.buffer(), 0,
                             claims_.size())
                .writeBuffer(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, parameters_.buffer(), 0,
                             parameters_.size())
                .writeBuffer(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, structures_.buffer(), 0,
                             structures_.size())
                .writeBuffer(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buildOutcomes_.buffer(), 0,
                             buildOutcomes_.size())
                .update(context.device(), stepSets_[readIndex]);

            resolveSets_[readIndex] = descriptors_.allocate(resolveLayout_.get());
            vkexp::DescriptorSetWriter{}
                .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, writeBuffer, 0, agentBytes)
                .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, occupancy_.buffer(), 0,
                             occupancy_.size())
                .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, claims_.buffer(), 0,
                             claims_.size())
                .writeBuffer(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, parameters_.buffer(), 0,
                             parameters_.size())
                .writeBuffer(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, structures_.buffer(), 0,
                             structures_.size())
                .writeBuffer(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buildOutcomes_.buffer(), 0,
                             buildOutcomes_.size())
                .update(context.device(), resolveSets_[readIndex]);
        }
        clearSet_ = descriptors_.allocate(clearLayout_.get());
        vkexp::DescriptorSetWriter{}
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, claims_.buffer(), 0, claims_.size())
            .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, parameters_.buffer(), 0,
                         parameters_.size())
            .update(context.device(), clearSet_);

        stepPipeline_ = vkexp::ComputePipelineBuilder{context.physicalDevice(), context.device()}
                            .shader(VKEXP_SHADER_DIR "/lattice_step.comp.spv")
                            .addDescriptorSetLayout(stepLayout_.get())
                            .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(std::uint32_t))
                            .build();
        resolvePipeline_ =
            vkexp::ComputePipelineBuilder{context.physicalDevice(), context.device()}
                .shader(VKEXP_SHADER_DIR "/lattice_resolve.comp.spv")
                .addDescriptorSetLayout(resolveLayout_.get())
                .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(std::uint32_t))
                .build();
        clearPipeline_ =
            vkexp::ComputePipelineBuilder{context.physicalDevice(), context.device()}
                .shader(VKEXP_SHADER_DIR "/lattice_clear.comp.spv")
                .addDescriptorSetLayout(clearLayout_.get())
                .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(std::uint32_t))
                .build();
    }

    void upload(const std::span<const vkexp::AgentState> agents,
                const std::span<const std::int32_t> occupancy) {
        agents_.read().write(agents.data(), agents.size_bytes());
        agents_.write().write(agents.data(), agents.size_bytes());
        occupancy_.write(occupancy.data(), occupancy.size_bytes());
        const std::vector<std::int32_t> empty(cellCount_, vkexp::lattice::kernel::LatticeNoStructure);
        structures_.write(empty.data(), empty.size() * sizeof(std::int32_t));
        const std::vector<std::uint32_t> outcomes(
            static_cast<std::size_t>(layout_.worldCount()) *
                vkexp::lattice::kernel::LatticeBuildOutcomeCount,
            0U);
        buildOutcomes_.write(outcomes.data(), outcomes.size() * sizeof(std::uint32_t));
    }

    void step() {
        context_.immediate().execute([&](const VkCommandBuffer commands) {
            const std::uint32_t readIndex = agents_.readIndex();
            const std::uint32_t stepIndex = parameterSlots - 1;
            const std::array<VkDeviceSize, 2> clearRanges{claims_.size(), parameters_.size()};
            const vkexp::DispatchSize clearGroups = vkexp::checkedDispatchSize(
                context_.physicalDevice(),
                {{cellCount_, 1, 1}, {256, 1, 1}, sizeof(std::uint32_t), clearRanges});
            vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, clearPipeline_.pipeline());
            vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    clearPipeline_.layout(), 0, 1, &clearSet_, 0, nullptr);
            vkCmdPushConstants(commands, clearPipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(stepIndex), &stepIndex);
            vkCmdDispatch(commands, clearGroups.x, 1, 1);
            vkexp::cmdComputeWriteToComputeRead(commands, claims_.buffer());

            const std::array<VkDeviceSize, 6> stepRanges{
                agents_.read().size(), agents_.write().size(), genome_.size(),
                occupancy_.size(),     claims_.size(),         parameters_.size()};
            const vkexp::DispatchSize stepGroups = vkexp::checkedDispatchSize(
                context_.physicalDevice(),
                {{layout_.agentCount(), 1, 1}, {64, 1, 1}, sizeof(std::uint32_t), stepRanges});
            vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, stepPipeline_.pipeline());
            vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    stepPipeline_.layout(), 0, 1, &stepSets_[readIndex], 0,
                                    nullptr);
            vkCmdPushConstants(commands, stepPipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(stepIndex), &stepIndex);
            vkCmdDispatch(commands, stepGroups.x, 1, 1);
            vkexp::cmdComputeWriteToComputeRead(commands, claims_.buffer());
            vkexp::cmdComputeWriteToComputeRead(commands, agents_.write().buffer());

            vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                              resolvePipeline_.pipeline());
            vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    resolvePipeline_.layout(), 0, 1, &resolveSets_[readIndex], 0,
                                    nullptr);
            vkCmdPushConstants(commands, resolvePipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(stepIndex), &stepIndex);
            vkCmdDispatch(commands, stepGroups.x, 1, 1);
            vkexp::cmdComputeWriteToComputeRead(commands, agents_.write().buffer());
            vkexp::cmdComputeWriteToComputeRead(commands, occupancy_.buffer());
        });
        context_.waitIdle();
        agents_.swap();
    }

    [[nodiscard]] std::vector<vkexp::AgentState> readAgents() {
        std::vector<vkexp::AgentState> result(layout_.agentCount());
        agents_.read().read(result.data(), result.size() * sizeof(vkexp::AgentState));
        return result;
    }

    [[nodiscard]] std::vector<std::int32_t> readOccupancy() {
        std::vector<std::int32_t> result(cellCount_);
        occupancy_.read(result.data(), result.size() * sizeof(std::int32_t));
        return result;
    }

    // Separate from upload(), which always clears the block field: a
    // construction step is the one case where what the previous step built has
    // to survive into the next one.
    void uploadStructures(const std::span<const std::int32_t> structures) {
        structures_.write(structures.data(), structures.size_bytes());
    }

    [[nodiscard]] std::vector<std::int32_t> readStructures() {
        std::vector<std::int32_t> result(cellCount_);
        structures_.read(result.data(), result.size() * sizeof(std::int32_t));
        return result;
    }

    [[nodiscard]] std::vector<std::uint32_t> readBuildOutcomes() {
        std::vector<std::uint32_t> result(static_cast<std::size_t>(layout_.worldCount()) *
                                          vkexp::lattice::kernel::LatticeBuildOutcomeCount);
        buildOutcomes_.read(result.data(), result.size() * sizeof(std::uint32_t));
        return result;
    }

private:
    void createLayout(const std::uint32_t bindingCount, vkexp::UniqueDescriptorSetLayout& layout) {
        std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount);
        for (std::uint32_t binding = 0; binding < bindingCount; ++binding) {
            bindings[binding].binding = binding;
            bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[binding].descriptorCount = 1;
            bindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = bindingCount;
        info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(context_.device(), &info, nullptr,
                                        layout.put(context_.device())) != VK_SUCCESS) {
            throw std::runtime_error("Unable to create a lattice parity descriptor layout");
        }
    }

    vkexp::HeadlessComputeContext& context_;
    vkexp::SimulationStep settings_;
    vkexp::lattice::PopulationLayout layout_;
    std::uint32_t cellCount_{};
    vkexp::PingPongBuffer agents_;
    vkexp::BufferResource genome_;
    vkexp::BufferResource occupancy_;
    vkexp::BufferResource claims_;
    // Bound even in the beacon world, where no pass reads them. A descriptor the
    // shader declares and the harness leaves unbound is undefined behaviour
    // rather than an unused slot, and the parity case that follows would be
    // measuring whatever the last test left in memory.
    vkexp::BufferResource structures_;
    vkexp::BufferResource buildOutcomes_;
    vkexp::BufferResource parameters_;
    vkexp::UniqueDescriptorSetLayout stepLayout_;
    vkexp::UniqueDescriptorSetLayout resolveLayout_;
    vkexp::UniqueDescriptorSetLayout clearLayout_;
    vkexp::DescriptorAllocator descriptors_;
    std::array<VkDescriptorSet, 2> stepSets_{};
    std::array<VkDescriptorSet, 2> resolveSets_{};
    VkDescriptorSet clearSet_{};
    vkexp::ComputePipeline stepPipeline_;
    vkexp::ComputePipeline resolvePipeline_;
    vkexp::ComputePipeline clearPipeline_;
};

// Every float the two sides both compute, summed signed across a run. Lockstep
// comparison alone cannot see a systematic bias smaller than the per-step
// tolerance, because resynchronising every step stops it accumulating: summing
// the signed differences restores it, since symmetric rounding noise cancels and
// a constant offset in a shader term does not.
using AgentDrift = std::array<double, 5 + vkexp::agentHiddenVectorCount * 4>;
constexpr double accumulatedDriftBudget = 1.0e-3;

// Cells and occupancy are compared exactly. They are integers, and the whole
// point of the arbitration rule is that both sides reach the same one: a
// tolerance here would hide the only failure that matters, which is two
// implementations putting an agent in different cells.
void compareAgents(const vkexp::AgentState& expected, const vkexp::AgentState& actual,
                   const std::string& context, const float tolerance, AgentDrift* drift = nullptr) {
    std::size_t slot = 0;
    const auto same = [&](const float left, const float right, const char* field) {
        if (drift != nullptr) {
            (*drift)[slot] += static_cast<double>(right) - static_cast<double>(left);
        }
        ++slot;
        if (std::abs(left - right) > tolerance) {
            throw std::runtime_error(context + ": " + field + " drifted, CPU " +
                                     std::to_string(left) + " vs GPU " + std::to_string(right));
        }
    };
    const auto identical = [&](const std::int32_t left, const std::int32_t right,
                               const char* field) {
        if (left != right) {
            throw std::runtime_error(context + ": " + field + " differs, CPU " +
                                     std::to_string(left) + " vs GPU " + std::to_string(right));
        }
    };
    identical(expected.cell.x, actual.cell.x, "cell.x");
    identical(expected.cell.y, actual.cell.y, "cell.y");
    identical(expected.cell.z, actual.cell.z, "cell.z");
    identical(expected.cell.w, actual.cell.w, "heading");
    identical(expected.intent.x, actual.intent.x, "intent.x");
    identical(expected.intent.y, actual.intent.y, "intent.y");
    identical(expected.intent.z, actual.intent.z, "intent.z");
    identical(expected.intent.w, actual.intent.w, "refusal flag");
    same(expected.signal.x, actual.signal.x, "broadcast");
    same(expected.metrics.x, actual.metrics.x, "best nearness");
    same(expected.metrics.y, actual.metrics.y, "contacts");
    same(expected.metrics.z, actual.metrics.z, "effort");
    same(expected.metrics.w, actual.metrics.w, "refusals");
    // The recurrent cells and the still-tick counter, neither of which was
    // compared before. The cells drift like any other float; the counter is a
    // whole number, so any difference at all in it is a difference of logic and
    // lands far outside the tolerance rather than inside it.
    same(expected.memory.x, actual.memory.x, "memory cell 1");
    same(expected.memory.y, actual.memory.y, "memory cell 2");
    same(expected.memory.z, actual.memory.z, "still ticks");
    for (std::size_t index = 0; index < expected.hidden.size(); ++index) {
        same(expected.hidden[index].x, actual.hidden[index].x, "hidden.x");
        same(expected.hidden[index].y, actual.hidden[index].y, "hidden.y");
        same(expected.hidden[index].z, actual.hidden[index].z, "hidden.z");
        same(expected.hidden[index].w, actual.hidden[index].w, "hidden.w");
    }
}

[[nodiscard]] double worstDrift(const AgentDrift& drift) {
    return *std::max_element(drift.begin(), drift.end(), [](const double left, const double right) {
        return std::abs(left) < std::abs(right);
    });
}

[[nodiscard]] const char* neighborhoodName(const vkexp::Neighborhood neighborhood) {
    return neighborhood == vkexp::Neighborhood::Faces ? "faces" : "moore";
}

[[nodiscard]] const char* neuronModelName(const vkexp::NeuronModel model) {
    switch (model) {
    case vkexp::NeuronModel::Reactive:
        return "reactive";
    case vkexp::NeuronModel::TimeConstant:
        return "time";
    case vkexp::NeuronModel::Gated:
        return "gated";
    case vkexp::NeuronModel::Spiking:
        return "spiking";
    }
    return "?";
}

// A lattice small enough that twelve agents crowd it. Contention is the thing
// under test, so the density is the test fixture: at 32x32x16 the same twelve
// agents would almost never meet, and every case would pass whatever the
// arbitration did.
[[nodiscard]] vkexp::SimulationStep paritySettings(const vkexp::Neighborhood neighborhood,
                                                   const vkexp::NeuronModel model) {
    vkexp::SimulationStep settings;
    settings.latticeWidth = 4;
    settings.latticeHeight = 4;
    settings.latticeDepth = 4;
    settings.neighborhood = neighborhood;
    settings.neuronModel = model;
    return settings;
}

// One step of the lattice, compared step for step, with the device resynchronised
// to the reference at the top of every step.
//
// Lockstep rather than two free runs, for a reason particular to a lattice: a
// float difference of one ulp in a drive that sits on the dead zone is the
// difference between moving and standing still, and one differing move puts the
// two populations in different cells for the rest of the run. That is a real
// property worth knowing -- the discrete outcome is not continuous in the
// arithmetic -- but measuring it is measuring chaos, not agreement. So each step
// starts from one state and the step itself is what is compared, and the
// accumulated signed drift below is what catches a systematic bias that the
// per-step tolerance is too loose to see.
void runLatticeTrajectoryParity(vkexp::HeadlessComputeContext& context,
                                const vkexp::Neighborhood neighborhood,
                                const vkexp::NeuronModel model, const std::uint32_t steps) {
    const vkexp::SimulationStep settings = paritySettings(neighborhood, model);
    const vkexp::lattice::PopulationLayout layout{12, 10, 2};
    const vkexp::neuro::BrainShape brain = vkexp::resolvedBrain(settings);
    const std::vector<float> weights = makeWeights(brain, layout.genomeCount, 0x9E37U);

    std::vector<vkexp::AgentState> expected = vkexp::lattice::makeInitialAgents(settings, layout);
    const std::uint32_t cells = vkexp::latticeCellsPerWorld(settings);
    std::vector<std::int32_t> occupancy(static_cast<std::size_t>(cells) * layout.worldCount());
    vkexp::lattice::buildOccupancy(expected, settings, layout, occupancy);
    std::vector<std::int32_t> claims(occupancy.size());

    LatticeHarness harness{context, settings, layout, weights};

    const std::string label = std::string{"Trajectory parity ["} + neighborhoodName(neighborhood) +
                              ", " + neuronModelName(model) + "]";
    AgentDrift drift{};
    for (std::uint32_t step = 0; step < steps; ++step) {
        harness.upload(expected, occupancy);
        harness.step();
        const std::vector<vkexp::AgentState> actual = harness.readAgents();
        const std::vector<std::int32_t> actualOccupancy = harness.readOccupancy();

        vkexp::stepLatticeCpu({expected, occupancy, claims, weights,
                               static_cast<std::uint32_t>(brain.weightCount()),
                               layout.groupSize(), layout.trialsPerGenome},
                              settings);
        for (std::size_t index = 0; index < expected.size(); ++index) {
            compareAgents(expected[index], actual[index],
                          label + " step " + std::to_string(step) + " agent " +
                              std::to_string(index),
                          4.0e-4F, &drift);
        }
        require(actualOccupancy == occupancy,
                label + ": the occupancy grids diverged at step " + std::to_string(step));
    }

    std::uint32_t movers = 0;
    std::uint32_t refused = 0;
    for (const vkexp::AgentState& agent : expected) {
        movers += agent.metrics.z > 0.0F ? 1U : 0U;
        refused += agent.metrics.w > 0.0F ? 1U : 0U;
    }
    // A run in which nobody ever moved, or nobody was ever refused, would agree
    // perfectly and prove nothing. Both halves of the rule have to have fired.
    require(movers > 0, label + ": no agent ever moved, so the movement rule was never exercised");
    require(refused > 0,
            label + ": no agent was ever refused, so the arbitration was never exercised");

    const double worst = worstDrift(drift);
    std::cout << "  drift" << label.substr(label.find('[')) << " = " << worst << '\n';
    require(std::abs(worst) <= accumulatedDriftBudget,
            label + " accumulated a systematic CPU/GPU drift of " + std::to_string(worst) +
                " over " + std::to_string(steps) + " steps (budget " +
                std::to_string(accumulatedDriftBudget) + ")");
}

// Two agents, one free cell between them, and nothing else in the world. The
// lower-numbered one must win on both sides -- not because that is a nice rule,
// but because it is the only one a GPU can be held to: the winner is a minimum
// over indices, so it does not depend on which invocation reached the cell
// first. This is the case a spec would be written against, so it is tested
// directly rather than only inside a trajectory.
void runContentionProbe(vkexp::HeadlessComputeContext& context) {
    vkexp::SimulationStep settings = paritySettings(vkexp::Neighborhood::Moore,
                                                    vkexp::NeuronModel::Reactive);
    // One agent drives hard toward +x and the other toward -x, so both ask for
    // the cell between them rather than whatever their weights would otherwise
    // have chosen. A large bias on the x-drive output and nothing else: tanh
    // saturates, which clears any dead zone, and the other two axes stay at zero.
    const vkexp::lattice::PopulationLayout layout{2, 2, 1};
    const vkexp::neuro::BrainShape brain = vkexp::resolvedBrain(settings);
    std::vector<float> weights(static_cast<std::size_t>(brain.weightCount()) * layout.genomeCount,
                               0.0F);
    const std::size_t moveBias = vkexp::neuro::kernel::brainOutputBiasIndex(
        0U, static_cast<std::uint32_t>(brain.inputCount), brain.packedLayers(),
        static_cast<std::uint32_t>(brain.outputCount), vkexp::neuro::kernel::BrainMoveOutput);
    weights[moveBias] = 8.0F;
    weights[static_cast<std::size_t>(brain.weightCount()) + moveBias] = -8.0F;

    const std::uint32_t cells = vkexp::latticeCellsPerWorld(settings);
    std::vector<vkexp::AgentState> agents(layout.agentCount());
    const vkexp::Int4 beacon = vkexp::lattice::beaconCell(settings, 0);
    for (std::uint32_t index = 0; index < agents.size(); ++index) {
        agents[index].beacon = beacon;
        agents[index].cell.w =
            static_cast<std::int32_t>(vkexp::lattice::kernel::LatticeNeighborCount);
    }
    // Agent 0 at x=0 and agent 1 at x=2, both on the same row: the cell at x=1
    // is the one they both want, and the one that is free.
    agents[0].cell = {0, 2, 2, agents[0].cell.w};
    agents[1].cell = {2, 2, 2, agents[1].cell.w};
    for (vkexp::AgentState& agent : agents) {
        agent.intent = {agent.cell.x, agent.cell.y, agent.cell.z, 0};
    }

    std::vector<std::int32_t> occupancy(static_cast<std::size_t>(cells) * layout.worldCount());
    vkexp::lattice::buildOccupancy(agents, settings, layout, occupancy);
    std::vector<std::int32_t> claims(occupancy.size());

    LatticeHarness harness{context, settings, layout, weights};
    harness.upload(agents, occupancy);

    std::vector<vkexp::AgentState> expected = agents;
    vkexp::stepLatticeCpu({expected, occupancy, claims, weights,
                           static_cast<std::uint32_t>(brain.weightCount()),
                           layout.groupSize(), layout.trialsPerGenome},
                          settings);
    harness.step();
    const std::vector<vkexp::AgentState> actual = harness.readAgents();

    require(expected[0].cell.x == 1,
            "Contention: the lower-numbered agent should have taken the contested cell, it is at x="
            + std::to_string(expected[0].cell.x));
    require(expected[1].cell.x == 2,
            "Contention: the higher-numbered agent should have stayed, it is at x=" +
                std::to_string(expected[1].cell.x));
    require(expected[1].metrics.w == 1.0F, "Contention: the loser was not charged a refusal");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        compareAgents(expected[index], actual[index],
                      "Contention agent " + std::to_string(index), 2.0e-3F);
    }
}

// A run whose lattice is full: every cell of a small world occupied, so no move
// can ever succeed. The interesting part is that both sides say so in the same
// way -- every agent refused, nobody moved, and the occupancy grid unchanged.
// Construction, which the probes above never reach: a different set of move
// rules, a different fitness and -- the reason this exists now -- a build gate
// that asks how full the cells around a site are. That question is written
// twice, once in C++ and once in GLSL, and nothing else in this file would
// notice the two answers drifting apart.
void runConstructionParityProbe(vkexp::HeadlessComputeContext& context,
                                const vkexp::WorldMode worldMode) {
    vkexp::SimulationStep settings =
        paritySettings(vkexp::Neighborhood::Moore, vkexp::NeuronModel::TimeConstant);
    settings.worldMode = worldMode;
    // The resource sits one level up and counts as reached from anywhere in a
    // 4x4x4 box. Both are deliberate: a probe where nobody ever picks up a load
    // compares the harvest rules without running them, and where the resource
    // lands is a hash this fixture does not get to choose.
    settings.resourceHeightLow = 1;
    settings.resourceHeightHigh = 1;
    settings.beaconContactRadius = 4;
    // Build often and on almost any signal: what is being compared is placement,
    // and a probe where nobody happens to build compares nothing. The final
    // check below refuses to pass if that is what happened.
    settings.buildIntervalTicks = 2;
    settings.buildThreshold = -1.0F;
    settings.allowSideSupportedBlocks = 1;
    // Tuned so the foundation rule actually refuses something in a box four
    // cells high, which the shipped defaults cannot: bedrock fills the bottom
    // course, so the foundation is never lower than one, and five levels of
    // headroom above that is more than this box has. A demanding fill and one
    // level of lead put the refusal where the group can reach it -- a block may
    // go on the platform, and nothing may go on top of that.
    //
    // This is a test of agreement, not of meaning: it asks whether both
    // implementations refuse the same attempts and file them under the same
    // outcome. What the fill and the radius mean is testConstructionLocalFoundation.
    settings.constructionSupportRadius = 1;
    settings.constructionCourseFill = 0.9F;
    settings.constructionHeightLead = 1;

    const vkexp::lattice::PopulationLayout layout{4, 4, 1};
    const vkexp::neuro::BrainShape brain = vkexp::resolvedBrain(settings);
    const std::vector<float> weights = makeWeights(brain, layout.genomeCount, 0xB10CU);

    std::vector<vkexp::AgentState> expected = vkexp::lattice::makeInitialAgents(settings, layout);
    const std::uint32_t cells = vkexp::latticeCellsPerWorld(settings);
    std::vector<std::int32_t> claims(static_cast<std::size_t>(cells) * layout.worldCount());
    std::vector<std::int32_t> structures;

    // The terrain this world would really start from: a bedrock course under
    // every column with ground, and nothing under the rest. Building worlds no
    // longer assume a floor, so a fixture that skipped this would stand its
    // agents on nothing and compare two different kinds of falling.
    structures = vkexp::lattice::makeTerrain(settings, layout.worldCount());
    // A block on the ground for the group to start beside, so that the very
    // first step already has something to climb, stand on and build against --
    // otherwise the probe spends its whole length comparing agents that have not
    // yet found anything to do.
    constexpr int platform = 2;
    for (int z = 0; z < platform; ++z) {
        for (int x = 0; x < platform; ++x) {
            structures[vkexp::lattice::kernel::latticeCellIndex(x, 1, z, settings.latticeWidth,
                                                                settings.latticeHeight)] = 1;
        }
    }
    // Where the group can actually stand. In a chasm only the first half of the
    // columns has bedrock, so the group goes there and the platform sits beside
    // it along z; everywhere else the floor is whole and it sits beside it
    // along x. Standing anybody over the void would make this a probe about
    // falling rather than about building.
    const bool chasm = worldMode == vkexp::WorldMode::Chasm;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        vkexp::AgentState& agent = expected[index];
        const std::int32_t row = static_cast<std::int32_t>(index) % platform;
        const std::int32_t column = static_cast<std::int32_t>(index) / platform;
        agent.cell.x = chasm ? row : row;
        agent.cell.z = chasm ? column + platform : column;
        // On top of the platform in the worlds that have a frontier, beside the
        // ground in the one that does not. Height is the point: bedrock fills
        // the bottom course, so the foundation is never lower than one, and in a
        // box four high a group standing at height one can never build far
        // enough above it to meet the rule at all.
        agent.cell.y = chasm ? 1 : platform;
        agent.intent = {agent.cell.x, agent.cell.y, agent.cell.z, 0};
    }
    std::vector<std::int32_t> occupancy(claims.size());
    vkexp::lattice::buildOccupancy(expected, settings, layout, occupancy);

    // upload() clears the outcome counters, so both sides are compared one step
    // at a time rather than as a running total -- which is the stronger check:
    // two attributions that differ and then differ back would cancel in a sum.
    std::vector<std::uint32_t> outcomes(
        static_cast<std::size_t>(layout.worldCount()) *
        vkexp::lattice::kernel::LatticeBuildOutcomeCount);
    static constexpr std::array<const char*, 10> reasonNames{
        "cooling",         "unwilling", "no facing",  "off the lattice", "blocked",
        "unsupported",     "above the frontier",      "in the way",      "placed",
        "contested"};

    std::vector<std::uint64_t> frontierRefusals(layout.worldCount(), 0);
    LatticeHarness harness{context, settings, layout, weights};
    for (std::uint32_t step = 0; step < 48; ++step) {
        harness.upload(expected, occupancy);
        harness.uploadStructures(structures);
        harness.step();
        std::fill(outcomes.begin(), outcomes.end(), 0U);
        vkexp::stepLatticeCpu({expected, occupancy, claims, weights,
                               static_cast<std::uint32_t>(brain.weightCount()),
                               layout.groupSize(), layout.trialsPerGenome, structures, outcomes},
                              settings);
        const std::vector<vkexp::AgentState> actual = harness.readAgents();
        const char* worldName = worldMode == vkexp::WorldMode::Harvest  ? "Harvest step "
                                : chasm                                 ? "Chasm step "
                                                                        : "Construction step ";
        const std::string where = std::string{worldName} + std::to_string(step);
        for (std::size_t index = 0; index < expected.size(); ++index) {
            compareAgents(expected[index], actual[index], where + " agent " + std::to_string(index),
                          2.0e-3F);
        }
        const std::vector<std::int32_t> device = harness.readStructures();
        for (std::size_t cell = 0; cell < structures.size(); ++cell) {
            require(structures[cell] == device[cell],
                    where + ": cell " + std::to_string(cell) + " holds " +
                        std::to_string(structures[cell]) + " on the host and " +
                        std::to_string(device[cell]) + " on the device");
        }
        // The reasons, not only the results. Two implementations can refuse the
        // same attempt for different reasons and agree on every block, and then
        // the counters that are supposed to explain the builders explain the
        // wrong thing.
        const std::vector<std::uint32_t> deviceOutcomes = harness.readBuildOutcomes();
        for (std::size_t at = 0; at < outcomes.size(); ++at) {
            require(outcomes[at] == deviceOutcomes[at],
                    where + ": " + reasonNames[at % reasonNames.size()] + " counted " +
                        std::to_string(outcomes[at]) + " on the host and " +
                        std::to_string(deviceOutcomes[at]) + " on the device");
        }
        std::uint32_t attempts = 0;
        for (const std::uint32_t reason : outcomes) {
            attempts += reason;
        }
        require(attempts == expected.size(),
                where + ": " + std::to_string(attempts) + " outcomes recorded for " +
                    std::to_string(expected.size()) +
                    " agents -- every agent gets exactly one reason per step");

        for (std::uint32_t world = 0; world < layout.worldCount(); ++world) {
            frontierRefusals[world] +=
                outcomes[static_cast<std::size_t>(world) *
                             vkexp::lattice::kernel::LatticeBuildOutcomeCount +
                         vkexp::lattice::kernel::LatticeBuildAboveFrontier];
        }

        if (chasm) {
            // Nobody stands on nothing. The rule is the same everywhere, but
            // only here can it be broken: with a floor all the way across, a
            // fall always finds one. The bug this catches let an agent walk off
            // the edge, stop at height zero because the landing search stopped
            // there, and then stroll along the bottom of the chasm -- which
            // makes the whole world pointless without failing anything else.
            const auto solid = [&](const int x, const int y, const int z) {
                if (!vkexp::lattice::kernel::latticeInBounds(x, y, z, settings.latticeWidth,
                                                             settings.latticeHeight,
                                                             settings.latticeDepth)) {
                    return false;
                }
                return structures[vkexp::lattice::kernel::latticeCellIndex(
                           x, y, z, settings.latticeWidth, settings.latticeHeight)] !=
                       vkexp::lattice::kernel::LatticeNoStructure;
            };
            for (std::size_t index = 0; index < expected.size(); ++index) {
                const vkexp::Int4 at = expected[index].cell;
                const bool held = solid(at.x, at.y - 1, at.z) || solid(at.x - 1, at.y, at.z) ||
                                  solid(at.x + 1, at.y, at.z) || solid(at.x, at.y, at.z - 1) ||
                                  solid(at.x, at.y, at.z + 1);
                require(held, where + " agent " + std::to_string(index) + " is standing at (" +
                                  std::to_string(at.x) + ", " + std::to_string(at.y) + ", " +
                                  std::to_string(at.z) + ") with nothing to hold it");
            }
        }
    }

    if (vkexp::worldHarvests(worldMode)) {
        // The two things only this world does. Without these the probe would
        // compare agents that happen to agree about rules neither side ran.
        std::size_t carrying = 0;
        float delivered = 0.0F;
        for (const vkexp::AgentState& agent : expected) {
            carrying += agent.memory.w > 0.0F ? 1 : 0;
            delivered += agent.metrics.w;
        }
        require(carrying > 0 || delivered > 0.0F,
                "A fetching parity probe never picked up a load, so it compared nothing new");
    }

    const auto placed = std::count_if(structures.begin(), structures.end(), [](const std::int32_t v) {
        // Strictly positive: bedrock lives in this field too, and counting the
        // ground as built work would let the check below pass a probe in which
        // nobody ever placed anything.
        return v > vkexp::lattice::kernel::LatticeNoStructure;
    });

    require(placed > 0, "Construction parity probe never placed a block, so it compared nothing");
    // And the gate is doing something: every cell filled would mean the local
    // fill test waved through anything, which is the failure mode a parity
    // check alone cannot see, because both sides would be wrong together.
    require(static_cast<std::size_t>(placed) < structures.size(),
            "Construction parity probe filled the whole world, so the build gate refused nothing");

    if (vkexp::lattice::kernel::latticeWorldFrontier(static_cast<std::uint32_t>(worldMode))) {
        // The frontier has to have fired at least once, or this probe compared
        // two implementations of a rule neither of them reached. That is exactly
        // how the rule went five refusals in two and a half million without
        // anyone noticing.
        std::uint64_t refusedAbove = 0;
        for (std::uint32_t world = 0; world < layout.worldCount(); ++world) {
            refusedAbove += frontierRefusals[world];
        }
        require(refusedAbove > 0,
                "The foundation rule never refused anything, so the probe did not test it");
    }
}

void runFullLatticeProbe(vkexp::HeadlessComputeContext& context) {
    vkexp::SimulationStep settings =
        paritySettings(vkexp::Neighborhood::Moore, vkexp::NeuronModel::TimeConstant);
    settings.latticeWidth = 2;
    settings.latticeHeight = 2;
    settings.latticeDepth = 2;
    // Seven agents in eight cells, and that is as full as a world gets: spawn
    // refuses to stand anybody on the beacon, so the beacon cell is the one cell
    // left. An eighth agent would find nowhere to go and stay at its default
    // corner on top of somebody else -- which is a fixture that tests nothing
    // rather than a crowd.
    const vkexp::lattice::PopulationLayout layout{7, 7, 1};
    const vkexp::neuro::BrainShape brain = vkexp::resolvedBrain(settings);
    const std::vector<float> weights = makeWeights(brain, layout.genomeCount, 0x1234U);

    std::vector<vkexp::AgentState> expected = vkexp::lattice::makeInitialAgents(settings, layout);
    const std::uint32_t cells = vkexp::latticeCellsPerWorld(settings);
    std::vector<std::int32_t> occupancy(static_cast<std::size_t>(cells) * layout.worldCount());
    vkexp::lattice::buildOccupancy(expected, settings, layout, occupancy);
    std::vector<std::int32_t> claims(occupancy.size());

    LatticeHarness harness{context, settings, layout, weights};
    harness.upload(expected, occupancy);

    // What "full" is allowed to mean: the same seven cells are occupied by the
    // same seven agents at every step, whichever of them happens to be standing
    // where. A cell that held two agents, or a count that changed, would be a
    // move into a cell that was not empty at the top of the step -- which is the
    // one rule the whole arbitration exists to keep.
    const auto occupiedCells = [&](const std::vector<std::int32_t>& grid, const char* what) {
        std::vector<std::int32_t> occupants;
        for (const std::int32_t owner : grid) {
            if (owner != vkexp::lattice::kernel::LatticeNoOccupant) {
                occupants.push_back(owner);
            }
        }
        std::sort(occupants.begin(), occupants.end());
        require(std::adjacent_find(occupants.begin(), occupants.end()) == occupants.end(),
                std::string{what} + ": one agent is standing in two cells");
        require(occupants.size() == expected.size(),
                std::string{what} + ": the number of occupied cells changed");
    };
    occupiedCells(occupancy, "Full lattice at spawn");

    bool everRefused = false;
    for (std::uint32_t step = 0; step < 8; ++step) {
        harness.upload(expected, occupancy);
        harness.step();
        vkexp::stepLatticeCpu({expected, occupancy, claims, weights,
                               static_cast<std::uint32_t>(brain.weightCount()),
                               layout.groupSize(), layout.trialsPerGenome},
                              settings);
        const std::vector<vkexp::AgentState> actual = harness.readAgents();
        for (std::size_t index = 0; index < expected.size(); ++index) {
            compareAgents(expected[index], actual[index],
                          "Full lattice agent " + std::to_string(index), 2.0e-3F);
            everRefused = everRefused || expected[index].metrics.w > 0.0F;
        }
        occupiedCells(occupancy, "Full lattice");
        occupiedCells(harness.readOccupancy(), "Full lattice on the device");
    }
    // Without this the probe could pass on a population that never tried to
    // move, which is the one outcome it must not accept: the point is the
    // refusal path, not the empty one.
    require(everRefused, "Full lattice: nobody was ever refused a cell in a world with one free");
}

// Every agent of a world addressing its own genome. The base offset is
// agentIndex / trialsPerGenome * stride on both sides, and an off-by-one there
// makes an agent run another genome's weights -- which produces a plausible run
// and a wrong one. Two genomes whose weights differ only in the move bias make
// the mistake visible as a direction.
// Every field of the two std430 mirrors, read by the shader and handed back.
//
// This is the cheapest test in the file and it is here because the most
// expensive bug so far was invisible to all the others. GpuStepParameters and
// AgentState are declared twice -- once in C++ and once in GLSL -- and the two
// languages align a struct by different rules, so a block that C++ pads to a
// sixteen-byte boundary can sit four bytes earlier in the shader's copy. Slot
// zero of an array then still very nearly works, which is why a parity case
// that records one step at a time can watch such a mistake for weeks.
//
// So: fill both structures with values that are all different from each other,
// put a decoy in slot zero, ask the shader for slot one, and compare raw bits.
// A field that arrives shifted by one word is a named mismatch rather than a
// rounding difference.
void runLayoutEchoProbe(vkexp::HeadlessComputeContext& context) {
    constexpr std::uint32_t slotCount = 2;
    constexpr std::uint32_t agentCount = 2;

    std::vector<const char*> names;
    std::vector<std::uint32_t> expected;
    const auto expectUint = [&](const char* name, const std::uint32_t value) {
        names.push_back(name);
        expected.push_back(value);
    };
    const auto expectFloat = [&](const char* name, const float value) {
        names.push_back(name);
        expected.push_back(std::bit_cast<std::uint32_t>(value));
    };

    // Distinct on purpose: two fields that happened to share a value would let
    // a swap between them pass.
    std::uint32_t counter = 101;
    const auto nextUint = [&] { return counter++; };
    float scalar = 1.5F;
    const auto nextFloat = [&] {
        const float value = scalar;
        scalar += 1.0F;
        return value;
    };

    vkexp::GpuStepParameters packed{};
    packed.deltaTime = nextFloat();
    packed.moveThreshold = nextFloat();
    packed.agentCount = nextUint();
    packed.brainLayout = nextUint();
    packed.trialsPerGenome = nextUint();
    packed.agentsPerWorld = nextUint();
    packed.worldCount = nextUint();
    packed.latticeWidth = nextUint();
    packed.latticeHeight = nextUint();
    packed.latticeDepth = nextUint();
    packed.cellsPerWorld = nextUint();
    packed.neighborhood = nextUint();
    packed.maximumDistance = nextUint();
    packed.beaconContactRadius = nextUint();
    packed.neuronModel = nextUint();
    packed.brainHiddenLayers = nextUint();
    packed.brainGenomeStride = nextUint();
    packed.worldMode = nextUint();
    packed.buildIntervalTicks = nextUint();
    packed.buildThreshold = nextFloat();
    packed.constructionCourseFill = nextFloat();
    packed.constructionHeightLead = nextUint();
    packed.constructionSupportRadius = nextUint();
    packed.allowSideSupportedBlocks = nextUint();
    packed.resourceHeightLow = nextUint();
    packed.resourceHeightHigh = nextUint();
    packed.groundWidth = nextUint();
    packed.beaconSeed = nextUint();
    packed.fitness.trackingReward = nextFloat();
    packed.fitness.objectiveBonus = nextFloat();
    packed.fitness.motorCostWeight = nextFloat();
    packed.fitness.refusalPenalty = nextFloat();
    packed.fitness.signalCostFactor = nextFloat();
    packed.fitness.reserved0 = nextFloat();
    packed.fitness.reserved1 = nextFloat();
    packed.fitness.reserved2 = nextFloat();

    expectFloat("deltaTime", packed.deltaTime);
    expectFloat("moveThreshold", packed.moveThreshold);
    expectUint("agentCount", packed.agentCount);
    expectUint("brainLayout", packed.brainLayout);
    expectUint("trialsPerGenome", packed.trialsPerGenome);
    expectUint("agentsPerWorld", packed.agentsPerWorld);
    expectUint("worldCount", packed.worldCount);
    expectUint("latticeWidth", packed.latticeWidth);
    expectUint("latticeHeight", packed.latticeHeight);
    expectUint("latticeDepth", packed.latticeDepth);
    expectUint("cellsPerWorld", packed.cellsPerWorld);
    expectUint("neighborhood", packed.neighborhood);
    expectUint("maximumDistance", packed.maximumDistance);
    expectUint("beaconContactRadius", packed.beaconContactRadius);
    expectUint("neuronModel", packed.neuronModel);
    expectUint("brainHiddenLayers", packed.brainHiddenLayers);
    expectUint("brainGenomeStride", packed.brainGenomeStride);
    expectUint("worldMode", packed.worldMode);
    expectUint("buildIntervalTicks", packed.buildIntervalTicks);
    expectFloat("buildThreshold", packed.buildThreshold);
    expectFloat("constructionCourseFill", packed.constructionCourseFill);
    expectUint("constructionHeightLead", packed.constructionHeightLead);
    expectUint("constructionSupportRadius", packed.constructionSupportRadius);
    expectUint("allowSideSupportedBlocks", packed.allowSideSupportedBlocks);
    expectUint("resourceHeightLow", packed.resourceHeightLow);
    expectUint("resourceHeightHigh", packed.resourceHeightHigh);
    expectUint("groundWidth", packed.groundWidth);
    expectUint("beaconSeed", packed.beaconSeed);
    expectFloat("fitness.trackingReward", packed.fitness.trackingReward);
    expectFloat("fitness.objectiveBonus", packed.fitness.objectiveBonus);
    expectFloat("fitness.motorCostWeight", packed.fitness.motorCostWeight);
    expectFloat("fitness.refusalPenalty", packed.fitness.refusalPenalty);
    expectFloat("fitness.signalCostFactor", packed.fitness.signalCostFactor);
    expectFloat("fitness.reserved0", packed.fitness.reserved0);
    expectFloat("fitness.reserved1", packed.fitness.reserved1);
    expectFloat("fitness.reserved2", packed.fitness.reserved2);

    // Slot zero holds something else entirely, so a shader that reads the wrong
    // slot fails every field rather than passing on a stride that only works
    // at index zero.
    std::array<vkexp::GpuStepParameters, slotCount> blocks{};
    blocks[0].deltaTime = -7.25F;
    blocks[0].agentCount = 0xDECAFU;
    blocks[1] = packed;

    std::array<vkexp::AgentState, agentCount> agents{};
    vkexp::AgentState& agent = agents[1];
    std::int32_t integer = 11;
    const auto nextInt = [&] { return integer++; };
    agent.cell = {nextInt(), nextInt(), nextInt(), nextInt()};
    agent.intent = {nextInt(), nextInt(), nextInt(), nextInt()};
    agent.beacon = {nextInt(), nextInt(), nextInt(), nextInt()};
    agent.signal = {nextFloat(), nextFloat(), nextFloat(), nextFloat()};
    agent.metrics = {nextFloat(), nextFloat(), nextFloat(), nextFloat()};
    agent.memory = {nextFloat(), nextFloat(), nextFloat(), nextFloat()};
    agent.hidden.front() = {nextFloat(), nextFloat(), nextFloat(), nextFloat()};
    agent.hidden.back() = {nextFloat(), nextFloat(), nextFloat(), nextFloat()};

    const auto expectInt4 = [&](const char* name, const vkexp::Int4& value) {
        expectUint(name, static_cast<std::uint32_t>(value.x));
        expectUint(name, static_cast<std::uint32_t>(value.y));
        expectUint(name, static_cast<std::uint32_t>(value.z));
        expectUint(name, static_cast<std::uint32_t>(value.w));
    };
    const auto expectFloat4 = [&](const char* name, const vkexp::Float4& value) {
        expectFloat(name, value.x);
        expectFloat(name, value.y);
        expectFloat(name, value.z);
        expectFloat(name, value.w);
    };
    expectInt4("agent.cell", agent.cell);
    expectInt4("agent.intent", agent.intent);
    expectInt4("agent.beacon", agent.beacon);
    expectFloat4("agent.signal", agent.signal);
    expectFloat4("agent.metrics", agent.metrics);
    expectFloat4("agent.memory", agent.memory);
    expectFloat4("agent.hidden.front", agent.hidden.front());
    expectFloat4("agent.hidden.back", agent.hidden.back());

    constexpr VkMemoryPropertyFlags hostMemory =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vkexp::BufferResource agentBuffer;
    vkexp::BufferResource blockBuffer;
    vkexp::BufferResource echoBuffer;
    agentBuffer.create(context.physicalDevice(), context.device(),
                       {sizeof(agents), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    blockBuffer.create(context.physicalDevice(), context.device(),
                       {sizeof(blocks), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    echoBuffer.create(
        context.physicalDevice(), context.device(),
        {expected.size() * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemory});
    agentBuffer.write(agents.data(), sizeof(agents));
    blockBuffer.write(blocks.data(), sizeof(blocks));

    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (std::uint32_t index = 0; index < bindings.size(); ++index) {
        bindings[index].binding = index;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    vkexp::UniqueDescriptorSetLayout layout;
    if (vkCreateDescriptorSetLayout(context.device(), &layoutInfo, nullptr,
                                    layout.put(context.device())) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create the layout echo descriptor layout");
    }
    vkexp::DescriptorAllocator descriptors;
    descriptors.create(context.device(), {1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}}});
    const VkDescriptorSet set = descriptors.allocate(layout.get());
    vkexp::DescriptorSetWriter{}
        .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, agentBuffer.buffer(), 0,
                     agentBuffer.size())
        .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, blockBuffer.buffer(), 0,
                     blockBuffer.size())
        .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, echoBuffer.buffer(), 0,
                     echoBuffer.size())
        .update(context.device(), set);

    struct EchoSelector {
        std::uint32_t stepIndex{};
        std::uint32_t agentIndex{};
    };
    const vkexp::ComputePipeline pipeline =
        vkexp::ComputePipelineBuilder{context.physicalDevice(), context.device()}
            .shader(VKEXP_SHADER_DIR "/layout_echo.comp.spv")
            .addDescriptorSetLayout(layout.get())
            .addPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(EchoSelector))
            .build();
    const EchoSelector selector{slotCount - 1, agentCount - 1};
    context.immediate().execute([&](const VkCommandBuffer commands) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline());
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0, 1,
                                &set, 0, nullptr);
        vkCmdPushConstants(commands, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(selector), &selector);
        vkCmdDispatch(commands, 1, 1, 1);
        vkexp::cmdBufferBarrier(commands, echoBuffer.buffer(),
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    });

    std::vector<std::uint32_t> actual(expected.size());
    echoBuffer.read(actual.data(), actual.size() * sizeof(std::uint32_t));
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(actual[index] == expected[index],
                std::string{"Layout echo: "} + names[index] + " came back as a different word (" +
                    std::to_string(actual[index]) + " for " + std::to_string(expected[index]) +
                    ", value " + std::to_string(index) + " of " + std::to_string(expected.size()) +
                    ")");
    }
    std::cout << "Layout echo: " << expected.size()
              << " fields of the shared structs survived the round trip" << std::endl;
}

void runGenomeAddressingProbe(vkexp::HeadlessComputeContext& context) {
    const vkexp::SimulationStep settings =
        paritySettings(vkexp::Neighborhood::Faces, vkexp::NeuronModel::Reactive);
    const vkexp::lattice::PopulationLayout layout{2, 2, 2};
    const vkexp::neuro::BrainShape brain = vkexp::resolvedBrain(settings);
    std::vector<float> weights(static_cast<std::size_t>(brain.weightCount()) * layout.genomeCount,
                               0.0F);
    const auto biasIndex = [&](const std::uint32_t output) {
        return vkexp::neuro::kernel::brainOutputBiasIndex(
            0U, static_cast<std::uint32_t>(brain.inputCount), brain.packedLayers(),
            static_cast<std::uint32_t>(brain.outputCount), output);
    };
    // Genome 0 drives +x, genome 1 drives -y. Under a faces-only neighbourhood
    // each of those is a single unambiguous step, so the resulting cell names
    // which genome the agent actually read.
    weights[biasIndex(vkexp::neuro::kernel::BrainMoveOutput)] = 8.0F;
    weights[static_cast<std::size_t>(brain.weightCount()) +
            biasIndex(vkexp::neuro::kernel::BrainMoveOutput + 1U)] = -8.0F;

    std::vector<vkexp::AgentState> agents(layout.agentCount());
    const std::uint32_t cells = vkexp::latticeCellsPerWorld(settings);
    for (std::uint32_t index = 0; index < agents.size(); ++index) {
        const std::uint32_t world =
            vkexp::logicalWorldForAgent(index, layout.groupSize(), layout.trialsPerGenome);
        agents[index].beacon = vkexp::lattice::beaconCell(settings, world);
        // One plane per genome, so the two agents that share a world start
        // apart and neither move can be refused: what is under test is which
        // genome was read, and a blocked step would hide the answer.
        const std::uint32_t genome = index / layout.trialsPerGenome;
        agents[index].cell = {1, static_cast<std::int32_t>(1 + genome),
                              static_cast<std::int32_t>(genome),
                              static_cast<std::int32_t>(
                                  vkexp::lattice::kernel::LatticeNeighborCount)};
        agents[index].intent = {agents[index].cell.x, agents[index].cell.y, agents[index].cell.z,
                                0};
    }
    std::vector<std::int32_t> occupancy(static_cast<std::size_t>(cells) * layout.worldCount());
    vkexp::lattice::buildOccupancy(agents, settings, layout, occupancy);
    std::vector<std::int32_t> claims(occupancy.size());

    LatticeHarness harness{context, settings, layout, weights};
    harness.upload(agents, occupancy);
    std::vector<vkexp::AgentState> expected = agents;
    vkexp::stepLatticeCpu({expected, occupancy, claims, weights,
                           static_cast<std::uint32_t>(brain.weightCount()),
                           layout.groupSize(), layout.trialsPerGenome},
                          settings);
    harness.step();
    const std::vector<vkexp::AgentState> actual = harness.readAgents();

    for (std::uint32_t index = 0; index < agents.size(); ++index) {
        const bool firstGenome = index / layout.trialsPerGenome == 0;
        const vkexp::AgentState& before = agents[index];
        const std::string who = "Genome addressing agent " + std::to_string(index);
        if (firstGenome) {
            require(expected[index].cell.x == before.cell.x + 1 &&
                        expected[index].cell.y == before.cell.y,
                    who + ": read a genome that does not drive +x");
        } else {
            require(expected[index].cell.y == before.cell.y - 1 &&
                        expected[index].cell.x == before.cell.x,
                    who + ": read a genome that does not drive -y");
        }
        compareAgents(expected[index], actual[index], who, 2.0e-3F);
    }
}

// The same trajectory under a three-layer plan. Split out rather than folded
// into the loop above because the plan changes the genome length, and a case
// that changed two things at once would not say which one drifted.
void runDeepPlanParity(vkexp::HeadlessComputeContext& context) {
    vkexp::SimulationStep settings =
        paritySettings(vkexp::Neighborhood::Moore, vkexp::NeuronModel::Gated);
    settings.hiddenLayers = {12, 8, 8};
    const vkexp::lattice::PopulationLayout layout{12, 10, 2};
    const vkexp::neuro::BrainShape brain = vkexp::resolvedBrain(settings);
    require(brain.hiddenLayerCount() == 3, "The deep parity case did not get three layers");
    const std::vector<float> weights = makeWeights(brain, layout.genomeCount, 0xBEEFU);

    std::vector<vkexp::AgentState> expected = vkexp::lattice::makeInitialAgents(settings, layout);
    const std::uint32_t cells = vkexp::latticeCellsPerWorld(settings);
    std::vector<std::int32_t> occupancy(static_cast<std::size_t>(cells) * layout.worldCount());
    vkexp::lattice::buildOccupancy(expected, settings, layout, occupancy);
    std::vector<std::int32_t> claims(occupancy.size());

    LatticeHarness harness{context, settings, layout, weights};
    AgentDrift drift{};
    for (std::uint32_t step = 0; step < 90; ++step) {
        harness.upload(expected, occupancy);
        harness.step();
        const std::vector<vkexp::AgentState> actual = harness.readAgents();
        const std::vector<std::int32_t> actualOccupancy = harness.readOccupancy();
        vkexp::stepLatticeCpu({expected, occupancy, claims, weights,
                               static_cast<std::uint32_t>(brain.weightCount()),
                               layout.groupSize(), layout.trialsPerGenome},
                              settings);
        for (std::size_t index = 0; index < expected.size(); ++index) {
            compareAgents(expected[index], actual[index],
                          "Deep plan parity step " + std::to_string(step) + " agent " +
                              std::to_string(index),
                          4.0e-4F, &drift);
        }
        require(actualOccupancy == occupancy,
                "Deep plan parity: the occupancy grids diverged at step " + std::to_string(step));
    }
    const double worst = worstDrift(drift);
    std::cout << "  drift[12 -> 8 -> 8, gated] = " << worst << '\n';
    require(std::abs(worst) <= accumulatedDriftBudget,
            "Deep plan parity accumulated a systematic CPU/GPU drift of " + std::to_string(worst));
}

int runAll() {
    vkexp::HeadlessComputeContext context{{"vkexp compute smoke"}};
    std::cout << "Compute smoke device: " << context.deviceName() << '\n';
    runGameOfLife(context);
    runImageRoundTrip(context);

    // First, because it is the cheapest and because every case after it is read
    // through the structures it checks.
    runLayoutEchoProbe(context);
    runContentionProbe(context);
    runGenomeAddressingProbe(context);
    runFullLatticeProbe(context);
    // Both building worlds: harvest shares every movement and build rule with
    // construction and differs in what it accumulates, which is exactly the kind
    // of difference a probe that only ran one of them would never see.
    runConstructionParityProbe(context, vkexp::WorldMode::Construction);
    runConstructionParityProbe(context, vkexp::WorldMode::Harvest);
    runConstructionParityProbe(context, vkexp::WorldMode::Chasm);

    // Both neighbourhoods, because the face-only reduction is a branch the Moore
    // case never takes, and all four neuron models, because each decides the
    // time constant somewhere different and only the integrator is shared.
    for (const vkexp::Neighborhood neighborhood :
         {vkexp::Neighborhood::Moore, vkexp::Neighborhood::Faces}) {
        for (const vkexp::NeuronModel model :
             {vkexp::NeuronModel::Reactive, vkexp::NeuronModel::TimeConstant,
              vkexp::NeuronModel::Gated, vkexp::NeuronModel::Spiking}) {
            runLatticeTrajectoryParity(context, neighborhood, model, 120);
        }
    }

    // A deeper plan, so the layer walk in brain_forward.glsl is exercised
    // against the one in vkexp::neuro::evaluate rather than only its first
    // layer. Drift accumulates through the layers, which is exactly what a
    // single-layer case cannot see.
    runDeepPlanParity(context);

    std::cout << "Lattice parity: CPU and GPU agree\n";
    return 0;
}

} // namespace

int main() {
    try {
        return runAll();
    } catch (const vkexp::HeadlessComputeUnavailable& unavailable) {
        std::cout << "Skipping compute smoke tests: " << unavailable.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "Compute smoke failure: " << error.what() << '\n';
        return 1;
    }
}
