#include "vkexp/graphics/AgentRenderer.hpp"

#include "vkexp/core/VulkanContext.hpp"
#include "vkexp/profiling/Profiler.hpp"
#include "vkexp/worlds/WorldScenario.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace vkexp {
namespace {

struct DrawParameters {
    float scaleX{};
    float scaleY{};
    float worldRadius{};
    float opacity{};
    std::uint32_t mode{};
    std::uint32_t worldShape{};
    std::uint32_t beaconScenario{};
    std::uint32_t beaconPhase{};
    std::uint32_t selectedWorld{};
    std::uint32_t agentsPerWorld{};
    std::uint32_t trialsPerGenome{};
    std::uint32_t trailWidth{};
    float trailRenderWidth{};
    float trailCellSize{};
    std::uint32_t trailRoundMarks{};
    float backgroundBrightness{};
    std::uint32_t beaconCount{};
    std::uint32_t uniformBeaconColor{};
    ScenarioParameterBlock scenario;
};
// Exactly the 128 bytes Vulkan guarantees for maxPushConstantsSize, with nothing
// left over: the next field to reach the drawing has to move to a uniform buffer
// the way the step parameters already did. Note that a smaller field buys
// nothing here -- the trailing scenario block is 16-aligned, so one more uint and
// three more cost the same.
// Segments in a round trail mark. Shared with the vertex shader's fan, which is
// why it is not written twice.
constexpr std::uint32_t trailDiscSegments = 8;

static_assert(sizeof(DrawParameters) == 128);
static_assert(offsetof(DrawParameters, scenario) == 80);

} // namespace

AgentRenderer::AgentRenderer(SimulationState& state, Profiler& profiler)
    : state_(state), metric_(profiler.registerMetric("Agent visualization")) {}

void AgentRenderer::onAttach(AppContext& context) {
    createPipeline(context);
    createTarget(context, state_.viewport.extent);
}

void AgentRenderer::createPipeline(AppContext& context) {
    if (state_.agents.buffers[0] == VK_NULL_HANDLE || state_.agents.buffers[1] == VK_NULL_HANDLE) {
        throw std::logic_error("AgentRenderer requires SimulationModule to be attached first");
    }
    const VkDevice device = context.vulkan.device();
    // 0: the agent state being displayed, 1: the trail field, 2: the pucks. All
    // read in the vertex stage, which keeps every draw on the same procedural
    // path -- nothing here is uploaded as geometry.
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (std::uint32_t index = 0; index < bindings.size(); ++index) {
        bindings[index].binding = index;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    }
    VkDescriptorSetLayoutCreateInfo descriptorInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptorInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    descriptorInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &descriptorInfo, nullptr,
                                    descriptorSetLayout_.put(device)) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create agent renderer descriptor layout");
    }
    descriptorAllocator_.create(device, {2, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6}}});
    for (std::size_t index = 0; index < descriptorSets_.size(); ++index) {
        descriptorSets_[index] = descriptorAllocator_.allocate(descriptorSetLayout_.get());
        DescriptorSetWriter{}
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, state_.agents.buffers[index], 0,
                         state_.agents.size)
            .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, state_.trail.buffer, 0,
                         state_.trail.size)
            .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, state_.puck.buffer, 0,
                         state_.puck.size)
            .update(device, descriptorSets_[index]);
    }
    boundTrailBuffer_ = state_.trail.buffer;

    VkPushConstantRange pushRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(DrawParameters)};
    const VkDescriptorSetLayout setLayout = descriptorSetLayout_.get();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, pipelineLayout_.put(device)) !=
        VK_SUCCESS) {
        throw std::runtime_error("Unable to create agent renderer pipeline layout");
    }

    const auto vertex = loadShaderModule(device, VKEXP_SHADER_DIR "/agents.vert.spv");
    const auto fragment = loadShaderModule(device, VKEXP_SHADER_DIR "/agents.frag.spv");
    const std::array stages{VkPipelineShaderStageCreateInfo{
                                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                VK_SHADER_STAGE_VERTEX_BIT, vertex.get(), "main", nullptr},
                            VkPipelineShaderStageCreateInfo{
                                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                VK_SHADER_STAGE_FRAGMENT_BIT, fragment.get(), "main", nullptr}};
    VkPipelineVertexInputStateCreateInfo vertexInput{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rasterizer{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0F;
    VkPipelineMultisampleStateCreateInfo multisampling{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable = VK_TRUE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.alphaBlendOp = VK_BLEND_OP_ADD;
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blending{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blending.attachmentCount = 1;
    blending.pAttachments = &blend;
    constexpr std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
    dynamic.pDynamicStates = dynamicStates.data();
    constexpr VkFormat colorFormat = targetFormat;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &colorFormat;
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.pNext = &rendering;
    info.stageCount = static_cast<std::uint32_t>(stages.size());
    info.pStages = stages.data();
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &rasterizer;
    info.pMultisampleState = &multisampling;
    info.pColorBlendState = &blending;
    info.pDynamicState = &dynamic;
    info.layout = pipelineLayout_.get();
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                  pipeline_.put(device)) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create agent renderer pipeline");
    }
}

void AgentRenderer::createTarget(AppContext& context, const VkExtent2D extent) {
    target_.create(
        context.vulkan.physicalDevice(), context.vulkan.device(),
        {extent, targetFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT});
    targetLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    state_.viewport.imageView = target_.view();
    state_.viewport.sampler = target_.sampler();
    state_.viewport.extent = extent;
    ++state_.viewport.generation;
}

void AgentRenderer::destroyTarget() {
    state_.viewport.imageView = VK_NULL_HANDLE;
    state_.viewport.sampler = VK_NULL_HANDLE;
    target_.reset();
    targetLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void AgentRenderer::onUpdate(AppContext& context, const FrameInfo&) {
    const VkExtent2D requested{std::clamp(state_.viewport.requestedWidth, 64U, 4096U),
                               std::clamp(state_.viewport.requestedHeight, 64U, 4096U)};
    if (requested.width != state_.viewport.extent.width ||
        requested.height != state_.viewport.extent.height) {
        context.vulkan.waitIdle();
        destroyTarget();
        createTarget(context, requested);
    }
}

void AgentRenderer::refreshTrailDescriptor(const VkDevice device) {
    if (state_.trail.buffer == boundTrailBuffer_ || state_.trail.buffer == VK_NULL_HANDLE) {
        return;
    }
    for (const VkDescriptorSet set : descriptorSets_) {
        DescriptorSetWriter{}
            .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, state_.trail.buffer, 0,
                         state_.trail.size)
            .update(device, set);
    }
    boundTrailBuffer_ = state_.trail.buffer;
}

void AgentRenderer::draw(const VkCommandBuffer commands, const float scaleX, const float scaleY,
                         const float worldRadius, const std::uint32_t mode, const float opacity,
                         const std::uint32_t vertices, const std::uint32_t instances) const {
    // Shared with the simulation so the drawn beacon cannot drift from the
    // simulated one: same resolver, same scenario packer.
    const SimulationStep settings = resolveStepSettings(state_.physics, state_.statistics.step,
                                                        state_.controls.stepsPerGeneration);
    const DrawParameters parameters{
        scaleX,
        scaleY,
        worldRadius,
        opacity,
        mode,
        static_cast<std::uint32_t>(settings.worldShape),
        static_cast<std::uint32_t>(settings.beaconScenario),
        settings.beaconPhase,
        state_.worlds.selectedWorld,
        state_.worlds.agentsPerWorld,
        state_.agents.trialsPerGenome,
        state_.trail.width,
        state_.physics.trailRenderWidth,
        state_.physics.trailCellSize,
        state_.display.roundTrailMarks ? 1U : 0U,
        state_.display.backgroundBrightness,
        scenarioDefinition(settings.beaconScenario).beaconCount,
        settings.uniformBeaconColor ? 1U : 0U,
        scenarioDefinition(settings.beaconScenario).gpuParameters(settings)};
    vkCmdPushConstants(commands, pipelineLayout_.get(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(parameters), &parameters);
    vkCmdDraw(commands, vertices, instances, 0, 0);
}

void AgentRenderer::onRender(AppContext& context, const FrameInfo&) {
    auto cpuScope = context.profiler.cpu().scope(metric_);
    auto gpuScope = context.profiler.gpu().scope(context.vulkan.commandBuffer(), metric_);
    const VkCommandBuffer commands = context.vulkan.commandBuffer();
    cmdImageBarrier(
        commands, target_.image(), targetLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        targetLayout_ == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                                                   : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        targetLayout_ == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    // The ground outside the arena, dimmed by the same control as the arena fill
    // so the two do not drift apart as it is pulled down.
    const float background = std::clamp(state_.display.backgroundBrightness, 0.0F, 1.0F);
    const VkClearColorValue clear{
        {0.001F * background, 0.002F * background, 0.005F * background, 1.0F}};
    VkRenderingAttachmentInfo attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    attachment.imageView = target_.view();
    attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.clearValue.color = clear;
    VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rendering.renderArea.extent = state_.viewport.extent;
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &attachment;
    vkCmdBeginRendering(commands, &rendering);
    const VkViewport viewport{0.0F,
                              0.0F,
                              static_cast<float>(state_.viewport.extent.width),
                              static_cast<float>(state_.viewport.extent.height),
                              0.0F,
                              1.0F};
    const VkRect2D scissor{{0, 0}, state_.viewport.extent};
    vkCmdSetViewport(commands, 0, 1, &viewport);
    vkCmdSetScissor(commands, 0, 1, &scissor);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.get());
    refreshTrailDescriptor(context.vulkan.device());
    const VkDescriptorSet descriptorSet = descriptorSets_[state_.agents.currentIndex];
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_.get(), 0, 1,
                            &descriptorSet, 0, nullptr);
    const float aspect = static_cast<float>(state_.viewport.extent.width) /
                         static_cast<float>(state_.viewport.extent.height);
    const float baseScale = 0.94F / state_.physics.worldRadius;
    const float scaleX = aspect >= 1.0F ? baseScale / aspect : baseScale;
    const float scaleY = aspect >= 1.0F ? baseScale : baseScale * aspect;
    const std::uint32_t arenaVertices = state_.physics.worldShape == WorldShape::Circle ? 192U : 6U;
    const std::uint32_t visibleAgentCount =
        agentsInLogicalWorld(state_.agents.genomeCount, state_.worlds.agentsPerWorld,
                             state_.agents.trialsPerGenome, state_.worlds.selectedWorld);
    draw(commands, scaleX, scaleY, state_.physics.worldRadius, 3, 1.0F, arenaVertices, 1);
    // The field goes on top of the arena and under everything that moves, so a
    // path reads as ground rather than as another agent.
    if (state_.display.trail && trailFieldActive(state_.physics.trailMode) &&
        state_.trail.cellsPerWorld > 0) {
        // A disc needs a fan of triangles, a square needs two: the vertex count
        // is the only thing that differs on this side.
        const std::uint32_t markVertices =
            state_.display.roundTrailMarks ? trailDiscSegments * 3 : 6;
        draw(commands, scaleX, scaleY, state_.physics.worldRadius, 5, 0.85F, markVertices,
             state_.trail.cellsPerWorld);
    }
    // The target disc is ground: the puck has to reach it, nothing collides with
    // it, so it goes under everything including the puck itself.
    const ScenarioDefinition& drawnScenario = scenarioDefinition(state_.physics.beaconScenario);
    if (drawnScenario.puck) {
        // 48 segments of three vertices each. Passing the segment count as the
        // vertex count drew a third of a disc -- a wedge that looked like a
        // scenario shape rather than a target -- because circleVertex builds one
        // triangle per segment rather than one vertex.
        draw(commands, scaleX, scaleY, state_.physics.worldRadius, 8, 0.55F, 48 * 3, 1);
    }
    // The plate is ground too: it is a place to stand on, not a body to walk
    // into. Drawn at the radius the press test uses rather than at the fixed
    // beacon radius, because the beacon marks where its light comes from and
    // this marks how much of the floor counts.
    if (drawnScenario.tunables.gateLatch) {
        draw(commands, scaleX, scaleY, state_.physics.worldRadius, 9, 0.75F, 32 * 3, 1);
    }
    // Above the ground and under everything that moves: an obstacle is scenery
    // an agent collides with, not a thing that acts.
    const std::uint32_t obstacleCount = drawnScenario.obstacleCount;
    if (obstacleCount > 0) {
        draw(commands, scaleX, scaleY, state_.physics.worldRadius, 6, 1.0F, 6, obstacleCount);
    }
    // The puck moves and is pushed, so it belongs with the agents rather than
    // with the scenery -- but under them, so a crowd around it stays readable.
    if (drawnScenario.puck) {
        draw(commands, scaleX, scaleY, state_.physics.worldRadius, 7, 0.95F, 24 * 3, 1);
    }
    if (state_.display.agents) {
        draw(commands, scaleX, scaleY, state_.physics.worldRadius, 2, 0.14F, 48, visibleAgentCount);
    }
    if (state_.display.beacons) {
        const std::uint32_t beaconInstances =
            scenarioDefinition(state_.physics.beaconScenario).beaconCount;
        draw(commands, scaleX, scaleY, state_.physics.worldRadius, 1, 0.90F, 48, beaconInstances);
    }
    if (state_.display.agents) {
        draw(commands, scaleX, scaleY, state_.physics.worldRadius, 0, 0.88F, 48, visibleAgentCount);
        draw(commands, scaleX, scaleY, state_.physics.worldRadius, 4, 0.95F, 3, visibleAgentCount);
    }
    vkCmdEndRendering(commands);
    cmdImageBarrier(commands, target_.image(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    targetLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void AgentRenderer::onDetach(AppContext&) {
    destroyTarget();
    pipeline_.reset();
    pipelineLayout_.reset();
    descriptorAllocator_.reset();
    descriptorSetLayout_.reset();
    descriptorSets_ = {};
}

} // namespace vkexp
