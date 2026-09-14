#include "vkexp/graphics/LatticeRenderer.hpp"

#include "vkexp/core/VulkanContext.hpp"
#include "vkexp/lattice/LatticeWorld.hpp"
#include "vkexp/profiling/Profiler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace vkexp {

// Column-major, the order GLSL reads a mat4 in. Written out rather than pulled
// in from a maths library: this program has no vector type of its own on the
// C++ side, the lattice is integers, and a perspective matrix is the only
// floating-point geometry in the whole build.
namespace {

using Mat4 = std::array<float, 16>;

struct Vec3 {
    float x{};
    float y{};
    float z{};
};

[[nodiscard]] Vec3 operator-(const Vec3& left, const Vec3& right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

[[nodiscard]] float dot(const Vec3& left, const Vec3& right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

[[nodiscard]] Vec3 cross(const Vec3& left, const Vec3& right) {
    return {left.y * right.z - left.z * right.y, left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x};
}

[[nodiscard]] Vec3 normalize(const Vec3& value) {
    const float length = std::sqrt(dot(value, value));
    if (length < 1.0e-6F) {
        return {0.0F, 0.0F, 1.0F};
    }
    return {value.x / length, value.y / length, value.z / length};
}

[[nodiscard]] Mat4 multiply(const Mat4& left, const Mat4& right) {
    Mat4 result{};
    for (std::size_t column = 0; column < 4; ++column) {
        for (std::size_t row = 0; row < 4; ++row) {
            float sum = 0.0F;
            for (std::size_t inner = 0; inner < 4; ++inner) {
                sum += left[inner * 4 + row] * right[column * 4 + inner];
            }
            result[column * 4 + row] = sum;
        }
    }
    return result;
}

// Right-handed, looking down -z in view space.
[[nodiscard]] Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
    const Vec3 back = normalize(eye - target);
    const Vec3 right = normalize(cross(up, back));
    const Vec3 above = cross(back, right);
    return Mat4{
        right.x, above.x, back.x, 0.0F, right.y,          above.y,          back.y,          0.0F,
        right.z, above.z, back.z, 0.0F, -dot(right, eye), -dot(above, eye), -dot(back, eye), 1.0F};
}

// Vulkan clip space: y points down and depth runs 0..1, which is the whole of
// the difference from the OpenGL form and the whole of what makes a picture
// come out upside down when it is copied from the wrong book.
[[nodiscard]] Mat4 perspective(const float verticalFieldOfView, const float aspect,
                               const float nearPlane, const float farPlane) {
    const float focal = 1.0F / std::tan(verticalFieldOfView * 0.5F);
    Mat4 result{};
    result[0] = focal / aspect;
    result[5] = -focal;
    result[10] = farPlane / (nearPlane - farPlane);
    result[11] = -1.0F;
    result[14] = (farPlane * nearPlane) / (nearPlane - farPlane);
    return result;
}

// Right-handed Vulkan orthographic clip space. `halfHeight` is chosen from the
// perspective camera's field of view below, so switching projection changes
// the geometry of the view without producing a surprising jump in scale.
[[nodiscard]] Mat4 orthographic(const float halfHeight, const float aspect, const float nearPlane,
                                const float farPlane) {
    const float halfWidth = halfHeight * aspect;
    Mat4 result{};
    result[0] = 1.0F / halfWidth;
    result[5] = -1.0F / halfHeight;
    result[10] = 1.0F / (nearPlane - farPlane);
    result[14] = nearPlane / (nearPlane - farPlane);
    result[15] = 1.0F;
    return result;
}

constexpr std::uint32_t cubeVertexCount = 36;    // six faces of two triangles
constexpr std::uint32_t boxEdgeVertexCount = 24;   // twelve edges of a line list
constexpr std::uint32_t goalGuideVertexCount = 6;  // a plumb line and a floor cross

constexpr std::uint32_t modeAgents = 0;
constexpr std::uint32_t modeBeacon = 1;
constexpr std::uint32_t modeBounds = 2;
constexpr std::uint32_t modeTrail = 3;
constexpr std::uint32_t modeStructure = 4;
constexpr std::uint32_t modeGoal = 5;
constexpr std::uint32_t modeTerrain = 6;

// Mirrors the push constant block in shaders/lattice/lattice_view.glsl. Exactly
// the 128 bytes Vulkan guarantees, with nothing spare: the mode, the slice axis
// and the agent stride share one word because the matrix takes half of it.
struct PushParameters {
    Mat4 viewProjection{};
    std::array<float, 4> camera{};
    std::array<std::int32_t, 4> lattice{};
    std::array<std::int32_t, 4> beacon{};
    std::array<float, 4> tint{};
};

static_assert(sizeof(PushParameters) == 128);

} // namespace

LatticeRenderer::LatticeRenderer(SimulationState& state, Profiler& profiler)
    : state_(state), metric_(profiler.registerMetric("Lattice view")) {}

void LatticeRenderer::onAttach(AppContext& context) {
    state_.display.camera = latticeHomeCamera(state_.settings);
    createPipelines(context);
    createTarget(context, state_.viewport.extent);
}

namespace {

[[nodiscard]] VkPipelineShaderStageCreateInfo stageInfo(const VkShaderStageFlagBits stage,
                                                        const VkShaderModule module) {
    VkPipelineShaderStageCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage = stage;
    info.module = module;
    info.pName = "main";
    return info;
}

} // namespace

void LatticeRenderer::createPipelines(AppContext& context) {
    if (state_.agents.buffers[0] == VK_NULL_HANDLE || state_.agents.buffers[1] == VK_NULL_HANDLE ||
        state_.trails.buffer == VK_NULL_HANDLE || state_.structures.buffer == VK_NULL_HANDLE) {
        throw std::logic_error("LatticeRenderer requires SimulationModule to be attached first");
    }
    const VkDevice device = context.vulkan.device();

    std::array<VkDescriptorSetLayoutBinding, 3> agentBindings{};
    for (std::uint32_t index = 0; index < agentBindings.size(); ++index) {
        agentBindings[index].binding = index;
        agentBindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        agentBindings[index].descriptorCount = 1;
        agentBindings[index].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    }
    VkDescriptorSetLayoutCreateInfo agentLayoutInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    agentLayoutInfo.bindingCount = static_cast<std::uint32_t>(agentBindings.size());
    agentLayoutInfo.pBindings = agentBindings.data();
    if (vkCreateDescriptorSetLayout(device, &agentLayoutInfo, nullptr,
                                    agentSetLayout_.put(device)) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create lattice view descriptor layout");
    }
    agentAllocator_.create(device, {2, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6}}});
    for (std::size_t index = 0; index < agentSets_.size(); ++index) {
        agentSets_[index] = agentAllocator_.allocate(agentSetLayout_.get());
        DescriptorSetWriter{}
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, state_.agents.buffers[index], 0,
                         state_.agents.size)
            .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, state_.trails.buffer, 0,
                         state_.trails.size)
            .writeBuffer(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, state_.structures.buffer, 0,
                         state_.structures.size)
            .update(device, agentSets_[index]);
    }

    std::array<VkDescriptorSetLayoutBinding, 2> resolveBindings{};
    for (std::uint32_t index = 0; index < resolveBindings.size(); ++index) {
        resolveBindings[index].binding = index;
        resolveBindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        resolveBindings[index].descriptorCount = 1;
        resolveBindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo resolveLayoutInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    resolveLayoutInfo.bindingCount = static_cast<std::uint32_t>(resolveBindings.size());
    resolveLayoutInfo.pBindings = resolveBindings.data();
    if (vkCreateDescriptorSetLayout(device, &resolveLayoutInfo, nullptr,
                                    resolveSetLayout_.put(device)) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create lattice resolve descriptor layout");
    }
    resolveAllocator_.create(device, {1, {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}}});
    resolveSet_ = resolveAllocator_.allocate(resolveSetLayout_.get());

    const VkPushConstantRange pushRange{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                        0, sizeof(PushParameters)};
    const VkDescriptorSetLayout agentSetLayout = agentSetLayout_.get();
    VkPipelineLayoutCreateInfo voxelLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    voxelLayoutInfo.setLayoutCount = 1;
    voxelLayoutInfo.pSetLayouts = &agentSetLayout;
    voxelLayoutInfo.pushConstantRangeCount = 1;
    voxelLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &voxelLayoutInfo, nullptr, voxelLayout_.put(device)) !=
        VK_SUCCESS) {
        throw std::runtime_error("Unable to create lattice view pipeline layout");
    }
    const VkDescriptorSetLayout resolveSetLayout = resolveSetLayout_.get();
    VkPipelineLayoutCreateInfo resolveLayoutCreate{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    resolveLayoutCreate.setLayoutCount = 1;
    resolveLayoutCreate.pSetLayouts = &resolveSetLayout;
    if (vkCreatePipelineLayout(device, &resolveLayoutCreate, nullptr, resolveLayout_.put(device)) !=
        VK_SUCCESS) {
        throw std::runtime_error("Unable to create lattice resolve pipeline layout");
    }

    const auto voxelVertex = loadShaderModule(device, VKEXP_SHADER_DIR "/view_voxels.vert.spv");
    const auto voxelFragment = loadShaderModule(device, VKEXP_SHADER_DIR "/view_voxels.frag.spv");
    const auto oitFragment = loadShaderModule(device, VKEXP_SHADER_DIR "/view_voxels_oit.frag.spv");
    const auto resolveVertex = loadShaderModule(device, VKEXP_SHADER_DIR "/view_resolve.vert.spv");
    const auto resolveFragment =
        loadShaderModule(device, VKEXP_SHADER_DIR "/view_resolve.frag.spv");

    VkPipelineVertexInputStateCreateInfo vertexInput{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineViewportStateCreateInfo viewportState{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rasterizer{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    // Nothing is culled. A see-through voxel needs the inside of its far face,
    // and an opaque one is a closed box whose back faces the depth test drops
    // anyway -- so culling would buy a little fill rate and cost the one case
    // that matters. The shading turns the normal toward the eye instead, which
    // is why the winding of the generated cube never has to be got right.
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0F;
    VkPipelineMultisampleStateCreateInfo multisampling{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    constexpr std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
    dynamic.pDynamicStates = dynamicStates.data();

    VkPipelineColorBlendAttachmentState opaqueBlend{};
    opaqueBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo opaqueBlending{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    opaqueBlending.attachmentCount = 1;
    opaqueBlending.pAttachments = &opaqueBlend;

    VkPipelineDepthStencilStateCreateInfo depthWrite{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthWrite.depthTestEnable = VK_TRUE;
    depthWrite.depthWriteEnable = VK_TRUE;
    depthWrite.depthCompareOp = VK_COMPARE_OP_LESS;
    depthWrite.maxDepthBounds = 1.0F;

    constexpr VkFormat colorAttachmentFormat = colorFormat;
    VkPipelineRenderingCreateInfo opaqueRendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    opaqueRendering.colorAttachmentCount = 1;
    opaqueRendering.pColorAttachmentFormats = &colorAttachmentFormat;
    opaqueRendering.depthAttachmentFormat = depthFormat;

    const std::array voxelStages{stageInfo(VK_SHADER_STAGE_VERTEX_BIT, voxelVertex.get()),
                                 stageInfo(VK_SHADER_STAGE_FRAGMENT_BIT, voxelFragment.get())};
    VkPipelineInputAssemblyStateCreateInfo triangles{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    triangles.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineInputAssemblyStateCreateInfo lines{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    lines.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    VkGraphicsPipelineCreateInfo voxelInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    voxelInfo.pNext = &opaqueRendering;
    voxelInfo.stageCount = static_cast<std::uint32_t>(voxelStages.size());
    voxelInfo.pStages = voxelStages.data();
    voxelInfo.pVertexInputState = &vertexInput;
    voxelInfo.pInputAssemblyState = &triangles;
    voxelInfo.pViewportState = &viewportState;
    voxelInfo.pRasterizationState = &rasterizer;
    voxelInfo.pMultisampleState = &multisampling;
    voxelInfo.pDepthStencilState = &depthWrite;
    voxelInfo.pColorBlendState = &opaqueBlending;
    voxelInfo.pDynamicState = &dynamic;
    voxelInfo.layout = voxelLayout_.get();
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &voxelInfo, nullptr,
                                  voxelPipeline_.put(device)) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create lattice voxel pipeline");
    }

    VkGraphicsPipelineCreateInfo boundsInfo = voxelInfo;
    boundsInfo.pInputAssemblyState = &lines;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &boundsInfo, nullptr,
                                  boundsPipeline_.put(device)) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create lattice bounds pipeline");
    }

    // The transparent pass writes two attachments and no depth: accumulation
    // sums, revealage multiplies, and neither is an order-dependent operation.
    std::array<VkPipelineColorBlendAttachmentState, 2> transparentBlends{};
    transparentBlends[0].blendEnable = VK_TRUE;
    transparentBlends[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    transparentBlends[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    transparentBlends[0].colorBlendOp = VK_BLEND_OP_ADD;
    transparentBlends[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    transparentBlends[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    transparentBlends[0].alphaBlendOp = VK_BLEND_OP_ADD;
    transparentBlends[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    transparentBlends[1].blendEnable = VK_TRUE;
    transparentBlends[1].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    transparentBlends[1].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    transparentBlends[1].colorBlendOp = VK_BLEND_OP_ADD;
    transparentBlends[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    transparentBlends[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    transparentBlends[1].alphaBlendOp = VK_BLEND_OP_ADD;
    transparentBlends[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    VkPipelineColorBlendStateCreateInfo transparentBlending{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    transparentBlending.attachmentCount = static_cast<std::uint32_t>(transparentBlends.size());
    transparentBlending.pAttachments = transparentBlends.data();

    VkPipelineDepthStencilStateCreateInfo depthRead = depthWrite;
    depthRead.depthWriteEnable = VK_FALSE;

    constexpr std::array transparentFormats{accumulationFormat, revealageFormat};
    VkPipelineRenderingCreateInfo transparentRendering{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    transparentRendering.colorAttachmentCount =
        static_cast<std::uint32_t>(transparentFormats.size());
    transparentRendering.pColorAttachmentFormats = transparentFormats.data();
    transparentRendering.depthAttachmentFormat = depthFormat;

    const std::array transparentStages{stageInfo(VK_SHADER_STAGE_VERTEX_BIT, voxelVertex.get()),
                                       stageInfo(VK_SHADER_STAGE_FRAGMENT_BIT, oitFragment.get())};
    VkGraphicsPipelineCreateInfo transparentInfo = voxelInfo;
    transparentInfo.pNext = &transparentRendering;
    transparentInfo.pStages = transparentStages.data();
    transparentInfo.pDepthStencilState = &depthRead;
    transparentInfo.pColorBlendState = &transparentBlending;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &transparentInfo, nullptr,
                                  transparentPipeline_.put(device)) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create lattice transparency pipeline");
    }

    VkPipelineColorBlendAttachmentState resolveBlend{};
    resolveBlend.blendEnable = VK_TRUE;
    resolveBlend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    resolveBlend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    resolveBlend.colorBlendOp = VK_BLEND_OP_ADD;
    resolveBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    resolveBlend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    resolveBlend.alphaBlendOp = VK_BLEND_OP_ADD;
    resolveBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo resolveBlending{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    resolveBlending.attachmentCount = 1;
    resolveBlending.pAttachments = &resolveBlend;
    VkPipelineDepthStencilStateCreateInfo noDepth{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    noDepth.maxDepthBounds = 1.0F;
    VkPipelineRenderingCreateInfo resolveRendering{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    resolveRendering.colorAttachmentCount = 1;
    resolveRendering.pColorAttachmentFormats = &colorAttachmentFormat;
    const std::array resolveStages{stageInfo(VK_SHADER_STAGE_VERTEX_BIT, resolveVertex.get()),
                                   stageInfo(VK_SHADER_STAGE_FRAGMENT_BIT, resolveFragment.get())};
    VkGraphicsPipelineCreateInfo resolveInfo = voxelInfo;
    resolveInfo.pNext = &resolveRendering;
    resolveInfo.pStages = resolveStages.data();
    resolveInfo.pDepthStencilState = &noDepth;
    resolveInfo.pColorBlendState = &resolveBlending;
    resolveInfo.layout = resolveLayout_.get();
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &resolveInfo, nullptr,
                                  resolvePipeline_.put(device)) != VK_SUCCESS) {
        throw std::runtime_error("Unable to create lattice resolve pipeline");
    }
}

void LatticeRenderer::createTarget(AppContext& context, const VkExtent2D extent) {
    const VkPhysicalDevice physicalDevice = context.vulkan.physicalDevice();
    const VkDevice device = context.vulkan.device();
    color_.create(
        physicalDevice, device,
        {extent, colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT});
    depth_.create(physicalDevice, device,
                  {extent, depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_FILTER_NEAREST,
                   VK_IMAGE_ASPECT_DEPTH_BIT});
    accumulation_.create(physicalDevice, device,
                         {extent, accumulationFormat,
                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT});
    revealage_.create(physicalDevice, device,
                      {extent, revealageFormat,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT});
    colorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    depthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    accumulationLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    revealageLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    writeResolveDescriptor(device);

    state_.viewport.imageView = color_.view();
    state_.viewport.sampler = color_.sampler();
    state_.viewport.extent = extent;
    ++state_.viewport.generation;
}

void LatticeRenderer::writeResolveDescriptor(const VkDevice device) {
    DescriptorSetWriter{}
        .writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, accumulation_.view(),
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, accumulation_.sampler())
        .writeImage(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, revealage_.view(),
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, revealage_.sampler())
        .update(device, resolveSet_);
}

void LatticeRenderer::destroyTarget() {
    state_.viewport.imageView = VK_NULL_HANDLE;
    state_.viewport.sampler = VK_NULL_HANDLE;
    revealage_.reset();
    accumulation_.reset();
    depth_.reset();
    color_.reset();
    colorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    depthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    accumulationLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    revealageLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void LatticeRenderer::onUpdate(AppContext& context, const FrameInfo& frame) {
    LatticeCamera& camera = state_.display.camera;
    if (camera.spin) {
        camera.yaw += camera.spinRate * frame.deltaSeconds;
    }
    // Wrapped rather than left to grow, so a run left spinning overnight does
    // not hand the sine a number with no fractional precision left in it.
    constexpr float tau = 6.283185307F;
    camera.yaw = std::fmod(camera.yaw, tau);
    camera.pitch = std::clamp(camera.pitch, -1.53F, 1.53F);
    camera.distance = std::clamp(camera.distance, 0.35F, 12.0F);
    // The box can be pushed off to the side but not lost: a target beyond its
    // own diagonal would leave an empty viewport and no way back but the reset
    // button, and a control that can strand the view is a control people stop
    // using.
    const float reach =
        std::sqrt(static_cast<float>(state_.settings.latticeWidth) *
                      static_cast<float>(state_.settings.latticeWidth) +
                  static_cast<float>(state_.settings.latticeHeight) *
                      static_cast<float>(state_.settings.latticeHeight) +
                  static_cast<float>(state_.settings.latticeDepth) *
                      static_cast<float>(state_.settings.latticeDepth));
    camera.targetX = std::clamp(camera.targetX, -reach, reach);
    camera.targetY = std::clamp(camera.targetY, -reach, reach);
    camera.targetZ = std::clamp(camera.targetZ, -reach, reach);

    const VkExtent2D requested{std::clamp(state_.viewport.requestedWidth, 64U, 4096U),
                               std::clamp(state_.viewport.requestedHeight, 64U, 4096U)};
    if (requested.width != state_.viewport.extent.width ||
        requested.height != state_.viewport.extent.height) {
        context.vulkan.waitIdle();
        destroyTarget();
        createTarget(context, requested);
    }
}

void LatticeRenderer::onRender(AppContext& context, const FrameInfo&) {
    auto cpuScope = context.profiler.cpu().scope(metric_);
    auto gpuScope = context.profiler.gpu().scope(context.vulkan.commandBuffer(), metric_);
    const VkCommandBuffer commands = context.vulkan.commandBuffer();
    const SimulationDisplay& display = state_.display;
    const SimulationStep& settings = state_.settings;

    const std::uint32_t visibleAgents =
        agentsInLogicalWorld(state_.agents.genomeCount, state_.worlds.agentsPerWorld,
                             state_.agents.trialsPerGenome, state_.worlds.selectedWorld);
    const std::uint32_t trials = std::max(1U, state_.agents.trialsPerGenome);
    const std::uint32_t group = state_.worlds.selectedWorld / trials;
    const std::uint32_t trial = state_.worlds.selectedWorld % trials;
    const std::uint32_t firstAgent = (group * state_.worlds.agentsPerWorld) * trials + trial;

    // The beacon is placed by the same pure function the simulation places it
    // with, from the seed the running generation is using, so the drawn beacon
    // cannot drift from the simulated one. Reading it off an agent record would
    // have meant a readback for a number that is already computable.
    // The one cell in the box the trial is about, whichever world this is.
    const Int4 beacon = worldHarvests(settings.worldMode)
                            ? lattice::resourceCell(settings, state_.worlds.selectedWorld)
                            : lattice::beaconCell(settings, state_.worlds.selectedWorld);

    const auto width = static_cast<float>(settings.latticeWidth);
    const auto height = static_cast<float>(settings.latticeHeight);
    const auto depth = static_cast<float>(settings.latticeDepth);
    const float halfDiagonal = 0.5F * std::sqrt(width * width + height * height + depth * depth);
    const LatticeCamera& camera = display.camera;
    const float radius = std::max(camera.distance * halfDiagonal, 0.2F);
    // The orbit is around whatever the camera is looking at, which is the middle
    // of the box until somebody drags it somewhere else.
    const Vec3 target{camera.targetX, camera.targetY, camera.targetZ};
    const Vec3 eye{target.x + radius * std::cos(camera.pitch) * std::sin(camera.yaw),
                   target.y + radius * std::sin(camera.pitch),
                   target.z + radius * std::cos(camera.pitch) * std::cos(camera.yaw)};
    const float aspect = static_cast<float>(state_.viewport.extent.width) /
                         static_cast<float>(state_.viewport.extent.height);
    constexpr float verticalFieldOfView = latticeCameraFieldOfView;
    const float nearPlane = std::max(0.05F, halfDiagonal * 0.01F);
    // Far enough to still hold the whole box once the target has been dragged
    // to a corner of it, which is as far as the clamp in onUpdate allows.
    const float farPlane = radius + halfDiagonal * 6.0F;
    const Mat4 projection =
        camera.projection == CameraProjection::Orthographic
            ? orthographic(std::max(radius * std::tan(verticalFieldOfView * 0.5F), 0.1F), aspect,
                           nearPlane, farPlane)
            : perspective(verticalFieldOfView, aspect, nearPlane, farPlane);
    const Mat4 viewProjection =
        multiply(projection, lookAt(eye, target, Vec3{0.0F, 1.0F, 0.0F}));

    const std::uint32_t sliceAxis = std::min(display.sliceAxis, 2U);
    const std::array<std::uint32_t, 3> extents{settings.latticeWidth, settings.latticeHeight,
                                               settings.latticeDepth};
    const std::uint32_t lastSlice = extents[sliceAxis] - 1U;
    const std::uint32_t clampedSliceLow = std::min(display.sliceLow, lastSlice);
    const auto sliceLow = static_cast<float>(clampedSliceLow);
    const auto sliceHigh =
        static_cast<float>(std::clamp(display.sliceHigh, clampedSliceLow, lastSlice));
    const float background = std::clamp(display.backgroundBrightness, 0.0F, 1.0F);

    PushParameters parameters{};
    parameters.viewProjection = viewProjection;
    parameters.camera = {eye.x, eye.y, eye.z, std::clamp(display.voxelScale, 0.05F, 1.0F)};
    parameters.lattice = {static_cast<std::int32_t>(settings.latticeWidth),
                          static_cast<std::int32_t>(settings.latticeHeight),
                          static_cast<std::int32_t>(settings.latticeDepth), 0};
    parameters.beacon = {beacon.x, beacon.y, beacon.z, static_cast<std::int32_t>(firstAgent)};
    parameters.tint = {std::clamp(display.voxelOpacity, 0.02F, 1.0F), background, sliceLow,
                       sliceHigh};

    const auto pushWord = [&](const std::uint32_t mode) {
        parameters.lattice[3] =
            static_cast<std::int32_t>(mode | (sliceAxis << 8U) | (trials << 16U));
    };
    const auto push = [&] {
        vkCmdPushConstants(commands, voxelLayout_.get(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(parameters), &parameters);
    };

    cmdImageBarrier(
        commands, color_.image(), colorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        colorLayout_ == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                                                  : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        colorLayout_ == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    cmdImageBarrier(
        commands, depth_.image(), depthLayout_, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        depthLayout_ == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                                                  : VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                                        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        depthLayout_ == VK_IMAGE_LAYOUT_UNDEFINED
            ? 0
            : VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1});
    colorLayout_ = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    depthLayout_ = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;

    const VkViewport viewport{0.0F,
                              0.0F,
                              static_cast<float>(state_.viewport.extent.width),
                              static_cast<float>(state_.viewport.extent.height),
                              0.0F,
                              1.0F};
    const VkRect2D scissor{{0, 0}, state_.viewport.extent};

    VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAttachment.imageView = color_.view();
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {
        {0.012F * background, 0.016F * background, 0.028F * background, 1.0F}};
    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView = depth_.view();
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil = {1.0F, 0};

    VkRenderingInfo opaquePass{VK_STRUCTURE_TYPE_RENDERING_INFO};
    opaquePass.renderArea.extent = state_.viewport.extent;
    opaquePass.layerCount = 1;
    opaquePass.colorAttachmentCount = 1;
    opaquePass.pColorAttachments = &colorAttachment;
    opaquePass.pDepthAttachment = &depthAttachment;

    const bool transparent = display.voxelStyle == VoxelStyle::Transparent;
    const bool drawAgents = display.agents && visibleAgents > 0 && state_.agents.agentCount > 0;
    const bool drawStructures = display.structures &&
                                worldBuilds(settings.worldMode) &&
                                state_.structures.buffer != VK_NULL_HANDLE;
    const std::uint32_t trailSamples =
        std::min({state_.trails.recordedTicks, display.trailLength, state_.trails.capacity});
    const bool drawTrails = display.trails && visibleAgents > 0 && trailSamples > 0 &&
                            state_.trails.buffer != VK_NULL_HANDLE;
    const VkDescriptorSet agentSet = agentSets_[state_.agents.currentIndex];

    vkCmdBeginRendering(commands, &opaquePass);
    vkCmdSetViewport(commands, 0, 1, &viewport);
    vkCmdSetScissor(commands, 0, 1, &scissor);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelLayout_.get(), 0, 1,
                            &agentSet, 0, nullptr);
    // Both are line lists, so they share one pipeline and one bind. The guide is
    // drawn wherever the objective is drawn, and for the same reason: it is the
    // one thing in the box whose position is the question.
    const bool drawObjective = display.beacons && settings.worldMode != WorldMode::Construction;
    if (display.bounds || drawObjective) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, boundsPipeline_.get());
        if (display.bounds) {
            pushWord(modeBounds);
            push();
            vkCmdDraw(commands, boxEdgeVertexCount, 1, 0, 0);
        }
        if (drawObjective) {
            pushWord(modeGoal);
            push();
            vkCmdDraw(commands, goalGuideVertexCount, 1, 0, 0);
        }
    }
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelPipeline_.get());
    if (drawObjective) {
        // Always opaque, and a little larger than a cell. It is the one thing in
        // the box whose position is the question rather than the answer, so it
        // should not be the thing that disappears when transparency is on. In
        // the harvest world it is the resource, which is the same claim.
        const float agentScale = parameters.camera[3];
        parameters.camera[3] = std::min(1.0F, agentScale + 0.18F);
        pushWord(modeBeacon);
        push();
        vkCmdDraw(commands, cubeVertexCount, 1, 0, 0);
        parameters.camera[3] = agentScale;
    }
    // The block field, drawn into whichever pass the style puts it in. One
    // instance per cell is deliberately simple: the field is already on the
    // device, and empty instances collapse in the vertex shader without a
    // compaction pass or readback. A tiny seam keeps a tower legible as masonry
    // instead of one featureless prism.
    // The ground, always opaque and always drawn, over the bottom course alone:
    // one instance per floor cell rather than per lattice cell, which is the
    // whole box divided by its height. It is drawn separately from the block
    // field because terrain is not work -- it should not go see-through when
    // work does, and a floor plane is the most expensive thing there is to put
    // through a blended pass, since every pixel of it costs a fragment whatever
    // stands in front of it.
    const auto drawTerrain = [&] {
        const std::array<std::int32_t, 4> savedBeacon = parameters.beacon;
        const float savedScale = parameters.camera[3];
        parameters.beacon[3] = static_cast<std::int32_t>(state_.worlds.selectedWorld);
        parameters.camera[3] = 1.0F;
        pushWord(modeTerrain);
        push();
        vkCmdDraw(commands, cubeVertexCount, settings.latticeWidth * settings.latticeDepth, 0, 0);
        parameters.beacon = savedBeacon;
        parameters.camera[3] = savedScale;
    };

    const auto drawStructureField = [&] {
        const std::array<std::int32_t, 4> savedBeacon = parameters.beacon;
        const float savedScale = parameters.camera[3];
        parameters.beacon[3] = static_cast<std::int32_t>(state_.worlds.selectedWorld);
        parameters.camera[3] = 0.94F;
        pushWord(modeStructure);
        push();
        vkCmdDraw(commands, cubeVertexCount, state_.lattice.cellsPerWorld, 0, 0);
        parameters.beacon = savedBeacon;
        parameters.camera[3] = savedScale;
    };

    if (drawStructures) {
        drawTerrain();
    }
    if (drawStructures && !transparent) {
        drawStructureField();
    }
    if (drawAgents && !transparent) {
        pushWord(modeAgents);
        push();
        vkCmdDraw(commands, cubeVertexCount, visibleAgents, 0, 0);
    }
    vkCmdEndRendering(commands);

    if (drawTrails || (transparent && (drawAgents || drawStructures))) {
        // The opaque pass clears and then writes depth for the beacon and box;
        // the transparent pass tests every agent against that result. Ending a
        // dynamic-rendering instance supplies no memory dependency of its own.
        cmdImageBarrier(commands, depth_.image(), depthLayout_, depthLayout_,
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                        {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1});
        cmdImageBarrier(commands, accumulation_.image(), accumulationLayout_,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        accumulationLayout_ == VK_IMAGE_LAYOUT_UNDEFINED
                            ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                            : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                        accumulationLayout_ == VK_IMAGE_LAYOUT_UNDEFINED
                            ? 0
                            : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        cmdImageBarrier(
            commands, revealage_.image(), revealageLayout_,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            revealageLayout_ == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                                                          : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            revealageLayout_ == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        accumulationLayout_ = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        revealageLayout_ = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        std::array<VkRenderingAttachmentInfo, 2> transparentAttachments{};
        transparentAttachments[0] =
            VkRenderingAttachmentInfo{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        transparentAttachments[0].imageView = accumulation_.view();
        transparentAttachments[0].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        transparentAttachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        transparentAttachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        transparentAttachments[0].clearValue.color = {{0.0F, 0.0F, 0.0F, 0.0F}};
        transparentAttachments[1] =
            VkRenderingAttachmentInfo{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        transparentAttachments[1].imageView = revealage_.view();
        transparentAttachments[1].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        transparentAttachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        transparentAttachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        // Nothing has been drawn, so all of the background still shows.
        transparentAttachments[1].clearValue.color = {{1.0F, 1.0F, 1.0F, 1.0F}};

        VkRenderingAttachmentInfo keptDepth = depthAttachment;
        keptDepth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        keptDepth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        VkRenderingInfo transparentPass{VK_STRUCTURE_TYPE_RENDERING_INFO};
        transparentPass.renderArea.extent = state_.viewport.extent;
        transparentPass.layerCount = 1;
        transparentPass.colorAttachmentCount =
            static_cast<std::uint32_t>(transparentAttachments.size());
        transparentPass.pColorAttachments = transparentAttachments.data();
        transparentPass.pDepthAttachment = &keptDepth;

        vkCmdBeginRendering(commands, &transparentPass);
        vkCmdSetViewport(commands, 0, 1, &viewport);
        vkCmdSetScissor(commands, 0, 1, &scissor);
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, transparentPipeline_.get());
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelLayout_.get(), 0, 1,
                                &agentSet, 0, nullptr);
        if (drawTrails) {
            // Trail mode reuses the beacon xyz lanes for its ring metadata; the
            // actual beacon was already drawn in the opaque pass. The fourth
            // lane remains the first visible agent, shared by both modes.
            const std::array<std::int32_t, 4> savedBeacon = parameters.beacon;
            const float savedScale = parameters.camera[3];
            const float savedOpacity = parameters.tint[0];
            parameters.beacon[0] = static_cast<std::int32_t>(trailSamples);
            parameters.beacon[1] = static_cast<std::int32_t>(state_.trails.newest);
            parameters.beacon[2] = static_cast<std::int32_t>(state_.trails.capacity);
            parameters.camera[3] = std::clamp(display.trailScale, 0.05F, 0.90F);
            parameters.tint[0] = std::clamp(display.trailOpacity, 0.01F, 1.0F);
            pushWord(modeTrail);
            push();
            vkCmdDraw(commands, cubeVertexCount, visibleAgents * trailSamples, 0, 0);
            parameters.beacon = savedBeacon;
            parameters.camera[3] = savedScale;
            parameters.tint[0] = savedOpacity;
        }
        // Blocks and agents go into the same weighted average, in no
        // particular order -- that is what the pass is for. The blocks are
        // listed first only because that is the order they are built in.
        if (drawStructures && transparent) {
            drawStructureField();
        }
        if (drawAgents && transparent) {
            pushWord(modeAgents);
            push();
            vkCmdDraw(commands, cubeVertexCount, visibleAgents, 0, 0);
        }
        vkCmdEndRendering(commands);

        cmdImageBarrier(
            commands, accumulation_.image(), accumulationLayout_,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        cmdImageBarrier(
            commands, revealage_.image(), revealageLayout_,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        accumulationLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        revealageLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        // The resolve reads what the opaque pass wrote and writes the same
        // image, and two render pass instances are only ordered by a dependency
        // that says so.
        cmdImageBarrier(
            commands, color_.image(), colorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);

        VkRenderingAttachmentInfo keptColor = colorAttachment;
        keptColor.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        VkRenderingInfo resolvePass{VK_STRUCTURE_TYPE_RENDERING_INFO};
        resolvePass.renderArea.extent = state_.viewport.extent;
        resolvePass.layerCount = 1;
        resolvePass.colorAttachmentCount = 1;
        resolvePass.pColorAttachments = &keptColor;
        vkCmdBeginRendering(commands, &resolvePass);
        vkCmdSetViewport(commands, 0, 1, &viewport);
        vkCmdSetScissor(commands, 0, 1, &scissor);
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, resolvePipeline_.get());
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, resolveLayout_.get(), 0,
                                1, &resolveSet_, 0, nullptr);
        vkCmdDraw(commands, 3, 1, 0, 0);
        vkCmdEndRendering(commands);
    }

    cmdImageBarrier(
        commands, color_.image(), colorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    colorLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void LatticeRenderer::onDetach(AppContext&) {
    destroyTarget();
    resolvePipeline_.reset();
    resolveLayout_.reset();
    transparentPipeline_.reset();
    boundsPipeline_.reset();
    voxelPipeline_.reset();
    voxelLayout_.reset();
    resolveAllocator_.reset();
    resolveSetLayout_.reset();
    resolveSet_ = VK_NULL_HANDLE;
    agentAllocator_.reset();
    agentSetLayout_.reset();
    agentSets_ = {};
}

} // namespace vkexp
