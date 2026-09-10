#pragma once

#include "vkexp/compute/ComputeResources.hpp"
#include "vkexp/core/Module.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"
#include "vkexp/simulation/SimulationState.hpp"

#include <array>

namespace vkexp {

class Profiler;

class AgentRenderer final : public Module {
public:
    AgentRenderer(SimulationState& state, Profiler& profiler);

    void onAttach(AppContext& context) override;
    void onUpdate(AppContext& context, const FrameInfo& frame) override;
    void onRender(AppContext& context, const FrameInfo& frame) override;
    void onDetach(AppContext& context) override;

private:
    static constexpr VkFormat targetFormat = VK_FORMAT_R8G8B8A8_UNORM;

    void createPipeline(AppContext& context);
    void createTarget(AppContext& context, VkExtent2D extent);
    void destroyTarget();
    // The three buffers these sets name -- agents, trail field, pucks -- are made
    // once by SimulationDriver and never remade, so the sets are written at attach
    // time and are correct for the life of the renderer. There is deliberately no
    // per-frame refresh: this class had one, and it was a patch over a driver that
    // freed those buffers on a brain-plan change. Resizing only what changes size
    // is the fix; reconfiguration_smoke pins the handles across every
    // reconfiguration the UI can produce.
    void draw(VkCommandBuffer commands, float scaleX, float scaleY, float worldRadius,
              std::uint32_t mode, float opacity, std::uint32_t vertices,
              std::uint32_t instances) const;

    SimulationState& state_;
    ProfileMetricId metric_{invalidProfileMetric};
    UniqueDescriptorSetLayout descriptorSetLayout_;
    DescriptorAllocator descriptorAllocator_;
    std::array<VkDescriptorSet, 2> descriptorSets_{};
    UniquePipelineLayout pipelineLayout_;
    UniquePipeline pipeline_;
    ImageResource target_;
    VkImageLayout targetLayout_{VK_IMAGE_LAYOUT_UNDEFINED};
};

} // namespace vkexp
