#pragma once

#include "vkexp/compute/ComputeResources.hpp"
#include "vkexp/core/Module.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"
#include "vkexp/simulation/SimulationState.hpp"

#include <array>

namespace vkexp {

class Profiler;

// A view of one logical lattice: the agents of the selected world as unit
// cubes, the beacon they are looking for, and the box they are inside.
//
// It is a read-only consumer of the agent buffer, like the 2D renderer before
// it: nothing here is written back, and turning a layer off changes the picture
// and not the run. What it does not share with that renderer is any code --
// that one drew normalised coordinates with no camera at all, and a lattice
// needs a camera before it needs anything else.
class LatticeRenderer final : public Module {
public:
    LatticeRenderer(SimulationState& state, Profiler& profiler);

    void onAttach(AppContext& context) override;
    void onUpdate(AppContext& context, const FrameInfo& frame) override;
    void onRender(AppContext& context, const FrameInfo& frame) override;
    void onDetach(AppContext& context) override;

private:
    static constexpr VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    static constexpr VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
    // Transparency resolves through a weighted sum, so the accumulation buffer
    // has to hold values well outside 0..1 -- an 8-bit target clips the weights
    // and the picture goes flat. See view_voxels_oit.frag for the method.
    static constexpr VkFormat accumulationFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    static constexpr VkFormat revealageFormat = VK_FORMAT_R16_SFLOAT;

    void createPipelines(AppContext& context);
    void createTarget(AppContext& context, VkExtent2D extent);
    void destroyTarget();
    void writeResolveDescriptor(VkDevice device);

    SimulationState& state_;
    ProfileMetricId metric_{invalidProfileMetric};

    // The agent and trail-history buffers are made once by SimulationDriver and
    // never remade, so these two sets are written at attach time and stay
    // correct for the life of the renderer. There is deliberately no per-frame
    // refresh: the 2D renderer had one, and it was a patch over a driver that
    // used to free those buffers on a brain-plan change.
    UniqueDescriptorSetLayout agentSetLayout_;
    DescriptorAllocator agentAllocator_;
    std::array<VkDescriptorSet, 2> agentSets_{};

    // The resolve set names the two transparency buffers, which are remade
    // whenever the viewport changes size, so this one is rewritten with them.
    UniqueDescriptorSetLayout resolveSetLayout_;
    DescriptorAllocator resolveAllocator_;
    VkDescriptorSet resolveSet_{};

    UniquePipelineLayout voxelLayout_;
    UniquePipeline voxelPipeline_;       // opaque cubes, depth written
    UniquePipeline boundsPipeline_;      // the box, as a line list
    UniquePipeline transparentPipeline_; // cubes into the two OIT buffers
    UniquePipelineLayout resolveLayout_;
    UniquePipeline resolvePipeline_;

    ImageResource color_;
    ImageResource depth_;
    ImageResource accumulation_;
    ImageResource revealage_;
    VkImageLayout colorLayout_{VK_IMAGE_LAYOUT_UNDEFINED};
    VkImageLayout depthLayout_{VK_IMAGE_LAYOUT_UNDEFINED};
    VkImageLayout accumulationLayout_{VK_IMAGE_LAYOUT_UNDEFINED};
    VkImageLayout revealageLayout_{VK_IMAGE_LAYOUT_UNDEFINED};
};

} // namespace vkexp
