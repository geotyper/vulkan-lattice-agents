#pragma once

#include "vkexp/core/Module.hpp"
#include "vkexp/profiling/ProfilerPanel.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"
#include "vkexp/simulation/SimulationState.hpp"

#include <vulkan/vulkan.h>

namespace vkexp {

namespace neuro {
struct BrainShape;
}

class ImGuiModule;
class Profiler;

class SimulationUiModule final : public Module {
public:
    SimulationUiModule(SimulationState& state, ImGuiModule& imgui, Profiler& profiler);

    void onAttach(AppContext& context) override;
    void onUpdate(AppContext& context, const FrameInfo& frame) override;
    void onDetach(AppContext& context) override;

private:
    void syncTexture();
    // The network's own window: how many hidden layers this run has and how wide
    // they are. Separate because it is a question about the brain rather than
    // about the world, and because it is edited rarely and read often.
    void drawBrainWindow(const neuro::BrainShape& brain);
    // Everything about the picture and nothing about the run. It has its own
    // settings window so the render window itself can remain an unobstructed
    // image, with no selectors or menus taking space from the lattice.
    void drawViewControls();
    void drawStructureShapes();
    void drawBuildOutcomes();

    SimulationState& state_;
    ImGuiModule& imgui_;
    ProfileMetricId metric_{invalidProfileMetric};
    VkDescriptorSet viewportDescriptor_{};
    std::uint64_t viewportGeneration_{};
    ProfilerPanel profilerPanel_;
};

} // namespace vkexp
