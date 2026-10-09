#include "vkexp/core/Application.hpp"
#include "vkexp/graphics/LatticeRenderer.hpp"
#include "vkexp/simulation/SimulationModule.hpp"
#include "vkexp/simulation/SimulationState.hpp"
#include "vkexp/ui/ImGuiModule.hpp"
#include "vkexp/ui/SimulationUiModule.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

namespace {

// In the order of vkexp::WorldMode, so a name's place in the list is its value.
constexpr std::array<std::string_view, vkexp::worldModeCount> worldNames{
    "beacon", "construction", "harvest", "chasm", "canopy"};

void printHelp(const char* executable) {
    std::cout << "Usage: " << executable << " [--no-validation] [--world <name>]\n"
              << "  --world <name>   beacon|construction|harvest|chasm|canopy: the world the\n"
              << "                   window opens on (default construction)\n";
}

} // namespace

int main(const int argc, char** argv) {
    try {
#ifdef VKEXP_ENABLE_VALIDATION
        bool validationEnabled = true;
#else
        bool validationEnabled = false;
#endif

        // This branch's interactive experiment opens on the construction task;
        // the domain default remains the beacon baseline for headless runs and
        // existing archives that choose no world explicitly.
        vkexp::WorldMode worldMode = vkexp::WorldMode::Construction;
        for (int i = 1; i < argc; ++i) {
            const std::string_view argument = argv[i];
            if (argument == "--no-validation") {
                validationEnabled = false;
            } else if (argument == "--world" && i + 1 < argc) {
                const auto* const named =
                    std::find(worldNames.begin(), worldNames.end(), std::string_view{argv[++i]});
                if (named == worldNames.end()) {
                    throw std::runtime_error("Unknown world: " + std::string{argv[i]});
                }
                worldMode = static_cast<vkexp::WorldMode>(named - worldNames.begin());
            } else if (argument == "--help" || argument == "-h") {
                printHelp(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("Unknown argument: " + std::string{argument});
            }
        }

        vkexp::SimulationState state;
        state.settings.worldMode = worldMode;
        // Only when a world was asked for by name: the construction task this
        // window has always opened on keeps the settings it has always had.
        if (worldMode != vkexp::WorldMode::Construction) {
            vkexp::applyWorldDefaults(state.settings);
        }
        if (worldMode == vkexp::WorldMode::Canopy) {
            state.controls.stepsPerGeneration = vkexp::canopyDefaultStepsPerGeneration;
        }
        vkexp::Application app{vkexp::ApplicationConfig{
            1440,
            900,
            "Vulkan Lattice Lab",
            validationEnabled,
        }};

        auto imgui = std::make_unique<vkexp::ImGuiModule>(app.profiler());
        auto& imguiBackend = *imgui;
        // Attachment and render order is an explicit dependency graph: the
        // simulation publishes its buffers, the view reads them into an
        // off-screen image, and ImGui composites that image into a panel. The
        // renderer never writes back, so the arrow only points one way.
        app.addModule(std::make_unique<vkexp::SimulationModule>(state, app.profiler()));
        app.addModule(std::make_unique<vkexp::LatticeRenderer>(state, app.profiler()));
        app.addModule(std::move(imgui));
        app.addModule(
            std::make_unique<vkexp::SimulationUiModule>(state, imguiBackend, app.profiler()));
        return app.run();
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
