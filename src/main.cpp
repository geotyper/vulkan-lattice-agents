#include "vkexp/core/Application.hpp"
#include "vkexp/graphics/LatticeRenderer.hpp"
#include "vkexp/simulation/SimulationModule.hpp"
#include "vkexp/simulation/SimulationState.hpp"
#include "vkexp/ui/ImGuiModule.hpp"
#include "vkexp/ui/SimulationUiModule.hpp"

#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

namespace {

void printHelp(const char* executable) {
    std::cout << "Usage: " << executable << " [--no-validation]\n";
}

} // namespace

int main(const int argc, char** argv) {
    try {
#ifdef VKEXP_ENABLE_VALIDATION
        bool validationEnabled = true;
#else
        bool validationEnabled = false;
#endif

        for (int i = 1; i < argc; ++i) {
            const std::string_view argument = argv[i];
            if (argument == "--no-validation") {
                validationEnabled = false;
            } else if (argument == "--help" || argument == "-h") {
                printHelp(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("Unknown argument: " + std::string{argument});
            }
        }

        vkexp::SimulationState state;
        // This branch's interactive experiment opens on the construction task;
        // the domain default remains the beacon baseline for headless runs and
        // existing archives that choose no world explicitly.
        state.settings.worldMode = vkexp::WorldMode::Construction;
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
