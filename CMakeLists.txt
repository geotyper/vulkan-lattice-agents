cmake_minimum_required(VERSION 3.24)

project(vulkan_neuroevolution_agents VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VKEXP_ENABLE_VALIDATION "Enable Vulkan validation layers in debug builds" ON)

include(CTest)

find_package(Vulkan 1.3 REQUIRED)
find_package(glfw3 3.3 REQUIRED)

include(FetchContent)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.8
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)

find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

set(SHADER_SOURCES
    game_of_life.comp
    agent_grid_clear.comp
    agent_grid_build.comp
    agent_step.comp
    trail_decay.comp
    trail_deposit.comp
    puck_step.comp
    agents.vert
    agents.frag
)
# Sources compiled into both the shaders and vkneuro_domain: the network preset
# and the scenario math.
file(GLOB SHADER_SHARED_KERNELS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vkexp/*/*.inl
)
file(GLOB SHADER_INCLUDE_SOURCES CONFIGURE_DEPENDS
    ${SHADER_SOURCE_DIR}/worlds/*.glsl
    ${SHADER_SOURCE_DIR}/worlds/steps/*.glsl
    ${SHADER_SOURCE_DIR}/neuro/*.glsl
    ${SHADER_SOURCE_DIR}/simulation/*.glsl
)
list(APPEND SHADER_INCLUDE_SOURCES ${SHADER_SHARED_KERNELS})

set(SHADER_BINARIES)
foreach(SHADER ${SHADER_SOURCES})
    set(SPIRV ${SHADER_BINARY_DIR}/${SHADER}.spv)
    # Every shader now shares at least the agent layout, and a shader rebuilt
    # against a stale copy of a shared contract is the failure this list exists
    # to prevent, so they all depend on all of it.
    set(SHADER_EXTRA_DEPENDENCIES ${SHADER_INCLUDE_SOURCES})
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BINARY_DIR}
        COMMAND ${GLSLANG_VALIDATOR} -V -I${SHADER_SOURCE_DIR}
                -I${CMAKE_CURRENT_SOURCE_DIR}/include
                ${SHADER_SOURCE_DIR}/${SHADER} -o ${SPIRV}
        DEPENDS ${SHADER_SOURCE_DIR}/${SHADER} ${SHADER_EXTRA_DEPENDENCIES}
        COMMENT "Compiling ${SHADER}"
        VERBATIM
    )
    list(APPEND SHADER_BINARIES ${SPIRV})
endforeach()
add_custom_target(vkexp_shaders ALL DEPENDS ${SHADER_BINARIES})

function(vkexp_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        # Vulkan C structs intentionally use partial aggregate initialization.
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Wno-missing-field-initializers
        )
    endif()
endfunction()

add_library(vkexp_profiling STATIC
    src/profiling/CpuProfiler.cpp
    src/profiling/GpuProfiler.cpp
    src/profiling/Profiler.cpp
)
target_include_directories(vkexp_profiling PUBLIC include)
target_link_libraries(vkexp_profiling PUBLIC Vulkan::Vulkan)
vkexp_enable_warnings(vkexp_profiling)

add_library(vkexp_core STATIC
    src/core/Application.cpp
    src/core/VulkanContext.cpp
    src/core/VulkanResource.cpp
    src/core/Window.cpp
)
target_include_directories(vkexp_core PUBLIC include)
target_link_libraries(vkexp_core PUBLIC Vulkan::Vulkan glfw vkexp_profiling)
vkexp_enable_warnings(vkexp_core)

add_library(vkexp_imgui STATIC
    src/ui/ImGuiModule.cpp
    src/profiling/ProfilerPanel.cpp
)
target_include_directories(vkexp_imgui PUBLIC include)
target_link_libraries(vkexp_imgui
    PUBLIC vkexp_core vkexp_profiling
    PRIVATE imgui
)
target_compile_definitions(vkexp_imgui PRIVATE
    VKEXP_IMGUI_INI="${CMAKE_CURRENT_BINARY_DIR}/imgui.ini"
)
vkexp_enable_warnings(vkexp_imgui)

add_library(vkexp_compute STATIC
    src/compute/ComputeResources.cpp
    src/compute/HeadlessComputeContext.cpp
)
target_include_directories(vkexp_compute PUBLIC include)
target_link_libraries(vkexp_compute PUBLIC vkexp_core)
vkexp_enable_warnings(vkexp_compute)

add_library(vkneuro_domain STATIC
    src/neuro/NeuralNetwork.cpp
    src/simulation/Sensors.cpp
    src/simulation/CpuSimulation.cpp
    src/evolution/GeneticAlgorithm.cpp
    src/evolution/GenomeArchive.cpp
    src/simulation/WorldSnapshot.cpp
    src/simulation/ExperimentSweep.cpp
    src/simulation/StepParameters.cpp
    src/worlds/WorldScenario.cpp
    src/worlds/scenarios/ShuttleScenario.cpp
    src/worlds/scenarios/StationaryScenario.cpp
    src/worlds/scenarios/TwoDoorsScenario.cpp
    src/worlds/scenarios/TwoGapsScenario.cpp
    src/worlds/scenarios/PuckPushScenario.cpp
    src/worlds/scenarios/AlternatingScenario.cpp
    src/worlds/scenarios/RotatingScenario.cpp
    src/worlds/scenarios/RandomMovementScenario.cpp
    src/worlds/scenarios/ForageHomeScenario.cpp
    src/worlds/scenarios/ScentRelayScenario.cpp
)
target_include_directories(vkneuro_domain PUBLIC include)
vkexp_enable_warnings(vkneuro_domain)

add_library(vkneuro_simulation STATIC
    src/simulation/SimulationDriver.cpp
    src/simulation/SimulationModule.cpp
)
add_dependencies(vkneuro_simulation vkexp_shaders)
target_include_directories(vkneuro_simulation PUBLIC include)
target_link_libraries(vkneuro_simulation PUBLIC vkneuro_domain vkexp_compute)
target_compile_definitions(vkneuro_simulation PRIVATE
    VKEXP_SHADER_DIR="${SHADER_BINARY_DIR}"
)
vkexp_enable_warnings(vkneuro_simulation)

add_library(vkneuro_visualization STATIC
    src/graphics/AgentRenderer.cpp
)
add_dependencies(vkneuro_visualization vkexp_shaders)
target_include_directories(vkneuro_visualization PUBLIC include)
target_link_libraries(vkneuro_visualization PUBLIC vkneuro_domain vkexp_compute)
target_compile_definitions(vkneuro_visualization PRIVATE
    VKEXP_SHADER_DIR="${SHADER_BINARY_DIR}"
)
vkexp_enable_warnings(vkneuro_visualization)

add_library(vkneuro_ui STATIC
    src/ui/SimulationUiModule.cpp
)
target_include_directories(vkneuro_ui PUBLIC include)
target_link_libraries(vkneuro_ui
    PUBLIC vkneuro_domain vkexp_imgui
    PRIVATE imgui
)
vkexp_enable_warnings(vkneuro_ui)

add_executable(vkneuro_headless src/headless/main.cpp)
add_dependencies(vkneuro_headless vkexp_shaders)
target_link_libraries(vkneuro_headless PRIVATE vkneuro_simulation)
vkexp_enable_warnings(vkneuro_headless)

add_executable(vulkan_neuroevolution_agents src/main.cpp)
target_link_libraries(vulkan_neuroevolution_agents PRIVATE
    vkneuro_simulation vkneuro_visualization vkneuro_ui
)
target_compile_definitions(vulkan_neuroevolution_agents PRIVATE
    $<$<AND:$<BOOL:${VKEXP_ENABLE_VALIDATION}>,$<CONFIG:Debug>>:VKEXP_ENABLE_VALIDATION>
)
vkexp_enable_warnings(vulkan_neuroevolution_agents)

if(BUILD_TESTING)
    add_executable(vkexp_unit_tests tests/vkexp_unit_tests.cpp)
    target_link_libraries(vkexp_unit_tests PRIVATE vkexp_compute vkneuro_domain)
    vkexp_enable_warnings(vkexp_unit_tests)

    add_test(NAME vkexp_unit_tests COMMAND vkexp_unit_tests)
    add_test(NAME vkexp_cli_help COMMAND vulkan_neuroevolution_agents --help)
    add_test(NAME vkneuro_headless_help COMMAND vkneuro_headless --help)

    # A short real evolution run: proves the headless path drives the same
    # driver end to end and that archives round-trip through the runner.
    add_test(NAME vkneuro_headless_smoke COMMAND vkneuro_headless
        --scenario forage --generations 2 --steps 24 --population 32
        --agents-per-world 8 --quiet
        --save-population ${CMAKE_CURRENT_BINARY_DIR}/headless_population.vkng
        --save-champion ${CMAKE_CURRENT_BINARY_DIR}/headless_champion.vkng
        --save-world ${CMAKE_CURRENT_BINARY_DIR}/headless_world.vknw)
    set_tests_properties(vkneuro_headless_smoke PROPERTIES SKIP_RETURN_CODE 77)

    # The scenario with static geometry, driven end to end. The parity test
    # covers one agent against one wall; this covers a population evolving in a
    # world that has walls at all, where a spawn hook and an obstacle loop run
    # every step for every agent.
    add_test(NAME vkneuro_headless_doors COMMAND vkneuro_headless
        --scenario doors --generations 2 --steps 48 --population 32
        --agents-per-world 8 --quiet)
    set_tests_properties(vkneuro_headless_doors PROPERTIES SKIP_RETURN_CODE 77)

    add_test(NAME vkneuro_headless_shuttle COMMAND vkneuro_headless
        --scenario shuttle --generations 2 --steps 48 --population 32
        --agents-per-world 8 --quiet)
    set_tests_properties(vkneuro_headless_shuttle PROPERTIES SKIP_RETURN_CODE 77)

    add_test(NAME vkneuro_headless_resume COMMAND vkneuro_headless
        --scenario forage --generations 1 --steps 24 --population 32
        --agents-per-world 8 --quiet
        --load-population ${CMAKE_CURRENT_BINARY_DIR}/headless_population.vkng)
    set_tests_properties(vkneuro_headless_resume PROPERTIES
        SKIP_RETURN_CODE 77
        DEPENDS vkneuro_headless_smoke)

    # No --scenario, no --world-size, no --steps: the snapshot carries the world
    # it ran in, and a run that needed the command line to restate it would not
    # be a resume.
    add_test(NAME vkneuro_headless_resume_world COMMAND vkneuro_headless
        --generations 1 --population 32 --agents-per-world 8 --quiet
        --load-world ${CMAKE_CURRENT_BINARY_DIR}/headless_world.vknw)
    set_tests_properties(vkneuro_headless_resume_world PROPERTIES
        SKIP_RETURN_CODE 77
        DEPENDS vkneuro_headless_smoke)

    add_executable(vkexp_compute_smoke tests/compute_smoke.cpp)
    add_dependencies(vkexp_compute_smoke vkexp_shaders)
    target_link_libraries(vkexp_compute_smoke PRIVATE vkexp_compute vkneuro_domain)
    target_compile_definitions(vkexp_compute_smoke PRIVATE
        VKEXP_SHADER_DIR="${SHADER_BINARY_DIR}"
    )
    vkexp_enable_warnings(vkexp_compute_smoke)
    add_test(NAME vkexp_compute_smoke COMMAND vkexp_compute_smoke)
    set_tests_properties(vkexp_compute_smoke PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(vkneuro_replay_smoke tests/replay_smoke.cpp)
    add_dependencies(vkneuro_replay_smoke vkexp_shaders)
    target_link_libraries(vkneuro_replay_smoke PRIVATE vkneuro_simulation)
    target_compile_definitions(vkneuro_replay_smoke PRIVATE
        VKEXP_SHADER_DIR="${SHADER_BINARY_DIR}"
    )
    vkexp_enable_warnings(vkneuro_replay_smoke)
    add_test(NAME vkneuro_replay_smoke COMMAND vkneuro_replay_smoke)
    set_tests_properties(vkneuro_replay_smoke PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(vkneuro_puck_smoke tests/puck_smoke.cpp)
    add_dependencies(vkneuro_puck_smoke vkexp_shaders)
    target_link_libraries(vkneuro_puck_smoke PRIVATE vkneuro_simulation)
    target_compile_definitions(vkneuro_puck_smoke PRIVATE
        VKEXP_SHADER_DIR="${SHADER_BINARY_DIR}"
    )
    vkexp_enable_warnings(vkneuro_puck_smoke)
    add_test(NAME vkneuro_puck_smoke COMMAND vkneuro_puck_smoke)
    set_tests_properties(vkneuro_puck_smoke PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(vkneuro_reconfiguration_smoke tests/reconfiguration_smoke.cpp)
    add_dependencies(vkneuro_reconfiguration_smoke vkexp_shaders)
    target_link_libraries(vkneuro_reconfiguration_smoke PRIVATE vkneuro_simulation)
    target_compile_definitions(vkneuro_reconfiguration_smoke PRIVATE
        VKEXP_SHADER_DIR="${SHADER_BINARY_DIR}"
    )
    vkexp_enable_warnings(vkneuro_reconfiguration_smoke)
    add_test(NAME vkneuro_reconfiguration_smoke COMMAND vkneuro_reconfiguration_smoke)
    # A reconfiguration bug shows up as a hang, not as a wrong answer, so the
    # timeout is part of the assertion rather than a safety net.
    set_tests_properties(vkneuro_reconfiguration_smoke PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 120
    )
endif()
