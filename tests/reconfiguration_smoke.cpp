// Reconfiguration coverage for SimulationDriver.
//
// Changing the lattice, the group size or the brain plan all resize or re-derive
// GPU resources under a running simulation, and in the 2D build each of those
// hung the application at least once. The failures were never in the settings
// but in ownership: buffers sized for a worst case that no longer matched what
// was running, descriptors left pointing at a reallocated handle, and a
// shrinking loop that had to terminate. None of that is reachable from a
// single-configuration test, because a fresh driver always agrees with itself.
//
// So this walks the reconfigurations the UI can produce and steps the simulation
// after each one. A hang shows up as the ctest timeout rather than as a wedged
// desktop, and anything that survives the step is checked for having actually
// simulated rather than quietly produced NaNs.

#include "vkexp/compute/HeadlessComputeContext.hpp"
#include "vkexp/lattice/LatticeKernel.hpp"
#include "vkexp/simulation/SimulationDriver.hpp"
#include "vkexp/simulation/SimulationState.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int skipExitCode = 77;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Runs a short generation and checks the population actually moved through it.
// The buffers below are allocated once and never resized, so their handles are
// the same for the life of the driver. That invariant is what makes every
// descriptor holding them correct by construction rather than by remembering to
// refresh them, which is what went wrong twice in the 2D build.
//
// The count of builds is what carries the invariant, and the handles are checked
// beside it. Comparing handles alone does not settle it: a buffer freed and
// immediately reallocated at the same size usually comes back as the same
// VkBuffer, so such a test passes while the driver is destroying buffers under
// live descriptors. An assertion that can only fail when an allocator declines
// to reuse a handle is not an assertion.
struct PublishedBuffers {
    VkBuffer lattice{};
    std::array<VkBuffer, 2> agents{};
};
PublishedBuffers expectedBuffers{};
bool expectedBuffersCaptured = false;

void stepAndCheck(vkexp::HeadlessComputeContext& context, vkexp::SimulationDriver& driver,
                  vkexp::SimulationState& state, const std::string& what) {
    std::cout << "  " << what << ": " << state.worlds.worldCount << " worlds of "
              << state.settings.latticeWidth << "x" << state.settings.latticeHeight << "x"
              << state.settings.latticeDepth << " = " << state.lattice.cellsPerWorld << " cells"
              << std::endl;
    require(state.lattice.buffer != VK_NULL_HANDLE, what + ": the occupancy grid is published");
    require(state.agents.buffers[0] != VK_NULL_HANDLE &&
                state.agents.buffers[1] != VK_NULL_HANDLE,
            what + ": both agent buffers are published");
    const PublishedBuffers published{state.lattice.buffer,
                                     {state.agents.buffers[0], state.agents.buffers[1]}};
    if (!expectedBuffersCaptured) {
        expectedBuffers = published;
        expectedBuffersCaptured = true;
    }
    require(driver.stepResourceBuilds() == 1,
            what + ": the fixed step resources were built once and not rebuilt");
    require(published.lattice == expectedBuffers.lattice,
            what + ": the occupancy grid was not reallocated, so no descriptor went stale");
    require(published.agents == expectedBuffers.agents,
            what + ": agent buffers were not reallocated, so no descriptor went stale");
    require(state.lattice.cellsPerWorld > 0, what + ": the lattice has cells");
    require(state.worlds.worldCount > 0, what + ": at least one logical world");
    // The clamp is the whole reason a slider cannot produce a configuration that
    // allocates: the allocation is fixed and the box gives way.
    require(static_cast<std::uint64_t>(state.lattice.cellsPerWorld) * state.worlds.worldCount *
                    sizeof(std::int32_t) <=
                state.lattice.size,
            what + ": the chosen lattice fits the fixed allocation");

    while (!driver.generationComplete()) {
        context.immediate().execute(
            [&](const VkCommandBuffer commands) { driver.recordSteps(commands, 64); });
    }
    const vkexp::GenerationSummary summary = driver.finishGeneration();
    require(std::isfinite(summary.bestFitness) && std::isfinite(summary.medianFitness),
            what + ": generation produced finite fitness");

    // A grid sized for a different configuration would leave the shader reading
    // or writing outside the agents' own world; the cheapest thing that catches
    // it is that every agent is still inside the box it was given.
    for (const vkexp::AgentState& agent : driver.agents()) {
        require(vkexp::lattice::kernel::latticeInBounds(
                    agent.cell.x, agent.cell.y, agent.cell.z, state.settings.latticeWidth,
                    state.settings.latticeHeight, state.settings.latticeDepth),
                what + ": every agent stayed inside its lattice");
    }
}

int run() {
    vkexp::HeadlessComputeContext context{
        vkexp::HeadlessComputeConfig{.applicationName = "vklat reconfiguration smoke"}};

    // The population has to be the real one. The per-world cost scales with the
    // world count, and the world count grows as the group shrinks: 512 genomes
    // at 10 agents per world across 4 trials is 208 lattices, which is the
    // configuration the budget clamp actually has to hold. A small population
    // makes every allocation here comfortable and proves nothing about the real
    // one.
    vkexp::SimulationState state{};
    state.controls.stepsPerGeneration = 48;
    state.worlds.requestedAgentsPerWorld = 29;

    vkexp::SimulationDriver driver{state, vkexp::EvolutionSettings{.populationSize = 512}, {}};
    driver.createResources(context.physicalDevice(), context.device());
    stepAndCheck(context, driver, state, "initial configuration");

    // 1. The lattice itself, largest first so the budget clamp is exercised while
    //    the world count is still high. The driver may shrink the request; what
    //    it must not do is fail to terminate, allocate, or keep a stale grid.
    for (const std::array<std::uint32_t, 3> box : {std::array<std::uint32_t, 3>{128, 128, 128},
                                                   std::array<std::uint32_t, 3>{64, 64, 32},
                                                   std::array<std::uint32_t, 3>{32, 32, 16},
                                                   std::array<std::uint32_t, 3>{8, 8, 8},
                                                   std::array<std::uint32_t, 3>{1, 1, 1}}) {
        state.settings.latticeWidth = box[0];
        state.settings.latticeHeight = box[1];
        state.settings.latticeDepth = box[2];
        driver.restart();
        require(state.settings.latticeWidth >= vkexp::latticeMinimumExtent &&
                    state.settings.latticeHeight >= vkexp::latticeMinimumExtent &&
                    state.settings.latticeDepth >= vkexp::latticeMinimumExtent,
                "lattice: never shrunk past the smallest extent");
        require(state.settings.latticeWidth <= std::max(box[0], vkexp::latticeMinimumExtent) &&
                    state.settings.latticeHeight <=
                        std::max(box[1], vkexp::latticeMinimumExtent) &&
                    state.settings.latticeDepth <= std::max(box[2], vkexp::latticeMinimumExtent),
                "lattice: clamped, never inflated");
        stepAndCheck(context, driver, state,
                     "lattice " + std::to_string(box[0]) + "x" + std::to_string(box[1]) + "x" +
                         std::to_string(box[2]));
    }

    // 2. Group size, in both directions. Shrinking it multiplies the number of
    //    logical worlds, which is what grows the per-world grids; growing it back
    //    must not leave the driver believing in the larger layout.
    state.settings.latticeWidth = 32;
    state.settings.latticeHeight = 32;
    state.settings.latticeDepth = 16;
    for (const std::uint32_t agentsPerWorld : {512U, 29U, 12U, 10U, 29U, 12U}) {
        state.worlds.requestedAgentsPerWorld = agentsPerWorld;
        driver.restart();
        require(state.worlds.agentsPerWorld <= agentsPerWorld,
                "group size " + std::to_string(agentsPerWorld) + ": clamped, never inflated");
        stepAndCheck(context, driver, state, "group size " + std::to_string(agentsPerWorld));
    }

    // 3. The brain plan. This is the reconfiguration that resizes a buffer for a
    //    reason that has nothing to do with the world: a different number of
    //    hidden layers is a different genome length. It used to remake every step
    //    resource, which freed the agent and world buffers under descriptors that
    //    were still naming them.
    //
    //    Each plan is checked for having actually changed the genome length,
    //    because a case that quietly resized nothing would walk none of this and
    //    still pass.
    {
        std::size_t previousWeights = driver.evolution().settings().weightCount;
        for (const std::array<std::uint32_t, 3> plan : {std::array<std::uint32_t, 3>{12, 8, 8},
                                                        std::array<std::uint32_t, 3>{10, 10, 0},
                                                        std::array<std::uint32_t, 3>{32, 0, 0},
                                                        std::array<std::uint32_t, 3>{0, 0, 0}}) {
            state.settings.hiddenLayers = plan;
            driver.restart();
            const std::size_t weights = driver.evolution().settings().weightCount;
            const std::string label = "brain plan " + std::to_string(plan[0]) + "," +
                                      std::to_string(plan[1]) + "," + std::to_string(plan[2]);
            require(weights != previousWeights, label + ": genome length actually changed");
            require(driver.evolution().population().front().weights.size() == weights,
                    label + ": the population was rebuilt at the new length");
            previousWeights = weights;
            stepAndCheck(context, driver, state, label);
        }
    }

    // 4. Every lattice against every group size the UI offers, under both
    //    neighbourhoods. The cross product is the point: the world count and the
    //    per-world cost move in opposite directions, and the pairs where they
    //    meet are the ones a single sweep of either axis never visits.
    for (const std::array<std::uint32_t, 3> box : {std::array<std::uint32_t, 3>{64, 64, 32},
                                                   std::array<std::uint32_t, 3>{32, 32, 16},
                                                   std::array<std::uint32_t, 3>{8, 8, 4}}) {
        for (const std::uint32_t agentsPerWorld : {29U, 24U, 12U, 10U}) {
            for (const vkexp::Neighborhood neighborhood :
                 {vkexp::Neighborhood::Moore, vkexp::Neighborhood::Faces}) {
                state.settings.latticeWidth = box[0];
                state.settings.latticeHeight = box[1];
                state.settings.latticeDepth = box[2];
                state.settings.neighborhood = neighborhood;
                state.worlds.requestedAgentsPerWorld = agentsPerWorld;
                driver.restart();
                stepAndCheck(context, driver, state,
                             "lattice " + std::to_string(box[0]) + "x" + std::to_string(box[1]) +
                                 "x" + std::to_string(box[2]) + ", " +
                                 std::to_string(agentsPerWorld) + " agents/world");
            }
        }
    }

    driver.destroyResources();
    std::cout << "Reconfiguration smoke passed on " << context.deviceName() << '\n';
    return EXIT_SUCCESS;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const vkexp::HeadlessComputeUnavailable& unavailable) {
        std::cout << "Skipping reconfiguration smoke: " << unavailable.what() << '\n';
        return skipExitCode;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
