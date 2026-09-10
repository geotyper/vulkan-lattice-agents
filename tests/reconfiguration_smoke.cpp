// Reconfiguration coverage for SimulationDriver.
//
// Changing the arena size, the group size or the trail resolution all resize GPU
// resources under a running simulation, and each of those hung the application at
// least once. The failures were not in the settings but in ownership: buffers
// sized for a worst case that no longer matched what was running, descriptors
// left pointing at a reallocated handle, and a coarsening loop that had to
// terminate. None of that is reachable from a single-configuration test, because
// a fresh driver always agrees with itself.
//
// So this walks the reconfigurations the UI can produce and steps the simulation
// after each one. A hang shows up as the ctest timeout rather than as a wedged
// desktop, and anything that survives the step is checked for having actually
// simulated rather than quietly produced NaNs.

#include "vkexp/compute/HeadlessComputeContext.hpp"
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
// descriptor holding them -- three compute sets and the renderer's three --
// correct by construction rather than by remembering to refresh them, which is
// what went wrong twice.
//
// Checking the trail alone was not enough, and it failed twice over. Giving the
// genome its own length made a plan change call createStepResources again, which
// freed the agent and puck buffers as well; the renderer kept naming all three
// and the GPU hung on the next frame. Nothing here changed the brain plan, so
// the path was never walked -- and when it was walked deliberately, the trail
// handle came back identical anyway, because a buffer freed and reallocated at
// the same size is usually the same VkBuffer. An assertion that can only fail
// when an allocator declines to reuse a handle is not an assertion.
//
// So the count of builds is what carries the invariant, and the handles are
// checked beside it: they catch the other failure, which is publishing a handle
// that no longer names what the driver is using.
struct PublishedBuffers {
    VkBuffer trail{};
    VkBuffer puck{};
    std::array<VkBuffer, 2> agents{};
};
PublishedBuffers expectedBuffers{};
bool expectedBuffersCaptured = false;

void stepAndCheck(vkexp::HeadlessComputeContext& context, vkexp::SimulationDriver& driver,
                  vkexp::SimulationState& state, const std::string& what) {
    std::cout << "  " << what << ": " << state.worlds.worldCount << " worlds, trail "
              << state.trail.width << "x" << state.trail.width << " cells of "
              << state.physics.trailCellSize << " m" << std::endl;
    require(state.trail.buffer != VK_NULL_HANDLE, what + ": trail field is published");
    require(state.puck.buffer != VK_NULL_HANDLE, what + ": puck field is published");
    require(state.agents.buffers[0] != VK_NULL_HANDLE &&
                state.agents.buffers[1] != VK_NULL_HANDLE,
            what + ": both agent buffers are published");
    const PublishedBuffers published{
        state.trail.buffer, state.puck.buffer, {state.agents.buffers[0], state.agents.buffers[1]}};
    if (!expectedBuffersCaptured) {
        expectedBuffers = published;
        expectedBuffersCaptured = true;
    }
    // The invariant itself, ahead of the handle checks below, which are only a
    // proxy for it: a buffer freed and immediately reallocated at the same size
    // usually comes back as the same VkBuffer, so comparing handles can pass
    // while the driver is destroying buffers under live descriptors. It did --
    // when the bug this test now covers was reintroduced deliberately, the trail
    // and puck handles matched and only the agent pair differed. The count does
    // not depend on an allocator's habits.
    require(driver.stepResourceBuilds() == 1,
            what + ": the fixed step resources were built once and not rebuilt");
    require(published.trail == expectedBuffers.trail,
            what + ": trail field was not reallocated, so no descriptor went stale");
    require(published.puck == expectedBuffers.puck,
            what + ": puck field was not reallocated, so no descriptor went stale");
    require(published.agents == expectedBuffers.agents,
            what + ": agent buffers were not reallocated, so no descriptor went stale");
    require(state.trail.cellsPerWorld > 0, what + ": trail field has cells");
    require(state.worlds.worldCount > 0, what + ": at least one logical world");

    while (!driver.generationComplete()) {
        context.immediate().execute(
            [&](const VkCommandBuffer commands) { driver.recordSteps(commands, 64); });
    }
    const vkexp::GenerationSummary summary = driver.finishGeneration();
    require(std::isfinite(summary.bestFitness) && std::isfinite(summary.medianFitness),
            what + ": generation produced finite fitness");

    // A field sized for a different configuration would leave the shader reading
    // or writing outside the agents' own world; the cheapest thing that catches
    // it is that every agent is still inside the arena it was given.
    for (const vkexp::AgentState& agent : driver.agents()) {
        require(std::isfinite(agent.pose.x) && std::isfinite(agent.pose.y),
                what + ": agent pose stayed finite");
        const float distance = std::hypot(agent.pose.x, agent.pose.y);
        require(distance <= state.physics.worldRadius * 1.5F + 1.0F,
                what + ": agent stayed inside its arena");
    }
}

int run() {
    vkexp::HeadlessComputeContext context{
        vkexp::HeadlessComputeConfig{.applicationName = "vkneuro reconfiguration smoke"}};

    // The population has to be the real one. An earlier version of this test ran
    // 96 genomes, which tops out at 40 logical worlds; the configuration that was
    // actually reported broken -- 512 genomes at 12 agents per world -- is 172
    // fields, and the field cost scales with that count. A small population makes
    // every allocation here comfortable and proves nothing about the real one.
    vkexp::SimulationState state{};
    state.controls.stepsPerGeneration = 48;
    state.worlds.requestedAgentsPerWorld = 29;
    state.physics.worldSize = vkexp::WorldSize::Small;
    state.physics.worldRadius = vkexp::worldRadiusForSize(state.physics.worldSize);
    state.physics.lightSensorRange = vkexp::lightRangeForWorld(state.physics);

    vkexp::SimulationDriver driver{state, vkexp::EvolutionSettings{.populationSize = 512}, {}};
    driver.createResources(context.physicalDevice(), context.device());
    stepAndCheck(context, driver, state, "initial configuration");

    // 1. Arena size. The neighbour grid, the trail field and the light range are
    //    all derived from it, and all three used to be sized from a different
    //    assumption about which world was running.
    for (const vkexp::WorldSize size :
         {vkexp::WorldSize::Large, vkexp::WorldSize::Medium, vkexp::WorldSize::Small}) {
        state.physics.worldSize = size;
        state.physics.worldRadius = vkexp::worldRadiusForSize(size);
        state.physics.lightSensorRange = vkexp::lightRangeForWorld(state.physics);
        driver.restart();
        stepAndCheck(context, driver, state,
                     "world size " + std::to_string(static_cast<int>(size)));
    }

    // 2. Group size, in both directions. Shrinking it multiplies the number of
    //    logical worlds, which is what grows the per-world buffers; growing it
    //    back must not leave the driver believing in the larger layout.
    // 29 is where the reporter said it still worked and 12 is where it stopped;
    // 10 is the floor the UI allows, which is the largest world count reachable.
    for (const std::uint32_t agentsPerWorld : {512U, 29U, 12U, 10U, 29U, 12U}) {
        state.worlds.requestedAgentsPerWorld = agentsPerWorld;
        driver.restart();
        require(state.worlds.agentsPerWorld <= agentsPerWorld,
                "group size " + std::to_string(agentsPerWorld) + ": clamped, never inflated");
        stepAndCheck(context, driver, state, "group size " + std::to_string(agentsPerWorld));
    }

    // 3. Trail resolution, finest first so the budget clamp is exercised while the
    //    world count is at its largest. The driver may coarsen the request; what
    //    it must not do is fail to terminate or keep a stale field.
    for (const float fraction :
         {vkexp::trailCellFractionFinest, 0.25F, 0.5F, vkexp::trailCellFractionCoarsest}) {
        state.physics.trailCellSize = vkexp::trailCellSizeForBodyFraction(fraction);
        driver.restart();
        require(state.physics.trailCellSize >=
                    vkexp::trailCellSizeForBodyFraction(vkexp::trailCellFractionFinest),
                "trail resolution: never refined past the finest setting");
        require(static_cast<std::uint64_t>(state.trail.cellsPerWorld) * state.worlds.worldCount *
                        vkexp::trail::kernel::TrailChannels * sizeof(std::uint32_t) <=
                    state.trail.size,
                "trail resolution: the chosen field fits the fixed allocation");
        stepAndCheck(context, driver, state, "trail resolution");
    }

    // 4. The brain plan. This is the reconfiguration that resizes a buffer for a
    //    reason that has nothing to do with the arena: a different number of
    //    hidden layers is a different genome length. It used to remake every step
    //    resource, which freed the agent, trail and puck buffers under the
    //    renderer's descriptors and hung the GPU on the next frame.
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
            state.physics.hiddenLayers = plan;
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

    // 5. Every arena against every group size the UI offers, at both ends of the
    //    resolution range. The reported crash was on a medium arena below 25
    //    agents per world at the *coarsest* grid, where the field is 31 MB --
    //    which is what ruled capacity out and pointed back at lifetimes.
    state.physics.trailCellSize =
        vkexp::trailCellSizeForBodyFraction(vkexp::trailCellFractionFinest);
    for (const vkexp::WorldSize size :
         {vkexp::WorldSize::Small, vkexp::WorldSize::Medium, vkexp::WorldSize::Large}) {
        for (const std::uint32_t agentsPerWorld : {29U, 24U, 12U, 10U}) {
            for (const float fraction :
                 {vkexp::trailCellFractionCoarsest, vkexp::trailCellFractionFinest}) {
                state.physics.worldSize = size;
                state.physics.worldRadius = vkexp::worldRadiusForSize(size);
                state.physics.lightSensorRange = vkexp::lightRangeForWorld(state.physics);
                state.worlds.requestedAgentsPerWorld = agentsPerWorld;
                state.physics.trailCellSize = vkexp::trailCellSizeForBodyFraction(fraction);
                driver.restart();
                stepAndCheck(context, driver, state,
                             "arena " + std::to_string(static_cast<int>(size)) + ", " +
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
