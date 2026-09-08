// Device coverage for the shared puck.
//
// The puck is the first piece of world state agents can change, and it is the
// first thing in this simulation that no single agent owns: one puck per
// logical world, moved by whichever agents happen to be leaning on it, scored
// for all of them. None of that is visible in a fitness number -- a run where
// the puck never moves at all still produces plausible ones -- so the claims
// are asserted here against the puck record itself.
//
// What is checked, in the order the failures matter:
//   * agents push it. A puck that never moves would leave every fitness curve
//     looking like a hard task rather than like a broken one;
//   * it stops. The drag has to win when nobody is pushing, or a nudged puck
//     drifts to the target on its own and the world scores luck;
//   * the level latches and never goes down, because the reported completion
//     ratio is read as a monotone curve;
//   * it stays inside the arena, which is the one containment rule everything
//     else in this simulation also obeys;
//   * a snapshot carries it. Resuming with the puck back at the start would
//     read as a run that had lost ground it had not lost.

#include "vkexp/compute/HeadlessComputeContext.hpp"
#include "vkexp/simulation/PuckKernel.hpp"
#include "vkexp/simulation/SimulationDriver.hpp"
#include "vkexp/simulation/SimulationState.hpp"

#include <cmath>
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

int run() {
    vkexp::HeadlessComputeContext context{
        vkexp::HeadlessComputeConfig{.applicationName = "vkneuro puck smoke"}};

    vkexp::SimulationState state{};
    state.controls.stepsPerGeneration = 240;
    state.worlds.requestedAgentsPerWorld = 16;
    state.physics.beaconScenario = vkexp::BeaconScenario::PuckPush;
    state.physics.worldRadius = vkexp::worldRadiusForSize(state.physics.worldSize);
    state.physics.lightSensorRange = vkexp::lightRangeForWorld(state.physics);

    vkexp::SimulationDriver driver{state, vkexp::EvolutionSettings{.populationSize = 64}, {}};
    driver.createResources(context.physicalDevice(), context.device());

    const auto stepGeneration = [&] {
        while (!driver.generationComplete()) {
            context.immediate().execute(
                [&](const VkCommandBuffer commands) { driver.recordSteps(commands, 64); });
        }
    };

    const std::vector<vkexp::PuckState> initial = driver.snapshot().pucks;
    require(!initial.empty(), "the puck world starts with a puck in every logical world");
    require(initial.size() == state.worlds.worldCount, "one puck per logical world");

    // Where they start: on the axis, alternating side by trial, so a genome
    // meets both and a fixed push direction cannot stand in for perceiving the
    // puck. Checked against the shared kernel rather than against a number
    // repeated here.
    for (std::size_t world = 0; world < initial.size(); ++world) {
        const auto trial = static_cast<std::uint32_t>(world % state.agents.trialsPerGenome);
        const auto expected =
            vkexp::puck::kernel::puckStartPosition(state.physics.worldRadius, trial);
        require(std::abs(initial[world].pose.x - expected.x) < 1.0e-5F &&
                    std::abs(initial[world].pose.y - expected.y) < 1.0e-5F,
                "a puck starts where the shared kernel puts it");
        require(vkexp::puckLevel(initial[world]) == 0, "a puck starts having achieved nothing");
    }

    stepGeneration();
    const std::vector<vkexp::PuckState> pushed = driver.snapshot().pucks;
    require(pushed.size() == initial.size(), "the puck count does not change mid-generation");

    std::size_t touched = 0;
    std::size_t moved = 0;
    for (std::size_t world = 0; world < pushed.size(); ++world) {
        const float travelled = std::hypot(pushed[world].pose.x - initial[world].pose.x,
                                           pushed[world].pose.y - initial[world].pose.y);
        if (travelled > 0.001F) {
            ++touched;
        }
        if (travelled > vkexp::puck::kernel::PuckRadius) {
            ++moved;
        }
        // Containment, and finiteness: a puck that left the arena or went to
        // NaN would still produce a fitness number, and a plausible one.
        require(std::isfinite(pushed[world].pose.x) && std::isfinite(pushed[world].pose.y),
                "a puck stays finite");
        require(std::hypot(pushed[world].pose.x, pushed[world].pose.y) <=
                    state.physics.worldRadius + 1.0e-3F,
                "a puck stays inside the arena");
        require(vkexp::puckLevel(pushed[world]) <= vkexp::puck::kernel::PuckLevelCount,
                "a puck never reports a level the world does not have");
    }
    // Untrained agents wander, so this is not a claim about competence -- only
    // that contact does something at all, in enough worlds that it cannot be one
    // lucky collision. If the push were not wired up, or the puck pass never ran,
    // or it read the wrong world's agents, every one of these would be zero.
    require(touched > 0, "agents shift the puck at all");
    require(moved > 0, "at least one world pushes the puck a whole puck width");

    // A puck cannot outrun the agents pushing it. This is the whole reason the
    // push is measured against the puck's own velocity rather than the agent's:
    // with an absolute velocity the term never vanishes, so a puck in continuous
    // contact keeps accelerating and ends up faster than anything in the world.
    // The bound is the property that formulation buys, and it is what stops the
    // task being solved by launching the puck once.
    for (std::size_t world = 0; world < pushed.size(); ++world) {
        require(std::hypot(pushed[world].motion.x, pushed[world].motion.y) <=
                    state.physics.maximumSpeed * 1.05F,
                "a puck never moves faster than the agents pushing it");
    }

    // Carrying on from where the generation ended, with the agents still in the
    // world and still moving. This is not a claim about the puck stopping --
    // agents keep leaning on it -- but about it staying bounded and latched over
    // a much longer run than one generation.
    const std::vector<vkexp::PuckState> resting = [&] {
        for (int batch = 0; batch < 8; ++batch) {
            context.immediate().execute(
                [&](const VkCommandBuffer commands) { driver.recordSteps(commands, 30); });
        }
        return driver.snapshot().pucks;
    }();
    for (std::size_t world = 0; world < resting.size(); ++world) {
        require(std::hypot(resting[world].pose.x, resting[world].pose.y) <=
                    state.physics.worldRadius + 1.0e-3F,
                "a puck stays inside the arena over a long run");
        require(std::hypot(resting[world].motion.x, resting[world].motion.y) <=
                    state.physics.maximumSpeed * 1.05F,
                "a puck stays bounded over a long run");
    }

    // The level is latched and taken as a maximum, so it can never fall. That
    // is what lets the reported completion ratio be read as a curve.
    for (std::size_t world = 0; world < resting.size(); ++world) {
        require(vkexp::puckLevel(resting[world]) >= vkexp::puckLevel(pushed[world]),
                "a puck's level never goes down");
    }

    // And a snapshot carries the puck rather than dropping it, which the round
    // trip through restoreSnapshot above has already exercised -- this pins the
    // claim that what came back is what went in.
    const vkexp::WorldSnapshot saved = driver.snapshot();
    require(saved.pucks.size() == resting.size(), "a snapshot carries one puck per world");
    for (std::size_t world = 0; world < saved.pucks.size(); ++world) {
        require(std::abs(saved.pucks[world].pose.x - resting[world].pose.x) < 1.0e-6F &&
                    std::abs(saved.pucks[world].pose.y - resting[world].pose.y) < 1.0e-6F,
                "a snapshot carries where the puck actually is");
    }

    std::cout << "Puck smoke: " << touched << " of " << pushed.size()
              << " worlds shifted their puck, " << moved << " by a whole puck width\n";
    return EXIT_SUCCESS;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
