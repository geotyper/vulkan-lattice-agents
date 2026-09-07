// Batch neuroevolution runner: the same SimulationDriver the windowed
// application uses, without a window, a swapchain or a frame loop. This is what
// makes overnight runs, parameter sweeps and ablation comparisons possible.

#include "vkexp/compute/HeadlessComputeContext.hpp"
#include "vkexp/evolution/GenomeArchive.hpp"
#include "vkexp/simulation/WorldSnapshot.hpp"
#include "vkexp/simulation/SimulationDriver.hpp"
#include "vkexp/simulation/SimulationState.hpp"
#include "vkexp/simulation/Units.hpp"
#include "vkexp/worlds/WorldScenario.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int skipExitCode = 77;

struct Options {
    vkexp::BeaconScenario scenario{vkexp::BeaconScenario::Stationary};
    vkexp::WorldShape worldShape{vkexp::WorldShape::Circle};
    vkexp::WorldSize worldSize{vkexp::WorldSize::Small};
    std::uint64_t generations{20};
    std::uint32_t stepsPerGeneration{900};
    std::uint32_t stepsPerBatch{128};
    std::uint32_t agentsPerWorld{12};
    std::size_t populationSize{512};
    std::uint32_t seed{0xC0FFEEU};
    bool agentCollisions{true};
    bool agentLight{true};
    bool swapDeliveryEnds{false};
    bool uniformBeaconColor{false};
    bool blockedDoorPerGeneration{false};
    vkexp::FitnessWeights fitness{};
    // Optional physics overrides. Absent means "keep the default", which lets a
    // sweep change one term without restating the rest of SimulationStep.
    std::optional<float> beaconAngularSpeed;
    std::optional<float> beaconRadiusRatio;
    std::optional<float> lightSensorRange;
    std::optional<float> maximumSpeed;
    std::optional<float> trailDepositRate;
    std::optional<float> beaconTrailDepositRate;
    std::optional<float> trailHalfLife;
    std::optional<float> trailCellSize;
    bool trailEnabled{true};
    vkexp::NeuronModel neuronModel{vkexp::NeuronModel::TimeConstant};
    bool quiet{};
    std::string savePopulation;
    std::string saveChampion;
    std::string loadPopulation;
    std::string saveWorld;
    std::string loadWorld;
    std::string csvPath;
};

std::string scenarioKeyList() {
    std::string keys;
    for (const vkexp::ScenarioDefinition* definition : vkexp::scenarioRegistry()) {
        keys += keys.empty() ? "" : "|";
        keys += definition->key;
    }
    return keys;
}

void printHelp(const char* executable) {
    // The scenario list comes from the registry, so a new scenario shows up in
    // --help without editing this text.
    std::cout << "Usage: " << executable
              << " [options]\n\n"
                 "Runs neuroevolution without a window and reports per-generation fitness.\n\n"
                 "Experiment:\n"
              << "  --scenario <name>        " << scenarioKeyList() << "\n"
              << "  --generations <n>        generations to run (default 20)\n"
                 "  --steps <n>              steps per generation (default 900 = 15.0 s)\n"
                 "  --population <n>         genomes (default 512)\n"
                 "  --agents-per-world <n>   agents sharing one logical world (default 12)\n"
                 "  --seed <n>               genetic algorithm seed (default 12648430)\n"
                 "  --world-size <name>      small|medium|large\n"
                 "  --world-shape <name>     circle|square\n"
                 "  --steps-per-batch <n>    steps recorded per submission (default 128)\n\n"
                 "World physics (default = the value the UI starts with):\n"
                 "  --beacon-speed <x>       beacon angular speed in rad/s (default 0.35)\n"
                 "  --orbit-ratio <x>        orbit radius as a fraction of the arena (0.72)\n"
                 "  --light-range <x>        light sensor range in metres (default 2.4)\n"
                 "  --max-speed <x>          agent speed limit in m/s (default 0.55)\n\n"
                 "Ablations:\n"
                 "  --no-agent-collisions    disable agent-agent collisions\n"
                 "  --no-agent-light         disable perception of other agents' signals\n"
                 "  --swap-ends              two gaps: trade the ends every other generation\n"
                 "  --uniform-beacon-color   ablate hue: both ends emit the average colour\n"
                 "  --doors-by-generation    two doors: the dead end changes per generation,\n"
                 "                           not per trial\n"
                 "  --no-trail               disable the ground trail field entirely\n"
                 "  --neuron-model <name>    reactive|time|gated: where a hidden neuron's "
                 "time\n"
                 "                           constant comes from. reactive pins it to the step, "
                 "which\n"
                 "                           is the memoryless network exactly; gated recomputes "
                 "it\n"
                 "                           from the inputs every step (default time)\n\n"
                 "Trail field:\n"
                 "  --trail-deposit <x>      agent mark laid per second (default 4.0)\n"
                 "  --beacon-deposit <x>     beacon mark laid per second (default 12.0)\n"
                 "  --trail-half-life <x>    seconds for a mark to fade to half (default 6)\n"
                 "  --trail-cell-size <x>    ground metres per trail cell, 0.02..0.08 "
                 "(default 0.06)\n\n"
                 "Fitness shaping (sweepable without rebuilding):\n"
                 "  --objective-bonus <x>    score per completed objective (default 2.0)\n"
                 "  --motor-cost <x>         fitness charged per unit of effort (default 0.002)\n"
                 "  --tracking-reward <x>    shaping for a moving beacon (default 0.25)\n"
                 "  --signal-cost <x>        emission cost relative to moving (default 0.25)\n"
                 "  --energy-drain <x>       battery drained per effort unit (default 0.0008)\n"
                 "  --fitness-sharing <x>    0 individual selection, 1 whole-world "
                 "(default 0)\n\n"
                 "Persistence:\n"
                 "  --load-population <path> resume from a genome archive\n"
                 "  --load-world <path>      resume a whole experiment mid-generation\n"
                 "  --save-world <path>      write the world state after the last "
                 "generation\n"
                 "  --save-population <path> write the final population\n"
                 "  --save-champion <path>   write the best genome of the final generation\n"
                 "  --csv <path>             append per-generation statistics as CSV\n\n"
                 "Other:\n"
                 "  --quiet                  only print the final summary line\n"
                 "  --help, -h               show this help\n";
}

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

template <typename T> T parseNumber(const std::string_view text, const std::string_view option) {
    T value{};
    const auto* const first = text.data();
    const auto* const last = first + text.size();
    const auto result = std::from_chars(first, last, value);
    if (result.ec != std::errc{} || result.ptr != last) {
        fail("Invalid numeric value for " + std::string{option} + ": " + std::string{text});
    }
    return value;
}

vkexp::BeaconScenario parseScenario(const std::string_view name) {
    for (const vkexp::ScenarioDefinition* definition : vkexp::scenarioRegistry()) {
        if (name == definition->key) {
            return definition->id;
        }
    }
    fail("Unknown scenario: " + std::string{name} + " (expected one of " + scenarioKeyList() + ")");
}

// Short names because they end up in run directories and CSV filenames.
[[nodiscard]] vkexp::NeuronModel parseNeuronModel(const std::string_view name) {
    if (name == "reactive") {
        return vkexp::NeuronModel::Reactive;
    }
    if (name == "time") {
        return vkexp::NeuronModel::TimeConstant;
    }
    if (name == "gated") {
        return vkexp::NeuronModel::Gated;
    }
    fail("Unknown neuron model '" + std::string{name} + "'; expected reactive, time or gated");
}

[[nodiscard]] const char* neuronModelName(const vkexp::NeuronModel model) {
    switch (model) {
    case vkexp::NeuronModel::Reactive:
        return "reactive (no state)";
    case vkexp::NeuronModel::TimeConstant:
        return "time constant (one evolved rate per neuron)";
    case vkexp::NeuronModel::Gated:
        return "gated (rate recomputed from the inputs each step)";
    }
    return "unknown";
}

Options parseOptions(const int argc, char** argv, bool& helpRequested) {
    Options options;
    const auto next = [&](int& index, const std::string_view option) -> std::string_view {
        if (index + 1 >= argc) {
            fail("Missing value for " + std::string{option});
        }
        return argv[++index];
    };
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            helpRequested = true;
            return options;
        } else if (argument == "--scenario") {
            options.scenario = parseScenario(next(index, argument));
        } else if (argument == "--generations") {
            options.generations = parseNumber<std::uint64_t>(next(index, argument), argument);
        } else if (argument == "--steps") {
            options.stepsPerGeneration =
                parseNumber<std::uint32_t>(next(index, argument), argument);
        } else if (argument == "--steps-per-batch") {
            options.stepsPerBatch = parseNumber<std::uint32_t>(next(index, argument), argument);
        } else if (argument == "--population") {
            options.populationSize = parseNumber<std::size_t>(next(index, argument), argument);
        } else if (argument == "--agents-per-world") {
            options.agentsPerWorld = parseNumber<std::uint32_t>(next(index, argument), argument);
        } else if (argument == "--seed") {
            options.seed = parseNumber<std::uint32_t>(next(index, argument), argument);
        } else if (argument == "--world-size") {
            const std::string_view name = next(index, argument);
            options.worldSize =
                name == "small"    ? vkexp::WorldSize::Small
                : name == "medium" ? vkexp::WorldSize::Medium
                : name == "large"
                    ? vkexp::WorldSize::Large
                    : throw std::runtime_error("Unknown world size: " + std::string{name});
        } else if (argument == "--world-shape") {
            const std::string_view name = next(index, argument);
            options.worldShape =
                name == "circle" ? vkexp::WorldShape::Circle
                : name == "square"
                    ? vkexp::WorldShape::Square
                    : throw std::runtime_error("Unknown world shape: " + std::string{name});
        } else if (argument == "--objective-bonus") {
            options.fitness.objectiveBonus = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--motor-cost") {
            options.fitness.motorCostWeight = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--tracking-reward") {
            options.fitness.trackingReward = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--signal-cost") {
            options.fitness.signalCostFactor = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--energy-drain") {
            options.fitness.energyDrain = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--fitness-sharing") {
            options.fitness.groupSharing = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--beacon-speed") {
            options.beaconAngularSpeed = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--orbit-ratio") {
            options.beaconRadiusRatio = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--light-range") {
            options.lightSensorRange = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--max-speed") {
            options.maximumSpeed = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--no-trail") {
            options.trailEnabled = false;
        } else if (argument == "--trail-deposit") {
            options.trailDepositRate = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--beacon-deposit") {
            options.beaconTrailDepositRate = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--trail-cell-size") {
            options.trailCellSize = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--trail-half-life") {
            options.trailHalfLife = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--no-agent-collisions") {
            options.agentCollisions = false;
        } else if (argument == "--neuron-model") {
            options.neuronModel = parseNeuronModel(next(index, argument));
        } else if (argument == "--no-agent-light") {
            options.agentLight = false;
        } else if (argument == "--swap-ends") {
            options.swapDeliveryEnds = true;
        } else if (argument == "--uniform-beacon-color") {
            options.uniformBeaconColor = true;
        } else if (argument == "--doors-by-generation") {
            options.blockedDoorPerGeneration = true;
        } else if (argument == "--quiet") {
            options.quiet = true;
        } else if (argument == "--save-population") {
            options.savePopulation = next(index, argument);
        } else if (argument == "--save-champion") {
            options.saveChampion = next(index, argument);
        } else if (argument == "--load-population") {
            options.loadPopulation = next(index, argument);
        } else if (argument == "--save-world") {
            options.saveWorld = next(index, argument);
        } else if (argument == "--load-world") {
            options.loadWorld = next(index, argument);
        } else if (argument == "--csv") {
            options.csvPath = next(index, argument);
        } else {
            fail("Unknown argument: " + std::string{argument});
        }
    }
    if (options.generations == 0 || options.stepsPerGeneration == 0 || options.stepsPerBatch == 0) {
        fail("Generations, steps and steps-per-batch must all be non-zero");
    }
    return options;
}

vkexp::GenomeArchiveMetadata makeMetadata(const vkexp::SimulationState& state,
                                          const vkexp::SimulationDriver& driver,
                                          const vkexp::neuro::BrainShape& brain) {
    return {driver.evolution().generation(),
            static_cast<std::uint32_t>(state.physics.beaconScenario),
            driver.evolution().settings().seed,
            state.statistics.bestFitness,
            state.statistics.meanFitness,
            static_cast<std::uint32_t>(brain.inputCount),
            static_cast<std::uint32_t>(brain.hiddenCount),
            static_cast<std::uint32_t>(brain.outputCount)};
}

int run(const Options& options) {
    vkexp::HeadlessComputeContext context{{"vkneuro headless evolution"}};

    vkexp::SimulationState state;
    state.controls.stepsPerGeneration = options.stepsPerGeneration;
    state.worlds.requestedAgentsPerWorld = options.agentsPerWorld;
    state.physics.beaconScenario = options.scenario;
    state.physics.worldShape = options.worldShape;
    state.physics.worldSize = options.worldSize;
    state.physics.worldRadius = vkexp::worldRadiusForSize(options.worldSize);
    state.physics.lightSensorRange = vkexp::lightRangeForWorld(state.physics);
    state.physics.agentCollisionsEnabled = options.agentCollisions;
    state.physics.agentLightEnabled = options.agentLight;
    state.physics.swapDeliveryEnds = options.swapDeliveryEnds;
    state.physics.uniformBeaconColor = options.uniformBeaconColor;
    state.physics.blockedDoorPerGeneration = options.blockedDoorPerGeneration;
    state.physics.fitness = options.fitness;
    if (options.beaconAngularSpeed) {
        state.physics.beaconAngularSpeed = *options.beaconAngularSpeed;
    }
    if (options.beaconRadiusRatio) {
        state.physics.beaconRadiusRatio = *options.beaconRadiusRatio;
    }
    if (options.lightSensorRange) {
        state.physics.lightSensorRange = *options.lightSensorRange;
    }
    if (options.maximumSpeed) {
        state.physics.maximumSpeed = *options.maximumSpeed;
    }
    state.physics.trailEnabled = options.trailEnabled;
    state.physics.neuronModel = options.neuronModel;
    if (options.trailDepositRate) {
        state.physics.trailDepositRate = *options.trailDepositRate;
    }
    if (options.beaconTrailDepositRate) {
        state.physics.beaconTrailDepositRate = *options.beaconTrailDepositRate;
    }
    if (options.trailHalfLife) {
        state.physics.trailHalfLife = *options.trailHalfLife;
    }
    if (options.trailCellSize) {
        state.physics.trailCellSize =
            std::clamp(*options.trailCellSize,
                       vkexp::trailCellSizeForBodyFraction(vkexp::trailCellFractionFinest),
                       vkexp::trailCellSizeForBodyFraction(vkexp::trailCellFractionCoarsest));
    }

    vkexp::EvolutionSettings evolution;
    evolution.populationSize = options.populationSize;
    evolution.seed = options.seed;

    vkexp::SimulationDriverConfig config;
    config.maximumStepsPerBatch = options.stepsPerBatch;

    vkexp::SimulationDriver driver{state, evolution, config};
    driver.createResources(context.physicalDevice(), context.device());

    // Before the scenario is resolved: a snapshot carries the world it ran in, so
    // it decides the scenario, the arena and the physics. The command line only
    // fills in what the file does not cover.
    if (!options.loadWorld.empty() && !options.loadPopulation.empty()) {
        fail("--load-world and --load-population both set the population; pass one");
    }
    if (!options.loadWorld.empty()) {
        const vkexp::WorldSnapshot world = vkexp::loadWorldSnapshot(options.loadWorld);
        driver.restoreSnapshot(world);
        if (!options.quiet) {
            std::cout << "Resumed " << world.genomes.size() << " genomes from "
                      << options.loadWorld << " at generation " << world.generation << ", step "
                      << world.step << " of " << world.stepsPerGeneration << " ("
                      << vkexp::scenarioDefinition(world.physics.beaconScenario).name << ")\n";
        }
    }

    const vkexp::ScenarioDefinition& scenario =
        vkexp::scenarioDefinition(state.physics.beaconScenario);

    if (!options.loadPopulation.empty()) {
        const vkexp::GenomeArchive archive = vkexp::loadGenomeArchive(options.loadPopulation);
        driver.loadPopulation(archive.genomes, archive.metadata.generation);
        if (!options.quiet) {
            std::cout << "Resumed " << archive.genomes.size() << " genomes from "
                      << options.loadPopulation << " at generation " << archive.metadata.generation
                      << '\n';
        }
    }

    std::optional<std::ofstream> csv;
    if (!options.csvPath.empty()) {
        const bool existed = std::ifstream{options.csvPath}.good();
        csv.emplace(options.csvPath, std::ios::app);
        if (!*csv) {
            fail("Unable to open CSV output: " + options.csvPath);
        }
        if (!existed) {
            *csv << "generation,scenario,seed,best,median,mean,arrival_ratio\n";
        }
    }

    if (!options.quiet) {
        std::cout << "Device:     " << context.deviceName() << '\n'
                  << "Scenario:   " << scenario.name << '\n'
                  << "Brain:      " << scenario.brain.inputCount << " -> "
                  << scenario.brain.hiddenCount << " -> " << scenario.brain.outputCount << '\n'
                  << "Trial:      " << state.controls.stepsPerGeneration
                  << " steps = " << std::fixed << std::setprecision(1)
                  << vkexp::units::secondsForSteps(state.controls.stepsPerGeneration,
                                                   vkexp::units::fixedTimeStep)
                  << " s at " << vkexp::units::simulationRateHz << " Hz\n"
                  << std::setprecision(2) << "World:      " << state.physics.worldRadius * 2.0F
                  << " m across, body "
                  << vkexp::units::metresToCentimetres(vkexp::agentBodyRadius * 2.0F) << " cm\n"
                  << std::defaultfloat << std::setprecision(6)
                  << "Population: " << driver.evolution().population().size() << " genomes x "
                  << driver.config().trialsPerGenome << " trials = " << state.agents.agentCount
                  << " agents in " << state.worlds.worldCount << " logical worlds\n"
                  << "Ablations:  agent collisions "
                  << (state.physics.agentCollisionsEnabled ? "on" : "OFF") << ", agent light "
                  << (state.physics.agentLightEnabled ? "on" : "OFF") << ", trail "
                  << (state.physics.trailEnabled ? "on" : "OFF") << ", beacon hue "
                  << (state.physics.uniformBeaconColor ? "ABLATED" : "on") << '\n'
                  << "Neurons:    " << neuronModelName(state.physics.neuronModel) << '\n';
        // Only when it is on, and only where it does something, so a default
        // run's output stays comparable with every run recorded before it.
        if (state.physics.swapDeliveryEnds &&
            vkexp::scenarioDefinition(state.physics.beaconScenario).tunables.swapDeliveryEnds) {
            std::cout << "World:      the two ends trade places on odd generations\n";
        }
        if (state.physics.blockedDoorPerGeneration &&
            vkexp::scenarioDefinition(state.physics.beaconScenario)
                .tunables.blockedDoorPerGeneration) {
            std::cout << "World:      the dead end changes by generation, not by trial\n";
        }
        // Only when it is on, so a default run's output stays byte-identical to
        // every run recorded before the option existed.
        if (state.physics.fitness.groupSharing > 0.0F) {
            std::cout << "Selection:  group fitness sharing " << std::fixed << std::setprecision(2)
                      << state.physics.fitness.groupSharing << std::defaultfloat
                      << std::setprecision(6) << " (plotted fitness stays individual)\n";
        }
        std::cout << '\n'
                  << "  gen        best      median        mean   arrival\n";
    }

    const std::uint64_t firstGeneration = driver.evolution().generation();
    const std::uint64_t lastGeneration = firstGeneration + options.generations;
    while (driver.evolution().generation() < lastGeneration) {
        const std::uint64_t generation = driver.evolution().generation();
        while (!driver.generationComplete()) {
            context.immediate().execute([&](const VkCommandBuffer commands) {
                driver.recordSteps(commands, options.stepsPerBatch);
            });
        }
        context.waitIdle();
        driver.finishGeneration();
        if (!options.quiet) {
            std::cout << std::setw(5) << generation << std::fixed << std::setprecision(4)
                      << std::setw(12) << state.statistics.bestFitness << std::setw(12)
                      << state.statistics.medianFitness << std::setw(12)
                      << state.statistics.meanFitness << std::setw(10)
                      << state.statistics.arrivalRatio << '\n';
        }
        if (csv) {
            *csv << generation << ',' << scenario.name << ',' << options.seed << ','
                 << state.statistics.bestFitness << ',' << state.statistics.medianFitness << ','
                 << state.statistics.meanFitness << ',' << state.statistics.arrivalRatio << '\n';
        }
    }
    if (csv) {
        csv->flush();
    }

    // finishGeneration() has already produced the next population, whose first
    // eliteCount entries are the ranked survivors, champion first.
    const std::vector<vkexp::Genome>& population = driver.evolution().population();
    const vkexp::GenomeArchiveMetadata metadata = makeMetadata(state, driver, scenario.brain);
    if (!options.savePopulation.empty()) {
        vkexp::saveGenomeArchive(options.savePopulation, population, metadata);
        if (!options.quiet) {
            std::cout << "Saved " << population.size() << " genomes to " << options.savePopulation
                      << '\n';
        }
    }
    if (!options.saveWorld.empty()) {
        // The device is idle here -- the last generation was waited on before it
        // was scored -- which is what snapshot() requires to read the agents back.
        vkexp::saveWorldSnapshot(options.saveWorld, driver.snapshot());
        if (!options.quiet) {
            std::cout << "Saved the world to " << options.saveWorld << '\n';
        }
    }
    if (!options.saveChampion.empty()) {
        vkexp::saveGenomeArchive(options.saveChampion, {population.data(), 1}, metadata);
        if (!options.quiet) {
            std::cout << "Saved champion to " << options.saveChampion << '\n';
        }
    }

    std::cout << "Final generation " << driver.evolution().generation() << ": best "
              << state.statistics.bestFitness << ", median " << state.statistics.medianFitness
              << ", mean " << state.statistics.meanFitness << ", arrival "
              << state.statistics.arrivalRatio << '\n';

    driver.destroyResources();
    return 0;
}

} // namespace

int main(const int argc, char** argv) {
    try {
        bool helpRequested = false;
        const Options options = parseOptions(argc, argv, helpRequested);
        if (helpRequested) {
            printHelp(argv[0]);
            return 0;
        }
        return run(options);
    } catch (const vkexp::HeadlessComputeUnavailable& unavailable) {
        std::cout << "Skipping headless evolution: " << unavailable.what() << '\n';
        return skipExitCode;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
