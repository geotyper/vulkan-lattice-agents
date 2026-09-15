// Batch neuroevolution runner: the same SimulationDriver the windowed
// application uses, without a window, a swapchain or a frame loop. This is what
// makes overnight runs, parameter sweeps and ablation comparisons possible.

#include "vkexp/compute/HeadlessComputeContext.hpp"
#include "vkexp/evolution/GenomeArchive.hpp"
#include "vkexp/neuro/BrainDescription.hpp"
#include "vkexp/simulation/RunSnapshot.hpp"
#include "vkexp/simulation/SimulationDriver.hpp"
#include "vkexp/simulation/SimulationState.hpp"
#include "vkexp/simulation/Units.hpp"

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
    std::uint64_t generations{20};
    std::uint32_t stepsPerGeneration{900};
    std::uint32_t stepsPerBatch{128};
    std::uint32_t agentsPerWorld{12};
    std::size_t populationSize{512};
    vkexp::WeightInit weightInit{vkexp::WeightInit::Saturating};
    std::uint32_t seed{0xC0FFEEU};

    // Absent means "leave the default", which lets a sweep change one term
    // without restating the rest of SimulationStep.
    std::optional<std::uint32_t> latticeWidth;
    std::optional<std::uint32_t> latticeHeight;
    std::optional<std::uint32_t> latticeDepth;
    std::optional<float> turnThreshold;
    std::optional<std::uint32_t> contactRadius;
    vkexp::Neighborhood neighborhood{vkexp::Neighborhood::Moore};
    vkexp::WorldMode worldMode{vkexp::WorldMode::Beacon};
    std::optional<std::uint32_t> buildIntervalTicks;
    float buildThreshold{0.55F};
    std::uint32_t resourceHeightLow{4};
    std::uint32_t resourceHeightHigh{8};
    std::uint32_t chasmGroundWidth{};
    float constructionCourseFill{0.5F};
    std::uint32_t constructionHeightLead{5};
    std::uint32_t constructionSupportRadius{2};
    bool allowSideSupportedBlocks{};

    vkexp::FitnessWeights fitness{};
    vkexp::NeuronModel neuronModel{vkexp::NeuronModel::TimeConstant};
    bool quiet{};
    std::string savePopulation;
    std::string saveChampion;
    std::string describeBrain;
    // Empty means the default hidden layers.
    std::vector<std::uint32_t> hiddenLayers;
    std::vector<std::uint32_t> hiddenSquashes;
    std::string loadPopulation;
    std::string saveRun;
    std::string loadRun;
    std::string csvPath;
};

void printHelp(const char* executable) {
    std::cout << "Usage: " << executable
              << " [options]\n\n"
                 "Runs neuroevolution on a discrete lattice without a window and reports\n"
                 "per-generation fitness.\n\n"
                 "Experiment:\n"
                 "  --generations <n>        generations to run (default 20)\n"
                 "  --steps <n>              steps per generation (default 900 = 15.0 s)\n"
                 "  --population <n>         genomes (default 512)\n"
                 "  --weight-init <name>     saturating|fan-in: how a fresh genome is drawn\n"
                 "                           (default saturating). fan-in scales each block by\n"
                 "                           its input count and needs both thresholds scaled\n"
                 "                           down with it\n"
                 "  --agents-per-world <n>   agents sharing one lattice (default 12)\n"
                 "  --seed <n>               genetic algorithm seed (default 12648430)\n"
                 "  --steps-per-batch <n>    steps recorded per submission (default 128)\n\n"
                 "The lattice:\n"
                 "  --world <name>          beacon|construction|harvest|chasm (default beacon)\n"
                 "  --lattice <WxHxD>        cells per world (default 32x32x16). Each world\n"
                 "                           costs W*H*D*4 bytes twice over, and there is one\n"
                 "                           world per group per trial, so a large box and a\n"
                 "                           large population are alternatives\n"
                 "  --neighbourhood <name>   faces|moore: whether a step may be diagonal\n"
                 "                           (default moore). The input vector is 26 cells wide\n"
                 "                           either way, so a population carries across\n"
                 "  --turn-threshold <x>     how sure both turn outputs must be before the\n"
                 "                           agent pivots, 0..1 (default 0.25). They have to\n"
                 "                           agree, and a turn costs the whole tick\n"
                 "  --contact-radius <n>     cells from the beacon that count as reaching it\n"
                 "                           (default 1). 0 means one agent per world can\n"
                 "                           score at a time\n"
                 "  --build-interval <n>     ticks between successful block placements (12)\n"
                 "  --build-threshold <x>    construction output required to place (0.55)\n"
                 "  --resource-band <a>-<b>  harvest/chasm: the band of heights the resource\n"
                 "                           hangs in, inclusive (4-8), hashed per world\n"
                 "  --ground-width <n>       chasm: columns of solid floor from x=0.\n"
                 "                           0 means half the lattice\n"
                 "  --course-fill <x>        fill a level needs, locally, to be stood on (0.5)\n"
                 "  --support-radius <n>     cells around a site that question covers (2). A\n"
                 "                           radius spanning the floor is the old global rule\n"
                 "  --height-lead <n>        build levels allowed above foundation (5)\n"
                 "  --side-support           allow cardinal face-supported bridge blocks\n"
                 "  --boundary-penalty <x>   charged per agent-tick on the x/z edge (0.002)\n\n"
                 "Ablations:\n"
                 "  --neuron-model <name>    reactive|time|gated|spiking: where a hidden\n"
                 "                           neuron's time constant comes from. reactive pins\n"
                 "                           it to the step; spiking uses leaky\n"
                 "                           integrate-and-fire pulses; gated recomputes it\n"
                 "                           from inputs (default time)\n"
                 "  --hidden <a[,b[,c]]>     hidden layer widths, front to back\n"
                 "  --hidden-squash <a[,b[,c]]>\n"
                 "                           tanh|sin|tanh-scaled|softsign per hidden layer\n"
                 "                           (default tanh). Outputs\n"
                 "                           are always tanh -- every threshold in the rules\n"
                 "                           reads one as how far and which way. Has no effect\n"
                 "                           under --neuron-model spiking, which writes 1 or 0\n"
                 "                           and never reaches a squash\n\n"
                 "Fitness shaping (sweepable without rebuilding):\n"
                 "  --tracking-reward <x>    worth of the nearest it ever got (default 1.0)\n"
                 "  --objective-bonus <x>    score per step within the contact radius (0.02)\n"
                 "  --motor-cost <x>         charged per move made (default 0.002)\n"
                 "  --refusal-penalty <x>    charged per move refused (default 0.01)\n"
                 "  --signal-cost <x>        broadcast cost relative to moving (default 0.25)\n"
                 "  --fitness-sharing <x>    0 individual selection, 1 whole-world (default 0)\n\n"
                 "Persistence:\n"
                 "  --load-population <path> resume from a genome archive\n"
                 "  --load-run <path>        resume a whole experiment mid-generation\n"
                 "  --save-run <path>        write the run state after the last generation\n"
                 "  --save-population <path> write the final population\n"
                 "  --save-champion <path>   write the best genome of the final generation\n"
                 "  --describe-brain <path>  write the network's structure as JSON and exit\n"
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

// "20", or "16,8", or "12,8,8": the hidden layers, front to back. Written this
// way because that is how the network reads out loud, and because it makes the
// depth visible in a run directory name.
[[nodiscard]] std::vector<std::uint32_t> parseHiddenLayers(const std::string_view text) {
    std::vector<std::uint32_t> widths;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string_view piece = text.substr(
            start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        if (piece.empty()) {
            fail("Empty hidden layer width in '" + std::string{text} + "'");
        }
        widths.push_back(parseNumber<std::uint32_t>(piece, "--hidden"));
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }
    if (widths.empty() || widths.size() > vkexp::neuro::Topology::hiddenLayerCount) {
        fail("--hidden takes 1 to " + std::to_string(vkexp::neuro::Topology::hiddenLayerCount) +
             " widths, front to back");
    }
    return widths;
}

// The squash each hidden layer uses, in the same front-to-back order as the
// widths beside it.
[[nodiscard]] std::vector<std::uint32_t> parseHiddenSquashes(const std::string_view text) {
    std::vector<std::uint32_t> squashes;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string_view piece = text.substr(
            start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        if (piece == "tanh") {
            squashes.push_back(vkexp::neuro::kernel::BrainActivationTanh);
        } else if (piece == "sin" || piece == "sine") {
            squashes.push_back(vkexp::neuro::kernel::BrainActivationSine);
        } else if (piece == "tanh-scaled" || piece == "scaled") {
            squashes.push_back(vkexp::neuro::kernel::BrainActivationTanhScaled);
        } else if (piece == "softsign") {
            squashes.push_back(vkexp::neuro::kernel::BrainActivationSoftsign);
        } else {
            fail("Unknown hidden squash '" + std::string{piece} +
                 "', expected tanh, sin, tanh-scaled or softsign");
        }
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }
    if (squashes.empty() || squashes.size() > vkexp::neuro::Topology::hiddenLayerCount) {
        fail("--hidden-squash takes 1 to " +
             std::to_string(vkexp::neuro::Topology::hiddenLayerCount) +
             " names, front to back");
    }
    return squashes;
}

// "32x32x16". Three numbers rather than three options because the box is one
// decision: a run directory named lattice-32x32x16 says what it was, and three
// separate flags invite two of them to be changed and the third forgotten.
void parseLatticeExtents(const std::string_view text, Options& options) {
    std::vector<std::uint32_t> extents;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t separator = text.find('x', start);
        const std::string_view piece =
            text.substr(start, separator == std::string_view::npos ? std::string_view::npos
                                                                   : separator - start);
        if (piece.empty()) {
            fail("Empty extent in '" + std::string{text} + "'");
        }
        extents.push_back(parseNumber<std::uint32_t>(piece, "--lattice"));
        if (separator == std::string_view::npos) {
            break;
        }
        start = separator + 1;
    }
    if (extents.size() != 3) {
        fail("--lattice takes three extents, as in 32x32x16");
    }
    options.latticeWidth = extents[0];
    options.latticeHeight = extents[1];
    options.latticeDepth = extents[2];
}

[[nodiscard]] std::string describeLayers(const vkexp::neuro::BrainShape& brain) {
    std::string text;
    for (std::size_t layer = 0; layer < brain.hiddenLayerCount(); ++layer) {
        text += text.empty() ? "" : " -> ";
        text += std::to_string(brain.hiddenLayer(layer));
    }
    return text;
}

[[nodiscard]] vkexp::Neighborhood parseNeighborhood(const std::string_view name) {
    if (name == "faces" || name == "6") {
        return vkexp::Neighborhood::Faces;
    }
    if (name == "moore" || name == "26") {
        return vkexp::Neighborhood::Moore;
    }
    fail("Unknown neighbourhood '" + std::string{name} + "'; expected faces or moore");
}

[[nodiscard]] vkexp::WorldMode parseWorldMode(const std::string_view name) {
    if (name == "beacon") {
        return vkexp::WorldMode::Beacon;
    }
    if (name == "construction" || name == "build") {
        return vkexp::WorldMode::Construction;
    }
    if (name == "harvest") {
        return vkexp::WorldMode::Harvest;
    }
    if (name == "chasm") {
        return vkexp::WorldMode::Chasm;
    }
    fail("Unknown world '" + std::string{name} +
         "'; expected beacon, construction, harvest or chasm");
}

[[nodiscard]] const char* worldModeName(const vkexp::WorldMode mode) {
    if (mode == vkexp::WorldMode::Construction) {
        return "construction";
    }
    if (mode == vkexp::WorldMode::Chasm) {
        return "chasm";
    }
    return mode == vkexp::WorldMode::Harvest ? "harvest" : "beacon";
}

[[nodiscard]] const char* neighborhoodName(const vkexp::Neighborhood neighborhood) {
    return neighborhood == vkexp::Neighborhood::Faces ? "faces only (6)" : "Moore (26)";
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
    if (name == "spiking") {
        return vkexp::NeuronModel::Spiking;
    }
    fail("Unknown neuron model '" + std::string{name} +
         "'; expected reactive, time, gated or spiking");
}

// The short form, for files rather than for reading.
[[nodiscard]] const char* neuronModelKey(const vkexp::NeuronModel model) {
    switch (model) {
    case vkexp::NeuronModel::Reactive:
        return "reactive";
    case vkexp::NeuronModel::TimeConstant:
        return "time";
    case vkexp::NeuronModel::Gated:
        return "gated";
    case vkexp::NeuronModel::Spiking:
        return "spiking";
    }
    return "time";
}

[[nodiscard]] const char* neuronModelName(const vkexp::NeuronModel model) {
    switch (model) {
    case vkexp::NeuronModel::Reactive:
        return "reactive (no state)";
    case vkexp::NeuronModel::TimeConstant:
        return "time constant (one evolved rate per neuron)";
    case vkexp::NeuronModel::Gated:
        return "gated (rate recomputed from the inputs each step)";
    case vkexp::NeuronModel::Spiking:
        return "spiking (leaky integrate-and-fire discrete pulses)";
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
        } else if (argument == "--generations") {
            options.generations = parseNumber<std::uint64_t>(next(index, argument), argument);
        } else if (argument == "--steps") {
            options.stepsPerGeneration =
                parseNumber<std::uint32_t>(next(index, argument), argument);
        } else if (argument == "--steps-per-batch") {
            options.stepsPerBatch = parseNumber<std::uint32_t>(next(index, argument), argument);
        } else if (argument == "--weight-init") {
            const std::string_view name = next(index, argument);
            if (name == "saturating" || name == "flat") {
                options.weightInit = vkexp::WeightInit::Saturating;
            } else if (name == "fan-in" || name == "fanin") {
                options.weightInit = vkexp::WeightInit::FanIn;
            } else {
                fail("Unknown weight initialisation: " + std::string{name});
            }
        } else if (argument == "--population") {
            options.populationSize = parseNumber<std::size_t>(next(index, argument), argument);
        } else if (argument == "--agents-per-world") {
            options.agentsPerWorld = parseNumber<std::uint32_t>(next(index, argument), argument);
        } else if (argument == "--seed") {
            options.seed = parseNumber<std::uint32_t>(next(index, argument), argument);
        } else if (argument == "--lattice") {
            parseLatticeExtents(next(index, argument), options);
        } else if (argument == "--world") {
            options.worldMode = parseWorldMode(next(index, argument));
        } else if (argument == "--neighbourhood" || argument == "--neighborhood") {
            options.neighborhood = parseNeighborhood(next(index, argument));
        } else if (argument == "--turn-threshold" || argument == "--move-threshold") {
            options.turnThreshold = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--contact-radius") {
            options.contactRadius = parseNumber<std::uint32_t>(next(index, argument), argument);
        } else if (argument == "--build-interval") {
            options.buildIntervalTicks =
                parseNumber<std::uint32_t>(next(index, argument), argument);
        } else if (argument == "--build-threshold") {
            options.buildThreshold = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--resource-band") {
            const std::string_view band = next(index, argument);
            const std::size_t dash = band.find('-');
            if (dash == std::string_view::npos) {
                fail("--resource-band wants <low>-<high>, for example 5-9");
            }
            options.resourceHeightLow = parseNumber<std::uint32_t>(band.substr(0, dash), argument);
            options.resourceHeightHigh =
                parseNumber<std::uint32_t>(band.substr(dash + 1), argument);
        } else if (argument == "--course-fill") {
            options.constructionCourseFill = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--support-radius") {
            options.constructionSupportRadius =
                parseNumber<std::uint32_t>(next(index, argument), argument);
        } else if (argument == "--height-lead") {
            options.constructionHeightLead =
                parseNumber<std::uint32_t>(next(index, argument), argument);
        } else if (argument == "--ground-width") {
            options.chasmGroundWidth = parseNumber<std::uint32_t>(next(index, argument), argument);
        } else if (argument == "--side-support") {
            options.allowSideSupportedBlocks = true;
        } else if (argument == "--boundary-penalty") {
            options.fitness.boundaryPenalty = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--tracking-reward") {
            options.fitness.trackingReward = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--objective-bonus") {
            options.fitness.objectiveBonus = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--motor-cost") {
            options.fitness.motorCostWeight = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--refusal-penalty") {
            options.fitness.refusalPenalty = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--signal-cost") {
            options.fitness.signalCostFactor = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--fitness-sharing") {
            options.fitness.groupSharing = parseNumber<float>(next(index, argument), argument);
        } else if (argument == "--neuron-model") {
            options.neuronModel = parseNeuronModel(next(index, argument));
        } else if (argument == "--quiet") {
            options.quiet = true;
        } else if (argument == "--save-population") {
            options.savePopulation = next(index, argument);
        } else if (argument == "--save-champion") {
            options.saveChampion = next(index, argument);
        } else if (argument == "--hidden") {
            options.hiddenLayers = parseHiddenLayers(next(index, argument));
        } else if (argument == "--hidden-squash") {
            options.hiddenSquashes = parseHiddenSquashes(next(index, argument));
        } else if (argument == "--describe-brain") {
            options.describeBrain = next(index, argument);
        } else if (argument == "--load-population") {
            options.loadPopulation = next(index, argument);
        } else if (argument == "--save-run") {
            options.saveRun = next(index, argument);
        } else if (argument == "--load-run") {
            options.loadRun = next(index, argument);
        } else if (argument == "--csv") {
            options.csvPath = next(index, argument);
        } else {
            fail("Unknown argument: " + std::string{argument});
        }
    }
    if (options.generations == 0 || options.stepsPerBatch == 0 ||
        options.buildIntervalTicks == 0U) {
        fail("Generations, steps and steps-per-batch must all be non-zero");
    }
    return options;
}

// Writing down the structure is not a run: it follows from the brain plan and
// the neuron model alone, needs no device, and answers a question about the
// build rather than about an experiment. So it is its own action, like --help,
// and the run options around it are not even validated.
void describeBrainAndExit(const Options& options) {
    vkexp::SimulationStep settings{};
    for (std::size_t layer = 0; layer < options.hiddenSquashes.size(); ++layer) {
        settings.hiddenActivation[layer] = options.hiddenSquashes[layer];
    }
    for (std::size_t layer = 0; layer < options.hiddenLayers.size(); ++layer) {
        settings.hiddenLayers[layer] = options.hiddenLayers[layer];
    }
    const vkexp::neuro::BrainDescription description = vkexp::neuro::describeBrain(
        vkexp::resolvedBrain(settings), neuronModelKey(options.neuronModel));
    std::ofstream stream{options.describeBrain, std::ios::trunc};
    if (!stream) {
        fail("Unable to write the brain description to " + options.describeBrain);
    }
    stream << vkexp::neuro::brainDescriptionToJson(description);
    if (!stream) {
        fail("Failed while writing " + options.describeBrain);
    }
    if (!options.quiet) {
        std::cout << "Wrote the network's structure to " << options.describeBrain << '\n';
    }
}

int run(const Options& options) {
    vkexp::HeadlessComputeContext context{{"vklat headless evolution"}};

    vkexp::SimulationState state;
    state.controls.stepsPerGeneration = options.stepsPerGeneration;
    state.worlds.requestedAgentsPerWorld = options.agentsPerWorld;
    if (options.latticeWidth) {
        state.settings.latticeWidth = *options.latticeWidth;
        state.settings.latticeHeight = *options.latticeHeight;
        state.settings.latticeDepth = *options.latticeDepth;
    }
    state.settings.neighborhood = options.neighborhood;
    state.settings.worldMode = options.worldMode;
    applyWorldDefaults(state.settings);
    // After the world's own defaults, and only when it was actually asked for:
    // a chasm builds four times as fast as the other worlds unless the command
    // line says otherwise, and a flag left off must not read as a request for
    // the shared default.
    if (options.buildIntervalTicks) {
        state.settings.buildIntervalTicks = *options.buildIntervalTicks;
    }
    state.settings.buildThreshold = std::clamp(options.buildThreshold, 0.0F, 1.0F);
    const std::uint32_t ceiling = std::max(state.settings.latticeHeight, 2U) - 1U;
    state.settings.resourceHeightLow = std::clamp(options.resourceHeightLow, 1U, ceiling);
    state.settings.resourceHeightHigh =
        std::clamp(options.resourceHeightHigh, state.settings.resourceHeightLow, ceiling);
    state.settings.chasmGroundWidth = options.chasmGroundWidth;
    state.settings.constructionCourseFill = std::clamp(options.constructionCourseFill, 0.0F, 1.0F);
    state.settings.constructionHeightLead =
        std::clamp(options.constructionHeightLead, 1U, vkexp::latticeMaximumExtent);
    state.settings.constructionSupportRadius =
        std::min(options.constructionSupportRadius, vkexp::latticeMaximumExtent);
    // The chasm forces it on and the flag can only add to that: a world whose
    // objective needs a cantilever must not be startable without one.
    state.settings.allowSideSupportedBlocks |= options.allowSideSupportedBlocks ? 1U : 0U;
    if (options.turnThreshold) {
        state.settings.turnThreshold = std::clamp(*options.turnThreshold, 0.0F, 1.0F);
    }
    if (options.contactRadius) {
        state.settings.beaconContactRadius = *options.contactRadius;
    }
    state.settings.fitness = options.fitness;
    for (std::size_t layer = 0; layer < options.hiddenSquashes.size(); ++layer) {
        state.settings.hiddenActivation[layer] = options.hiddenSquashes[layer];
    }
    for (std::size_t layer = 0; layer < options.hiddenLayers.size(); ++layer) {
        state.settings.hiddenLayers[layer] = options.hiddenLayers[layer];
    }
    state.settings.neuronModel = options.neuronModel;

    vkexp::EvolutionSettings evolution;
    evolution.populationSize = options.populationSize;
    evolution.weightInit = options.weightInit;
    evolution.seed = options.seed;

    vkexp::SimulationDriverConfig config;
    config.maximumStepsPerBatch = options.stepsPerBatch;

    vkexp::SimulationDriver driver{state, evolution, config};
    driver.createResources(context.physicalDevice(), context.device());

    // A snapshot carries the lattice it ran in, so it decides the box, the
    // beacon placement and the shaping. The command line only fills in what the
    // file does not cover.
    if (!options.loadRun.empty() && !options.loadPopulation.empty()) {
        fail("--load-run and --load-population both set the population; pass one");
    }
    if (!options.loadRun.empty()) {
        const vkexp::RunSnapshot snapshot = vkexp::loadRunSnapshot(options.loadRun);
        driver.restoreSnapshot(snapshot);
        if (!options.quiet) {
            std::cout << "Resumed " << snapshot.genomes.size() << " genomes from "
                      << options.loadRun << " at generation " << snapshot.generation << ", step "
                      << snapshot.step << " of " << snapshot.stepsPerGeneration << '\n';
        }
    }

    const vkexp::neuro::BrainShape runBrain = vkexp::resolvedBrain(state.settings);
    if (!options.hiddenLayers.empty() &&
        runBrain.hiddenLayerCount() != options.hiddenLayers.size()) {
        fail("The requested hidden layers do not fit the genome capacity");
    }

    if (!options.loadPopulation.empty()) {
        const vkexp::GenomeArchive archive = vkexp::loadGenomeArchive(options.loadPopulation);
        driver.loadPopulation(archive.genomes, archive.metadata.generation);
        if (!options.quiet) {
            std::cout << "Resumed " << archive.genomes.size() << " genomes from "
                      << options.loadPopulation << " at generation " << archive.metadata.generation
                      << '\n';
            // Whether anything about the layout was actually checked, rather
            // than only the number of weights. A file that cannot say what its
            // weights mean still produces a plausible curve when reinterpreted.
            std::cout << "Structure:  "
                      << (archive.describedStructure
                              ? "checked against the file"
                              : "NOT STATED by the file -- only the weight count matched")
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
            // The shape columns describe the best-scoring world only, and are
            // empty in beacon mode. They are here because the score alone
            // cannot tell a slab from a spire, and a CSV that cannot either is
            // a CSV that has to be re-run to answer the question.
            *csv << "generation,lattice,seed,best,median,mean,arrival_ratio,"
                    "blocks,footprint,peak,mean_height,height_spread,compactness,overhangs,"
                    "roofed,turning,walking,edge,void,ceiling,crowded,cooling,off_lattice,"
                    "blocked,unsupported,above_frontier,in_the_way,placed,contested\n";
        }
    }

    // After the driver has had its say: refreshLattice clamps the box to what
    // the occupancy allocation holds at this population size, and reporting the
    // box that was asked for rather than the one that is running is how a run
    // gets filed under the wrong name.
    const std::string latticeText = std::to_string(state.settings.latticeWidth) + "x" +
                                    std::to_string(state.settings.latticeHeight) + "x" +
                                    std::to_string(state.settings.latticeDepth);

    if (!options.quiet) {
        std::cout << "Device:     " << context.deviceName() << '\n'
                  << "Lattice:    " << latticeText << " = " << latticeCellsPerWorld(state.settings)
                  << " cells per world, " << neighborhoodName(state.settings.neighborhood) << '\n'
                  << "Brain:      " << runBrain.inputCount << " -> " << describeLayers(runBrain)
                  << " -> " << runBrain.outputCount << '\n'
                  << "Trial:      " << state.controls.stepsPerGeneration
                  << " steps = " << std::fixed << std::setprecision(1)
                  << vkexp::units::secondsForSteps(state.controls.stepsPerGeneration,
                                                   vkexp::units::fixedTimeStep)
                  << " s at " << vkexp::units::simulationRateHz << " Hz\n"
                  << std::defaultfloat << std::setprecision(6)
                  << "Population: " << driver.evolution().population().size() << " genomes x "
                  << driver.config().trialsPerGenome << " trials = " << state.agents.agentCount
                  << " agents in " << state.worlds.worldCount << " lattices\n"
                  << "World:      "
                  << worldModeName(state.settings.worldMode)
                  << '\n'
                  << "Movement:   turn threshold " << std::fixed << std::setprecision(2)
                  << state.settings.turnThreshold;
        if (vkexp::worldBuilds(state.settings.worldMode)) {
            std::cout << '\n'
                      << "Construction: one supported block every "
                      << state.settings.buildIntervalTicks << " ticks, output > "
                      << state.settings.buildThreshold << ", boundary penalty "
                      << state.settings.fitness.boundaryPenalty << " per agent-tick\n"
                      << "Support:    "
                      << (state.settings.allowSideSupportedBlocks != 0U
                              ? "a block directly below, or a cardinal side face"
                              : "a block directly below")
                      << '\n';
            if (vkexp::lattice::kernel::latticeWorldFrontier(
                    static_cast<std::uint32_t>(state.settings.worldMode))) {
                std::cout << "Frontier:   " << state.settings.constructionCourseFill * 100.0F
                          << "% fill within " << state.settings.constructionSupportRadius
                          << " cells, " << state.settings.constructionHeightLead
                          << " levels of headroom above it\n";
            }
            if (vkexp::worldHarvests(state.settings.worldMode)) {
                std::cout << "Resource:   between " << state.settings.resourceHeightLow << " and "
                          << state.settings.resourceHeightHigh << " levels up, collected within "
                          << state.settings.beaconContactRadius
                          << " cell(s) and carried back to the ground\n";
            }
            if (state.settings.worldMode == vkexp::WorldMode::Chasm) {
                std::cout << "Ground:     " << vkexp::latticeGroundWidth(state.settings) << " of "
                          << state.settings.latticeWidth
                          << " columns; the rest is open air, and the resource hangs over it\n";
            }
        } else {
            std::cout << ", beacon reached within " << std::defaultfloat << std::setprecision(6)
                      << state.settings.beaconContactRadius << " cell(s)\n";
        }
        std::cout << "Neurons:    " << neuronModelName(state.settings.neuronModel) << '\n';
        // Only when it is on, so a default run's output stays comparable with
        // every run recorded before the option existed.
        if (state.settings.worldMode == vkexp::WorldMode::Beacon &&
            state.settings.fitness.groupSharing > 0.0F) {
            std::cout << "Selection:  group fitness sharing " << std::fixed << std::setprecision(2)
                      << state.settings.fitness.groupSharing << std::defaultfloat
                      << std::setprecision(6) << " (plotted fitness stays individual)\n";
        }
        std::cout << '\n'
                  << (state.settings.worldMode == vkexp::WorldMode::Construction
                          ? "  gen    weighted      median        mean weighted.fill\n"
                          : "  gen        best      median        mean   arrival\n");
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
            *csv << generation << ',' << latticeText << ',' << options.seed << ','
                 << state.statistics.bestFitness << ',' << state.statistics.medianFitness << ','
                 << state.statistics.meanFitness << ',' << state.statistics.arrivalRatio;
            const auto& shapes = state.statistics.worldShapes;
            if (shapes.empty()) {
                *csv << ",,,,,,,";
            } else {
                const vkexp::StructureShape& shape =
                    shapes[std::min<std::size_t>(state.statistics.bestWorld, shapes.size() - 1)];
                *csv << ',' << shape.blocks << ',' << shape.footprint << ',' << shape.peak << ','
                     << shape.meanHeight << ',' << shape.heightSpread << ',' << shape.compactness
                     << ',' << shape.overhangs << ',' << shape.enclosed;
            }
            // Summed over worlds, unlike the shape beside it: a refusal is a
            // statement about the rules, and the rules are the same everywhere.
            const auto reasons =
                static_cast<std::size_t>(vkexp::lattice::kernel::LatticeBuildOutcomeCount);
            for (std::size_t reason = 0; reason < reasons; ++reason) {
                std::uint64_t total = 0;
                for (std::size_t at = reason; at < state.statistics.buildOutcomes.size();
                     at += reasons) {
                    total += state.statistics.buildOutcomes[at];
                }
                *csv << ',';
                if (!state.statistics.buildOutcomes.empty()) {
                    *csv << total;
                }
            }
            *csv << '\n';
        }
    }
    if (csv) {
        csv->flush();
    }

    // finishGeneration() has already produced the next population, whose first
    // eliteCount entries are the ranked survivors, champion first.
    const std::vector<vkexp::Genome>& population = driver.evolution().population();
    const vkexp::GenomeArchiveMetadata metadata =
        vkexp::genomeArchiveMetadata(state, driver, runBrain);
    if (!options.savePopulation.empty()) {
        vkexp::saveGenomeArchive(options.savePopulation, population, metadata);
        if (!options.quiet) {
            std::cout << "Saved " << population.size() << " genomes to " << options.savePopulation
                      << '\n';
        }
    }
    if (!options.saveRun.empty()) {
        // The device is idle here -- the last generation was waited on before it
        // was scored -- which is what snapshot() requires to read the agents back.
        vkexp::saveRunSnapshot(options.saveRun, driver.snapshot());
        if (!options.quiet) {
            std::cout << "Saved the run to " << options.saveRun << '\n';
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
        if (!options.describeBrain.empty()) {
            describeBrainAndExit(options);
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
