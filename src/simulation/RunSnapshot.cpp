#include "vkexp/simulation/RunSnapshot.hpp"

#include "vkexp/neuro/NeuralNetwork.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <span>
#include <string>

namespace vkexp {
namespace {

static_assert(std::endian::native == std::endian::little,
              "Run snapshots are written little-endian");

constexpr std::array<char, 4> snapshotMagic{'V', 'K', 'L', 'R'};

// The settings block is written field by field rather than as a struct blob.
// SimulationStep is free to be reordered by whoever adds the next tunable; a
// blob would keep loading and quietly mean something else.
struct SnapshotHeader {
    std::array<char, 4> magic{};
    std::uint32_t version{};
    std::uint32_t genomeCount{};
    std::uint32_t weightCount{};
    std::uint32_t agentCount{};
    std::uint32_t agentStateBytes{};
    std::uint32_t settingsFieldCount{};
    std::uint32_t stepsPerGeneration{};
    std::uint32_t step{};
    std::uint32_t requestedAgentsPerWorld{};
    std::uint32_t trialsPerGenome{};
    std::uint32_t seed{};
    std::uint64_t generation{};
};

static_assert(sizeof(SnapshotHeader) == 56);

// One list, walked in both directions, so a field can never be saved and loaded
// in a different order. Adding a tunable means adding a line here and bumping
// runSnapshotVersion.
template <typename Visit> void visitSettings(SimulationStep& settings, Visit&& visit) {
    visit(settings.deltaTime);
    visit(settings.moveThreshold);
    visit(settings.fitness.trackingReward);
    visit(settings.fitness.objectiveBonus);
    visit(settings.fitness.motorCostWeight);
    visit(settings.fitness.refusalPenalty);
    visit(settings.fitness.signalCostFactor);
    visit(settings.fitness.groupSharing);
    visit(settings.fitness.boundaryPenalty);
    visit(settings.buildThreshold);
    visit(settings.constructionCourseFill);
}

constexpr std::uint32_t settingsFloatCount = 11;

// visitSettings and SettingsIntegers together have to name every field of
// SimulationStep, and this is what notices when a new tunable is added and
// quietly not saved. If it fires: add the field to one of the two lists above,
// bump runSnapshotVersion, then update this number.
static_assert(sizeof(SimulationStep) == 108,
              "SimulationStep changed shape -- update the run snapshot field lists");

// The fields that are not floats, kept apart so the float list above stays a
// plain sequence.
struct SettingsIntegers {
    std::uint32_t latticeWidth{};
    std::uint32_t latticeHeight{};
    std::uint32_t latticeDepth{};
    std::uint32_t beaconContactRadius{};
    std::uint32_t beaconSeed{};
    std::uint32_t neighborhood{};
    std::uint32_t firstHiddenLayer{};
    std::uint32_t secondHiddenLayer{};
    std::uint32_t thirdHiddenLayer{};
    std::uint32_t neuronModel{};
    std::uint32_t worldMode{};
    std::uint32_t buildIntervalTicks{};
    std::uint32_t constructionHeightLead{};
    std::uint32_t allowSideSupportedBlocks{};
    std::uint32_t constructionSupportRadius{};
    std::uint32_t resourceHeight{};
};

static_assert(sizeof(SettingsIntegers) == 64);

void readExactly(std::ifstream& stream, void* destination, const std::size_t bytes,
                 const std::filesystem::path& path) {
    stream.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    if (stream.gcount() != static_cast<std::streamsize>(bytes)) {
        throw RunSnapshotError("Run snapshot is truncated: " + path.string());
    }
}

} // namespace

void saveRunSnapshot(const std::filesystem::path& path, const RunSnapshot& snapshot) {
    if (snapshot.genomes.empty() || snapshot.agents.empty()) {
        throw RunSnapshotError("Refusing to write an empty run snapshot");
    }
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code directoryError;
        std::filesystem::create_directories(parent, directoryError);
    }
    std::ofstream stream{path, std::ios::binary | std::ios::trunc};
    if (!stream) {
        throw RunSnapshotError("Unable to open run snapshot for writing: " + path.string());
    }

    const SnapshotHeader header{snapshotMagic,
                                runSnapshotVersion,
                                static_cast<std::uint32_t>(snapshot.genomes.size()),
                                static_cast<std::uint32_t>(snapshot.genomes.front().weights.size()),
                                static_cast<std::uint32_t>(snapshot.agents.size()),
                                static_cast<std::uint32_t>(sizeof(AgentState)),
                                settingsFloatCount,
                                snapshot.stepsPerGeneration,
                                snapshot.step,
                                snapshot.requestedAgentsPerWorld,
                                snapshot.trialsPerGenome,
                                snapshot.seed,
                                snapshot.generation};
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));

    SimulationStep settings = snapshot.settings;
    std::vector<float> floats;
    floats.reserve(settingsFloatCount);
    visitSettings(settings, [&](const float value) { floats.push_back(value); });
    if (floats.size() != settingsFloatCount) {
        throw RunSnapshotError("Run snapshot settings list disagrees with its declared size");
    }
    stream.write(reinterpret_cast<const char*>(floats.data()),
                 static_cast<std::streamsize>(floats.size() * sizeof(float)));

    const SettingsIntegers integers{settings.latticeWidth,
                                    settings.latticeHeight,
                                    settings.latticeDepth,
                                    settings.beaconContactRadius,
                                    settings.beaconSeed,
                                    static_cast<std::uint32_t>(settings.neighborhood),
                                    settings.hiddenLayers[0],
                                    settings.hiddenLayers[1],
                                    settings.hiddenLayers[2],
                                    static_cast<std::uint32_t>(settings.neuronModel),
                                    static_cast<std::uint32_t>(settings.worldMode),
                                    settings.buildIntervalTicks,
                                    settings.constructionHeightLead,
                                    settings.allowSideSupportedBlocks,
                                    settings.constructionSupportRadius,
                                    settings.resourceHeight};
    stream.write(reinterpret_cast<const char*>(&integers), sizeof(integers));

    for (const Genome& genome : snapshot.genomes) {
        stream.write(reinterpret_cast<const char*>(genome.weights.data()),
                     static_cast<std::streamsize>(genome.weights.size() * sizeof(float)));
    }
    stream.write(reinterpret_cast<const char*>(snapshot.agents.data()),
                 static_cast<std::streamsize>(snapshot.agents.size() * sizeof(AgentState)));
    const std::uint32_t floorCapacity =
        snapshot.settings.latticeWidth * snapshot.settings.latticeDepth;
    const std::uint32_t requestedAgents =
        worldBuilds(snapshot.settings.worldMode)
            ? std::min(snapshot.requestedAgentsPerWorld, floorCapacity)
            : snapshot.requestedAgentsPerWorld;
    const std::uint32_t agentsPerWorld =
        clampAgentsPerWorld(static_cast<std::uint32_t>(snapshot.genomes.size()), requestedAgents);
    const std::uint32_t worlds =
        logicalWorldCount(static_cast<std::uint32_t>(snapshot.genomes.size()), agentsPerWorld,
                          snapshot.trialsPerGenome);
    const std::size_t structureCount =
        static_cast<std::size_t>(latticeCellsPerWorld(snapshot.settings)) * worlds;
    std::vector<std::int32_t> emptyStructures;
    std::span<const std::int32_t> structures = snapshot.structures;
    if (structures.empty()) {
        emptyStructures.assign(structureCount, lattice::kernel::LatticeNoStructure);
        structures = emptyStructures;
    } else if (structures.size() != structureCount) {
        throw RunSnapshotError("Run snapshot construction field has the wrong size");
    }
    stream.write(reinterpret_cast<const char*>(structures.data()),
                 static_cast<std::streamsize>(structures.size() * sizeof(std::int32_t)));
    stream.flush();
    if (!stream) {
        throw RunSnapshotError("Failed while writing run snapshot: " + path.string());
    }
}

RunSnapshot loadRunSnapshot(const std::filesystem::path& path) {
    std::ifstream stream{path, std::ios::binary};
    if (!stream) {
        throw RunSnapshotError("Unable to open run snapshot: " + path.string());
    }
    SnapshotHeader header{};
    readExactly(stream, &header, sizeof(header), path);
    if (header.magic != snapshotMagic) {
        throw RunSnapshotError("Not a run snapshot: " + path.string());
    }
    if (header.version != runSnapshotVersion) {
        throw RunSnapshotError("Unsupported run snapshot version in " + path.string());
    }
    // The length is the file's own, not one number every run shares, so what is
    // checked here is only that it is a length this build could produce. The
    // brain plan itself travels in the settings below and is compared where it
    // can be explained -- see the archive's structure document.
    if (header.weightCount == 0 || header.weightCount > neuro::Topology::maximumWeightCount) {
        throw RunSnapshotError("Run snapshot claims a genome length no plan this build can run "
                               "produces: " +
                               path.string());
    }
    if (header.agentStateBytes != sizeof(AgentState)) {
        throw RunSnapshotError("Run snapshot was written for a different agent layout: " +
                               path.string());
    }
    if (header.settingsFieldCount != settingsFloatCount) {
        throw RunSnapshotError("Run snapshot was written for a different settings list: " +
                               path.string());
    }
    if (header.genomeCount == 0 || header.agentCount == 0) {
        throw RunSnapshotError("Run snapshot contains no population: " + path.string());
    }

    RunSnapshot snapshot;
    snapshot.generation = header.generation;
    snapshot.step = header.step;
    snapshot.stepsPerGeneration = header.stepsPerGeneration;
    snapshot.requestedAgentsPerWorld = header.requestedAgentsPerWorld;
    snapshot.trialsPerGenome = header.trialsPerGenome;
    snapshot.seed = header.seed;

    std::vector<float> floats(settingsFloatCount);
    readExactly(stream, floats.data(), floats.size() * sizeof(float), path);
    std::size_t cursor = 0;
    visitSettings(snapshot.settings, [&](float& value) { value = floats[cursor++]; });

    SettingsIntegers integers{};
    readExactly(stream, &integers, sizeof(integers), path);
    if (integers.neighborhood >= neighborhoodCount) {
        throw RunSnapshotError("Run snapshot names a neighbourhood this build does not have: " +
                               path.string());
    }
    if (integers.neuronModel >= neuronModelCount) {
        throw RunSnapshotError("Run snapshot names neuron model " +
                               std::to_string(integers.neuronModel) +
                               ", which this build has no rule for: " + path.string());
    }
    if (integers.worldMode >= worldModeCount) {
        throw RunSnapshotError("Run snapshot names a world mode this build does not have: " +
                               path.string());
    }
    if (integers.allowSideSupportedBlocks > 1U) {
        throw RunSnapshotError("Run snapshot has an invalid side-support switch: " + path.string());
    }
    // A lattice outside the range this build allocates is refused rather than
    // clamped: resuming into a differently sized box would put every agent's
    // recorded cell somewhere else, which is a different experiment reported
    // under the old one's name.
    if (integers.latticeWidth < latticeMinimumExtent ||
        integers.latticeWidth > latticeMaximumExtent ||
        integers.latticeHeight < latticeMinimumExtent ||
        integers.latticeHeight > latticeMaximumExtent ||
        integers.latticeDepth < latticeMinimumExtent ||
        integers.latticeDepth > latticeMaximumExtent) {
        throw RunSnapshotError("Run snapshot names a lattice this build cannot allocate: " +
                               path.string());
    }
    snapshot.settings.latticeWidth = integers.latticeWidth;
    snapshot.settings.latticeHeight = integers.latticeHeight;
    snapshot.settings.latticeDepth = integers.latticeDepth;
    snapshot.settings.beaconContactRadius = integers.beaconContactRadius;
    snapshot.settings.beaconSeed = integers.beaconSeed;
    snapshot.settings.neighborhood = static_cast<Neighborhood>(integers.neighborhood);
    snapshot.settings.hiddenLayers = {integers.firstHiddenLayer, integers.secondHiddenLayer,
                                      integers.thirdHiddenLayer};
    snapshot.settings.neuronModel = static_cast<NeuronModel>(integers.neuronModel);
    snapshot.settings.worldMode = static_cast<WorldMode>(integers.worldMode);
    snapshot.settings.buildIntervalTicks = integers.buildIntervalTicks;
    snapshot.settings.constructionHeightLead = integers.constructionHeightLead;
    snapshot.settings.constructionSupportRadius = integers.constructionSupportRadius;
    snapshot.settings.resourceHeight = integers.resourceHeight;
    snapshot.settings.allowSideSupportedBlocks = integers.allowSideSupportedBlocks;

    snapshot.genomes.assign(header.genomeCount, Genome{neuro::Weights(header.weightCount, 0.0F)});
    for (Genome& genome : snapshot.genomes) {
        readExactly(stream, genome.weights.data(), genome.weights.size() * sizeof(float), path);
    }
    snapshot.agents.resize(header.agentCount);
    readExactly(stream, snapshot.agents.data(), snapshot.agents.size() * sizeof(AgentState), path);
    const std::uint32_t floorCapacity =
        snapshot.settings.latticeWidth * snapshot.settings.latticeDepth;
    const std::uint32_t requestedAgents =
        worldBuilds(snapshot.settings.worldMode)
            ? std::min(snapshot.requestedAgentsPerWorld, floorCapacity)
            : snapshot.requestedAgentsPerWorld;
    const std::uint32_t agentsPerWorld = clampAgentsPerWorld(header.genomeCount, requestedAgents);
    const std::uint32_t worlds =
        logicalWorldCount(header.genomeCount, agentsPerWorld, snapshot.trialsPerGenome);
    snapshot.structures.resize(static_cast<std::size_t>(latticeCellsPerWorld(snapshot.settings)) *
                               worlds);
    readExactly(stream, snapshot.structures.data(),
                snapshot.structures.size() * sizeof(std::int32_t), path);
    return snapshot;
}

} // namespace vkexp
