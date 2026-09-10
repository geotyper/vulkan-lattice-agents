#include "vkexp/simulation/WorldSnapshot.hpp"

#include "vkexp/neuro/NeuralNetwork.hpp"

#include <array>
#include <bit>
#include <fstream>

namespace vkexp {
namespace {

static_assert(std::endian::native == std::endian::little,
              "World snapshots are written little-endian");

constexpr std::array<char, 4> snapshotMagic{'V', 'K', 'N', 'W'};

// The physics block is written field by field rather than as a struct blob.
// SimulationStep carries bools and is free to be reordered by whoever adds the
// next tunable; a blob would keep loading and quietly mean something else.
struct SnapshotHeader {
    std::array<char, 4> magic{};
    std::uint32_t version{};
    std::uint32_t genomeCount{};
    std::uint32_t weightCount{};
    std::uint32_t agentCount{};
    std::uint32_t agentStateBytes{};
    std::uint32_t puckCount{};
    std::uint32_t physicsFieldCount{};
    std::uint32_t stepsPerGeneration{};
    std::uint32_t step{};
    std::uint32_t requestedAgentsPerWorld{};
    std::uint32_t trialsPerGenome{};
    std::uint32_t seed{};
    std::uint64_t generation{};
};

static_assert(sizeof(SnapshotHeader) == 64);

// One list, walked in both directions, so a field can never be saved and loaded
// in a different order. Adding a tunable means adding a line here and bumping
// worldSnapshotVersion.
template <typename Visit> void visitPhysics(SimulationStep& physics, Visit&& visit) {
    visit(physics.deltaTime);
    visit(physics.worldRadius);
    visit(physics.thrust);
    visit(physics.turnAcceleration);
    visit(physics.linearDrag);
    visit(physics.angularDrag);
    visit(physics.sensorFieldOfView);
    visit(physics.arrivalRadiusMultiplier);
    visit(physics.maximumSpeed);
    visit(physics.maximumAngularSpeed);
    visit(physics.lightSensorRange);
    visit(physics.lightRangeRatio);
    visit(physics.lightExposure);
    visit(physics.collisionRestitution);
    visit(physics.contactStiffness);
    visit(physics.wallCollisionPenalty);
    visit(physics.beaconAngularSpeed);
    visit(physics.beaconRotationAngle);
    visit(physics.beaconRadiusRatio);
    visit(physics.beaconMotionTime);
    visit(physics.beaconTeleportProbability);
    visit(physics.beaconRandomSpeed);
    visit(physics.forageCargoDecayRate);
    visit(physics.foragePickupReward);
    visit(physics.forageDeliveryReward);
    visit(physics.puckTargetRadiusRatio);
    visit(physics.puckRadiusRatio);
    visit(physics.puckBreakawayPushes);
    visit(physics.gateLatchSeconds);
    visit(physics.trailDepositRate);
    visit(physics.trailHalfLife);
    visit(physics.beaconTrailDepositRate);
    visit(physics.trailRenderWidth);
    visit(physics.trailCellSize);
    visit(physics.fitness.objectiveBonus);
    visit(physics.fitness.motorCostWeight);
    visit(physics.fitness.signalCostFactor);
    visit(physics.fitness.energyDrain);
    visit(physics.fitness.trackingReward);
    visit(physics.fitness.groupSharing);
}

constexpr std::uint32_t physicsFloatCount = 40;

// visitPhysics and PhysicsIntegers together have to name every field of
// SimulationStep, and this is what notices when a new tunable is added and
// quietly not saved. If it fires: add the field to one of the two lists above,
// bump worldSnapshotVersion, then update this number.
//
// It catches every float and every integer, but not a bool: a bool can land in
// padding the trailing bools already carry and move this number not at all, as
// neuronMemoryEnabled did before it became an enum, and as swapDeliveryEnds did
// after it. A new bool is therefore
// covered by testWorldSnapshotRoundTrip naming it in both polarities, which is
// the check that does not depend on the size changing.
static_assert(sizeof(SimulationStep) == 212,
              "SimulationStep changed shape -- update the world snapshot field lists");

// The handful of fields that are not floats, kept apart so the float list above
// stays a plain sequence.
struct PhysicsIntegers {
    std::uint32_t beaconMotionSeed{};
    std::uint32_t worldShape{};
    std::uint32_t worldSize{};
    std::uint32_t beaconScenario{};
    std::uint32_t beaconPhase{};
    std::uint32_t beaconPhaseChanged{};
    std::uint32_t agentCollisionsEnabled{};
    std::uint32_t agentLightEnabled{};
    std::uint32_t trailMode{};
    std::uint32_t firstHiddenLayer{};
    std::uint32_t secondHiddenLayer{};
    std::uint32_t thirdHiddenLayer{};
    std::uint32_t neuronModel{};
    std::uint32_t swapDeliveryEnds{};
    std::uint32_t uniformBeaconColor{};
    std::uint32_t blockedDoorPerGeneration{};
    std::uint32_t puckRandomStart{};
};

static_assert(sizeof(PhysicsIntegers) == 68);

void readExactly(std::ifstream& stream, void* destination, const std::size_t bytes,
                 const std::filesystem::path& path) {
    stream.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    if (stream.gcount() != static_cast<std::streamsize>(bytes)) {
        throw WorldSnapshotError("World snapshot is truncated: " + path.string());
    }
}

} // namespace

void saveWorldSnapshot(const std::filesystem::path& path, const WorldSnapshot& snapshot) {
    if (snapshot.genomes.empty() || snapshot.agents.empty()) {
        throw WorldSnapshotError("Refusing to write an empty world snapshot");
    }
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code directoryError;
        std::filesystem::create_directories(parent, directoryError);
    }
    std::ofstream stream{path, std::ios::binary | std::ios::trunc};
    if (!stream) {
        throw WorldSnapshotError("Unable to open world snapshot for writing: " + path.string());
    }

    const SnapshotHeader header{snapshotMagic,
                                worldSnapshotVersion,
                                static_cast<std::uint32_t>(snapshot.genomes.size()),
                                static_cast<std::uint32_t>(neuro::Topology::weightCount),
                                static_cast<std::uint32_t>(snapshot.agents.size()),
                                static_cast<std::uint32_t>(sizeof(AgentState)),
                                static_cast<std::uint32_t>(snapshot.pucks.size()),
                                physicsFloatCount,
                                snapshot.stepsPerGeneration,
                                snapshot.step,
                                snapshot.requestedAgentsPerWorld,
                                snapshot.trialsPerGenome,
                                snapshot.seed,
                                snapshot.generation};
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));

    SimulationStep physics = snapshot.physics;
    std::vector<float> floats;
    floats.reserve(physicsFloatCount);
    visitPhysics(physics, [&](const float value) { floats.push_back(value); });
    if (floats.size() != physicsFloatCount) {
        throw WorldSnapshotError("World snapshot physics list disagrees with its declared size");
    }
    stream.write(reinterpret_cast<const char*>(floats.data()),
                 static_cast<std::streamsize>(floats.size() * sizeof(float)));

    const PhysicsIntegers integers{physics.beaconMotionSeed,
                                   static_cast<std::uint32_t>(physics.worldShape),
                                   static_cast<std::uint32_t>(physics.worldSize),
                                   static_cast<std::uint32_t>(physics.beaconScenario),
                                   physics.beaconPhase,
                                   physics.beaconPhaseChanged ? 1U : 0U,
                                   physics.agentCollisionsEnabled ? 1U : 0U,
                                   physics.agentLightEnabled ? 1U : 0U,
                                   static_cast<std::uint32_t>(physics.trailMode),
                                   physics.hiddenLayers[0],
                                   physics.hiddenLayers[1],
                                   physics.hiddenLayers[2],
                                   static_cast<std::uint32_t>(physics.neuronModel),
                                   physics.swapDeliveryEnds ? 1U : 0U,
                                   physics.uniformBeaconColor ? 1U : 0U,
                                   physics.blockedDoorPerGeneration ? 1U : 0U,
                                   physics.puckRandomStart ? 1U : 0U};
    stream.write(reinterpret_cast<const char*>(&integers), sizeof(integers));

    for (const Genome& genome : snapshot.genomes) {
        stream.write(reinterpret_cast<const char*>(genome.weights.data()),
                     static_cast<std::streamsize>(genome.weights.size() * sizeof(float)));
    }
    stream.write(reinterpret_cast<const char*>(snapshot.agents.data()),
                 static_cast<std::streamsize>(snapshot.agents.size() * sizeof(AgentState)));
    stream.write(reinterpret_cast<const char*>(snapshot.pucks.data()),
                 static_cast<std::streamsize>(snapshot.pucks.size() * sizeof(PuckState)));
    stream.flush();
    if (!stream) {
        throw WorldSnapshotError("Failed while writing world snapshot: " + path.string());
    }
}

WorldSnapshot loadWorldSnapshot(const std::filesystem::path& path) {
    std::ifstream stream{path, std::ios::binary};
    if (!stream) {
        throw WorldSnapshotError("Unable to open world snapshot: " + path.string());
    }
    SnapshotHeader header{};
    readExactly(stream, &header, sizeof(header), path);
    if (header.magic != snapshotMagic) {
        throw WorldSnapshotError("Not a world snapshot: " + path.string());
    }
    if (header.version != worldSnapshotVersion) {
        throw WorldSnapshotError("Unsupported world snapshot version in " + path.string());
    }
    if (header.weightCount != neuro::Topology::weightCount) {
        throw WorldSnapshotError("World snapshot was written for a different brain topology: " +
                                 path.string());
    }
    if (header.agentStateBytes != sizeof(AgentState)) {
        throw WorldSnapshotError("World snapshot was written for a different agent layout: " +
                                 path.string());
    }
    if (header.physicsFieldCount != physicsFloatCount) {
        throw WorldSnapshotError("World snapshot was written for a different settings list: " +
                                 path.string());
    }
    if (header.genomeCount == 0 || header.agentCount == 0) {
        throw WorldSnapshotError("World snapshot contains no population: " + path.string());
    }

    WorldSnapshot snapshot;
    snapshot.generation = header.generation;
    snapshot.step = header.step;
    snapshot.stepsPerGeneration = header.stepsPerGeneration;
    snapshot.requestedAgentsPerWorld = header.requestedAgentsPerWorld;
    snapshot.trialsPerGenome = header.trialsPerGenome;
    snapshot.seed = header.seed;

    std::vector<float> floats(physicsFloatCount);
    readExactly(stream, floats.data(), floats.size() * sizeof(float), path);
    std::size_t cursor = 0;
    visitPhysics(snapshot.physics, [&](float& value) { value = floats[cursor++]; });

    PhysicsIntegers integers{};
    readExactly(stream, &integers, sizeof(integers), path);
    if (integers.beaconScenario >= beaconScenarioCount) {
        throw WorldSnapshotError("World snapshot names a scenario this build does not have: " +
                                 path.string());
    }
    snapshot.physics.beaconMotionSeed = integers.beaconMotionSeed;
    snapshot.physics.worldShape = static_cast<WorldShape>(integers.worldShape);
    snapshot.physics.worldSize = static_cast<WorldSize>(integers.worldSize);
    snapshot.physics.beaconScenario = static_cast<BeaconScenario>(integers.beaconScenario);
    snapshot.physics.beaconPhase = integers.beaconPhase;
    snapshot.physics.beaconPhaseChanged = integers.beaconPhaseChanged != 0;
    snapshot.physics.agentCollisionsEnabled = integers.agentCollisionsEnabled != 0;
    snapshot.physics.agentLightEnabled = integers.agentLightEnabled != 0;
    snapshot.physics.trailMode = static_cast<TrailMode>(integers.trailMode);
    snapshot.physics.hiddenLayers = {integers.firstHiddenLayer, integers.secondHiddenLayer,
                                     integers.thirdHiddenLayer};
    snapshot.physics.swapDeliveryEnds = integers.swapDeliveryEnds != 0;
    snapshot.physics.uniformBeaconColor = integers.uniformBeaconColor != 0;
    snapshot.physics.blockedDoorPerGeneration = integers.blockedDoorPerGeneration != 0;
    snapshot.physics.puckRandomStart = integers.puckRandomStart != 0;
    if (integers.neuronModel >= neuronModelCount) {
        throw WorldSnapshotError("World snapshot names neuron model " +
                                 std::to_string(integers.neuronModel) + ", which this build has "
                                 "no rule for: " + path.string());
    }
    snapshot.physics.neuronModel = static_cast<NeuronModel>(integers.neuronModel);

    snapshot.genomes.resize(header.genomeCount);
    for (Genome& genome : snapshot.genomes) {
        readExactly(stream, genome.weights.data(), genome.weights.size() * sizeof(float), path);
    }
    snapshot.agents.resize(header.agentCount);
    readExactly(stream, snapshot.agents.data(), snapshot.agents.size() * sizeof(AgentState), path);
    snapshot.pucks.resize(header.puckCount);
    if (header.puckCount > 0) {
        readExactly(stream, snapshot.pucks.data(), snapshot.pucks.size() * sizeof(PuckState), path);
    }
    return snapshot;
}

} // namespace vkexp
