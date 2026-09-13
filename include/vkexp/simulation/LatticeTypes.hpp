#pragma once

#include "vkexp/lattice/LatticeKernel.hpp"
#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/simulation/Units.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vkexp {

enum class NeuronModel : std::uint32_t {
    Reactive = neuro::kernel::NeuronModelReactive,
    TimeConstant = neuro::kernel::NeuronModelTimeConstant,
    Gated = neuro::kernel::NeuronModelGated,
    Spiking = neuro::kernel::NeuronModelSpiking,
};

inline constexpr std::size_t neuronModelCount = neuro::kernel::NeuronModelCount;

// Which cells an agent may step into. Declared in LatticeKernel.inl so the
// shader reads the same numbers; see there for why the input vector keeps its
// width under both.
enum class Neighborhood : std::uint32_t {
    Faces = lattice::kernel::LatticeNeighborhoodFaces,
    Moore = lattice::kernel::LatticeNeighborhoodMoore,
};

inline constexpr std::size_t neighborhoodCount = lattice::kernel::LatticeNeighborhoodCount;

enum class WorldMode : std::uint32_t {
    Beacon = lattice::kernel::LatticeWorldBeacon,
    Construction = lattice::kernel::LatticeWorldConstruction,
};

inline constexpr std::size_t worldModeCount = lattice::kernel::LatticeWorldCount;

// The brain's neighbourhood block has to be as wide as the lattice's
// neighbourhood. The two constants live in different kernels because they
// compile into different namespaces in C++ and into one flat scope in GLSL; this
// is where they are held to each other.
static_assert(neuro::kernel::BrainNeighborCount == lattice::kernel::LatticeNeighborCount,
              "The brain's neighbourhood block and the lattice's neighbourhood disagree");

// --- how big a lattice may be -----------------------------------------------
//
// Variant A of the plan: a small lattice and many worlds. 32x32x16 is 16384
// cells, 64 KiB of occupancy per world, so the ~170 logical worlds a population
// of 512 splits into cost about 11 MiB -- against the 340 MiB a 100x100x50 box
// would have cost for the same population, which is what ruled that size out for
// anything but a single display world.
//
// Density is the real argument, not bytes: twelve agents in 16384 cells is
// 0.07%, which is already sparse. The plan's neighbourhood work -- chains,
// formations, reading what a neighbour broadcasts -- needs agents that meet.
inline constexpr std::uint32_t latticeMinimumExtent = 4;
inline constexpr std::uint32_t latticeMaximumExtent = 128;
inline constexpr std::uint32_t latticeDefaultWidth = 32;
inline constexpr std::uint32_t latticeDefaultHeight = 32;
inline constexpr std::uint32_t latticeDefaultDepth = 16;

// The occupancy and bid grids are one int per cell per world and are both
// touched in full on every step, so the budget is set by what can be streamed
// twice per step rather than by what fits. Capped against the device's own
// memory too, since a headless CI GPU may have far less than a desktop one.
inline constexpr std::uint64_t latticeFieldByteBudget = 256ULL * 1024ULL * 1024ULL;
inline constexpr std::uint64_t latticeFieldHeapFraction = 16; // at most a sixteenth of VRAM

inline constexpr std::uint32_t minimumAgentsPerWorld = 1;

[[nodiscard]] constexpr std::uint32_t
clampAgentsPerWorld(const std::uint32_t genomeCount, const std::uint32_t requestedAgentsPerWorld) {
    if (genomeCount == 0) {
        return 0;
    }
    const std::uint32_t minimum = std::min(minimumAgentsPerWorld, genomeCount);
    return std::clamp(requestedAgentsPerWorld, minimum, genomeCount);
}

[[nodiscard]] constexpr std::uint32_t worldGroupCount(const std::uint32_t genomeCount,
                                                      const std::uint32_t agentsPerWorld) {
    const std::uint32_t clamped = clampAgentsPerWorld(genomeCount, agentsPerWorld);
    return clamped == 0 ? 0 : (genomeCount + clamped - 1) / clamped;
}

[[nodiscard]] constexpr std::uint32_t logicalWorldCount(const std::uint32_t genomeCount,
                                                        const std::uint32_t agentsPerWorld,
                                                        const std::uint32_t trialsPerGenome) {
    return worldGroupCount(genomeCount, agentsPerWorld) * trialsPerGenome;
}

[[nodiscard]] constexpr std::uint32_t logicalWorldForAgent(const std::uint32_t agentIndex,
                                                           const std::uint32_t agentsPerWorld,
                                                           const std::uint32_t trialsPerGenome) {
    if (agentsPerWorld == 0 || trialsPerGenome == 0) {
        return 0;
    }
    const std::uint32_t genome = agentIndex / trialsPerGenome;
    const std::uint32_t trial = agentIndex % trialsPerGenome;
    return (genome / agentsPerWorld) * trialsPerGenome + trial;
}

[[nodiscard]] constexpr std::uint32_t agentsInLogicalWorld(const std::uint32_t genomeCount,
                                                           const std::uint32_t agentsPerWorld,
                                                           const std::uint32_t trialsPerGenome,
                                                           const std::uint32_t worldIndex) {
    const std::uint32_t clamped = clampAgentsPerWorld(genomeCount, agentsPerWorld);
    if (clamped == 0 || trialsPerGenome == 0 ||
        worldIndex >= logicalWorldCount(genomeCount, clamped, trialsPerGenome)) {
        return 0;
    }
    const std::uint32_t group = worldIndex / trialsPerGenome;
    const std::uint32_t firstGenome = group * clamped;
    return std::min(clamped, genomeCount - firstGenome);
}

// One vector per four hidden neurons, so the block follows the brain preset
// instead of being resized by hand when the hidden layers change width. Every
// layer's states live end to end in here, so a plan with three layers needs no
// storage a plan with one does not: only a different division of the same block.
inline constexpr std::size_t agentHiddenVectorCount =
    (neuro::kernel::BrainHiddenNeuronCapacity + 3U) / 4U;

struct alignas(16) Float4 {
    float x{};
    float y{};
    float z{};
    float w{};
};

struct alignas(16) Int4 {
    std::int32_t x{};
    std::int32_t y{};
    std::int32_t z{};
    std::int32_t w{};
};

// std430-compatible data shared verbatim with the compute shaders.
//
// This is what the 2D record collapsed into. `pose` was x, y, angle and a
// collision radius; `motion` was a velocity, an angular velocity and a battery;
// there were eight wall-contact sectors and eight agent-contact sectors, a
// mirrored puck position, a cargo level and a beacon. None of it survives the
// move to a lattice: a cell has no fractional position, a step has no velocity,
// and "am I touching something" is answered by reading the neighbouring cells
// rather than by carrying the answer between steps.
//
// 224 bytes against 304, and the 128 of that which is the hidden block is the
// part that did not change.
struct alignas(16) AgentState {
    // Where it stands, and which way it last went. The heading is a neighbour
    // index, and LatticeNeighborCount means "has not moved yet" -- the one value
    // outside the range, so a fresh agent is distinguishable from one that
    // happens to be pointing at neighbour zero.
    Int4 cell;
    // Where it asked to go this step, and whether it was refused. Kept on the
    // record rather than recomputed because the two halves of a step run in
    // separate dispatches: the pass that decides cannot be the pass that moves,
    // since every agent has to have bid before any agent may win.
    Int4 intent;
    // The beacon of the world this agent lives in, and which world that is.
    // Mirrored onto the agent for the same reason the puck's position used to
    // be: the beacon is per-world state, and copying it onto each agent is what
    // lets fitness and the shaping see one agent at a time.
    Int4 beacon;
    // What this agent broadcasts, in .x, with three lanes spare. One channel and
    // not three: a neighbour block of 26 cells costs 26 slots per channel, and a
    // colour would put the input vector past the 127 the packed brain layout can
    // name. Three spare lanes rather than a float so the record stays a run of
    // 16-byte vectors, which is what keeps the std430 offsets below stable.
    Float4 signal;
    // best nearness reached, steps in contact, effort spent, moves refused.
    // What each is worth is a weight; see latticeTrialFitness.
    // Construction reuses .x for best relative height, .y for blocks placed
    // and .w for horizontal-perimeter ticks. Scoring interprets the shared
    // record according to worldMode.
    Float4 metrics;
    // The two output-layer recurrent cells, fed back as inputs next step, and
    // two spare lanes. Kept even though every hidden neuron now carries its own
    // time constant: the two mechanisms are different -- a time constant decides
    // how fast a neuron forgets, a recurrent cell decides what to remember --
    // and dropping one of them would make the ablation between them impossible
    // rather than merely unnecessary.
    Float4 memory;
    // Continuous-time state of every hidden neuron, carried between steps and
    // zero at the start of a generation -- which is the whole of the reset
    // semantics: an agent begins each trial remembering nothing of the last.
    // Four neurons per vector, derived from the preset rather than sized by
    // hand, and mirrored by shaders/simulation/agent_layout.glsl.
    std::array<Float4, agentHiddenVectorCount> hidden{};
};

static_assert(std::is_trivially_copyable_v<AgentState>);
static_assert(sizeof(AgentState) == 224);
static_assert(offsetof(AgentState, intent) == 16);
static_assert(offsetof(AgentState, beacon) == 32);
static_assert(offsetof(AgentState, signal) == 48);
static_assert(offsetof(AgentState, metrics) == 64);
static_assert(offsetof(AgentState, memory) == 80);
static_assert(offsetof(AgentState, hidden) == 96,
              "The hidden block goes last, so every earlier offset the shaders "
              "use is unchanged when the neuron capacity grows");

// Reading and writing one neuron's state. The shader does the same arithmetic on
// its own vec4 array; this is index maths on a different substrate rather than a
// second copy of the layout.
[[nodiscard]] inline float agentHiddenState(const AgentState& agent, const std::size_t neuron) {
    const Float4& block = agent.hidden[neuron / 4];
    switch (neuron % 4) {
    case 0:
        return block.x;
    case 1:
        return block.y;
    case 2:
        return block.z;
    default:
        return block.w;
    }
}

inline void setAgentHiddenState(AgentState& agent, const std::size_t neuron, const float value) {
    Float4& block = agent.hidden[neuron / 4];
    switch (neuron % 4) {
    case 0:
        block.x = value;
        break;
    case 1:
        block.y = value;
        break;
    case 2:
        block.z = value;
        break;
    default:
        block.w = value;
        break;
    }
}

// Shaping coefficients. Sliders rather than literals, because every fitness
// experiment would otherwise be a two-language edit plus a parity re-check.
struct FitnessWeights {
    // What closing the distance to the beacon is worth, scored once at the end
    // against the nearest the agent ever got. Shaping and not the objective: an
    // agent that reaches the beacon and is pushed off it keeps this.
    float trackingReward{1.0F};
    // Score per step spent within the contact radius. Per step rather than per
    // arrival, because a lattice cell holds one agent: "got there" is a race
    // that eleven of twelve lose whatever they did, and "stayed near" is not.
    float objectiveBonus{0.02F};
    // Charged per move actually made. Small: moving is how the task is solved,
    // and this only has to make a policy that jitters in place lose to one that
    // does not.
    float motorCostWeight{0.002F};
    // Charged per move refused, which is the whole of the pressure toward not
    // crowding. Larger than the cost of moving on purpose -- walking into a
    // neighbour is worse than walking around it.
    float refusalPenalty{0.01F};
    // What broadcasting costs, relative to moving. Signalling is free to a
    // sender otherwise, and a channel nobody pays for is one every genome
    // saturates.
    float signalCostFactor{0.25F};

    // How much of a genome's score comes from the logical world it lives in
    // rather than from itself. 0 is pure individual selection; 1 gives every
    // genome sharing a world the same score, so selection acts on the group and
    // a broadcast that only helps a neighbour finally pays its sender back.
    //
    // Scoring-time only, and deliberately absent from GpuFitnessWeights: the
    // other weights act on an agent during the step, this one acts on a
    // population at the generation boundary and has nothing to say to a shader.
    float groupSharing{0.0F};

    // Construction-only group charge for every agent-tick spent on the
    // horizontal perimeter. The ceiling is intentionally excluded: reaching
    // upward is the objective, while camping at x/z edges is not.
    // Scoring-time only; the shader merely accumulates perimeter ticks.
    float boundaryPenalty{0.002F};
};

// Seconds are still the unit the neuron time constants are expressed in, so the
// step keeps a deltaTime even though nothing moves continuously any more; see
// vkexp/simulation/Units.hpp for the rule that keeps per-step and per-second
// quantities apart.
struct SimulationStep {
    float deltaTime{units::fixedTimeStep}; // s

    // The box. Sliders, because what the plan leaves open is exactly how much
    // room the neighbourhood work needs -- and because the answer is different
    // for a chain than for a formation.
    std::uint32_t latticeWidth{latticeDefaultWidth};
    std::uint32_t latticeHeight{latticeDefaultHeight};
    std::uint32_t latticeDepth{latticeDefaultDepth};

    // How sure a drive has to be before it becomes a step. At zero an agent
    // moves every step whatever it thinks; near one it has to commit. This is
    // the whole of the decision to stand still, which is why it is a parameter.
    float moveThreshold{lattice::kernel::LatticeMoveThresholdDefault};

    // How near the beacon counts as reached. One rather than zero, so a group
    // can crowd a beacon that only one of them can stand on.
    std::uint32_t beaconContactRadius{1};

    // Where the beacon goes. Per trial, from this seed and the world index, so
    // a genome is scored on several placements rather than on one it could
    // memorise.
    std::uint32_t beaconSeed{0x5EEDU};

    WorldMode worldMode{WorldMode::Beacon};

    // A successful placement starts this many ticks of cooldown. Direct
    // support from below is always valid; cardinal side support is an opt-in
    // construction experiment below.
    std::uint32_t buildIntervalTicks{12};
    float buildThreshold{0.55F};
    // A course advances the shared foundation height after this fraction of
    // its x/z area is occupied. Courses count consecutively from the floor, so
    // sparse blocks high above an unfinished course cannot raise the frontier.
    float constructionCourseFill{0.25F};
    // The highest legal target is this many courses above that foundation.
    std::uint32_t constructionHeightLead{5};
    // Opt-in cantilevers: a block may use a cardinal x/z face as support. Edge
    // and corner contact remain insufficient.
    std::uint32_t allowSideSupportedBlocks{};

    Neighborhood neighborhood{Neighborhood::Moore};

    // The hidden layers to run, widest question first: how many, and how wide.
    // All three zero means the default plan. Layers are dense from the front; a
    // hole is refused rather than closed up, because {20, 0, 8} could mean two
    // readings and guessing between them is worse than saying no.
    //
    // Only the hidden layers, deliberately. The two ends are the world's own
    // business: how many sensors a lattice offers and how many actuators it
    // needs are statements about the world, not about how much brain to spend.
    std::array<std::uint32_t, neuro::kernel::BrainHiddenLayerCapacity> hiddenLayers{};
    FitnessWeights fitness{};
    // Where a hidden neuron's time constant comes from. Reactive pins it to
    // deltaTime, which makes the update y = activation and reproduces the
    // memoryless network exactly, so every model is the same code path with one
    // parameter changed rather than a separate network.
    NeuronModel neuronModel{NeuronModel::TimeConstant};
};

[[nodiscard]] constexpr std::uint32_t latticeCellsPerWorld(const SimulationStep& settings) {
    return lattice::kernel::latticeCellCount(settings.latticeWidth, settings.latticeHeight,
                                             settings.latticeDepth);
}

// How many agents one world can actually be given a cell of its own.
//
// Placement is not free to use every cell. A beacon world never stands anybody
// on the beacon -- an agent that starts on the objective has solved the world
// before the first step -- so it is one cell short of its own size. A
// construction world starts everybody on the floor, so it is one course rather
// than a volume. Asking for more than this does not fail: the probe in
// makeInitialAgents simply runs out of candidates and the surplus agents keep
// their default corner, standing inside each other, which breaks the one
// invariant the whole arbitration rests on. So the number is named here and
// clamped once, where the settings are known.
[[nodiscard]] constexpr std::uint32_t latticeSpawnCapacity(const SimulationStep& settings) {
    if (settings.worldMode == WorldMode::Construction) {
        return std::max(settings.latticeWidth * settings.latticeDepth, 1U);
    }
    return std::max(latticeCellsPerWorld(settings), 2U) - 1U;
}

[[nodiscard]] constexpr std::uint32_t latticeMaximumDistance(const SimulationStep& settings) {
    return lattice::kernel::latticeMaximumDistance(
        static_cast<std::uint32_t>(settings.neighborhood), settings.latticeWidth,
        settings.latticeHeight, settings.latticeDepth);
}

// A lattice big enough to hold every agent a world can be given, and inside the
// extent the shader's addressing allows. Clamping here rather than at each
// caller is what keeps a slider from producing a configuration that allocates.
[[nodiscard]] constexpr std::uint32_t clampLatticeExtent(const std::uint32_t extent) {
    return std::clamp(extent, latticeMinimumExtent, latticeMaximumExtent);
}

// std430 mirror of FitnessWeights, minus the scoring-time one.
struct alignas(16) GpuFitnessWeights {
    float trackingReward{};
    float objectiveBonus{};
    float motorCostWeight{};
    float refusalPenalty{};
    float signalCostFactor{};
    float reserved0{};
    float reserved1{};
    float reserved2{};
};

static_assert(sizeof(GpuFitnessWeights) == 32);

// std430-compatible per-step parameters, in a storage buffer indexed by step
// rather than in push constants: one batched upload per frame costs less than
// one vkCmdPushConstants per step, and the three passes of a step share it.
struct alignas(16) GpuStepParameters {
    float deltaTime{};
    float moveThreshold{};
    std::uint32_t agentCount{};
    std::uint32_t brainLayout{}; // packed active input and output counts
    std::uint32_t trialsPerGenome{};
    std::uint32_t agentsPerWorld{};
    std::uint32_t worldCount{};
    std::uint32_t latticeWidth{};
    std::uint32_t latticeHeight{};
    std::uint32_t latticeDepth{};
    std::uint32_t cellsPerWorld{};
    std::uint32_t neighborhood{};
    // Precomputed on the host: it is a function of the extents and the
    // neighbourhood, and every invocation of every pass would otherwise derive
    // it again from the same four numbers.
    std::uint32_t maximumDistance{};
    std::uint32_t beaconContactRadius{};
    std::uint32_t neuronModel{};
    // The three hidden layer widths, six bits each, and the genome stride. The
    // stride travels as its own uint: three layers of the neuron capacity put it
    // past the twelve bits it used to share with the layout word, and a stride
    // that wrapped would address another genome's weights and still produce
    // numbers.
    std::uint32_t brainHiddenLayers{};
    std::uint32_t brainGenomeStride{};
    std::uint32_t worldMode{};
    std::uint32_t buildIntervalTicks{};
    float buildThreshold{};
    float constructionCourseFill{};
    std::uint32_t constructionHeightLead{};
    std::uint32_t allowSideSupportedBlocks{};
    GpuFitnessWeights fitness;
};

static_assert(sizeof(GpuStepParameters) == 128);
static_assert(offsetof(GpuStepParameters, latticeWidth) == 28);
// The one offset the GLSL mirror cannot derive for itself. Twenty-three scalars
// come to 92 bytes and `alignas(16)` pushes this block to 96; the shader only
// lands on the same number because its copy of the block is declared as vec4s,
// which std430 aligns to 16 as well. Eight floats there would align to 4, and
// the two strides would differ by one word -- which is invisible at step index
// zero and total nonsense at every index after it.
static_assert(offsetof(GpuStepParameters, fitness) == 96);
static_assert(offsetof(GpuStepParameters, neuronModel) == 56);
static_assert(offsetof(GpuStepParameters, fitness) == 96);

// The network this run actually builds. The two ends are the lattice's own: how
// many cells surround one, and how many drives a move needs. Only the hidden
// plan is a setting, and all-zero means the default one -- so a run keeps the
// brain it was tuned with unless someone says otherwise.
[[nodiscard]] inline neuro::BrainShape resolvedBrain(const SimulationStep& settings) {
    neuro::BrainShape shape = neuro::defaultBrainShape;
    if (settings.hiddenLayers[0] != 0) {
        shape.hiddenCount = settings.hiddenLayers[0];
        shape.secondHiddenCount = settings.hiddenLayers[1];
        shape.thirdHiddenCount = settings.hiddenLayers[2];
    }
    return shape;
}

[[nodiscard]] constexpr GpuFitnessWeights packFitnessWeights(const FitnessWeights& weights) {
    return {weights.trackingReward,
            weights.objectiveBonus,
            weights.motorCostWeight,
            weights.refusalPenalty,
            weights.signalCostFactor,
            0.0F,
            0.0F,
            0.0F};
}

} // namespace vkexp
