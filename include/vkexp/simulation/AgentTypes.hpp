#pragma once

#include "vkexp/neuro/BrainKernel.hpp"
#include "vkexp/simulation/PuckKernel.hpp"
#include "vkexp/simulation/TrailKernel.hpp"
#include "vkexp/simulation/Units.hpp"
#include "vkexp/worlds/ScenarioKernel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vkexp {

// How a hidden neuron decides its time constant. One integrator, three sources
// for the rate it runs at; declared in BrainKernel.inl so the shader gets the
// same numbers. See there for what each one is and why Gated contains the other
// two as special cases.
enum class NeuronModel : std::uint32_t {
    Reactive = neuro::kernel::NeuronModelReactive,
    TimeConstant = neuro::kernel::NeuronModelTimeConstant,
    Gated = neuro::kernel::NeuronModelGated,
};

inline constexpr std::size_t neuronModelCount = neuro::kernel::NeuronModelCount;

enum class WorldShape : std::uint32_t {
    Circle = 0,
    Square = 1,
};

enum class WorldSize : std::uint32_t {
    Small = 0,
    Medium = 1,
    Large = 2,
};

enum class BeaconScenario : std::uint32_t {
    Stationary = 0,
    AlternatingDiagonals = 1,
    Rotating = 2,
    RandomMovement = 3,
    ForageHome = 4,
    ScentRelay = 5,
    TwoDoors = 6,
    Shuttle = 7,
    TwoGaps = 8,
    PuckPush = 9,
    GatePlate = 10,
};

inline constexpr std::size_t beaconScenarioCount = 11;

// Body radius in metres: a 4.4 cm disc, roughly an e-puck-class table robot.
// Stored per agent in `pose.w`, so a scenario may vary it; this is the spawn
// value and the scale every other length is chosen against.
//
// Also declared as ScenarioAgentBodyRadius in ScenarioKernel.inl, because
// scenario geometry has to be sizeable by the agent and the shared kernel
// cannot reach in here. It cannot be defined once and referenced: the kernel's
// constants are plain `const float`, which GLSL needs and which C++ will not
// accept as a constant expression. testTwoDoorsGeometry asserts the two agree.
inline constexpr float agentBodyRadius = 0.022F;

inline constexpr float agentBodyDiameter = agentBodyRadius * 2.0F;

// Trail resolution is chosen in units of the body that leaves the trail. One
// body diameter per cell is the coarsest useful setting -- a track cannot then
// be drawn narrower than the thing that made it -- and a fifth of a body is a
// thin line. Both ends are far inside the correctness bound: the outer antenna
// tips are 15.5 cm apart, three and a half cells even at the coarsest.
//
// What actually limits the fine end is bandwidth, not capacity: the decay pass
// touches every value of every world on every step, so the field is read and
// written in full sixty times a second, and halving the cell size quadruples
// that. The field also scales with the world count, which grows as the group
// size shrinks -- 512 genomes at 12 agents per world is 172 fields, not one.
//
// So the budget is set by what can be streamed twice per step, not by what fits:
// 256 MiB of field is half a gigabyte of traffic per step, which is a few
// milliseconds. It is additionally capped against the device's own memory, since
// a headless CI GPU may have far less than a desktop one.
// The resolutions on offer, in body diameters per cell. Shared with the UI so a
// clamped choice always lands on a setting the menu can name: coarsening by
// doubling would leave the label saying one thing and the field being another.
inline constexpr std::array<float, 5> trailCellFractions{0.2F, 0.25F, 1.0F / 3.0F, 0.5F, 1.0F};
inline constexpr float trailCellFractionFinest = trailCellFractions.front();
inline constexpr float trailCellFractionCoarsest = trailCellFractions.back();
inline constexpr std::uint64_t trailFieldByteBudget = 256ULL * 1024ULL * 1024ULL;
inline constexpr std::uint64_t trailFieldHeapFraction = 16; // at most a sixteenth of VRAM

[[nodiscard]] constexpr float trailCellSizeForBodyFraction(const float fraction) {
    return agentBodyDiameter * fraction;
}

// Arena radius in metres: the small world is 3.7 m across, which puts a 4.4 cm
// body roughly 84 body-lengths from the far wall.
inline constexpr float smallWorldRadius = 1.84F;
// Single source of truth shared with the vertex shader.
inline const float beaconVisualRadius = worlds::kernel::BeaconVisualRadius;
inline constexpr std::uint32_t minimumAgentsPerWorld = 10;

[[nodiscard]] constexpr float worldRadiusForSize(const WorldSize size) {
    switch (size) {
    case WorldSize::Small:
        return smallWorldRadius;
    case WorldSize::Medium:
        return smallWorldRadius * 1.5F;
    case WorldSize::Large:
        return smallWorldRadius * 3.0F;
    }
    return smallWorldRadius;
}

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
// instead of being resized by hand when the hidden layer changes width.
inline constexpr std::size_t agentHiddenVectorCount =
    (neuro::kernel::BrainHiddenCapacity + 3U) / 4U;

struct alignas(16) Float4 {
    float x{};
    float y{};
    float z{};
    float w{};
};

// std430-compatible data shared verbatim with compute and vertex shaders.
struct alignas(16) AgentState {
    Float4 pose;        // position.xy, angle, circular collision radius
    Float4 motion;      // velocity.xy, angular velocity, normalized energy
    Float4 signal;      // emitted RGB and intensity
    // Base beacon.xy, trial id, completed mask or forage-cycle count. In the
    // puck world .xy is the puck's velocity instead, mirrored beside its
    // position below: nothing there reads a base beacon, and the work reward
    // needs the puck's motion to measure an approach against it.
    Float4 target;
    Float4 metrics;     // phase start/min distance, motor cost, completed-phase progress
    // .x penalty total, .yz the world's puck position mirrored onto the agent,
    // .w logical world id. The mirror exists because fitness and the progress
    // shaping only ever see one agent: the puck is shared state on the device,
    // and copying its position onto each agent in its world is what lets the
    // ordinary machinery -- best-approach shaping, objective counting -- score a
    // joint outcome without a second mechanism beside it.
    Float4 penalties;
    Float4 internal;    // cargo level, seeking-home flag, and two recurrent memory cells
    Float4 wallTouch0;  // wall contact sectors 0..3
    Float4 wallTouch1;  // wall contact sectors 4..7
    Float4 agentTouch0; // agent contact sectors 0..3
    Float4 agentTouch1; // agent contact sectors 4..7
    // Continuous-time state of every hidden neuron, carried between steps and
    // zero at the start of a generation -- which is the whole of the reset
    // semantics: an agent begins each trial remembering nothing of the last.
    // Four neurons per vector, derived from the preset rather than sized by
    // hand, and mirrored by shaders/simulation/agent_layout.glsl.
    std::array<Float4, agentHiddenVectorCount> hidden{};
};

static_assert(std::is_trivially_copyable_v<AgentState>);
static_assert(sizeof(AgentState) == 256);
static_assert(offsetof(AgentState, metrics) == 64);
static_assert(offsetof(AgentState, penalties) == 80);
static_assert(offsetof(AgentState, internal) == 96);
static_assert(offsetof(AgentState, hidden) == 176,
              "The hidden block goes last, so every earlier offset the shaders "
              "and the renderer use is unchanged");

// One puck per logical world -- shared state several agents act on at once, and
// the first thing in this simulation they can change rather than only read.
//
// pose:   x, y, radius, how far from the middle it was placed
// motion: vx, vy, the highest rung reached so far, unused
//
// The rung is latched into the record rather than recomputed at the end because
// it is a milestone: "the puck got at least this far" is what the run is
// measuring, and a puck nudged toward the middle and back out again still got
// there. It is stored as a float beside the velocity for the same reason the
// agent's cargo flag is: this is a std430 block the shaders read, and a float
// keeps the four-lane layout the rest of the file uses.
//
// The starting distance travels with the puck for the same reason: the rungs are
// equal fractions of the journey from where it was placed to the target disc, so
// the reported level has to know where the journey began.
struct PuckState {
    Float4 pose{};
    Float4 motion{};
};

static_assert(std::is_trivially_copyable_v<PuckState>);
static_assert(sizeof(PuckState) == 32);
static_assert(offsetof(PuckState, motion) == 16);

[[nodiscard]] inline std::uint32_t puckLevel(const PuckState& puck) {
    return static_cast<std::uint32_t>(std::max(puck.motion.z, 0.0F) + 0.5F);
}

// Reading and writing one neuron's state. The shader does the same arithmetic on
// its own vec4 array; this is index maths on a different substrate rather than a
// second copy of the layout, the same way the tactile sectors are addressed.
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
static_assert(offsetof(AgentState, wallTouch0) == 112);

// Shaping and energy coefficients. They used to be literals split between
// CpuSimulation.cpp and agent_step.comp, which made every fitness experiment a
// two-language edit plus a parity re-check; as parameters they are sliders.
struct FitnessWeights {
    float objectiveBonus{2.0F};    // score per completed objective
    float motorCostWeight{0.002F}; // fitness charged per unit of motor and signal effort
    float signalCostFactor{0.25F}; // cost of emitting, relative to moving
    float energyDrain{0.0008F};    // battery drained per unit of effort
    float trackingReward{0.25F};   // shaping for scenarios whose beacon keeps moving

    // How much of a genome's score comes from the logical world it lives in
    // rather than from itself. 0 is pure individual selection, the behaviour
    // this project has always had; 1 gives every genome sharing a world the
    // same score, so selection acts on the group and a signal that only helps
    // a neighbour finally pays its sender back.
    //
    // Scoring-time only, and deliberately absent from GpuFitnessWeights: the
    // other weights act on an agent during the step, this one acts on a
    // population at the generation boundary and has nothing to say to a shader.
    float groupSharing{0.0F};
};

// Units are metres, seconds and radians throughout; see vkexp/simulation/Units.hpp
// for why, and for the rule that keeps per-step and per-second quantities apart.
struct SimulationStep {
    float deltaTime{units::fixedTimeStep}; // s
    float worldRadius{smallWorldRadius};   // m
    float thrust{1.9F};                    // m/s^2 at full forward drive
    float turnAcceleration{5.0F};          // rad/s^2 at full differential drive
    float linearDrag{1.7F};                // 1/s, applied as exp(-drag * dt)
    float angularDrag{2.4F};               // 1/s, same form
    float sensorFieldOfView{1.8F};         // rad, total arc spanned by the receptors
    float arrivalRadiusMultiplier{1.0F};   // dimensionless
    float maximumSpeed{0.55F};             // m/s
    float maximumAngularSpeed{3.0F};       // rad/s
    float lightSensorRange{2.4F};          // m, derived from lightRangeRatio
    // Light range as a fraction of the arena radius. Fixing it in metres made a
    // bigger world a blind-search task stacked on top of the intended one: at
    // 2.4 m the beacon is invisible from most of the large arena, so there is no
    // gradient to follow for most of a trial. Scaling it keeps perception
    // proportional, so world size changes how far things are, not whether they
    // can be seen at all. 2.4 m small, 3.6 m medium, 7.2 m large.
    float lightRangeRatio{2.4F / smallWorldRadius};
    float lightExposure{1.25F};        // dimensionless tone-mapping gain
    float collisionRestitution{0.35F}; // dimensionless
    // 1/s. Overlap is resolved as 1 - exp(-rate * dt), the same exponential form
    // the drags use, so contact resolution no longer depends on the step rate.
    // 113.8/s reproduces the old fixed 0.85-per-step at 60 Hz to within 0.03%.
    float contactStiffness{113.8F};
    // Fitness charged per second of full-strength wall contact. This used to be
    // charged per step, which quietly made deltaTime a fitness parameter.
    float wallCollisionPenalty{0.6F};
    float beaconAngularSpeed{0.35F};        // rad/s
    float beaconRotationAngle{};            // rad
    float beaconRadiusRatio{0.72F};         // fraction of worldRadius
    float beaconMotionTime{};               // s
    float beaconTeleportProbability{0.25F}; // dimensionless, per teleport epoch
    float beaconRandomSpeed{0.18F};         // m/s
    float forageCargoDecayRate{0.08F};      // cargo fraction lost per second
    float foragePickupReward{0.25F};        // fitness per pickup event
    float forageDeliveryReward{4.0F};       // fitness per unit of cargo delivered
    // Puck world: the radius of the disc in the middle that counts as delivered,
    // as a fraction of the arena radius. A quarter is a target a group can hit
    // and one agent shoving blindly cannot, which is the difficulty the world is
    // for; it is a slider because where that boundary sits is the experiment.
    float puckTargetRadiusRatio{0.25F};
    // How big the puck is, as a fraction of the arena radius. Bigger is easier
    // in two ways at once -- more contact arc for a group to share, and a larger
    // thing to find -- so it is the first knob to reach for when the world is
    // not being learned at all.
    float puckRadiusRatio{puck::kernel::PuckRadiusRatio};
    // Gate world: how long the gate keeps running after the plate is released.
    // This is the difficulty of the world in one number. Above zero one agent
    // presses and runs, and nothing has to be shared; at zero the gate shuts the
    // instant the plate is let go, only the far side scores, and somebody has to
    // stay behind for nothing -- which is the condition group fitness sharing
    // exists for, reached by moving a slider rather than by adding a scenario.
    float gateLatchSeconds{4.0F}; // s
    // Trail field. The deposit is per second and the lifetime is a half-life in
    // seconds, so neither becomes a function of the step rate.
    // Deposit rates come from what a single pass has to leave behind, not from a
    // round number: at 0.55 m/s an agent is over a 6 cm cell for about six steps,
    // so 1.0/s would leave a 1.6% mark -- invisible on screen and near-nothing to
    // the antennae. These leave a pass at roughly a third of full scale, while
    // standing still still saturates the cell.
    float trailDepositRate{4.0F};        // agent mark laid per second
    float trailHalfLife{6.0F};           // s for a mark to fade to half
    float beaconTrailDepositRate{12.0F}; // beacons mark harder than agents do
    // How much of its cell a mark fills when drawn. Display only: the antennae
    // read whole cells either way, and the deposit is already the narrowest it
    // can be at one cell. The default is the body diameter over the cell size, so
    // a track is as wide as whatever left it; 1.0 fills the cell, which reads
    // better on a large arena where a cell is only a few pixels across.
    // Filling the cell is now the right default: a cell is at most a body
    // diameter, so a full cell is never wider than whatever left the mark.
    float trailRenderWidth{1.0F};
    float trailCellSize{trailCellSizeForBodyFraction(trailCellFractionCoarsest)};
    bool trailEnabled{true};
    FitnessWeights fitness{};
    std::uint32_t beaconMotionSeed{};
    WorldShape worldShape{WorldShape::Circle};
    WorldSize worldSize{WorldSize::Small};
    BeaconScenario beaconScenario{BeaconScenario::Stationary};
    std::uint32_t beaconPhase{};
    bool beaconPhaseChanged{};
    bool agentCollisionsEnabled{true};
    bool agentLightEnabled{true};
    // Two gaps only: trade the resource and home ends every other generation, so
    // a genome cannot bake in "carry north, deliver south" and must read which
    // colour it is heading for. Off by default, because it is the harder task
    // and a run that has not learned the easy one first says nothing.
    bool swapDeliveryEnds{false};
    // Ablation: give every beacon the average of the two beacon colours, so the
    // ends stay lit and stop being told apart by hue. The control for "the agents
    // read the colour" -- a policy that does not read it scores the same with
    // this on, and a policy that does collapses.
    bool uniformBeaconColor{false};
    // Two doors only: run the dead end off the generation instead of the trial,
    // so a whole population trains on one door and its successors on the other.
    // Cleaner selection inside a generation, at the risk of the population
    // thrashing between the two; see twoDoorsBlockedDoor for the trade.
    bool blockedDoorPerGeneration{false};
    // Where a hidden neuron's time constant comes from. Reactive pins it to
    // deltaTime, which makes the update y = activation and reproduces the
    // memoryless network exactly, so every model is the same code path with one
    // parameter changed rather than a separate network.
    NeuronModel neuronModel{NeuronModel::TimeConstant};
};

// Cells across the arena's bounding square. Constant in metres, so world size
// changes how much ground there is, not how finely it is smelled.
[[nodiscard]] inline std::uint32_t trailWidthForWorld(const float worldRadius,
                                                      const float cellSize) {
    return static_cast<std::uint32_t>(std::ceil((worldRadius * 2.0F) / cellSize));
}

[[nodiscard]] inline float lightRangeForWorld(const SimulationStep& settings) {
    return settings.worldRadius * settings.lightRangeRatio;
}

// The neighbour grid scales with the arena for the same reason: at a fixed cell
// size the large world holds 8464 cells against the small world's 961 at the
// same agent density, so the per-agent sweep would grow with world size for no
// gain in resolution. Scaling keeps the grid 31x31 in every world.
[[nodiscard]] inline float gridCellSizeForWorld(const float baseCellSize, const float worldRadius) {
    return baseCellSize * (worldRadius / smallWorldRadius);
}

[[nodiscard]] inline float beaconArrivalRadius(const SimulationStep& settings) {
    return beaconVisualRadius * settings.arrivalRadiusMultiplier;
}

// Opaque per-scenario transport. Each scenario packs and unpacks this block
// itself on both sides, so adding a scenario never widens the shared structs.
// Mirrors `ScenarioParameters` in shaders/worlds/scenario_params.glsl.
struct alignas(16) ScenarioParameterBlock {
    Float4 floats0;
    Float4 floats1;
    std::array<std::uint32_t, 4> integers{};
};

static_assert(sizeof(ScenarioParameterBlock) == 48);
static_assert(offsetof(ScenarioParameterBlock, integers) == 32);

// std430 mirror of FitnessWeights; padded so the scenario block stays aligned.
struct alignas(16) GpuFitnessWeights {
    float objectiveBonus{};
    float motorCostWeight{};
    float signalCostFactor{};
    float energyDrain{};
    float trackingReward{};
    float reserved0{};
    float reserved1{};
    float reserved2{};
};

static_assert(sizeof(GpuFitnessWeights) == 32);

// std430-compatible per-step parameters. This lives in a storage buffer rather
// than push constants: the scenario block already pushes the structure past the
// 128-byte size Vulkan guarantees for maxPushConstantsSize, and one batched
// upload per frame costs less than one vkCmdPushConstants per step.
struct alignas(16) GpuStepParameters {
    float deltaTime{};
    float worldRadius{};
    float thrust{};
    float turnAcceleration{};
    float linearDrag{};
    float angularDrag{};
    float sensorFieldOfView{};
    float arrivalRadius{};
    float maximumSpeed{};
    float maximumAngularSpeed{};
    float lightSensorRange{};
    float lightExposure{};
    float collisionRestitution{};
    float contactStiffness{};
    float gridCellSize{};
    float wallCollisionPenalty{};
    std::uint32_t agentCount{};
    std::uint32_t brainLayout{}; // packed genome stride and active input/hidden/output counts
    std::uint32_t trialsPerGenome{};
    std::uint32_t worldShape{};
    std::uint32_t gridWidth{};
    std::uint32_t gridCellsPerWorld{};
    std::uint32_t agentCollisionsEnabled{};
    std::uint32_t agentLightEnabled{};
    std::uint32_t beaconScenario{};
    std::uint32_t beaconPhase{};
    std::uint32_t beaconPhaseChanged{};
    std::uint32_t beaconCount{};
    float trailCellSize{};
    float trailSurvival{}; // per-step factor, already resolved from the half-life
    float trailDeposit{};  // fixed-point units an agent adds this step
    float beaconTrailDeposit{};
    std::uint32_t trailWidth{};
    std::uint32_t trailCellsPerWorld{};
    std::uint32_t trailEnabled{};
    std::uint32_t agentsPerWorld{}; // lets one agent per world deposit the beacon
    GpuFitnessWeights fitness;
    ScenarioParameterBlock scenario;
    // Appended rather than slotted in beside the other flags: every offset above
    // is asserted and mirrored by the shader's struct, and moving one to save
    // twelve bytes of padding would be paid for in a silent misread.
    std::uint32_t neuronModel{};
    std::uint32_t obstacleCount{};
    std::uint32_t uniformBeaconColor{};
    // The puck pass runs one thread per logical world, so unlike every other
    // pass it has to be told how many there are.
    std::uint32_t worldCount{};
    float puckTargetRadiusRatio{};
    std::uint32_t puckEnabled{};
    std::uint32_t reserved3{};
    std::uint32_t reserved4{};
};

static_assert(sizeof(GpuStepParameters) == 256);
static_assert(offsetof(GpuStepParameters, agentCount) == 64);
static_assert(offsetof(GpuStepParameters, beaconScenario) == 96);
static_assert(offsetof(GpuStepParameters, trailCellSize) == 112);
static_assert(offsetof(GpuStepParameters, fitness) == 144);
static_assert(offsetof(GpuStepParameters, scenario) == 176);
static_assert(offsetof(GpuStepParameters, neuronModel) == 224);

[[nodiscard]] constexpr GpuFitnessWeights packFitnessWeights(const FitnessWeights& weights) {
    return {weights.objectiveBonus,
            weights.motorCostWeight,
            weights.signalCostFactor,
            weights.energyDrain,
            weights.trackingReward,
            0.0F,
            0.0F,
            0.0F};
}

} // namespace vkexp
