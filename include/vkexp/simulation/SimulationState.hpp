#pragma once

#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/simulation/ExperimentSweep.hpp"
#include "vkexp/simulation/LatticeTypes.hpp"
#include "vkexp/simulation/StructureShape.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace vkexp {

struct SimulationControls {
    bool paused{};
    bool resetRequested{};
    std::uint32_t stepsPerFrame{4};
    std::uint32_t stepsPerGeneration{900};

    // Snapshot requests, handled the same way as resetRequested: the UI raises a
    // flag and SimulationModule acts on it between frames, where the device can
    // be made idle. `snapshotStatus` is what the last attempt did, shown back in
    // the panel so a failed load is visible rather than silent.
    std::string snapshotPath{"run.vklr"};
    std::string snapshotStatus;
    bool saveRequested{};
    bool loadRequested{};

    // A genome archive carries weights and nothing else, which is exactly what
    // replaying a champion needs: the world is whatever is set up here, and the
    // brain is the one that was trained elsewhere.
    std::string genomePath{"champion.vkng"};
    bool loadGenomesRequested{};
    bool saveGenomesRequested{};
    // Whether that save is the whole population or the champion alone.
    bool saveWholePopulation{};
    // Writes what the weights *mean* rather than the weights: which slot of the
    // input vector is which sensor, which span of the genome is which weight
    // block. Goes next to the archive with a .json extension.
    bool saveBrainStructureRequested{};

    // The hidden-layer plan being edited in the Brain window, before it is
    // applied. Kept beside the other controls rather than in the settings block
    // because it is a draft: what the run is actually using lives in
    // SimulationStep, and these two differing is exactly what "not applied yet"
    // means. All zero means "not started editing".
    std::array<int, 3> hiddenLayerDraft{};

    // Watching rather than training. The generation is still scored and
    // reported -- that is how loaded weights get judged -- but nothing is
    // selected or mutated, so the same population respawns and the run repeats
    // instead of drifting away from the weights that were loaded.
    bool replay{};

    // Raised by the UI and acted on between frames, like the flags above.
    bool sweepStartRequested{};
    bool sweepStopRequested{};
};

struct SimulationStatistics {
    std::uint64_t generation{};
    // Generations actually evaluated since the run started. Not the same as
    // `generation`, which does not advance under replay, and emphatically not
    // the same as the length of the history below, which stops at
    // maximumSamples and would otherwise report a run as having frozen the
    // moment the plots filled up.
    std::uint64_t evaluatedGenerations{};
    std::uint32_t step{};
    float bestFitness{};
    float meanFitness{};
    float medianFitness{};
    // Beacon mode: fraction of agents that reached the contact radius.
    // Construction mode: mean height-weighted block fill across worlds,
    // normalised by the theoretical full lattice. The historic name remains
    // part of CSV/sweep storage.
    float arrivalRatio{};
    // Construction mode: what each world had built when the last generation
    // ended, and which world earned the top score. Measured and reported, never
    // scored -- the reason is in StructureShape.hpp. Empty in beacon mode, and
    // empty until the first generation finishes.
    std::vector<StructureShape> worldShapes;
    std::uint32_t bestWorld{};
};

struct EvolutionHistory {
    std::vector<float> bestFitness;
    std::vector<float> medianFitness;
    std::vector<float> meanFitness;
    std::vector<float> arrivalRatio;
    // The plots keep a window, not the whole run. Anything reading a length
    // here is reading how much is plotted; how far the run has got is
    // SimulationStatistics::evaluatedGenerations.
    std::size_t maximumSamples{256};
};

// Solid voxels hide each other; see-through ones let a box be read from
// outside. Which is wanted depends on how crowded the world is, so it is a
// setting rather than a decision. See LatticeRenderer for why the transparent
// path resolves without sorting anything.
enum class VoxelStyle : std::uint32_t {
    Solid = 0,
    Transparent = 1,
};

enum class CameraProjection : std::uint32_t {
    Perspective = 0,
    Orthographic = 1,
};

// Where the camera is, in the lattice's own terms. An orbit rather than a free
// camera: the thing being looked at is a box with a known centre, and every
// control that cannot lose the box is one fewer way to end up staring at
// nothing.
// The vertical angle the view covers, in radians. Declared here rather than in
// the renderer because the panel needs it too: how far one dragged pixel moves
// the box is a question about the projection, and two places computing it from
// two constants is two places that drift.
inline constexpr float latticeCameraFieldOfView = 0.87F;

struct LatticeCamera {
    // Start exactly side-on. The default box is wider on x, so the eye begins
    // on +z and sees that widest horizontal edge across the viewport.
    float yaw{};   // radians around the up axis
    float pitch{}; // radians above the horizon, clamped short of the poles
    // Multiples of the box's half-diagonal, so the default frames any lattice
    // rather than the one it was tuned on.
    float distance{2.3F};
    CameraProjection projection{CameraProjection::Perspective};
    bool spin{};
    float spinRate{0.15F}; // radians per second while spinning

    // What the camera looks at, in cells from the centre of the box. An orbit
    // alone keeps this at zero and can only ever study the middle of a lattice;
    // with it, a corner of a 32x32x16 box can be brought to the middle of the
    // viewport and examined at a distance that would otherwise frame the whole
    // thing.
    float targetX{};
    float targetY{};
    float targetZ{};

    // Slide that point across the screen. The arguments are world units along
    // the camera's own right and up axes, so a caller converts pixels to world
    // units once and never has to know how yaw and pitch become a basis --
    // which is written here, once.
    void slide(const float right, const float up) {
        const float cosPitch = std::cos(pitch);
        const float sinPitch = std::sin(pitch);
        const float cosYaw = std::cos(yaw);
        const float sinYaw = std::sin(yaw);
        // right = (cos yaw, 0, -sin yaw); up = (-sin pitch sin yaw, cos pitch,
        // -sin pitch cos yaw). Both fall out of the eye direction and world up,
        // and both are already unit length, so there is nothing to normalise.
        targetX += right * cosYaw - up * sinPitch * sinYaw;
        targetY += up * cosPitch;
        targetZ += -right * sinYaw - up * sinPitch * cosYaw;
    }
};

// Face the broader horizontal side: look along the shorter of x/z so the
// longer one spans the picture. Height remains vertical and pitch stays zero.
// Kept next to the camera state so startup and the Reset view button cannot
// quietly acquire different definitions of "home".
[[nodiscard]] constexpr LatticeCamera latticeHomeCamera(const SimulationStep& settings) {
    LatticeCamera camera{};
    constexpr float halfPi = 1.570796327F;
    camera.yaw = settings.latticeWidth >= settings.latticeDepth ? 0.0F : halfPi;
    return camera;
}

// What the viewport draws. None of this reaches the simulation -- turning the
// agents off changes the picture, not the run.
struct SimulationDisplay {
    bool agents{true};
    bool beacons{true};
    bool structures{true};
    bool trails{true};
    // The lattice as a wireframe box, so a sparse world still reads as a volume
    // rather than as points floating in nothing.
    bool bounds{true};
    float backgroundBrightness{1.0F};

    VoxelStyle voxelStyle{VoxelStyle::Solid};
    // How much of its cell a voxel fills. Below 1 the lattice reads as a grid of
    // separate bodies; at 1 a pair of neighbours is one block.
    float voxelScale{0.78F};
    // Only read in the transparent style.
    float voxelOpacity{0.34F};
    // A breadcrumb history rather than simulation state: it is written after
    // movement and never read by an agent. Each agent has the same fixed ring;
    // this chooses how much of its newest end is drawn.
    std::uint32_t trailLength{160};
    float trailOpacity{0.20F};
    float trailScale{0.32F};

    // A slab of the lattice, so the inside of a box can be seen without making
    // everything see-through. The axis is 0, 1 or 2; the bounds are in cells and
    // are clamped to the lattice when the extents change.
    std::uint32_t sliceAxis{2};
    std::uint32_t sliceLow{};
    std::uint32_t sliceHigh{latticeMaximumExtent};

    LatticeCamera camera;
};

struct SimulationWorlds {
    std::uint32_t requestedAgentsPerWorld{12};
    std::uint32_t agentsPerWorld{12};
    std::uint32_t groupCount{};
    std::uint32_t worldCount{};
    std::uint32_t selectedWorld{};
};

struct AgentBufferView {
    std::array<VkBuffer, 2> buffers{};
    VkDeviceSize size{};
    std::uint32_t currentIndex{};
    std::uint32_t agentCount{};
    std::uint32_t genomeCount{};
    std::uint32_t trialsPerGenome{};
    std::uint64_t generation{};

    [[nodiscard]] VkBuffer currentBuffer() const { return buffers[currentIndex]; }
};

// The occupancy grid, published so a view can draw the lattice it never writes.
// Sized once for the budget and never reallocated, so the handle never goes
// stale; `cellsPerWorld` says how much of it the running lattice actually uses.
struct LatticeBufferView {
    VkBuffer buffer{};
    VkDeviceSize size{};
    std::uint32_t cellsPerWorld{};
};

// Persistent blocks built during the current generation. Zero is empty and a
// positive value identifies the builder, which lets the view add subtle colour
// variation without turning ownership into simulation behaviour.
struct StructureBufferView {
    VkBuffer buffer{};
    VkDeviceSize size{};
};

inline constexpr std::uint32_t trailHistoryCapacity = 256;

// One ivec4 position per agent per recorded tick, arranged as fixed-size rings.
// The fourth lane is a validity flag, so a restored mid-generation snapshot can
// start with an empty visual history rather than displaying old records.
struct TrailBufferView {
    VkBuffer buffer{};
    VkDeviceSize size{};
    std::uint32_t capacity{};
    std::uint32_t recordedTicks{};
    std::uint32_t newest{};
};

struct SimulationViewport {
    VkImageView imageView{};
    VkSampler sampler{};
    VkExtent2D extent{960, 720};
    std::uint32_t requestedWidth{960};
    std::uint32_t requestedHeight{720};
    std::uint64_t generation{};
};

struct SimulationState {
    SimulationControls controls;
    SimulationStatistics statistics;
    EvolutionSettings evolution;
    EvolutionHistory history;
    SimulationWorlds worlds;
    SimulationStep settings;
    SweepState sweep;
    SimulationDisplay display;
    AgentBufferView agents;
    LatticeBufferView lattice;
    StructureBufferView structures;
    TrailBufferView trails;
    SimulationViewport viewport;
};

} // namespace vkexp
