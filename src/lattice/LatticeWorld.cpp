#include "vkexp/lattice/LatticeWorld.hpp"

#include "vkexp/lattice/LatticeKernel.hpp"

#include <algorithm>

namespace vkexp::lattice {
namespace {

namespace kern = ::vkexp::lattice::kernel;

// One round of a bijective integer mix. Not a generator: the point is that
// placement is a pure function of what it is asked about, so the beacon of world
// 91 can be computed without having computed the beacon of world 90.
// The shared one, so the resource the shader places and the beacon the host
// places cannot drift onto different hashes.
[[nodiscard]] std::uint32_t mix(const std::uint32_t first, const std::uint32_t second) {
    return kern::latticeMix(first, second);
}

[[nodiscard]] Int4 cellFromHash(const SimulationStep& settings, const std::uint32_t hash) {
    return Int4{
        static_cast<std::int32_t>(hash % settings.latticeWidth),
        static_cast<std::int32_t>((hash / settings.latticeWidth) % settings.latticeHeight),
        static_cast<std::int32_t>((hash / (settings.latticeWidth * settings.latticeHeight)) %
                                  settings.latticeDepth),
        0};
}

[[nodiscard]] std::uint32_t cellIndex(const SimulationStep& settings, const Int4& cell) {
    return kern::latticeCellIndex(cell.x, cell.y, cell.z, settings.latticeWidth,
                                  settings.latticeHeight);
}

[[nodiscard]] Int4 cellFromIndex(const SimulationStep& settings, const std::uint32_t index) {
    const std::uint32_t plane = settings.latticeWidth * settings.latticeHeight;
    return Int4{static_cast<std::int32_t>(index % settings.latticeWidth),
                static_cast<std::int32_t>((index % plane) / settings.latticeWidth),
                static_cast<std::int32_t>(index / plane), 0};
}

// Standing on the ground rather than in it: the bedrock course occupies height
// zero, so a building world's group starts one level up, and only over columns
// that have ground under them.
// Indexes the ground plan rather than the whole floor: over a chasm the open
// columns are not candidates, so the run is groundWidth by depth.
[[nodiscard]] Int4 floorCellFromIndex(const std::uint32_t groundWidth, const std::uint32_t index) {
    return Int4{static_cast<std::int32_t>(index % groundWidth), 1,
                static_cast<std::int32_t>(index / groundWidth), 0};
}

} // namespace

Int4 beaconCell(const SimulationStep& settings, const std::uint32_t world) {
    // Hashed twice against different constants so a run whose seed differs by one
    // does not produce beacons a cell apart: the seed is a slider, and a slider
    // whose neighbouring values are near-identical worlds is a slider that
    // cannot be used to ask whether placement mattered.
    Int4 cell = cellFromHash(settings, mix(settings.beaconSeed ^ 0xB1AC0FU, world));
    cell.w = static_cast<std::int32_t>(world);
    return cell;
}

Int4 resourceCell(const SimulationStep& settings, const std::uint32_t world) {
    const std::uint32_t hash = kern::latticeResourceHash(world, settings.beaconSeed);
    return Int4{kern::latticeResourceX(hash, settings.latticeWidth, latticeGroundWidth(settings)),
                kern::latticeResourceY(hash, settings.resourceHeightLow,
                                       settings.resourceHeightHigh, settings.latticeHeight),
                kern::latticeResourceZ(hash, settings.latticeWidth, settings.latticeDepth),
                static_cast<std::int32_t>(world)};
}

std::vector<std::int32_t> makeTerrain(const SimulationStep& settings,
                                      const std::uint32_t worldCount) {
    const std::uint32_t cells = latticeCellsPerWorld(settings);
    std::vector<std::int32_t> field(static_cast<std::size_t>(cells) * worldCount,
                                    kern::LatticeNoStructure);
    if (!worldBuilds(settings.worldMode)) {
        return field;
    }
    // A course of bedrock where there is ground, and nothing where there is a
    // chasm. Every building world gets one, because support no longer assumes a
    // floor: what an agent stands on is always a block, and terrain is only the
    // blocks that were there before anybody built.
    const std::uint32_t ground = latticeGroundWidth(settings);
    for (std::uint32_t world = 0; world < worldCount; ++world) {
        const std::size_t base = static_cast<std::size_t>(world) * cells;
        for (std::uint32_t z = 0; z < settings.latticeDepth; ++z) {
            for (std::uint32_t x = 0; x < settings.latticeWidth; ++x) {
                if (!kern::latticeGroundColumn(static_cast<int>(x), ground)) {
                    continue;
                }
                field[base + kern::latticeCellIndex(static_cast<int>(x), 0, static_cast<int>(z),
                                                    settings.latticeWidth,
                                                    settings.latticeHeight)] = kern::LatticeBedrock;
            }
        }
    }
    return field;
}

std::vector<AgentState> makeInitialAgents(const SimulationStep& settings,
                                          const PopulationLayout& layout) {
    const std::uint32_t cells = latticeCellsPerWorld(settings);
    const std::uint32_t worlds = layout.worldCount();
    const std::uint32_t groupSize = layout.groupSize();
    std::vector<AgentState> agents(layout.agentCount());
    if (cells == 0 || worlds == 0 || agents.empty()) {
        return agents;
    }

    // One grid for the whole population, so a probe cannot walk out of its own
    // world into the next one's cells.
    std::vector<std::int32_t> occupancy(static_cast<std::size_t>(cells) * worlds,
                                        kern::LatticeNoOccupant);

    for (std::uint32_t index = 0; index < agents.size(); ++index) {
        const std::uint32_t genome = index / layout.trialsPerGenome;
        const std::uint32_t world = logicalWorldForAgent(index, groupSize, layout.trialsPerGenome);
        const std::uint32_t slot = genome % groupSize;
        Int4 beacon = beaconCell(settings, world);
        if (worldBuilds(settings.worldMode)) {
            // xyz becomes the per-step build intent in construction mode. Only
            // w is permanent: it keeps naming the logical world.
            beacon = Int4{-1, -1, -1, static_cast<std::int32_t>(world)};
        }

        AgentState& agent = agents[index];
        agent.beacon = beacon;
        // The one heading value outside the neighbour range: an agent that has
        // not moved is not an agent pointing at neighbour zero.
        agent.cell.w = static_cast<std::int32_t>(kern::LatticeNeighborCount);
        agent.intent.w = 0;
        agent.metrics = {};
        agent.memory = {};
        agent.signal = {};

        // Probed forward from the hashed cell rather than redrawn, so placing an
        // agent costs one hash whatever the lattice's occupancy is -- a redraw
        // loop has no bound when a world is nearly full, and a world that is
        // nearly full is exactly the configuration worth being able to run.
        const std::uint32_t start = mix(settings.beaconSeed ^ 0x5CA1EDU, world * 1021U + slot);
        const bool construction = worldBuilds(settings.worldMode);
        const std::uint32_t groundWidth = latticeGroundWidth(settings);
        const std::uint32_t candidateCount =
            construction ? groundWidth * settings.latticeDepth : cells;
        const std::uint32_t base = construction
                                       ? start % candidateCount
                                       : cellIndex(settings, cellFromHash(settings, start));
        const std::size_t worldBase = static_cast<std::size_t>(world) * cells;
        for (std::uint32_t probe = 0; probe < candidateCount; ++probe) {
            const std::uint32_t candidate = (base + probe) % candidateCount;
            Int4 cell = construction ? floorCellFromIndex(groundWidth, candidate)
                                     : cellFromIndex(settings, candidate);
            const std::uint32_t candidateCell = cellIndex(settings, cell);
            // Never on the beacon: an agent that starts on the objective has
            // solved the world before the first step, which would make the
            // shaping unreadable for the whole group it is scored beside.
            if (!construction && cell.x == beacon.x && cell.y == beacon.y && cell.z == beacon.z) {
                continue;
            }
            if (occupancy[worldBase + candidateCell] != kern::LatticeNoOccupant) {
                continue;
            }
            occupancy[worldBase + candidateCell] = static_cast<std::int32_t>(index);
            cell.w = agent.cell.w;
            agent.cell = cell;
            agent.intent = Int4{cell.x, cell.y, cell.z, 0};
            break;
        }

        if (!construction) {
            const std::uint32_t distance = kern::latticeStepDistance(
                static_cast<std::uint32_t>(settings.neighborhood), beacon.x - agent.cell.x,
                beacon.y - agent.cell.y, beacon.z - agent.cell.z);
            agent.metrics.x = kern::latticeNearness(distance, latticeMaximumDistance(settings));
        }
    }
    return agents;
}

void buildOccupancy(const std::span<const AgentState> agents, const SimulationStep& settings,
                    const PopulationLayout& layout, const std::span<std::int32_t> occupancy) {
    std::fill(occupancy.begin(), occupancy.end(), kern::LatticeNoOccupant);
    const std::uint32_t cells = latticeCellsPerWorld(settings);
    if (cells == 0) {
        return;
    }
    const std::uint32_t groupSize = layout.groupSize();
    for (std::uint32_t index = 0; index < agents.size(); ++index) {
        const AgentState& agent = agents[index];
        const std::uint32_t world = logicalWorldForAgent(index, groupSize, layout.trialsPerGenome);
        if (!kern::latticeInBounds(agent.cell.x, agent.cell.y, agent.cell.z, settings.latticeWidth,
                                   settings.latticeHeight, settings.latticeDepth)) {
            continue;
        }
        const std::size_t slot =
            static_cast<std::size_t>(world) * cells + cellIndex(settings, agent.cell);
        if (slot < occupancy.size()) {
            occupancy[slot] = static_cast<std::int32_t>(index);
        }
    }
}

} // namespace vkexp::lattice
