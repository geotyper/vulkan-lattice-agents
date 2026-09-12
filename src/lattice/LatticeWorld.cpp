#include "vkexp/lattice/LatticeWorld.hpp"

#include "vkexp/lattice/LatticeKernel.hpp"

#include <algorithm>

namespace vkexp::lattice {
namespace {

namespace kern = ::vkexp::lattice::kernel;

// One round of a bijective integer mix. Not a generator: the point is that
// placement is a pure function of what it is asked about, so the beacon of world
// 91 can be computed without having computed the beacon of world 90.
[[nodiscard]] std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7FEB352DU;
    value ^= value >> 15U;
    value *= 0x846CA68BU;
    value ^= value >> 16U;
    return value;
}

[[nodiscard]] std::uint32_t mix(const std::uint32_t first, const std::uint32_t second) {
    return mix(first ^ (mix(second) + 0x9E3779B9U + (first << 6U) + (first >> 2U)));
}

[[nodiscard]] Int4 cellFromHash(const SimulationStep& settings, const std::uint32_t hash) {
    return Int4{static_cast<std::int32_t>(hash % settings.latticeWidth),
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

std::vector<AgentState> makeInitialAgents(const SimulationStep& settings,
                                          const PopulationLayout& layout) {
    const std::uint32_t cells = latticeCellsPerWorld(settings);
    const std::uint32_t worlds = layout.worldCount();
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
        const std::uint32_t world =
            logicalWorldForAgent(index, layout.agentsPerWorld, layout.trialsPerGenome);
        const std::uint32_t slot = genome % layout.agentsPerWorld;
        const Int4 beacon = beaconCell(settings, world);

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
        const std::uint32_t base = cellIndex(settings, cellFromHash(settings, start));
        const std::size_t worldBase = static_cast<std::size_t>(world) * cells;
        for (std::uint32_t probe = 0; probe < cells; ++probe) {
            const std::uint32_t candidate = (base + probe) % cells;
            Int4 cell = cellFromIndex(settings, candidate);
            // Never on the beacon: an agent that starts on the objective has
            // solved the world before the first step, which would make the
            // shaping unreadable for the whole group it is scored beside.
            if (cell.x == beacon.x && cell.y == beacon.y && cell.z == beacon.z) {
                continue;
            }
            if (occupancy[worldBase + candidate] != kern::LatticeNoOccupant) {
                continue;
            }
            occupancy[worldBase + candidate] = static_cast<std::int32_t>(index);
            cell.w = agent.cell.w;
            agent.cell = cell;
            agent.intent = Int4{cell.x, cell.y, cell.z, 0};
            break;
        }

        const std::uint32_t distance = kern::latticeStepDistance(
            static_cast<std::uint32_t>(settings.neighborhood), beacon.x - agent.cell.x,
            beacon.y - agent.cell.y, beacon.z - agent.cell.z);
        agent.metrics.x = kern::latticeNearness(distance, latticeMaximumDistance(settings));
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
    for (std::uint32_t index = 0; index < agents.size(); ++index) {
        const AgentState& agent = agents[index];
        const std::uint32_t world =
            logicalWorldForAgent(index, layout.agentsPerWorld, layout.trialsPerGenome);
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
