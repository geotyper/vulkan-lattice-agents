#pragma once

#include "vkexp/simulation/LatticeTypes.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace vkexp::lattice {

// How a population is divided into logical worlds. The same three numbers
// SimulationWorlds publishes, gathered here so nothing that only needs to place
// agents has to reach into the driver's state.
struct PopulationLayout {
    std::uint32_t genomeCount{};
    std::uint32_t agentsPerWorld{};
    std::uint32_t trialsPerGenome{1};

    [[nodiscard]] std::uint32_t agentCount() const { return genomeCount * trialsPerGenome; }

    // The group size actually in force, which is not always the one asked for:
    // clampAgentsPerWorld keeps it between one and the population. Everything
    // that divides agents into worlds asks here rather than reading the field,
    // because a caller that indexed with the unclamped number would address
    // worlds that worldCount() says do not exist -- which is a write past the
    // end of the occupancy grid rather than a wrong answer.
    [[nodiscard]] std::uint32_t groupSize() const {
        return clampAgentsPerWorld(genomeCount, agentsPerWorld);
    }
    [[nodiscard]] std::uint32_t worldCount() const {
        return logicalWorldCount(genomeCount, groupSize(), trialsPerGenome);
    }
};

// Where the beacon of one logical world stands.
//
// Hashed from the seed and the world index rather than drawn from a generator,
// so the answer does not depend on what was asked before it. That is what lets
// the parity test place one world without simulating the ones ahead of it, and
// what makes a snapshot resumable: a run reconstructs the same lattice from four
// numbers rather than from a saved stream position.
//
// Per world and not per group means per trial too: the same genome is scored on
// several placements, so a policy that memorised one corner scores as badly as
// it should.
[[nodiscard]] Int4 beaconCell(const SimulationStep& settings, std::uint32_t world);

// Where the harvest world's resource stands. The device works this out for
// itself from the same shared kernel functions rather than being told, so this
// exists for the host alone -- to draw it, and to say in a test where it should
// be.
[[nodiscard]] Int4 resourceCell(const SimulationStep& settings, std::uint32_t world);

// Every agent of a fresh generation, in buffer order, with its beacon mirrored
// on and its metrics cleared. Placement is the same hash, probed forward until a
// free cell -- so two agents never start in one cell, and the whole arrangement
// is a function of the seed and the layout.
[[nodiscard]] std::vector<AgentState> makeInitialAgents(const SimulationStep& settings,
                                                        const PopulationLayout& layout);

// The occupancy grid that goes with a set of agents: -1 everywhere, and the
// agent index in the cell it stands in. Separate from the spawn because a
// snapshot restores agents without respawning them, and the grid is derived
// state that would otherwise have to be saved alongside.
void buildOccupancy(std::span<const AgentState> agents, const SimulationStep& settings,
                    const PopulationLayout& layout, std::span<std::int32_t> occupancy);

} // namespace vkexp::lattice
