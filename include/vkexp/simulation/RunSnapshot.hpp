#pragma once

#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/simulation/LatticeTypes.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace vkexp {

class RunSnapshotError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A whole experiment frozen mid-flight: the population, where every agent
// stands, how far into the generation it got, and the settings that produced all
// of it. A genome archive holds only the weights, which is the right thing to
// carry between runs; this is the right thing to carry between sessions.
//
// The occupancy grid is deliberately not here. It is device-local, it is the
// largest thing in the simulation, and it is derived: lattice::buildOccupancy
// rebuilds it exactly from the agents on load. Storing it would make a snapshot
// an order of magnitude bigger to save an O(agents) loop.
struct RunSnapshot {
    SimulationStep settings{};
    std::vector<Genome> genomes;
    std::vector<AgentState> agents;
    // Unlike agent occupancy, construction is not derivable from agent cells.
    // It is the work the group has already done and must survive a resume.
    std::vector<std::int32_t> structures;
    std::uint64_t generation{};
    std::uint32_t step{};
    std::uint32_t stepsPerGeneration{};
    std::uint32_t requestedAgentsPerWorld{};
    std::uint32_t trialsPerGenome{};
    std::uint32_t seed{};
};

// Version 13 lets a hidden layer choose its squash, which is a field in the
// settings and a pair of bits in the packed plan the GPU reads.
// Version 12 is the body frame: an agent owns a facing, its commands are a turn
// and an action rather than five world-axis drives, and its senses are read
// relative to where it is looking. A stored agent's cell.w meant "the way it
// last moved" and now means "the way it is looking", so an older file would
// load and turn every climber's grip into nonsense.
// Version 9 adds the chasm world: a bedrock floor stored in the block field, a
// resource hung in a band over the missing half, and how much floor there is.
// Version 8 drops the construction frontier, which refused nothing the support
// rule had not already refused. Version 7 adds the harvest world and where its
// resource sits. Version 5 added optional cardinal side support. The per-course
// counters introduced with version 4 are derived from the stored structure
// field.
// It is not a continuation of the old 2D format: the agent record, settings and
// world all changed at once, and that format has different magic.
inline constexpr std::uint32_t runSnapshotVersion = 15;

// Versioned and little-endian, like the genome archive, and just as strict: a
// file from another brain topology or another agent layout is rejected rather
// than reinterpreted.
void saveRunSnapshot(const std::filesystem::path& path, const RunSnapshot& snapshot);

[[nodiscard]] RunSnapshot loadRunSnapshot(const std::filesystem::path& path);

} // namespace vkexp
