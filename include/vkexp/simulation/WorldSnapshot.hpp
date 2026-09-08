#pragma once

#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/simulation/AgentTypes.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <vector>

namespace vkexp {

class WorldSnapshotError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A whole experiment frozen mid-flight: the population, where every agent is,
// how far into the generation it got, and the settings that produced all of it.
// A genome archive holds only the weights, which is the right thing to carry
// between runs; this is the right thing to carry between sessions.
//
// The trail field is deliberately not here. It is device-local, it is the
// largest thing in the simulation by two orders of magnitude, and it is derived:
// a couple of half-lives of stepping rebuilds it. Storing it would make a
// snapshot a hundred times bigger to save a few seconds of simulation.
struct WorldSnapshot {
    SimulationStep physics{};
    std::vector<Genome> genomes;
    std::vector<AgentState> agents;
    // One per logical world, and empty for every world that has no puck. This
    // is experiment state, not a derived field like the trail: a snapshot that
    // dropped it would resume with the puck back at the start and the run would
    // read as having lost ground it had not lost.
    std::vector<PuckState> pucks;
    std::uint64_t generation{};
    std::uint32_t step{};
    std::uint32_t stepsPerGeneration{};
    std::uint32_t requestedAgentsPerWorld{};
    std::uint32_t trialsPerGenome{};
    std::uint32_t seed{};
};

// 2 added the group fitness sharing weight. 3 added neuron time constants: the
// genome, the agent record and the settings all changed shape at once, so a
// version 2 file cannot be reinterpreted into this run under any default. A
// file that predates a setting is not a file that had it turned off, so both
// bumps reject rather than fill in.
// 4 replaced the neuron-memory flag with a three-valued model and grew the
// genome by a gate block, so a version 3 file names a different network.
inline constexpr std::uint32_t worldSnapshotVersion = 8;

// Versioned and little-endian, like the genome archive, and just as strict: a
// file from another brain topology, another agent layout or another scenario
// count is rejected rather than reinterpreted.
void saveWorldSnapshot(const std::filesystem::path& path, const WorldSnapshot& snapshot);

[[nodiscard]] WorldSnapshot loadWorldSnapshot(const std::filesystem::path& path);

} // namespace vkexp
