#pragma once

#include "vkexp/simulation/LatticeTypes.hpp"

#include <cstdint>
#include <span>

namespace vkexp {

// Everything one reference step reads and writes. Spans and not owners: the
// driver keeps these buffers on the device and mirrors them here, and the parity
// test builds them itself.
//
// A step is no longer a per-agent function. It cannot be: which cell an agent
// ends up in depends on who else asked for it, so the unit of simulation is a
// population and the reference has to run the same three passes the device does.
// That is the whole reason this file exists rather than a `stepAgent` -- a
// per-agent reference could not reproduce contention, and contention is the
// collision model.
struct LatticePopulation {
    std::span<AgentState> agents;
    // worldCount * cellsPerWorld, agent index or LatticeNoOccupant.
    std::span<std::int32_t> occupancy;
    // The same shape. Cleared at the top of every step, so a caller never has to
    // know it exists beyond allocating it.
    std::span<std::int32_t> claims;
    // genomeCount * genomeStride, laid out exactly as the device buffer is.
    std::span<const float> weights;
    std::uint32_t genomeStride{};
    std::uint32_t agentsPerWorld{1};
    std::uint32_t trialsPerGenome{1};
    // Construction blocks, zero for empty. Empty in beacon-only references.
    std::span<std::int32_t> structures;
    // worldCount * LatticeBuildOutcomeCount, accumulated rather than cleared.
    // Optional: leave it empty and the step records nothing, which is what every
    // caller that is not diagnosing the builders wants.
    std::span<std::uint32_t> buildOutcomes;
};

// One step of a whole population: sense, decide and bid; then resolve and move.
// The same order, the same arbitration and the same accumulators as the three
// compute dispatches, which is what the parity test checks step for step.
void stepLatticeCpu(const LatticePopulation& population, const SimulationStep& settings);

// How much foundation stands under one build site: the highest level below
// `buildY` at which `constructionCourseFill` of the cells within
// `constructionSupportRadius` of the column are occupied, plus one, or zero if
// no level qualifies. The height lead is then measured from there.
//
// The question used to be asked of the world's whole floor area, which made the
// frontier a global synchroniser and a layer cake the only buildable shape. A
// radius that spans the floor asks the old question again, so the old rule is
// still reachable rather than deleted.
//
// `worldStructures` is one world's slice of the block field. Mirrored by
// constructionLocalFoundation in lattice_step.comp; the parity test is what
// keeps the two honest.
[[nodiscard]] std::uint32_t constructionLocalFoundation(
    std::span<const std::int32_t> worldStructures, const SimulationStep& settings, int x,
    int buildY, int z);

// Scores a finished trial from what the steps accumulated. Pure arithmetic on
// the record, so it is the same function whether the steps ran here or on the
// device -- the metrics come back from the agent buffer either way.
[[nodiscard]] float agentFitness(const AgentState& agent, const FitnessWeights& weights = {});

} // namespace vkexp
