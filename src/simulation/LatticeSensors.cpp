#include "vkexp/simulation/LatticeSensors.hpp"

#include "vkexp/lattice/LatticeKernel.hpp"
#include "vkexp/neuro/BrainKernel.hpp"

#include <algorithm>

namespace vkexp {
namespace {

namespace brain = neuro::kernel;
namespace kern = lattice::kernel;

} // namespace

neuro::Inputs sampleAgentInputs(const AgentState& agent, const std::span<const float> signals,
                                const std::span<const std::int32_t> occupancy,
                                const SimulationStep& settings) {
    neuro::Inputs inputs{};

    // All twenty-six under both movement settings. See LatticeKernel.inl for why
    // the width does not follow the neighbourhood: a population trained to walk
    // the faces still has to be loadable into a run that walks the diagonals.
    for (brain::uint neighbor = 0; neighbor < brain::BrainNeighborCount; ++neighbor) {
        const int x = agent.cell.x + kern::latticeNeighborX(neighbor);
        const int y = agent.cell.y + kern::latticeNeighborY(neighbor);
        const int z = agent.cell.z + kern::latticeNeighborZ(neighbor);

        float occupied = 0.0F;
        float blocked = 0.0F;
        float signal = 0.0F;
        if (!kern::latticeInBounds(x, y, z, settings.latticeWidth, settings.latticeHeight,
                                   settings.latticeDepth)) {
            // The edge of the lattice reads as a wall rather than as an empty
            // cell. There is no boundary geometry and no push-out: a lattice
            // ends, and the one place that has to be said is here and in the
            // move rule, not in a containment pass after the fact.
            blocked = 1.0F;
        } else {
            const std::uint32_t index = kern::latticeCellIndex(
                x, y, z, settings.latticeWidth, settings.latticeHeight);
            const std::int32_t occupant =
                index < occupancy.size() ? occupancy[index] : kern::LatticeNoOccupant;
            if (occupant != kern::LatticeNoOccupant &&
                static_cast<std::size_t>(occupant) < signals.size()) {
                occupied = 1.0F;
                signal = std::clamp(signals[static_cast<std::size_t>(occupant)], 0.0F, 1.0F);
            }
        }
        inputs[brain::brainNeighborChannelIndex(neighbor, kern::LatticeNeighborOccupied)] = occupied;
        inputs[brain::brainNeighborChannelIndex(neighbor, kern::LatticeNeighborBlocked)] = blocked;
        inputs[brain::brainNeighborChannelIndex(neighbor, kern::LatticeNeighborSignal)] = signal;
    }

    const int deltaX = agent.beacon.x - agent.cell.x;
    const int deltaY = agent.beacon.y - agent.cell.y;
    const int deltaZ = agent.beacon.z - agent.cell.z;
    const float length = kern::latticeVectorLength(deltaX, deltaY, deltaZ);
    const std::uint32_t distance = kern::latticeStepDistance(
        static_cast<std::uint32_t>(settings.neighborhood), deltaX, deltaY, deltaZ);
    inputs[brain::brainBeaconInputIndex(0)] = kern::latticeDirectionComponent(deltaX, length);
    inputs[brain::brainBeaconInputIndex(1)] = kern::latticeDirectionComponent(deltaY, length);
    inputs[brain::brainBeaconInputIndex(2)] = kern::latticeDirectionComponent(deltaZ, length);
    inputs[brain::brainBeaconInputIndex(3)] =
        kern::latticeNearness(distance, latticeMaximumDistance(settings));

    // The heading, as the unit step it last took. An agent that has not moved
    // reads zero on all three, which is a distinguishable state and not a
    // direction -- see the note on the heading field.
    const auto heading = static_cast<brain::uint>(agent.cell.w);
    if (heading < brain::BrainNeighborCount) {
        const int headingX = kern::latticeNeighborX(heading);
        const int headingY = kern::latticeNeighborY(heading);
        const int headingZ = kern::latticeNeighborZ(heading);
        const float headingLength = kern::latticeVectorLength(headingX, headingY, headingZ);
        inputs[brain::BrainSelfOffset] =
            kern::latticeDirectionComponent(headingX, headingLength);
        inputs[brain::BrainSelfOffset + 1] =
            kern::latticeDirectionComponent(headingY, headingLength);
        inputs[brain::BrainSelfOffset + 2] =
            kern::latticeDirectionComponent(headingZ, headingLength);
    }
    inputs[brain::BrainSelfOffset + 3] = agent.intent.w != 0 ? 1.0F : 0.0F;

    inputs[brain::BrainRecurrentInputOffset] = std::clamp(agent.memory.x, -1.0F, 1.0F);
    inputs[brain::BrainRecurrentInputOffset + 1] = std::clamp(agent.memory.y, -1.0F, 1.0F);
    return inputs;
}

} // namespace vkexp
