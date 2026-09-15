#include "vkexp/simulation/LatticeSensors.hpp"

#include "vkexp/lattice/LatticeWorld.hpp"

#include "vkexp/lattice/LatticeKernel.hpp"
#include "vkexp/neuro/BrainKernel.hpp"

#include <algorithm>
#include <array>

namespace vkexp {
namespace {

namespace brain = neuro::kernel;
namespace kern = lattice::kernel;

} // namespace

neuro::Inputs sampleAgentInputs(const AgentState& agent, const std::span<const float> signals,
                                const std::span<const std::int32_t> occupancy,
                                const SimulationStep& settings,
                                const std::span<const std::int32_t> structures) {
    neuro::Inputs inputs{};

    // All twenty-six under both movement settings. See LatticeKernel.inl for why
    // the width does not follow the neighbourhood: a population trained to walk
    // the faces still has to be loadable into a run that walks the diagonals.
    // Read in the body frame: slot n is always the same direction relative to
    // the agent, whichever way it is facing.
    const auto facing = static_cast<std::uint32_t>(agent.cell.w) % kern::LatticeFacingCount;
    for (brain::uint neighbor = 0; neighbor < brain::BrainNeighborCount; ++neighbor) {
        const int bodyX = kern::latticeNeighborX(neighbor);
        const int bodyZ = kern::latticeNeighborZ(neighbor);
        const int x = agent.cell.x + kern::latticeRotatedX(facing, bodyX, bodyZ);
        const int y = agent.cell.y + kern::latticeNeighborY(neighbor);
        const int z = agent.cell.z + kern::latticeRotatedZ(facing, bodyX, bodyZ);

        float occupied = 0.0F;
        float edge = 0.0F;
        float structure = 0.0F;
        float signal = 0.0F;
        if (!kern::latticeInBounds(x, y, z, settings.latticeWidth, settings.latticeHeight,
                                   settings.latticeDepth)) {
            // The edge of the lattice reads as a wall rather than as an empty
            // cell. There is no boundary geometry and no push-out: a lattice
            // ends, and the one place that has to be said is here and in the
            // move rule, not in a containment pass after the fact.
            edge = 1.0F;
        } else {
            const std::uint32_t index =
                kern::latticeCellIndex(x, y, z, settings.latticeWidth, settings.latticeHeight);
            const std::int32_t occupant =
                index < occupancy.size() ? occupancy[index] : kern::LatticeNoOccupant;
            if (worldBuilds(settings.worldMode) && index < structures.size() &&
                structures[index] != kern::LatticeNoStructure) {
                structure = 1.0F;
            } else if (occupant != kern::LatticeNoOccupant &&
                       static_cast<std::size_t>(occupant) < signals.size()) {
                occupied = 1.0F;
                signal = std::clamp(signals[static_cast<std::size_t>(occupant)], 0.0F, 1.0F);
            }
        }
        inputs[brain::brainNeighborChannelIndex(neighbor, kern::LatticeNeighborOccupied)] =
            occupied;
        inputs[brain::brainNeighborChannelIndex(neighbor, kern::LatticeNeighborEdge)] = edge;
        inputs[brain::brainNeighborChannelIndex(neighbor, kern::LatticeNeighborStructure)] =
            structure;
        inputs[brain::brainNeighborChannelIndex(neighbor, kern::LatticeNeighborSignal)] = signal;
    }

    if (settings.worldMode == WorldMode::Construction) {
        inputs[brain::brainBeaconInputIndex(0)] =
            static_cast<float>(agent.cell.y) /
            static_cast<float>(std::max(settings.latticeHeight - 1U, 1U));
        inputs[brain::brainBeaconInputIndex(1)] = agent.signal.z <= 0.0F ? 1.0F : 0.0F;
        inputs[brain::brainBeaconInputIndex(2)] = std::clamp(agent.signal.w, 0.0F, 1.0F);

        // The same five faces as constructionSupported plus the wall in front
        // of the feet, and no implicit floor: the floor is a course of bedrock
        // in the block field like anything else, so an agent over a chasm reads
        // unsupported and is right.
        bool supported = false;
        const std::array<std::array<int, 3>, 6> supportOffsets{
            {{{0, -1, 0}},
             {{-1, 0, 0}},
             {{1, 0, 0}},
             {{0, 0, -1}},
             {{0, 0, 1}},
             {{kern::latticeFacingX(facing), -1, kern::latticeFacingZ(facing)}}}};
        for (const auto& offset : supportOffsets) {
            const int x = agent.cell.x + offset[0];
            const int y = agent.cell.y + offset[1];
            const int z = agent.cell.z + offset[2];
            if (kern::latticeInBounds(x, y, z, settings.latticeWidth, settings.latticeHeight,
                                      settings.latticeDepth)) {
                const std::uint32_t cell =
                    kern::latticeCellIndex(x, y, z, settings.latticeWidth, settings.latticeHeight);
                supported |=
                    cell < structures.size() && structures[cell] != kern::LatticeNoStructure;
            }
        }
        inputs[brain::brainBeaconInputIndex(3)] = supported ? 1.0F : 0.0F;
    } else {
        // Beacon and harvest point at the same kind of thing -- a cell somewhere
        // else that the trial is about -- so they are sensed by the same four
        // numbers. Where the cell is differs: a beacon is mirrored onto the
        // agent, and a resource is derived from the world it stands in.
        Int4 objective = agent.beacon;
        if (worldHarvests(settings.worldMode)) {
            objective = lattice::resourceCell(settings,
                                              static_cast<std::uint32_t>(agent.beacon.w));
        }
        const int deltaX = objective.x - agent.cell.x;
        const int deltaY = objective.y - agent.cell.y;
        const int deltaZ = objective.z - agent.cell.z;
        const float length = kern::latticeVectorLength(deltaX, deltaY, deltaZ);
        const std::uint32_t distance = kern::latticeStepDistance(
            static_cast<std::uint32_t>(settings.neighborhood), deltaX, deltaY, deltaZ);
        // Turned into the body frame like everything else. The inverse rotation,
        // since this converts a world offset into body axes rather than the
        // other way round.
        const std::uint32_t inverse =
            (kern::LatticeFacingCount - facing) % kern::LatticeFacingCount;
        const int bodyAheadX = kern::latticeRotatedX(inverse, deltaX, deltaZ);
        const int bodyAheadZ = kern::latticeRotatedZ(inverse, deltaX, deltaZ);
        inputs[brain::brainBeaconInputIndex(0)] =
            kern::latticeDirectionComponent(bodyAheadX, length);
        inputs[brain::brainBeaconInputIndex(1)] = kern::latticeDirectionComponent(deltaY, length);
        inputs[brain::brainBeaconInputIndex(2)] =
            kern::latticeDirectionComponent(bodyAheadZ, length);
        inputs[brain::brainBeaconInputIndex(3)] =
            kern::latticeNearness(distance, latticeMaximumDistance(settings));
        if (worldHarvests(settings.worldMode)) {
            inputs[brain::brainBeaconInputIndex(4)] = agent.signal.z <= 0.0F ? 1.0F : 0.0F;
            inputs[brain::brainBeaconInputIndex(5)] = std::clamp(agent.memory.w, 0.0F, 1.0F);
        }
    }

    // No heading channel: in the body frame the agent faces forward by
    // definition, so its absolute orientation is unobservable and could only
    // ever have carried a constant.
    inputs[brain::BrainSelfOffset] = agent.intent.w != 0 ? 1.0F : 0.0F;
    inputs[brain::BrainSelfOffset + 1] = kern::latticeStillness(agent.memory.z);

    inputs[brain::BrainRecurrentInputOffset] = std::clamp(agent.memory.x, -1.0F, 1.0F);
    inputs[brain::BrainRecurrentInputOffset + 1] = std::clamp(agent.memory.y, -1.0F, 1.0F);
    return inputs;
}

} // namespace vkexp
