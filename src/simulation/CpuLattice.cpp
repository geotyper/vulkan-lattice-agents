#include "vkexp/simulation/CpuLattice.hpp"

#include "vkexp/lattice/LatticeKernel.hpp"
#include "vkexp/simulation/LatticeSensors.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace vkexp {
namespace {

namespace brain = neuro::kernel;
namespace kern = lattice::kernel;

[[nodiscard]] bool sameCell(const Int4& first, const Int4& second) {
    return first.x == second.x && first.y == second.y && first.z == second.z;
}

[[nodiscard]] bool hasStructure(const std::span<const std::int32_t> structures,
                                const SimulationStep& settings, const int x, const int y,
                                const int z) {
    if (!kern::latticeInBounds(x, y, z, settings.latticeWidth, settings.latticeHeight,
                               settings.latticeDepth)) {
        return false;
    }
    const std::uint32_t cell =
        kern::latticeCellIndex(x, y, z, settings.latticeWidth, settings.latticeHeight);
    return cell < structures.size() && structures[cell] != kern::LatticeNoStructure;
}

[[nodiscard]] bool constructionSupported(const std::span<const std::int32_t> structures,
                                         const SimulationStep& settings, const int x, const int y,
                                         const int z) {
    return y <= 0 || hasStructure(structures, settings, x, y - 1, z) ||
           hasStructure(structures, settings, x - 1, y, z) ||
           hasStructure(structures, settings, x + 1, y, z) ||
           hasStructure(structures, settings, x, y, z - 1) ||
           hasStructure(structures, settings, x, y, z + 1);
}

[[nodiscard]] bool constructionBlockSupported(const std::span<const std::int32_t> structures,
                                              const SimulationStep& settings, const int x,
                                              const int y, const int z) {
    if (y <= 0 || hasStructure(structures, settings, x, y - 1, z)) {
        return true;
    }
    return settings.allowSideSupportedBlocks != 0U &&
           (hasStructure(structures, settings, x - 1, y, z) ||
            hasStructure(structures, settings, x + 1, y, z) ||
            hasStructure(structures, settings, x, y, z - 1) ||
            hasStructure(structures, settings, x, y, z + 1));
}

[[nodiscard]] int constructionLandingY(const std::span<const std::int32_t> structures,
                                       const SimulationStep& settings, const int x, const int y,
                                       const int z) {
    int landing = std::clamp(y, 0, static_cast<int>(settings.latticeHeight) - 1);
    while (landing > 0 && !constructionSupported(structures, settings, x, landing, z)) {
        --landing;
    }
    return landing;
}

[[nodiscard]] std::array<int, 2> constructionFacing(const AgentState& agent) {
    const auto heading = static_cast<std::uint32_t>(agent.cell.w);
    if (heading >= kern::LatticeNeighborCount) {
        return {0, 0};
    }
    const int x = kern::latticeNeighborX(heading);
    const int z = kern::latticeNeighborZ(heading);
    return x != 0 ? std::array<int, 2>{x, 0} : std::array<int, 2>{0, z};
}

} // namespace

void stepLatticeCpu(const LatticePopulation& population, const SimulationStep& settings) {
    const std::uint32_t cells = latticeCellsPerWorld(settings);
    if (cells == 0 || population.agents.empty()) {
        return;
    }
    const neuro::BrainShape shape = resolvedBrain(settings);
    const auto neighborhood = static_cast<std::uint32_t>(settings.neighborhood);
    const std::uint32_t maximumDistance = latticeMaximumDistance(settings);
    const std::size_t neuronCount = shape.hiddenTotal();

    std::fill(population.claims.begin(), population.claims.end(), kern::LatticeNoClaim);

    // What everybody was broadcasting when the step began. Taken before the
    // decide loop because that loop writes each agent's new broadcast as it
    // goes: on the device the two are different buffers, so a neighbour always
    // reads last step's value, and a reference that read the live record would
    // drift from it by an amount that depends on the agent numbering.
    std::vector<float> signals(population.agents.size());
    for (std::size_t index = 0; index < population.agents.size(); ++index) {
        signals[index] = population.agents[index].signal.x;
    }

    // A compact mirror of the GPU's per-course counter buffer. It is rebuilt
    // from the reference structure field once per step, before any placement,
    // so every agent decides against the same construction frontier.
    const std::uint32_t worldCount =
        static_cast<std::uint32_t>(population.occupancy.size() / cells);
    std::vector<std::uint32_t> foundationHeights(worldCount, 0U);
    if (settings.worldMode == WorldMode::Construction && !population.structures.empty()) {
        std::vector<std::uint32_t> courseCounts(
            static_cast<std::size_t>(worldCount) * settings.latticeHeight, 0U);
        for (std::size_t absolute = 0; absolute < population.structures.size(); ++absolute) {
            if (population.structures[absolute] == kern::LatticeNoStructure) {
                continue;
            }
            const std::uint32_t world = static_cast<std::uint32_t>(absolute / cells);
            const std::uint32_t local = static_cast<std::uint32_t>(absolute % cells);
            const std::uint32_t y = (local / settings.latticeWidth) % settings.latticeHeight;
            ++courseCounts[static_cast<std::size_t>(world) * settings.latticeHeight + y];
        }
        const std::uint32_t courseArea = settings.latticeWidth * settings.latticeDepth;
        const std::uint32_t required =
            std::max(static_cast<std::uint32_t>(
                         std::ceil(static_cast<float>(courseArea) *
                                   std::clamp(settings.constructionCourseFill, 0.0F, 1.0F))),
                     1U);
        for (std::uint32_t world = 0; world < worldCount; ++world) {
            while (foundationHeights[world] < settings.latticeHeight &&
                   courseCounts[static_cast<std::size_t>(world) * settings.latticeHeight +
                                foundationHeights[world]] >= required) {
                ++foundationHeights[world];
            }
        }
    }

    // --- decide ---------------------------------------------------------------
    //
    // Every agent senses the lattice as it stands at the top of the step and bids
    // for where it wants to be. Nothing moves here, which is the point: an agent
    // that moved would change what the agents after it see, and the answer would
    // then depend on the order -- which a GPU does not fix and a reference could
    // not reproduce.
    for (std::uint32_t index = 0; index < population.agents.size(); ++index) {
        AgentState& agent = population.agents[index];
        const std::uint32_t world =
            logicalWorldForAgent(index, population.agentsPerWorld, population.trialsPerGenome);
        const std::size_t worldBase = static_cast<std::size_t>(world) * cells;
        const std::span<std::int32_t> worldStructures =
            population.structures.size() >= worldBase + cells
                ? population.structures.subspan(worldBase, cells)
                : std::span<std::int32_t>{};
        const std::span<std::int32_t> worldOccupancy =
            population.occupancy.subspan(worldBase, cells);

        const std::uint32_t genome = index / population.trialsPerGenome;
        const std::span<const float> weights = population.weights.subspan(
            static_cast<std::size_t>(genome) * population.genomeStride, population.genomeStride);

        neuro::HiddenState hidden{};
        for (std::size_t neuron = 0; neuron < neuronCount; ++neuron) {
            hidden[neuron] = agentHiddenState(agent, neuron);
        }
        const neuro::Outputs output = neuro::evaluate(
            weights, sampleAgentInputs(agent, signals, worldOccupancy, settings, worldStructures),
            hidden, settings.deltaTime, static_cast<brain::uint>(settings.neuronModel), shape);
        for (std::size_t neuron = 0; neuron < neuronCount; ++neuron) {
            setAgentHiddenState(agent, neuron, hidden[neuron]);
        }

        agent.signal.x = std::clamp(output[brain::BrainSignalIntensityOutput], 0.0F, 1.0F);
        agent.signal.y = output[brain::BrainBuildOutput];
        agent.signal.z = std::max(agent.signal.z - 1.0F, 0.0F);
        agent.signal.w = 0.0F;
        if (shape.outputCount >=
            neuro::Topology::actuatorOutputCount + neuro::Topology::recurrentMemoryCount) {
            agent.memory.x = output[neuro::Topology::recurrentOutputOffset];
            agent.memory.y = output[neuro::Topology::recurrentOutputOffset + 1];
        } else {
            agent.memory.x = 0.0F;
            agent.memory.y = 0.0F;
        }

        const float driveX = output[brain::BrainMoveOutput];
        const float driveY = output[brain::BrainMoveOutput + 1];
        const float driveZ = output[brain::BrainMoveOutput + 2];
        const int stepX = kern::latticeMoveComponent(neighborhood, 0U, driveX, driveY, driveZ,
                                                     settings.moveThreshold);
        const int stepY = kern::latticeMoveComponent(neighborhood, 1U, driveX, driveY, driveZ,
                                                     settings.moveThreshold);
        const int stepZ = kern::latticeMoveComponent(neighborhood, 2U, driveX, driveY, driveZ,
                                                     settings.moveThreshold);

        agent.intent = Int4{agent.cell.x, agent.cell.y, agent.cell.z, 0};
        if (settings.worldMode == WorldMode::Construction) {
            int wantedX = agent.cell.x;
            int wantedY = agent.cell.y;
            int wantedZ = agent.cell.z;
            const bool horizontal = stepX != 0 || stepZ != 0;
            bool attempted = horizontal || stepY != 0;

            if (horizontal) {
                wantedX += stepX;
                wantedZ += stepZ;
                if (!kern::latticeInBounds(wantedX, wantedY, wantedZ, settings.latticeWidth,
                                           settings.latticeHeight, settings.latticeDepth)) {
                    agent.intent.w = 1;
                    attempted = false;
                } else if (hasStructure(worldStructures, settings, wantedX, wantedY, wantedZ)) {
                    ++wantedY;
                    if (!kern::latticeInBounds(wantedX, wantedY, wantedZ, settings.latticeWidth,
                                               settings.latticeHeight, settings.latticeDepth) ||
                        hasStructure(worldStructures, settings, wantedX, wantedY, wantedZ)) {
                        attempted = false;
                    }
                } else {
                    wantedY =
                        constructionLandingY(worldStructures, settings, wantedX, wantedY, wantedZ);
                }
            } else if (stepY > 0) {
                const auto [faceX, faceZ] = constructionFacing(agent);
                const bool hasFace = (faceX != 0 || faceZ != 0) &&
                                     (hasStructure(worldStructures, settings, agent.cell.x + faceX,
                                                   agent.cell.y, agent.cell.z + faceZ) ||
                                      hasStructure(worldStructures, settings, agent.cell.x + faceX,
                                                   agent.cell.y + 1, agent.cell.z + faceZ));
                ++wantedY;
                if (!hasFace ||
                    !kern::latticeInBounds(wantedX, wantedY, wantedZ, settings.latticeWidth,
                                           settings.latticeHeight, settings.latticeDepth) ||
                    hasStructure(worldStructures, settings, wantedX, wantedY, wantedZ)) {
                    attempted = false;
                }
            } else if (stepY < 0) {
                wantedY =
                    constructionLandingY(worldStructures, settings, wantedX, wantedY - 1, wantedZ);
            } else if (!constructionSupported(worldStructures, settings, wantedX, wantedY,
                                              wantedZ)) {
                wantedY =
                    constructionLandingY(worldStructures, settings, wantedX, wantedY - 1, wantedZ);
                attempted = true;
            }

            if (!attempted && agent.intent.w == 0 &&
                !constructionSupported(worldStructures, settings, agent.cell.x, agent.cell.y,
                                       agent.cell.z)) {
                wantedX = agent.cell.x;
                wantedY = constructionLandingY(worldStructures, settings, agent.cell.x,
                                               agent.cell.y - 1, agent.cell.z);
                wantedZ = agent.cell.z;
                attempted = true;
            }

            if (attempted && agent.intent.w == 0 &&
                (wantedX != agent.cell.x || wantedY != agent.cell.y || wantedZ != agent.cell.z)) {
                const std::uint32_t wanted = kern::latticeCellIndex(
                    wantedX, wantedY, wantedZ, settings.latticeWidth, settings.latticeHeight);
                if (worldOccupancy[wanted] == kern::LatticeNoOccupant &&
                    !hasStructure(worldStructures, settings, wantedX, wantedY, wantedZ)) {
                    agent.intent = Int4{wantedX, wantedY, wantedZ, 0};
                    population.claims[worldBase + wanted] = kern::latticeBetterClaim(
                        population.claims[worldBase + wanted], static_cast<std::int32_t>(index));
                } else if (!hasStructure(worldStructures, settings, wantedX, wantedY, wantedZ)) {
                    agent.intent.w = 1;
                }
            }

            agent.beacon.x = -1;
            agent.beacon.y = -1;
            agent.beacon.z = -1;
            if (agent.signal.z <= 0.0F && agent.signal.y > settings.buildThreshold) {
                const auto [faceX, faceZ] = constructionFacing(agent);
                const int buildX = agent.cell.x + faceX;
                const int buildZ = agent.cell.z + faceZ;
                if ((faceX != 0 || faceZ != 0) && buildX >= 0 && buildZ >= 0 &&
                    buildX < static_cast<int>(settings.latticeWidth) &&
                    buildZ < static_cast<int>(settings.latticeDepth)) {
                    int buildY = agent.cell.y;
                    bool supported = constructionBlockSupported(worldStructures, settings, buildX,
                                                                buildY, buildZ);
                    if (!supported && settings.allowSideSupportedBlocks != 0U && buildY > 0) {
                        --buildY;
                        supported = constructionBlockSupported(worldStructures, settings, buildX,
                                                               buildY, buildZ);
                    }
                    const std::uint32_t target = kern::latticeCellIndex(
                        buildX, buildY, buildZ, settings.latticeWidth, settings.latticeHeight);
                    const bool empty =
                        !hasStructure(worldStructures, settings, buildX, buildY, buildZ);
                    const bool belowFrontier =
                        static_cast<std::uint32_t>(buildY) <
                        foundationHeights[world] + std::max(settings.constructionHeightLead, 1U);
                    if (empty && supported && belowFrontier &&
                        worldOccupancy[target] == kern::LatticeNoOccupant) {
                        agent.beacon.x = buildX;
                        agent.beacon.y = buildY;
                        agent.beacon.z = buildZ;
                        population.claims[worldBase + target] = kern::latticeBetterClaim(
                            population.claims[worldBase + target],
                            kern::latticeBuildClaim(
                                index, static_cast<std::uint32_t>(population.agents.size())));
                    }
                }
            }
            continue;
        }
        if (stepX == 0 && stepY == 0 && stepZ == 0) {
            continue;
        }

        const int wantedX = agent.cell.x + stepX;
        const int wantedY = agent.cell.y + stepY;
        const int wantedZ = agent.cell.z + stepZ;
        if (!kern::latticeInBounds(wantedX, wantedY, wantedZ, settings.latticeWidth,
                                   settings.latticeHeight, settings.latticeDepth)) {
            agent.intent.w = 1; // walked into the edge of the lattice
            continue;
        }
        const std::uint32_t wanted = kern::latticeCellIndex(
            wantedX, wantedY, wantedZ, settings.latticeWidth, settings.latticeHeight);
        if (!kern::latticeCellEnterable(worldOccupancy[wanted])) {
            agent.intent.w = 1; // walked into somebody who was already standing there
            continue;
        }
        agent.intent = Int4{wantedX, wantedY, wantedZ, 0};
        std::int32_t& claim = population.claims[worldBase + wanted];
        claim = kern::latticeBetterClaim(claim, static_cast<std::int32_t>(index));
    }

    // --- resolve --------------------------------------------------------------
    //
    // Every bid is in, so who won is settled and every agent can be told the
    // answer independently. The occupancy writes below never collide: only an
    // agent's own former cell is cleared by it, and a cell can only be won if it
    // was empty when the step began.
    for (std::uint32_t index = 0; index < population.agents.size(); ++index) {
        AgentState& agent = population.agents[index];
        const std::uint32_t world =
            logicalWorldForAgent(index, population.agentsPerWorld, population.trialsPerGenome);
        const std::size_t worldBase = static_cast<std::size_t>(world) * cells;
        const std::span<std::int32_t> worldStructures =
            population.structures.size() >= worldBase + cells
                ? population.structures.subspan(worldBase, cells)
                : std::span<std::int32_t>{};

        bool refused = agent.intent.w != 0;
        bool moved = false;
        if (!sameCell(agent.intent, agent.cell)) {
            const std::uint32_t wanted =
                kern::latticeCellIndex(agent.intent.x, agent.intent.y, agent.intent.z,
                                       settings.latticeWidth, settings.latticeHeight);
            if (population.claims[worldBase + wanted] == static_cast<std::int32_t>(index)) {
                const std::uint32_t vacated =
                    kern::latticeCellIndex(agent.cell.x, agent.cell.y, agent.cell.z,
                                           settings.latticeWidth, settings.latticeHeight);
                population.occupancy[worldBase + vacated] = kern::LatticeNoOccupant;
                population.occupancy[worldBase + wanted] = static_cast<std::int32_t>(index);
                const int movedX = agent.intent.x - agent.cell.x;
                const int movedY = agent.intent.y - agent.cell.y;
                const int movedZ = agent.intent.z - agent.cell.z;
                if (settings.worldMode != WorldMode::Construction || movedX != 0 || movedZ != 0) {
                    agent.cell.w = static_cast<std::int32_t>(
                        kern::latticeNeighborIndex(movedX, movedY, movedZ));
                }
                agent.cell.x = agent.intent.x;
                agent.cell.y = agent.intent.y;
                agent.cell.z = agent.intent.z;
                moved = true;
            } else {
                refused = settings.worldMode != WorldMode::Construction ||
                          population.claims[worldBase + wanted] >= 0;
            }
        }
        // The flag the next step reads back as a self input, so a policy can
        // notice it is stuck without having to infer it from the neighbourhood.
        agent.intent = Int4{agent.cell.x, agent.cell.y, agent.cell.z, refused ? 1 : 0};

        if (settings.worldMode == WorldMode::Construction && agent.beacon.x >= 0 &&
            !worldStructures.empty()) {
            const std::uint32_t target =
                kern::latticeCellIndex(agent.beacon.x, agent.beacon.y, agent.beacon.z,
                                       settings.latticeWidth, settings.latticeHeight);
            if (population.claims[worldBase + target] ==
                    kern::latticeBuildClaim(index,
                                            static_cast<std::uint32_t>(population.agents.size())) &&
                worldStructures[target] == kern::LatticeNoStructure) {
                worldStructures[target] = static_cast<std::int32_t>(index) + 1;
                agent.signal.z = static_cast<float>(settings.buildIntervalTicks);
                agent.signal.w = 1.0F;
                agent.metrics.y += 1.0F;
            }
        }

        agent.metrics.z +=
            (moved ? 1.0F : 0.0F) + settings.fitness.signalCostFactor * agent.signal.x;
        if (settings.worldMode == WorldMode::Construction) {
            const bool onHorizontalPerimeter =
                agent.cell.x == 0 || agent.cell.z == 0 ||
                agent.cell.x == static_cast<std::int32_t>(settings.latticeWidth) - 1 ||
                agent.cell.z == static_cast<std::int32_t>(settings.latticeDepth) - 1;
            agent.metrics.w += onHorizontalPerimeter ? 1.0F : 0.0F;
            agent.metrics.x = std::max(
                agent.metrics.x, static_cast<float>(agent.cell.y + 1) /
                                     static_cast<float>(std::max(settings.latticeHeight, 1U)));
        } else {
            if (refused) {
                agent.metrics.w += 1.0F;
            }
            const std::uint32_t distance = kern::latticeStepDistance(
                neighborhood, agent.beacon.x - agent.cell.x, agent.beacon.y - agent.cell.y,
                agent.beacon.z - agent.cell.z);
            agent.metrics.x =
                std::max(agent.metrics.x, kern::latticeNearness(distance, maximumDistance));
            if (kern::latticeBeaconReached(distance, settings.beaconContactRadius)) {
                agent.metrics.y += 1.0F;
            }
        }
    }
}

float agentFitness(const AgentState& agent, const FitnessWeights& weights) {
    return kern::latticeTrialFitness(
        agent.metrics.x, agent.metrics.y, agent.metrics.z, agent.metrics.w, weights.trackingReward,
        weights.objectiveBonus, weights.motorCostWeight, weights.refusalPenalty);
}

} // namespace vkexp
