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

// Mirrors constructionFacing in lattice_step.comp.
[[nodiscard]] std::array<int, 2> constructionFacing(const SimulationStep& settings,
                                                    const float aimX, const float aimZ) {
    return {kern::latticeAimComponent(0U, aimX, aimZ, settings.moveThreshold),
            kern::latticeAimComponent(1U, aimX, aimZ, settings.moveThreshold)};
}

} // namespace

std::uint32_t constructionLocalFoundation(const std::span<const std::int32_t> worldStructures,
                                          const SimulationStep& settings, const int x,
                                          const int buildY, const int z) {
    const auto radius = static_cast<int>(settings.constructionSupportRadius);
    const float fill = std::clamp(settings.constructionCourseFill, 0.0F, 1.0F);
    for (int y = buildY - 1; y >= 0; --y) {
        std::uint32_t sampled = 0;
        std::uint32_t filled = 0;
        for (int dz = -radius; dz <= radius; ++dz) {
            for (int dx = -radius; dx <= radius; ++dx) {
                const int nx = x + dx;
                const int nz = z + dz;
                if (nx < 0 || nz < 0 || nx >= static_cast<int>(settings.latticeWidth) ||
                    nz >= static_cast<int>(settings.latticeDepth)) {
                    continue;
                }
                ++sampled;
                filled += hasStructure(worldStructures, settings, nx, y, nz) ? 1U : 0U;
            }
        }
        const auto required = std::max(
            static_cast<std::uint32_t>(std::ceil(static_cast<float>(sampled) * fill)), 1U);
        if (filled >= required) {
            return static_cast<std::uint32_t>(y) + 1U;
        }
    }
    return 0;
}

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
        const float aimDriveX = output[brain::BrainFaceOutput];
        const float aimDriveZ = output[brain::BrainFaceOutput + 1];
        const int stepX = kern::latticeMoveComponent(neighborhood, 0U, driveX, driveY, driveZ,
                                                     settings.moveThreshold);
        const int stepY = kern::latticeMoveComponent(neighborhood, 1U, driveX, driveY, driveZ,
                                                     settings.moveThreshold);
        const int stepZ = kern::latticeMoveComponent(neighborhood, 2U, driveX, driveY, driveZ,
                                                     settings.moveThreshold);

        agent.intent = Int4{agent.cell.x, agent.cell.y, agent.cell.z, 0};
        if (worldBuilds(settings.worldMode)) {
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
                const auto [faceX, faceZ] = constructionFacing(settings, aimDriveX, aimDriveZ);
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
            // One outcome per agent per step, filed under the first test the
            // attempt fails. Mirrors the same chain in lattice_step.comp; the
            // construction parity probe compares the counters as well as the
            // blocks, so a reason recorded differently is a failure and not a
            // difference of opinion.
            // LatticeBuildOutcomeCount means "not decided here": the agent
            // placed a bid and the resolve loop will say whether it won.
            std::uint32_t outcome = kern::LatticeBuildOutcomeCount;
            if (agent.signal.z > 0.0F) {
                outcome = kern::LatticeBuildCooling;
            } else if (agent.signal.y <= settings.buildThreshold) {
                outcome = kern::LatticeBuildUnwilling;
            } else {
                const auto [faceX, faceZ] = constructionFacing(settings, aimDriveX, aimDriveZ);
                const int buildX = agent.cell.x + faceX;
                const int buildZ = agent.cell.z + faceZ;
                if (faceX == 0 && faceZ == 0) {
                    outcome = kern::LatticeBuildNoFacing;
                } else if (buildX < 0 || buildZ < 0 ||
                           buildX >= static_cast<int>(settings.latticeWidth) ||
                           buildZ >= static_cast<int>(settings.latticeDepth)) {
                    outcome = kern::LatticeBuildOffLattice;
                } else {
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
                    const std::uint32_t foundation = constructionLocalFoundation(
                        worldStructures, settings, buildX, buildY, buildZ);
                    if (hasStructure(worldStructures, settings, buildX, buildY, buildZ)) {
                        outcome = kern::LatticeBuildBlocked;
                    } else if (!supported) {
                        outcome = kern::LatticeBuildUnsupported;
                    } else if (static_cast<std::uint32_t>(buildY) >=
                               foundation + std::max(settings.constructionHeightLead, 1U)) {
                        outcome = kern::LatticeBuildAboveFrontier;
                    } else if (worldOccupancy[target] != kern::LatticeNoOccupant) {
                        outcome = kern::LatticeBuildInTheWay;
                    } else {
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
            if (!population.buildOutcomes.empty() && outcome < kern::LatticeBuildOutcomeCount) {
                ++population.buildOutcomes[static_cast<std::size_t>(world) *
                                               kern::LatticeBuildOutcomeCount +
                                           outcome];
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
                if (!worldBuilds(settings.worldMode) || movedX != 0 || movedZ != 0) {
                    agent.cell.w = static_cast<std::int32_t>(
                        kern::latticeNeighborIndex(movedX, movedY, movedZ));
                }
                agent.cell.x = agent.intent.x;
                agent.cell.y = agent.intent.y;
                agent.cell.z = agent.intent.z;
                moved = true;
            } else {
                refused = !worldBuilds(settings.worldMode) ||
                          population.claims[worldBase + wanted] >= 0;
            }
        }
        // The flag the next step reads back as a self input, so a policy can
        // notice it is stuck without having to infer it from the neighbourhood.
        agent.intent = Int4{agent.cell.x, agent.cell.y, agent.cell.z, refused ? 1 : 0};
        // And how long it has been standing still, which the refusal flag cannot
        // say: an agent that never asked to move was never refused.
        agent.memory.z = moved ? 0.0F : agent.memory.z + 1.0F;

        if (worldBuilds(settings.worldMode) && agent.beacon.x >= 0 &&
            !worldStructures.empty()) {
            const std::uint32_t target =
                kern::latticeCellIndex(agent.beacon.x, agent.beacon.y, agent.beacon.z,
                                       settings.latticeWidth, settings.latticeHeight);
            bool placed = false;
            if (population.claims[worldBase + target] ==
                    kern::latticeBuildClaim(index,
                                            static_cast<std::uint32_t>(population.agents.size())) &&
                worldStructures[target] == kern::LatticeNoStructure) {
                worldStructures[target] = static_cast<std::int32_t>(index) + 1;
                agent.signal.z = static_cast<float>(settings.buildIntervalTicks);
                agent.signal.w = 1.0F;
                agent.metrics.y += 1.0F;
                placed = true;
            }
            if (!population.buildOutcomes.empty()) {
                ++population.buildOutcomes[static_cast<std::size_t>(world) *
                                               kern::LatticeBuildOutcomeCount +
                                           (placed ? kern::LatticeBuildPlaced
                                                   : kern::LatticeBuildContested)];
            }
        }

        agent.metrics.z +=
            (moved ? 1.0F : 0.0F) + settings.fitness.signalCostFactor * agent.signal.x;
        if (settings.worldMode == WorldMode::Harvest) {
            // Mirrors the harvest block of lattice_resolve.comp: pick up at the
            // resource, score on the floor.
            const std::uint32_t hash =
                kern::latticeResourceHash(world, settings.beaconSeed);
            const int resourceX = kern::latticeResourceX(hash, settings.latticeWidth);
            const int resourceY =
                kern::latticeResourceY(settings.resourceHeight, settings.latticeHeight);
            const int resourceZ =
                kern::latticeResourceZ(hash, settings.latticeWidth, settings.latticeDepth);
            const std::uint32_t distance = kern::latticeStepDistance(
                static_cast<std::uint32_t>(settings.neighborhood), resourceX - agent.cell.x,
                resourceY - agent.cell.y, resourceZ - agent.cell.z);
            agent.metrics.x = std::max(agent.metrics.x,
                                       kern::latticeNearness(distance, maximumDistance));
            if (agent.memory.w <= 0.0F) {
                if (kern::latticeBeaconReached(distance, settings.beaconContactRadius)) {
                    agent.memory.w = 1.0F;
                }
            } else if (agent.cell.y == 0) {
                agent.memory.w = 0.0F;
                agent.metrics.w += 1.0F;
            }
        } else if (settings.worldMode == WorldMode::Construction) {
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
