#include "vkexp/simulation/CpuLattice.hpp"

#include "vkexp/lattice/LatticeWorld.hpp"

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

// Mirrors constructionSupported in lattice_step.comp. There is no special case
// for height zero: the floor is a course of bedrock in the block field, so
// "something below me" is the whole of the rule.
[[nodiscard]] bool constructionSupported(const std::span<const std::int32_t> structures,
                                         const SimulationStep& settings, const int x, const int y,
                                         const int z, const std::uint32_t facing) {
    if (hasStructure(structures, settings, x, y - 1, z) ||
        hasStructure(structures, settings, x - 1, y, z) ||
        hasStructure(structures, settings, x + 1, y, z) ||
        hasStructure(structures, settings, x, y, z - 1) ||
        hasStructure(structures, settings, x, y, z + 1)) {
        return true;
    }
    // Holding the wall in front of your feet: an edge rather than a face, and
    // read through the facing, so a climber holds the wall it is looking at and
    // turning away lets go.
    return hasStructure(structures, settings, x + kern::latticeFacingX(facing), y - 1,
                        z + kern::latticeFacingZ(facing));
}

[[nodiscard]] bool constructionBlockSupported(const std::span<const std::int32_t> structures,
                                              const SimulationStep& settings, const int x,
                                              const int y, const int z) {
    if (hasStructure(structures, settings, x, y - 1, z)) {
        return true;
    }
    return settings.allowSideSupportedBlocks != 0U &&
           (hasStructure(structures, settings, x - 1, y, z) ||
            hasStructure(structures, settings, x + 1, y, z) ||
            hasStructure(structures, settings, x, y, z - 1) ||
            hasStructure(structures, settings, x, y, z + 1));
}

// Mirrors constructionLandingY in lattice_step.comp, including the -1 that says
// this column has no bottom at all.
[[nodiscard]] int constructionLandingY(const std::span<const std::int32_t> structures,
                                       const SimulationStep& settings, const int x, const int y,
                                       const int z, const std::uint32_t facing) {
    int landing = std::clamp(y, 0, static_cast<int>(settings.latticeHeight) - 1);
    while (landing >= 0 && !constructionSupported(structures, settings, x, landing, z, facing)) {
        --landing;
    }
    return landing;
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
        agent.signal.y = output[brain::BrainActionOutput];
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

        // Three commands and no more: turn, or spend the tick on the one thing
        // the action output asks for. Mirrors the same grammar in
        // lattice_step.comp -- a turn ends the tick, so pointing somewhere else
        // costs a tick per ninety degrees.
        const auto facing = static_cast<std::uint32_t>(agent.cell.w) % kern::LatticeFacingCount;
        // Two outputs that have to agree, read through the same dead zone as
        // everything else. Sign and not a left/right pair: the two turns are one
        // axis, and a network that had to learn "not both at once" would be
        // learning the encoding rather than the task.
        const int turn = kern::latticeTurnStep(output[brain::BrainTurnOutput],
                                               output[brain::BrainTurnOutput + 1],
                                               settings.turnThreshold);
        const bool turning = turn != 0;
        if (turning) {
            agent.cell.w = static_cast<std::int32_t>(kern::latticeTurn(facing, turn > 0));
        }
        const auto heading = static_cast<std::uint32_t>(agent.cell.w) % kern::LatticeFacingCount;

        agent.intent = Int4{agent.cell.x, agent.cell.y, agent.cell.z, 0};
        // One outcome per agent per step, filed under the first test the
        // attempt fails. Mirrors the same chain in lattice_step.comp; the
        // construction parity probe compares the counters as well as the
        // blocks, so a reason recorded differently is a failure and not a
        // difference of opinion.
        // LatticeBuildOutcomeCount means "not decided here": the agent placed a
        // build bid and the resolve loop will say whether it won.
        std::uint32_t outcome = kern::LatticeActionTurning;
        // Walk or build, on one output and one dead zone. In a world that builds
        // only the upper band means anything; in one that does not, the lower
        // band is the descent -- see the note where it is used.
        const int action = kern::latticeAxisStep(agent.signal.y, settings.buildThreshold);

        if (worldBuilds(settings.worldMode)) {
            int wantedX = agent.cell.x;
            int wantedY = agent.cell.y;
            int wantedZ = agent.cell.z;
            const int forwardX = kern::latticeFacingX(heading);
            const int forwardZ = kern::latticeFacingZ(heading);
            bool stepping = !turning;

            agent.beacon.x = -1;
            agent.beacon.y = -1;
            agent.beacon.z = -1;

            if (!turning && action > 0) {
                stepping = false;
                if (agent.signal.z > 0.0F) {
                    // Still on the cooldown from the last block. The tick is
                    // spent either way, so the agent walks it off rather than
                    // standing in place waiting for the counter.
                    outcome = kern::LatticeBuildCooling;
                    stepping = true;
                } else {
                    const int buildX = agent.cell.x + forwardX;
                    const int buildZ = agent.cell.z + forwardZ;
                    if (buildX < 0 || buildZ < 0 ||
                        buildX >= static_cast<int>(settings.latticeWidth) ||
                        buildZ >= static_cast<int>(settings.latticeDepth)) {
                        outcome = kern::LatticeBuildOffLattice;
                    } else {
                        int buildY = agent.cell.y;
                        bool supported = constructionBlockSupported(worldStructures, settings,
                                                                    buildX, buildY, buildZ);
                        if (!supported && settings.allowSideSupportedBlocks != 0U && buildY > 0) {
                            --buildY;
                            supported = constructionBlockSupported(worldStructures, settings,
                                                                   buildX, buildY, buildZ);
                        }
                        const std::uint32_t target = kern::latticeCellIndex(
                            buildX, buildY, buildZ, settings.latticeWidth, settings.latticeHeight);
                        if (hasStructure(worldStructures, settings, buildX, buildY, buildZ)) {
                            outcome = kern::LatticeBuildBlocked;
                        } else if (!supported) {
                            outcome = kern::LatticeBuildUnsupported;
                        } else if (kern::latticeWorldFrontier(
                                       static_cast<std::uint32_t>(settings.worldMode)) &&
                                   static_cast<std::uint32_t>(buildY) >=
                                       constructionLocalFoundation(worldStructures, settings,
                                                                   buildX, buildY, buildZ) +
                                           std::max(settings.constructionHeightLead, 1U)) {
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
                            outcome = kern::LatticeBuildOutcomeCount; // the resolve loop decides
                        }
                    }
                }
            }

            if (stepping) {
                wantedX += forwardX;
                wantedZ += forwardZ;
                std::uint32_t refusal = kern::LatticeBuildOutcomeCount; // no refusal yet
                if (!kern::latticeInBounds(wantedX, wantedY, wantedZ, settings.latticeWidth,
                                           settings.latticeHeight, settings.latticeDepth)) {
                    refusal = kern::LatticeActionEdge;
                } else if (hasStructure(worldStructures, settings, wantedX, wantedY, wantedZ)) {
                    // A wall in front is climbed, not stepped onto: the agent
                    // rises one level in its own column and holds the block
                    // with its feet. The step after that carries it over the
                    // top.
                    wantedX = agent.cell.x;
                    wantedZ = agent.cell.z;
                    ++wantedY;
                    if (!kern::latticeInBounds(wantedX, wantedY, wantedZ, settings.latticeWidth,
                                               settings.latticeHeight, settings.latticeDepth) ||
                        hasStructure(worldStructures, settings, wantedX, wantedY, wantedZ)) {
                        refusal = kern::LatticeActionCeiling;
                    }
                } else {
                    const int landing = constructionLandingY(worldStructures, settings, wantedX,
                                                             wantedY, wantedZ, heading);
                    // The far side of a chasm edge, told apart from the edge of
                    // the lattice: one is a wall to turn away from and the other
                    // is a gap to build across.
                    if (landing < 0) {
                        refusal = kern::LatticeActionVoid;
                    }
                    wantedY = landing;
                }
                if (refusal < kern::LatticeBuildOutcomeCount) {
                    wantedX = agent.cell.x;
                    wantedY = agent.cell.y;
                    wantedZ = agent.cell.z;
                    agent.intent.w = 1;
                    outcome = refusal;
                } else {
                    outcome = kern::LatticeActionWalking;
                }
            }

            // Gravity, applied to wherever the tick left the agent: a climber
            // that turned away from the wall it was holding has let go of it.
            if (!constructionSupported(worldStructures, settings, wantedX, wantedY, wantedZ,
                                       heading)) {
                const int landing = constructionLandingY(worldStructures, settings, wantedX,
                                                         wantedY - 1, wantedZ, heading);
                if (landing >= 0) {
                    wantedY = landing;
                }
            }

            if (wantedX != agent.cell.x || wantedY != agent.cell.y || wantedZ != agent.cell.z) {
                const std::uint32_t wanted = kern::latticeCellIndex(
                    wantedX, wantedY, wantedZ, settings.latticeWidth, settings.latticeHeight);
                if (worldOccupancy[wanted] == kern::LatticeNoOccupant &&
                    !hasStructure(worldStructures, settings, wantedX, wantedY, wantedZ)) {
                    agent.intent = Int4{wantedX, wantedY, wantedZ, 0};
                    population.claims[worldBase + wanted] = kern::latticeBetterClaim(
                        population.claims[worldBase + wanted], static_cast<std::int32_t>(index));
                } else if (!hasStructure(worldStructures, settings, wantedX, wantedY, wantedZ)) {
                    agent.intent.w = 1; // another agent, not a wall
                    outcome = kern::LatticeActionCrowded;
                }
            }
        } else if (!turning) {
            // A world with no gravity and nothing to build in: the action
            // output still has two ends, so it spends them on the vertical the
            // body frame otherwise cannot reach. Above the threshold rises,
            // below the negative threshold sinks, and the band between them is
            // the step forward.
            const int stepX = action == 0 ? kern::latticeFacingX(heading) : 0;
            const int stepY = action;
            const int stepZ = action == 0 ? kern::latticeFacingZ(heading) : 0;
            const int wantedX = agent.cell.x + stepX;
            const int wantedY = agent.cell.y + stepY;
            const int wantedZ = agent.cell.z + stepZ;
            outcome = kern::LatticeActionWalking;
            if (!kern::latticeInBounds(wantedX, wantedY, wantedZ, settings.latticeWidth,
                                       settings.latticeHeight, settings.latticeDepth)) {
                agent.intent.w = 1; // walked into the edge of the lattice
                outcome = kern::LatticeActionEdge;
            } else {
                const std::uint32_t wanted = kern::latticeCellIndex(
                    wantedX, wantedY, wantedZ, settings.latticeWidth, settings.latticeHeight);
                if (!kern::latticeCellEnterable(worldOccupancy[wanted])) {
                    agent.intent.w = 1; // somebody was already standing there
                    outcome = kern::LatticeActionCrowded;
                } else {
                    agent.intent = Int4{wantedX, wantedY, wantedZ, 0};
                    std::int32_t& claim = population.claims[worldBase + wanted];
                    claim = kern::latticeBetterClaim(claim, static_cast<std::int32_t>(index));
                }
            }
        }

        if (!population.buildOutcomes.empty() && outcome < kern::LatticeBuildOutcomeCount) {
            ++population.buildOutcomes[static_cast<std::size_t>(world) *
                                           kern::LatticeBuildOutcomeCount +
                                       outcome];
        }
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
                // cell.w is untouched: facing is state the agent owns and only
                // a turn changes it. A move that rewrote it would mean an agent
                // could never walk one way while looking another, and climbing
                // a wall is exactly that.
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
        if (worldHarvests(settings.worldMode)) {
            // Mirrors the harvest block of lattice_resolve.comp: pick up at the
            // resource, score on the floor.
            const Int4 resource = lattice::resourceCell(settings, world);
            const std::uint32_t distance = kern::latticeStepDistance(
                static_cast<std::uint32_t>(settings.neighborhood), resource.x - agent.cell.x,
                resource.y - agent.cell.y, resource.z - agent.cell.z);
            agent.metrics.x = std::max(agent.metrics.x,
                                       kern::latticeNearness(distance, maximumDistance));
            if (agent.memory.w <= 0.0F) {
                if (kern::latticeBeaconReached(distance, settings.beaconContactRadius)) {
                    agent.memory.w = 1.0F;
                }
            } else if (kern::latticeGroundColumn(agent.cell.x, latticeGroundWidth(settings)) &&
                       agent.cell.y <= 1) {
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
