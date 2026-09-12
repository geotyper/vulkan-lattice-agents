#include "vkexp/simulation/CpuLattice.hpp"

#include "vkexp/lattice/LatticeKernel.hpp"
#include "vkexp/simulation/LatticeSensors.hpp"

#include <algorithm>
#include <vector>

namespace vkexp {
namespace {

namespace brain = neuro::kernel;
namespace kern = lattice::kernel;

[[nodiscard]] bool sameCell(const Int4& first, const Int4& second) {
    return first.x == second.x && first.y == second.y && first.z == second.z;
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
            weights, sampleAgentInputs(agent, signals, worldOccupancy, settings), hidden,
            settings.deltaTime, static_cast<brain::uint>(settings.neuronModel), shape);
        for (std::size_t neuron = 0; neuron < neuronCount; ++neuron) {
            setAgentHiddenState(agent, neuron, hidden[neuron]);
        }

        agent.signal.x = std::clamp(output[brain::BrainSignalIntensityOutput], 0.0F, 1.0F);
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
        const std::uint32_t wanted = kern::latticeCellIndex(wantedX, wantedY, wantedZ,
                                                            settings.latticeWidth,
                                                            settings.latticeHeight);
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
                agent.cell.w = static_cast<std::int32_t>(kern::latticeNeighborIndex(
                    agent.intent.x - agent.cell.x, agent.intent.y - agent.cell.y,
                    agent.intent.z - agent.cell.z));
                agent.cell.x = agent.intent.x;
                agent.cell.y = agent.intent.y;
                agent.cell.z = agent.intent.z;
                moved = true;
            } else {
                refused = true; // lost the cell to a lower-numbered agent
            }
        }
        // The flag the next step reads back as a self input, so a policy can
        // notice it is stuck without having to infer it from the neighbourhood.
        agent.intent = Int4{agent.cell.x, agent.cell.y, agent.cell.z, refused ? 1 : 0};

        agent.metrics.z += (moved ? 1.0F : 0.0F) +
                           settings.fitness.signalCostFactor * agent.signal.x;
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

float agentFitness(const AgentState& agent, const FitnessWeights& weights) {
    return kern::latticeTrialFitness(agent.metrics.x, agent.metrics.y, agent.metrics.z,
                                     agent.metrics.w, weights.trackingReward,
                                     weights.objectiveBonus, weights.motorCostWeight,
                                     weights.refusalPenalty);
}

} // namespace vkexp
