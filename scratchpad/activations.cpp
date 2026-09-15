// What the squash actually does to the numbers, measured on real trajectories.
//
// Three questions, none of which the fitness column can answer:
//   * how big the pre-activation z gets per hidden layer, since sine is only a
//     periodic function if z covers more than a period -- below pi/2 it is a
//     smooth non-saturating one and none of the objections to it apply;
//   * what the hidden layer's values look like once squashed, since a uniform
//     phase makes sine pile up at +-1 rather than near zero;
//   * what the sum feeding the output tanh looks like, and how often the
//     decision lands in the band that means "walk".
// And one about behaviour: how many ticks in a row an agent keeps the same
// decision, which is the only way to ask whether a non-monotone hidden layer
// costs an agent its ability to hold one.
#include "vkexp/lattice/LatticeWorld.hpp"
#include "vkexp/neuro/BrainKernel.hpp"
#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/simulation/CpuLattice.hpp"
#include "vkexp/simulation/LatticeSensors.hpp"
#include "vkexp/simulation/LatticeTypes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace bk = vkexp::neuro::kernel;
namespace lk = vkexp::lattice::kernel;

namespace {

struct Tally {
    std::vector<float> values;
    void add(const float value) { values.push_back(value); }
    [[nodiscard]] float quantile(const double fraction) {
        if (values.empty()) {
            return 0.0F;
        }
        std::sort(values.begin(), values.end());
        const auto at = static_cast<std::size_t>(fraction * double(values.size() - 1));
        return values[at];
    }
    [[nodiscard]] double fractionAbove(const float bound) const {
        if (values.empty()) {
            return 0.0;
        }
        const auto count = std::count_if(values.begin(), values.end(),
                                         [&](const float v) { return std::abs(v) > bound; });
        return double(count) / double(values.size());
    }
    [[nodiscard]] double meanMagnitude() const {
        double sum = 0.0;
        for (const float v : values) {
            sum += std::abs(double(v));
        }
        return values.empty() ? 0.0 : sum / double(values.size());
    }
};

// The forward pass again, with the intermediates kept. Cross-checked against
// vkexp::neuro::evaluate on every call, so this is an instrument reading the
// real network rather than a second opinion about it.
struct Probe {
    std::vector<Tally> preActivation; // per hidden layer
    std::vector<Tally> activation;    // per hidden layer, after the squash
    Tally outputSum;                  // the argument of the output tanh, action slot
    Tally turnSum;                    // the same for the first turn vote
};

void instrument(const std::span<const float> weights, const vkexp::neuro::Inputs& inputs,
                vkexp::neuro::HiddenState state, const float deltaTime, const bk::uint model,
                const vkexp::neuro::BrainShape shape, Probe& probe,
                const vkexp::neuro::Outputs& truth) {
    const auto inputCount = static_cast<bk::uint>(shape.inputCount);
    const auto outputCount = static_cast<bk::uint>(shape.outputCount);
    const bk::uint layers = shape.packedLayers();
    const bk::uint layerCount = bk::brainHiddenLayerCount(layers);

    std::vector<float> source(inputs.begin(), inputs.end());
    bk::uint sourceCount = inputCount;
    for (bk::uint layer = 0; layer < layerCount; ++layer) {
        const bk::uint width = bk::brainHiddenLayerSize(layers, layer);
        const bk::uint stateOffset = bk::brainHiddenLayerStateOffset(layers, layer);
        const bk::uint squash = bk::brainLayerActivation(layers, layer);
        std::vector<float> produced(width, 0.0F);
        for (bk::uint neuron = 0; neuron < width; ++neuron) {
            float z = weights[bk::brainLayerBiasIndex(0U, inputCount, layers, layer, neuron)];
            for (bk::uint index = 0; index < sourceCount; ++index) {
                z += weights[bk::brainLayerWeightIndex(0U, inputCount, layers, layer, neuron,
                                                       index)] *
                     source[index];
            }
            const bk::uint global = stateOffset + neuron;
            float timeConstant = deltaTime;
            if (model == bk::NeuronModelTimeConstant || model == bk::NeuronModelSpiking) {
                timeConstant = bk::brainTimeConstant(
                    weights[bk::brainTimeConstantGeneIndex(0U, inputCount, layers, outputCount,
                                                           global)]);
            } else if (model == bk::NeuronModelGated) {
                float gate = weights[bk::brainGateBiasIndex(0U, inputCount, layers, outputCount,
                                                            layer, neuron)];
                for (bk::uint index = 0; index < sourceCount; ++index) {
                    gate += weights[bk::brainGateWeightIndex(0U, inputCount, layers, outputCount,
                                                             layer, neuron, index)] *
                            source[index];
                }
                timeConstant = bk::brainTimeConstant(gate);
            }
            state[global] = bk::brainIntegrateNeuron(state[global], z, timeConstant, deltaTime);
            // The integrated state is what the squash actually reads, so that is
            // what "pre-activation" means here.
            probe.preActivation[layer].add(state[global]);
            if (model == bk::NeuronModelSpiking) {
                produced[neuron] = state[global] >= 1.0F ? 1.0F : 0.0F;
                if (state[global] >= 1.0F) {
                    state[global] = 0.0F;
                } else if (state[global] < -1.0F) {
                    state[global] = -1.0F;
                }
            } else {
                produced[neuron] = bk::brainLayerActivate(squash, state[global], sourceCount);
            }
            probe.activation[layer].add(produced[neuron]);
        }
        source.assign(produced.begin(), produced.end());
        sourceCount = width;
    }

    for (bk::uint neuron = 0; neuron < outputCount; ++neuron) {
        float z = weights[bk::brainOutputBiasIndex(0U, inputCount, layers, outputCount, neuron)];
        for (bk::uint index = 0; index < sourceCount; ++index) {
            z += weights[bk::brainOutputWeightIndex(0U, inputCount, layers, neuron, index)] *
                 source[index];
        }
        if (neuron == bk::BrainActionOutput) {
            probe.outputSum.add(z);
        }
        if (neuron == bk::BrainTurnOutput) {
            probe.turnSum.add(z);
        }
        const float produced = bk::brainActivation(z);
        if (std::abs(produced - truth[neuron]) > 2.0e-4F) {
            std::printf("  ПРОБА РАСХОДИТСЯ с evaluate на выходе %u: %g против %g\n", neuron,
                        double(produced), double(truth[neuron]));
        }
    }
}

const char* decisionName(const int decision) {
    switch (decision) {
    case 0:
        return "walk";
    case 1:
        return "build";
    case 2:
        return "left";
    default:
        return "right";
    }
}

// biasOffset shifts every hidden bias down, which for a rectified layer is a
// firing threshold: the neuron is silent unless its input clears the offset. It
// is how this asks whether sparsity alone accounts for the spiking model,
// without adopting the spiking neuron. Zero leaves the draw as it was.
void run(const char* label, const std::array<std::uint32_t, 3>& squashes,
         const std::array<std::uint32_t, 3>& widths, const vkexp::NeuronModel model,
         const float biasOffset = 0.0F) {
    vkexp::SimulationStep settings{};
    settings.worldMode = vkexp::WorldMode::Construction;
    settings.latticeWidth = 32;
    settings.latticeHeight = 16;
    settings.latticeDepth = 32;
    settings.neuronModel = model;
    settings.hiddenLayers = widths;
    settings.hiddenActivation = squashes;

    const vkexp::lattice::PopulationLayout layout{64, 12, 1};
    const vkexp::neuro::BrainShape brain = vkexp::resolvedBrain(settings);
    const auto stride = static_cast<std::uint32_t>(brain.weightCount());

    std::mt19937 random{0xC0FFEEU};
    std::vector<float> weights;
    weights.reserve(std::size_t(stride) * layout.genomeCount);
    for (std::uint32_t genome = 0; genome < layout.genomeCount; ++genome) {
        vkexp::neuro::Weights drawn = vkexp::neuro::randomWeights(brain, random, false);
        if (biasOffset != 0.0F) {
            const bk::uint layers = brain.packedLayers();
            for (std::size_t layer = 0; layer < brain.hiddenLayerCount(); ++layer) {
                for (std::size_t neuron = 0; neuron < brain.hiddenLayer(layer); ++neuron) {
                    drawn[bk::brainLayerBiasIndex(0U, static_cast<bk::uint>(brain.inputCount),
                                                  layers, static_cast<bk::uint>(layer),
                                                  static_cast<bk::uint>(neuron))] += biasOffset;
                }
            }
        }
        weights.insert(weights.end(), drawn.begin(), drawn.end());
    }

    std::vector<vkexp::AgentState> agents = vkexp::lattice::makeInitialAgents(settings, layout);
    const std::uint32_t cells = vkexp::latticeCellsPerWorld(settings);
    std::vector<std::int32_t> structures =
        vkexp::lattice::makeTerrain(settings, layout.worldCount());
    std::vector<std::int32_t> occupancy(std::size_t(cells) * layout.worldCount());
    vkexp::lattice::buildOccupancy(agents, settings, layout, occupancy);
    std::vector<std::int32_t> claims(occupancy.size());

    Probe probe;
    probe.preActivation.resize(brain.hiddenLayerCount());
    probe.activation.resize(brain.hiddenLayerCount());

    // One run length per agent, closed when its decision changes.
    std::vector<int> lastDecision(agents.size(), -1);
    std::vector<std::uint32_t> holding(agents.size(), 0);
    std::vector<std::uint32_t> holdLengths;
    std::map<int, std::uint64_t> decisionCounts;

    constexpr std::uint32_t steps = 300;
    for (std::uint32_t step = 0; step < steps; ++step) {
        std::vector<float> signals(agents.size());
        for (std::size_t index = 0; index < agents.size(); ++index) {
            signals[index] = agents[index].signal.x;
        }
        for (std::size_t index = 0; index < agents.size(); ++index) {
            const std::uint32_t world = vkexp::logicalWorldForAgent(
                static_cast<std::uint32_t>(index), layout.agentsPerWorld, layout.trialsPerGenome);
            const std::size_t base = std::size_t(world) * cells;
            const std::span<const std::int32_t> worldOccupancy{occupancy.data() + base, cells};
            const std::span<const std::int32_t> worldStructures{structures.data() + base, cells};
            const vkexp::neuro::Inputs inputs = vkexp::sampleAgentInputs(
                agents[index], signals, worldOccupancy, settings, worldStructures);
            const std::uint32_t genome =
                static_cast<std::uint32_t>(index) / layout.trialsPerGenome;
            const std::span<const float> genomeWeights{weights.data() + std::size_t(genome) * stride,
                                                       stride};
            vkexp::neuro::HiddenState state{};
            vkexp::neuro::HiddenState auxLane{};
            vkexp::neuro::HiddenState emittedLane{};
            for (std::size_t neuron = 0; neuron < brain.hiddenTotal(); ++neuron) {
                state[neuron] = vkexp::agentHiddenState(agents[index], neuron);
                auxLane[neuron] = vkexp::agentHiddenAux(agents[index], neuron);
                emittedLane[neuron] = vkexp::agentHiddenOut(agents[index], neuron);
            }
            vkexp::neuro::HiddenState truthState = state;
            const vkexp::neuro::Outputs truth = vkexp::neuro::evaluate(
                genomeWeights, inputs, truthState, auxLane, emittedLane, settings.deltaTime,
                static_cast<bk::uint>(settings.neuronModel), brain);
            // Only a sample: three hundred steps times seven hundred agents is
            // more numbers than the question needs, and the first world is as
            // representative as any other. The probe below carries its own copy
            // of the forward pass, which knows the three squashed models and not
            // the ones that fire, so it is skipped for those: their decisions
            // and their hold lengths still come from the real evaluator above.
            const bool probeable = settings.neuronModel == vkexp::NeuronModel::Reactive ||
                                   settings.neuronModel == vkexp::NeuronModel::TimeConstant ||
                                   settings.neuronModel == vkexp::NeuronModel::Gated;
            if (index < 16 && probeable) {
                instrument(genomeWeights, inputs, state, settings.deltaTime,
                           static_cast<bk::uint>(settings.neuronModel), brain, probe, truth);
            }

            const int turn = lk::latticeTurnStep(truth[bk::BrainTurnOutput],
                                                 truth[bk::BrainTurnOutput + 1],
                                                 settings.turnThreshold);
            int decision = 0;
            if (turn != 0) {
                decision = turn > 0 ? 3 : 2;
            } else if (truth[bk::BrainActionOutput] > settings.buildThreshold) {
                decision = 1;
            }
            ++decisionCounts[decision];
            if (decision == lastDecision[index]) {
                ++holding[index];
            } else {
                if (lastDecision[index] >= 0) {
                    holdLengths.push_back(holding[index]);
                }
                lastDecision[index] = decision;
                holding[index] = 1;
            }
        }
        vkexp::stepLatticeCpu({agents, occupancy, claims, weights, stride, layout.agentsPerWorld,
                               layout.trialsPerGenome, structures},
                              settings);
    }

    std::printf("\n=== %s ===\n", label);
    for (std::size_t layer = 0; layer < probe.preActivation.size(); ++layer) {
        Tally& z = probe.preActivation[layer];
        const Tally& a = probe.activation[layer];
        std::printf("  слой %zu: |z| сред %.2f, медиана %.2f, 99%% %.2f | доля |z|>pi/2 %.1f%%, "
                    ">pi %.1f%%, >2pi %.1f%%\n",
                    layer, z.meanMagnitude(), double(std::abs(z.quantile(0.5))),
                    double(std::abs(z.quantile(0.99))), 100.0 * z.fractionAbove(1.5708F),
                    100.0 * z.fractionAbove(3.1416F), 100.0 * z.fractionAbove(6.2832F));
        std::printf("           после squash: |h| сред %.2f, доля |h|>0.9 %.1f%%, "
                    "|h|<0.1 %.1f%%\n",
                    a.meanMagnitude(), 100.0 * a.fractionAbove(0.9F),
                    100.0 * (1.0 - a.fractionAbove(0.1F)));
    }
    std::printf("  сумма перед выходным tanh: действие |z| сред %.2f, >2 %.1f%% | поворот |z| "
                "сред %.2f, >2 %.1f%%\n",
                probe.outputSum.meanMagnitude(), 100.0 * probe.outputSum.fractionAbove(2.0F),
                probe.turnSum.meanMagnitude(), 100.0 * probe.turnSum.fractionAbove(2.0F));

    std::uint64_t total = 0;
    for (const auto& [decision, count] : decisionCounts) {
        total += count;
    }
    std::string breakdown;
    for (const auto& [decision, count] : decisionCounts) {
        breakdown += std::string{decisionName(decision)} + " " +
                     std::to_string(100 * count / total) + "%  ";
    }
    std::printf("  решения: %s\n", breakdown.c_str());

    double holdSum = 0.0;
    for (const std::uint32_t length : holdLengths) {
        holdSum += length;
    }
    std::sort(holdLengths.begin(), holdLengths.end());
    const double mean = holdLengths.empty() ? 0.0 : holdSum / double(holdLengths.size());
    const std::uint32_t median =
        holdLengths.empty() ? 0 : holdLengths[holdLengths.size() / 2];
    const std::uint32_t p99 =
        holdLengths.empty() ? 0 : holdLengths[holdLengths.size() * 99 / 100];
    std::printf("  держит решение: сред %.2f тика, медиана %u, 99%% %u, самое долгое %u\n", mean,
                median, p99, holdLengths.empty() ? 0 : holdLengths.back());
}

} // namespace

int main() {
    const std::array<std::uint32_t, 3> flat{35, 0, 0};
    run("35 tanh        (time)", {bk::BrainActivationTanh, 0, 0}, flat,
        vkexp::NeuronModel::TimeConstant);
    run("35 relu-unit -3 (time)", {bk::BrainActivationReluUnit, 0, 0}, flat,
        vkexp::NeuronModel::TimeConstant, -3.0F);
    run("35            (spiking)", {0, 0, 0}, flat, vkexp::NeuronModel::Spiking);
    run("35           (adaptive)", {0, 0, 0}, flat, vkexp::NeuronModel::Adaptive);
    run("35         (oscillator)", {0, 0, 0}, flat, vkexp::NeuronModel::Oscillator);
    return 0;
}
