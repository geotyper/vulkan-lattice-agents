#include "vkexp/neuro/NeuralNetwork.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace vkexp::neuro {

Outputs evaluate(const std::span<const float> weights, const Inputs& inputs,
                 HiddenState& state, HiddenState& aux, const float deltaTime,
                 const kernel::uint model, const BrainShape shape) {
    if (!shape.fitsCapacity()) {
        throw std::invalid_argument("Neural-network shape exceeds genome capacity");
    }
    if (weights.size() < shape.weightCount()) {
        throw std::invalid_argument("Genome is shorter than the plan it is evaluated under");
    }
    const auto inputCount = static_cast<kernel::uint>(shape.inputCount);
    const auto outputCount = static_cast<kernel::uint>(shape.outputCount);
    const kernel::uint layers = shape.packedLayers();
    const kernel::uint layerCount = kernel::brainHiddenLayerCount(layers);
    constexpr kernel::uint base = 0; // one genome, so it starts at zero here

    // What the current layer reads. It starts as the input vector and becomes
    // each layer's output in turn, which is the whole of what makes a deep plan
    // work: one loop, and only the source changes. Copied rather than aliased
    // because the shader cannot alias either, and the two have to agree.
    std::array<float, Topology::inputCount> source{};
    kernel::uint sourceCount = inputCount;
    for (kernel::uint index = 0; index < inputCount; ++index) {
        source[index] = inputs[index];
    }

    for (kernel::uint layer = 0; layer < layerCount; ++layer) {
        const kernel::uint width = kernel::brainHiddenLayerSize(layers, layer);
        const kernel::uint stateOffset = kernel::brainHiddenLayerStateOffset(layers, layer);
        const kernel::uint squash = kernel::brainLayerActivation(layers, layer);
        std::array<float, Topology::hiddenNeuronCapacity> produced{};
        for (kernel::uint neuron = 0; neuron < width; ++neuron) {
            float activation =
                weights[kernel::brainLayerBiasIndex(base, inputCount, layers, layer, neuron)];
            for (kernel::uint index = 0; index < sourceCount; ++index) {
                activation += weights[kernel::brainLayerWeightIndex(base, inputCount, layers, layer,
                                                                    neuron, index)] *
                              source[index];
            }
            // Where the time constant comes from is the only thing the model
            // changes. Reactive pins it to the step, which the shared integrator
            // turns into a plain assignment; gated recomputes it from what the
            // layer reads, through the same mapping the gene uses, so a constant
            // gate is exactly the fixed-time-constant neuron.
            const kernel::uint global = stateOffset + neuron;
            float timeConstant = deltaTime;
            if (model == kernel::NeuronModelTimeConstant || model == kernel::NeuronModelSpiking) {
                timeConstant = kernel::brainTimeConstant(weights[kernel::brainTimeConstantGeneIndex(
                    base, inputCount, layers, outputCount, global)]);
            } else if (model == kernel::NeuronModelGated) {
                float gate = weights[kernel::brainGateBiasIndex(base, inputCount, layers,
                                                                outputCount, layer, neuron)];
                for (kernel::uint index = 0; index < sourceCount; ++index) {
                    gate += weights[kernel::brainGateWeightIndex(base, inputCount, layers,
                                                                 outputCount, layer, neuron,
                                                                 index)] *
                            source[index];
                }
                timeConstant = kernel::brainTimeConstant(gate);
            }
            state[global] =
                kernel::brainIntegrateNeuron(state[global], activation, timeConstant, deltaTime);
            if (model == kernel::NeuronModelSpiking || model == kernel::NeuronModelAdaptive) {
                // Spiking is the bump-free case of the same discharge, so both
                // models take the same call and differ only in what they feed
                // it. A spiking neuron has no auxiliary lane of its own: it is
                // handed a zero and writes one back.
                float excess = model == kernel::NeuronModelAdaptive ? aux[global] : 0.0F;
                float bump = 0.0F;
                float relax = deltaTime;
                if (model == kernel::NeuronModelAdaptive) {
                    bump = kernel::brainAdaptationBump(
                        weights[kernel::brainAdaptationGeneIndex(base, inputCount, layers,
                                                                 outputCount, global,
                                                                 kernel::BrainAdaptationBumpGene)]);
                    relax = kernel::brainTimeConstant(weights[kernel::brainAdaptationGeneIndex(
                        base, inputCount, layers, outputCount, global,
                        kernel::BrainAdaptationDecayGene)]);
                }
                produced[neuron] =
                    kernel::brainDischarge(state[global], excess, bump, relax, deltaTime);
                aux[global] = excess;
            } else {
                produced[neuron] = kernel::brainLayerActivate(squash, state[global], sourceCount);
            }
        }
        for (kernel::uint neuron = 0; neuron < width; ++neuron) {
            source[neuron] = produced[neuron];
        }
        sourceCount = width;
    }

    Outputs outputs{};
    for (kernel::uint neuron = 0; neuron < outputCount; ++neuron) {
        float activation =
            weights[kernel::brainOutputBiasIndex(base, inputCount, layers, outputCount, neuron)];
        for (kernel::uint index = 0; index < sourceCount; ++index) {
            activation +=
                weights[kernel::brainOutputWeightIndex(base, inputCount, layers, neuron, index)] *
                source[index];
        }
        outputs[neuron] = kernel::brainActivation(activation);
    }
    return outputs;
}

Outputs evaluate(const std::span<const float> weights, const Inputs& inputs,
                 const BrainShape shape) {
    HiddenState state{};
    HiddenState aux{};
    // Any positive step works: the reactive model assigns the activation
    // outright, so the value cannot reach the result.
    return evaluate(weights, inputs, state, aux, 1.0F, kernel::NeuronModelReactive, shape);
}


// A fresh genome, block by block, with each weight drawn at a width that depends
// on how many things its neuron adds up.
//
// One width for every gene was what this did before, and for a network read
// through a threshold it is the wrong shape of wrong. A neuron summing seventy
// eight inputs at a width of 0.55 reaches tanh's flat part on almost any input,
// and so does the output reading twenty of those -- so a population nobody has
// trained yet is not undecided, it is maximally decided, at random, on every
// tick. That is what made a fresh group spend its generation pivoting on the
// spot: not indecision, but saturation.
//
// So each weight block is drawn at `spread / sqrt(fan-in)`, which is the
// standard answer and the one that keeps a sum the same size whatever it sums.
// Biases are drawn narrow rather than zeroed: zero is the same starting neuron
// for everyone, and a population that begins identical has nothing for selection
// to sort. The time-constant genes keep the original width, because they are not
// weights -- each is read through a sigmoid onto a time constant, and narrowing
// them would pull every neuron toward the same middle rate.
//
// The evolution is free to leave this behind. Nothing here caps a weight: a
// lineage that needs to be decisive grows into it, which is the difference
// between a population that can become sure and one that starts that way.
Weights randomWeights(const BrainShape shape, std::mt19937& random, const bool fanIn,
                      const float spread) {
    if (!shape.fitsCapacity()) {
        throw std::invalid_argument("Neural-network shape exceeds genome capacity");
    }
    Weights weights = makeWeights(shape);
    const auto inputCount = static_cast<kernel::uint>(shape.inputCount);
    const auto outputCount = static_cast<kernel::uint>(shape.outputCount);
    const kernel::uint layers = shape.packedLayers();
    const kernel::uint layerCount = kernel::brainHiddenLayerCount(layers);
    // Biases follow the weights: narrowing one and not the other would leave the
    // bias deciding the neuron on its own.
    const float biasSpread = fanIn ? spread * 0.2F : spread;
    const auto blockWidth = [&](const kernel::uint sources) {
        if (!fanIn || sources == 0) {
            return spread;
        }
        return spread / std::sqrt(static_cast<float>(sources));
    };

    const auto draw = [&](const float width) {
        std::normal_distribution<float> distribution{0.0F, width};
        return distribution(random);
    };

    // The forward block and the gate block that mirrors it, layer by layer. Both
    // are written in one pass because they are the same shape reading the same
    // sources, so a gate drawn at a different width would be a second opinion
    // held to a different standard.
    for (kernel::uint layer = 0; layer < layerCount; ++layer) {
        const kernel::uint size = kernel::brainHiddenLayerSize(layers, layer);
        const kernel::uint sources = kernel::brainLayerSourceCount(inputCount, layers, layer);
        const float width = blockWidth(sources);
        for (kernel::uint neuron = 0; neuron < size; ++neuron) {
            for (kernel::uint source = 0; source < sources; ++source) {
                weights[kernel::brainLayerWeightIndex(0U, inputCount, layers, layer, neuron,
                                                      source)] = draw(width);
                weights[kernel::brainGateWeightIndex(0U, inputCount, layers, outputCount, layer,
                                                     neuron, source)] = draw(width);
            }
            weights[kernel::brainLayerBiasIndex(0U, inputCount, layers, layer, neuron)] =
                draw(biasSpread);
            weights[kernel::brainGateBiasIndex(0U, inputCount, layers, outputCount, layer,
                                               neuron)] = draw(biasSpread);
        }
    }

    const kernel::uint lastHidden = kernel::brainLastHiddenSize(layers);
    const float outputWidth = blockWidth(lastHidden);
    for (kernel::uint neuron = 0; neuron < outputCount; ++neuron) {
        for (kernel::uint hidden = 0; hidden < lastHidden; ++hidden) {
            weights[kernel::brainOutputWeightIndex(0U, inputCount, layers, neuron, hidden)] =
                draw(outputWidth);
        }
        weights[kernel::brainOutputBiasIndex(0U, inputCount, layers, outputCount, neuron)] =
            draw(biasSpread);
    }

    const kernel::uint neurons = kernel::brainHiddenNeuronCount(layers);
    for (kernel::uint neuron = 0; neuron < neurons; ++neuron) {
        weights[kernel::brainTimeConstantGeneIndex(0U, inputCount, layers, outputCount, neuron)] =
            draw(spread);
        // The adaptation pair, always at the full spread like the time constant
        // beside it: both enter their range through a squash, so narrowing them
        // for a fan-in policy would narrow a range and not a sum.
        for (kernel::uint gene = 0; gene < kernel::BrainAdaptationGeneCount; ++gene) {
            weights[kernel::brainAdaptationGeneIndex(0U, inputCount, layers, outputCount, neuron,
                                                     gene)] = draw(spread);
        }
    }
    return weights;
}

} // namespace vkexp::neuro
