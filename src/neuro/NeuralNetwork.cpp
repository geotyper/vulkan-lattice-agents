#include "vkexp/neuro/NeuralNetwork.hpp"

#include <cmath>
#include <stdexcept>

namespace vkexp::neuro {

Outputs evaluate(const std::span<const float, Topology::weightCount> weights, const Inputs& inputs,
                 HiddenState& state, const float deltaTime, const kernel::uint model,
                 const BrainShape shape) {
    if (!shape.fitsCapacity()) {
        throw std::invalid_argument("Neural-network shape exceeds genome capacity");
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
        std::array<float, Topology::hiddenCount> produced{};
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
            if (model == kernel::NeuronModelTimeConstant) {
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
            produced[neuron] = kernel::brainActivation(state[global]);
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

Outputs evaluate(const std::span<const float, Topology::weightCount> weights, const Inputs& inputs,
                 const BrainShape shape) {
    HiddenState state{};
    // Any positive step works: the reactive model assigns the activation
    // outright, so the value cannot reach the result.
    return evaluate(weights, inputs, state, 1.0F, kernel::NeuronModelReactive, shape);
}

} // namespace vkexp::neuro
