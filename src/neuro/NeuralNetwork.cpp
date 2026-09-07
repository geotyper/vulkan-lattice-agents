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
    const auto hiddenCount = static_cast<kernel::uint>(shape.hiddenCount);
    const auto outputCount = static_cast<kernel::uint>(shape.outputCount);
    constexpr kernel::uint base = 0; // one genome, so it starts at zero here

    std::array<float, Topology::hiddenCount> hidden{};
    for (kernel::uint neuron = 0; neuron < hiddenCount; ++neuron) {
        float activation =
            weights[kernel::brainHiddenBiasIndex(base, inputCount, hiddenCount, neuron)];
        for (kernel::uint input = 0; input < inputCount; ++input) {
            activation += weights[kernel::brainHiddenWeightIndex(base, inputCount, neuron, input)] *
                          inputs[input];
        }
        // Where the time constant comes from is the only thing the model
        // changes. Reactive pins it to the step, which the shared integrator
        // turns into a plain assignment; gated recomputes it from the inputs
        // through the same mapping the gene uses, so a constant gate is exactly
        // the fixed-time-constant neuron.
        float timeConstant = deltaTime;
        if (model == kernel::NeuronModelTimeConstant) {
            timeConstant = kernel::brainTimeConstant(weights[kernel::brainTimeConstantGeneIndex(
                base, inputCount, hiddenCount, outputCount, neuron)]);
        } else if (model == kernel::NeuronModelGated) {
            float gate = weights[kernel::brainGateBiasIndex(base, inputCount, hiddenCount,
                                                            outputCount, neuron)];
            for (kernel::uint input = 0; input < inputCount; ++input) {
                gate += weights[kernel::brainGateWeightIndex(base, inputCount, hiddenCount,
                                                             outputCount, neuron, input)] *
                        inputs[input];
            }
            timeConstant = kernel::brainTimeConstant(gate);
        }
        state[neuron] =
            kernel::brainIntegrateNeuron(state[neuron], activation, timeConstant, deltaTime);
        hidden[neuron] = std::tanh(state[neuron]);
    }

    Outputs outputs{};
    for (kernel::uint neuron = 0; neuron < outputCount; ++neuron) {
        float activation = weights[kernel::brainOutputBiasIndex(base, inputCount, hiddenCount,
                                                                outputCount, neuron)];
        for (kernel::uint hiddenIndex = 0; hiddenIndex < hiddenCount; ++hiddenIndex) {
            activation += weights[kernel::brainOutputWeightIndex(base, inputCount, hiddenCount,
                                                                 neuron, hiddenIndex)] *
                          hidden[hiddenIndex];
        }
        outputs[neuron] = std::tanh(activation);
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
