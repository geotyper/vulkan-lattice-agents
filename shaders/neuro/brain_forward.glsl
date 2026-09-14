#ifndef VKEXP_BRAIN_FORWARD_GLSL
#define VKEXP_BRAIN_FORWARD_GLSL

// One forward pass of the network, layer by layer, integrating every hidden
// neuron and leaving its new state on the agent.
//
// Its counterpart is vkexp::neuro::evaluate, and the two are held together by
// the parity test rather than by being one piece of code: this one has to read a
// storage buffer and write into a struct the shader owns, which is the part that
// cannot be shared. Everything it does *with* those numbers -- where a weight
// lives, what a time constant is, how a neuron integrates -- comes from
// BrainKernel.inl, so the arithmetic is shared even though the plumbing is not.
//
// Must be included after the genome buffer is declared: it reads `weights[]`
// directly rather than taking it as a parameter, because GLSL has no way to pass
// an unsized buffer.

void brainForward(inout Agent agent, float sensed[BrainInputCapacity], uint base, uint inputCount,
                  uint outputCount, uint layers, uint neuronModel, float deltaTime,
                  out float produced[BrainOutputCapacity]) {
    const uint layerCount = brainHiddenLayerCount(layers);

    // The input vector, then each layer's output in turn. One scratch array
    // sized for the widest thing that can land in it, because GLSL cannot size
    // a local array from a runtime value and the evaluator has to do the same.
    float source[BrainInputCapacity];
    for (uint index = 0; index < inputCount; ++index) {
        source[index] = sensed[index];
    }
    uint sourceCount = inputCount;

    for (uint layer = 0; layer < layerCount; ++layer) {
        const uint width = brainHiddenLayerSize(layers, layer);
        const uint stateOffset = brainHiddenLayerStateOffset(layers, layer);
        const uint squash = brainLayerActivation(layers, layer);
        float hidden[BrainHiddenNeuronCapacity];
        for (uint neuron = 0; neuron < width; ++neuron) {
            float activation =
                weights[brainLayerBiasIndex(base, inputCount, layers, layer, neuron)];
            for (uint index = 0; index < sourceCount; ++index) {
                activation +=
                    weights[brainLayerWeightIndex(base, inputCount, layers, layer, neuron, index)] *
                    source[index];
            }
            // Where the time constant comes from is the only thing the model
            // changes, exactly as in vkexp::neuro::evaluate. Reactive pins it to
            // the step, which the shared integrator turns into a plain
            // assignment; gated recomputes it from what this layer reads,
            // through the same mapping the gene uses.
            const uint global = stateOffset + neuron;
            float timeConstant = deltaTime;
            if (neuronModel == NeuronModelTimeConstant || neuronModel == NeuronModelSpiking) {
                timeConstant = brainTimeConstant(weights[brainTimeConstantGeneIndex(
                    base, inputCount, layers, outputCount, global)]);
            } else if (neuronModel == NeuronModelGated) {
                float gate =
                    weights[brainGateBiasIndex(base, inputCount, layers, outputCount, layer,
                                               neuron)];
                for (uint index = 0; index < sourceCount; ++index) {
                    gate += weights[brainGateWeightIndex(base, inputCount, layers, outputCount,
                                                         layer, neuron, index)] *
                            source[index];
                }
                timeConstant = brainTimeConstant(gate);
            }
            float state = brainIntegrateNeuron(agentHiddenState(agent, global), activation,
                                               timeConstant, deltaTime);
            if (neuronModel == NeuronModelSpiking) {
                if (state >= 1.0) {
                    hidden[neuron] = 1.0;
                    state = 0.0;
                } else {
                    hidden[neuron] = 0.0;
                    if (state < -1.0) {
                        state = -1.0;
                    }
                }
            } else {
                hidden[neuron] = brainLayerActivate(squash, state);
            }
            agent.hidden[global >> 2u][global & 3u] = state;
        }
        for (uint neuron = 0; neuron < width; ++neuron) {
            source[neuron] = hidden[neuron];
        }
        sourceCount = width;
    }

    for (uint neuron = 0; neuron < outputCount; ++neuron) {
        float activation =
            weights[brainOutputBiasIndex(base, inputCount, layers, outputCount, neuron)];
        for (uint index = 0; index < sourceCount; ++index) {
            activation +=
                weights[brainOutputWeightIndex(base, inputCount, layers, neuron, index)] *
                source[index];
        }
        produced[neuron] = brainActivation(activation);
    }
}

#endif
