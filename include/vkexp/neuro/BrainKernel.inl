// The network preset: how many sensors feed the brain, how wide it is, what its
// outputs mean, and how a flat genome is addressed.
//
// Compiled twice -- as C++ through BrainKernel.hpp and as GLSL through
// shaders/neuro/brain_kernel.glsl -- so both sides build the same network from
// the same declaration. Adding a sensor changes the counts here and nothing
// else: the construction below is written once and derives every offset.
//
// Same common-subset rules as LatticeKernel.inl: VKEXP_BRAIN_FN in front of
// every function, only uint/bool across boundaries, no standard library. Note
// that GLSL reserves more words than C++ does -- `input`, `output`, `layout`,
// `filter`, `active` and friends cannot be identifiers here.
//
// What this file is *not* is where the world is described. It knows the input
// vector has a block for the neighbourhood and how wide that block is; what a
// neighbour is, and how one is reached, is LatticeKernel.inl's business. That
// split is what lets the lattice change shape without the genome changing
// length.

// --- preset: change these when the sensor suite or brain width changes -------

// The Moore neighbourhood, one slot per surrounding cell. Restated rather than
// included from LatticeKernel.inl because the two kernels compile into separate
// namespaces on the C++ side and GLSL has no namespaces at all; testLatticeBrain
// asserts the two agree, which is the same arrangement the 2D build used for the
// body radius it shared with the scenario kernel.
const uint BrainNeighborCount = 26u;
// Something is standing there, the lattice ends there, a block stands there, and
// how loudly its occupant is signalling. The last channel is the whole of
// agent-to-agent perception: an agent reads its neighbour's broadcast, never its
// state.
//
// The wall and the block used to share one channel, and an agent could not tell
// them apart. That is a real difference: a wall is a cell that can never hold a
// block and a block is a cell that already does, and the two call for opposite
// responses -- turn away from one, climb or build beside the other. The
// refusal counters show a quarter of all genuine build attempts aimed at a
// block and two fifths aimed out of the world, which is what being unable to
// tell them apart costs.
const uint BrainNeighborChannels = 4u;
// What the world's task tells the agent. Six slots, read differently by each
// world and named once here rather than in three sensing functions:
//
//   beacon        0-2 unit vector to the beacon, 3 nearness, 4-5 unused
//   construction  0 own height, 1 may build now, 2 built last step,
//                 3 standing on something, 4-5 unused
//   harvest       0-2 unit vector to the resource, 3 nearness,
//                 4 may build now, 5 carrying a load
//
// A world that uses fewer leaves the rest at zero. Two spare slots across two
// modes is a cheaper price than a block whose width depends on the mode, which
// would make a population unloadable across worlds -- the same trade the
// neighbourhood block already makes for the movement setting.
const uint BrainBeaconInputCount = 6u;
// Whether the last action was refused, and how long the agent has been standing
// still. The stillness channel is a ramp rather than a flag, and that is the
// whole of why it is worth a slot. See LatticeStillnessSpan.
//
// The heading used to be here as a unit vector, because a move was chosen in
// lattice axes and the network had no other way to know which way it was
// already going. In a body frame it is not information at all: the agent faces
// forward by definition, and everything it senses is measured from there, so
// its absolute orientation is unobservable and cannot matter. Three slots that
// could only ever have carried a constant.
const uint BrainSelfInputCount = 2u;
const uint BrainRecurrentCount = 2u; // memory cells, fed back as inputs

// Two actuators and a voice, where there used to be six actuators.
//
// Turn is one signed drive with a dead zone, not two positive ones. The dead
// zone is the point: "keep going straight" has to be what an agent gets by not
// committing, exactly as "stay put" used to be. Two positive outputs would make
// straight a conjunction -- both quiet at once -- which is harder to hold and
// drifts. They would also need a rule for what happens when both fire, and an
// arbitrary rule is one more thing two implementations have to agree on.
const uint BrainTurnOutputCount = 1u;
// One output, two actions, and no third: under the threshold the agent steps
// forward, over it the agent builds in front of itself. Standing still is not
// in the set at all, and that is deliberate. It used to be reachable by not
// committing on any axis, which made "the network declined to act" and "the
// network was not asked" the same state -- invisible in the counters, and
// exactly what agents that froze were doing. Now doing nothing is walking.
//
// It also puts the build/step alternation on a single value crossing a
// threshold, which is what a tower is: build, climb, build, climb. That
// sequence had to be coordinated across five continuous outputs before.
const uint BrainActionOutputCount = 1u;
const uint BrainSignalOutputCount = 1u;
const uint BrainActuatorOutputCount =
    BrainTurnOutputCount + BrainActionOutputCount + BrainSignalOutputCount;

// Hidden neurons in total, across however many layers there are, and how many
// layers there may be. Both are compile-time because both size arrays: the
// shader's scratch buffers, and the state block on the agent record. A plan
// chooses how to spend the total; it cannot raise it.
//
// 32 rather than 20, and three layers rather than one. The cost of the headroom
// is paid in the genome stride, which is sized for the widest plan the capacity
// allows -- one 32-wide layer -- so a run using fewer neurons carries weights it
// never reads. That is the same trade the gate block already makes, and for the
// same reason: one buffer size and one genome length means a population stays
// loadable across plans, and comparing two plans stays possible at all.
const uint BrainHiddenNeuronCapacity = 32u;
const uint BrainHiddenLayerCapacity = 3u;

// The one hidden layer this network had before plans existed, and still what a
// world means when it does not say otherwise. Separate from the capacity on
// purpose: raising how many neurons there *may* be must not quietly widen every
// world's brain, which is exactly what sharing one constant would have done.
const uint BrainDefaultHiddenWidth = 20u;

// --- derived layout: never edited by hand ------------------------------------

const uint BrainNeighborBlockSize = BrainNeighborCount * BrainNeighborChannels;

// The neighbourhood goes first and keeps its width whatever the movement
// setting is: "how many directions can I see" and "how many can I walk" are
// separate claims, and only the second changes what the brain has to solve.
const uint BrainNeighborOffset = 0u;
const uint BrainBeaconOffset = BrainNeighborOffset + BrainNeighborBlockSize;
const uint BrainSelfOffset = BrainBeaconOffset + BrainBeaconInputCount;
const uint BrainRecurrentInputOffset = BrainSelfOffset + BrainSelfInputCount;
const uint BrainInputCapacity = BrainRecurrentInputOffset + BrainRecurrentCount;

const uint BrainTurnOutput = 0u;
const uint BrainActionOutput = BrainTurnOutputCount;
const uint BrainSignalIntensityOutput = BrainActionOutput + BrainActionOutputCount;
const uint BrainRecurrentOutputOffset = BrainActuatorOutputCount;
const uint BrainOutputCapacity = BrainActuatorOutputCount + BrainRecurrentCount;

VKEXP_BRAIN_FN uint brainNeighborChannelIndex(uint neighbor, uint channel) {
    return BrainNeighborOffset + neighbor * BrainNeighborChannels + channel;
}

VKEXP_BRAIN_FN uint brainBeaconInputIndex(uint channel) { return BrainBeaconOffset + channel; }

// --- activation --------------------------------------------------------------
//
// The squash on a hidden neuron's state and on every output. It lives here
// because it was written twice -- once in the CPU evaluator and once in the
// shader -- and a change applied to one of them would be found by nothing but
// accumulated parity drift, which says a number is wrong without saying which
// line wrote it.
//
// It is tanh and not something periodic on purpose, and the reasons are worth
// keeping next to the code that would be edited to change it:
//
//   * Monotone. More input is more output, everywhere. Under a periodic
//     activation the sign of the response flips every half period, so a mutation
//     that raises a weight helps or hurts depending on where the neuron happens
//     to sit, and "brighter on the left" stops meaning one thing.
//   * Saturating, which is what makes a memory possible at all here. A neuron
//     driven hard sits at +/-1 and stops responding, which is a decision that
//     holds; a periodic activation cycles back through zero instead, so a
//     neuron cannot commit. That is the whole point of the time constants
//     undone. Note this is not an argument about the Lipschitz bound -- tanh and
//     sin are both 1-Lipschitz -- but about where the derivative vanishes.
//   * Insensitive to input error exactly where the input is large, which is what
//     keeps the CPU/GPU drift budget tight. A periodic activation is maximally
//     sensitive there, and large-argument reduction is where implementations
//     differ most, so drift would become a function of how confident the network
//     is.
//
// Periodic activations are a real tool where repetition is wanted -- CPPNs use
// them for symmetry, SIREN for high-frequency detail. This network is a
// controller that has to hold decisions, and it can already oscillate through a
// recurrent loop with two different time constants when it wants to.
VKEXP_BRAIN_MATH_FN float brainActivation(float value) { return tanh(value); }

// --- neuron time constants ---------------------------------------------------
//
// Every hidden neuron carries its own state and its own time constant:
//
//     y += (dt / tau) * (-y + activation);   h = tanh(y)
//
// tau is a gene, so how long a neuron remembers is selected for rather than
// designed, and different time scales become a trait evolution can separate: a
// fast neuron is a reflex that tracks its input within a step, a slow one holds
// a fact across seconds. dt enters explicitly, so a memory is measured in
// seconds and not in steps -- the same rule 3e the rest of the physics follows.
//
// This is where memory belongs. The two recurrent cells put it in the *output*
// layer, which cost two of the eight output slots and squeezed everything a
// brain might remember through a two-number bottleneck. Time constants give all
// twenty neurons a state and take no output slot at all.
//
// tau = dt makes the update y = activation exactly, which is the memoryless
// network this replaced. That is what makes the whole feature ablatable without
// a second code path: turned off it is literally the old behaviour rather than
// a reimplementation of it.
const float BrainTimeConstantMinimum = 0.0166666667f; // s, one step at 60 Hz
const float BrainTimeConstantMaximum = 4.0f;          // s

// Genes are unbounded, so the range is entered through a squash. It is
// logarithmic because what matters about a memory is its order of magnitude,
// not its linear length -- the useful settings crowd the short end, and a
// linear map would spend most of the gene range between two and four seconds.
// A gene of zero lands on the geometric middle, about 0.26 s.
VKEXP_BRAIN_MATH_FN float brainTimeConstant(float gene) {
    const float unit = 1.0f / (1.0f + exp(-gene));
    return BrainTimeConstantMinimum *
           pow(BrainTimeConstantMaximum / BrainTimeConstantMinimum, unit);
}

// One step of the continuous-time update, shared so the CPU evaluator and the
// shader cannot integrate the neuron differently. The ratio is clamped at 1 so
// a time constant shorter than the step cannot overshoot into oscillation --
// which is also why tau bottoms out at one step rather than at zero.
VKEXP_BRAIN_MATH_FN float brainIntegrateNeuron(float state, float activation, float timeConstant,
                                               float deltaTime) {
    const float rate = clamp(deltaTime / timeConstant, 0.0f, 1.0f);
    return state + rate * (activation - state);
}

// --- neuron models -----------------------------------------------------------
//
// Three ways to decide the time constant, and one integrator. What changes
// between them is only where tau comes from, which is why the UI can switch
// them at runtime and why the ablation is exact rather than approximate.
//
//   Reactive      tau = dt. The update collapses to y = activation: the
//                 memoryless network, reached through the same arithmetic.
//   TimeConstant  tau from a gene, fixed for the neuron's life. A neuron
//                 forgets at one rate whatever is happening to it.
//   Gated         tau recomputed every step from the inputs, through the same
//                 mapping the gene uses. A neuron can hold a value and then let
//                 go of it when something tells it to -- which a fixed rate
//                 cannot do, because "keep this until the trial ends" and "track
//                 this closely" are the same neuron at different moments.
//
// Note the sign. The gate asks for a time constant, not for an update fraction:
// driving it up makes the neuron hold, leaving it low makes the neuron follow.
// That is the opposite of a GRU update gate, and it is this way round because
// the gate and the gene go through the same mapping -- which is what makes the
// two models comparable at all.
//
// Gated is a strict generalisation: feed it a constant and it is TimeConstant.
// It costs one extra weight row and one bias per hidden neuron and no extra
// state, because the state it needs is the one the neuron already carries.
const uint NeuronModelReactive = 0u;
const uint NeuronModelTimeConstant = 1u;
const uint NeuronModelGated = 2u;
const uint NeuronModelSpiking = 3u;
const uint NeuronModelCount = 4u;

// --- genome addressing: one dense network laid out flat ----------------------
//
// [input->hidden weights][hidden biases][hidden->output weights][output biases]
// [hidden time constants][gate weights][gate biases]
//
// Each block goes after the last so every earlier offset is unchanged and a
// scenario that trims inputs or outputs still addresses a dense prefix. The gate
// block is carried by every genome whatever model is selected: it is the same
// genome under all three, so a model can be switched mid-experiment without the
// population meaning something different afterwards.

// --- how many hidden layers, and how wide -----------------------------------
//
// The plan is three widths packed into one uint, six bits each. A width of zero
// means the layer is not there, and the layers are dense from the front, so
// {20, 0, 0} is the single 20-wide layer this network had for its whole life.
//
// Packed rather than passed as an array because it crosses into GLSL, where the
// common subset this file is written in has no structs and no arrays of
// parameters. Six bits is the capacity above with one to spare.
const uint BrainLayerSizeMask = 0x3fu;
const uint BrainLayerSizeBits = 6u;

VKEXP_BRAIN_FN uint brainPackHiddenLayers(uint first, uint second, uint third) {
    return (first & BrainLayerSizeMask) | ((second & BrainLayerSizeMask) << BrainLayerSizeBits) |
           ((third & BrainLayerSizeMask) << (2u * BrainLayerSizeBits));
}

VKEXP_BRAIN_FN uint brainHiddenLayerSize(uint layers, uint layer) {
    if (layer >= BrainHiddenLayerCapacity) {
        return 0u;
    }
    return (layers >> (layer * BrainLayerSizeBits)) & BrainLayerSizeMask;
}

// Layers are dense from the front, so the count is where the widths stop.
VKEXP_BRAIN_FN uint brainHiddenLayerCount(uint layers) {
    uint count = 0u;
    for (uint layer = 0u; layer < BrainHiddenLayerCapacity; ++layer) {
        if (brainHiddenLayerSize(layers, layer) == 0u) {
            return count;
        }
        count = layer + 1u;
    }
    return count;
}

VKEXP_BRAIN_FN uint brainHiddenNeuronCount(uint layers) {
    uint total = 0u;
    for (uint layer = 0u; layer < BrainHiddenLayerCapacity; ++layer) {
        total += brainHiddenLayerSize(layers, layer);
    }
    return total;
}

// Where a layer's neuron states begin in the agent's hidden block. The states
// of all layers live end to end in one array, so a deeper plan needs no new
// storage on the agent -- only a different division of the same block.
VKEXP_BRAIN_FN uint brainHiddenLayerStateOffset(uint layers, uint layer) {
    uint offset = 0u;
    for (uint earlier = 0u; earlier < layer; ++earlier) {
        offset += brainHiddenLayerSize(layers, earlier);
    }
    return offset;
}

// What a layer reads: the input vector for the first, the layer before it after
// that. This one function is why a deep plan needs no special case anywhere.
VKEXP_BRAIN_FN uint brainLayerSourceCount(uint inputCount, uint layers, uint layer) {
    if (layer == 0u) {
        return inputCount;
    }
    return brainHiddenLayerSize(layers, layer - 1u);
}

// --- genome addressing: the layers laid out flat -----------------------------
//
// [layer 0 weights][layer 0 biases] ... [layer n weights][layer n biases]
// [hidden->output weights][output biases][time constants]
// [gate 0 weights][gate 0 biases] ... [gate n weights][gate n biases]
//
// The gate block mirrors the forward block layer for layer, because a gate is a
// second opinion about the same inputs. It is carried whatever neuron model is
// selected, so switching a model is a parameter change and not a
// reinterpretation of the population -- see the model notes above.

// Every forward weight and bias, for all layers: the size of the first block,
// and also of the gate block that mirrors it.
VKEXP_BRAIN_FN uint brainForwardBlockSize(uint inputCount, uint layers) {
    uint total = 0u;
    for (uint layer = 0u; layer < BrainHiddenLayerCapacity; ++layer) {
        const uint size = brainHiddenLayerSize(layers, layer);
        total += size * brainLayerSourceCount(inputCount, layers, layer) + size;
    }
    return total;
}

// The width the output layer reads from: the last hidden layer.
VKEXP_BRAIN_FN uint brainLastHiddenSize(uint layers) {
    const uint count = brainHiddenLayerCount(layers);
    if (count == 0u) {
        return 0u;
    }
    return brainHiddenLayerSize(layers, count - 1u);
}

VKEXP_BRAIN_FN uint brainWeightCount(uint inputCount, uint layers, uint outputCount) {
    const uint forward = brainForwardBlockSize(inputCount, layers);
    return forward + brainLastHiddenSize(layers) * outputCount + outputCount +
           brainHiddenNeuronCount(layers) + forward;
}

// Start of a layer's own weights, walking the layers before it.
VKEXP_BRAIN_FN uint brainLayerBlockOffset(uint base, uint inputCount, uint layers, uint layer) {
    uint offset = base;
    for (uint earlier = 0u; earlier < layer; ++earlier) {
        const uint size = brainHiddenLayerSize(layers, earlier);
        offset += size * brainLayerSourceCount(inputCount, layers, earlier) + size;
    }
    return offset;
}

VKEXP_BRAIN_FN uint brainLayerWeightIndex(uint base, uint inputCount, uint layers, uint layer,
                                          uint neuron, uint sourceIndex) {
    return brainLayerBlockOffset(base, inputCount, layers, layer) +
           neuron * brainLayerSourceCount(inputCount, layers, layer) + sourceIndex;
}

VKEXP_BRAIN_FN uint brainLayerBiasIndex(uint base, uint inputCount, uint layers, uint layer,
                                        uint neuron) {
    const uint size = brainHiddenLayerSize(layers, layer);
    return brainLayerBlockOffset(base, inputCount, layers, layer) +
           size * brainLayerSourceCount(inputCount, layers, layer) + neuron;
}

VKEXP_BRAIN_FN uint brainOutputBlockOffset(uint base, uint inputCount, uint layers) {
    return base + brainForwardBlockSize(inputCount, layers);
}

VKEXP_BRAIN_FN uint brainOutputWeightIndex(uint base, uint inputCount, uint layers, uint neuron,
                                           uint hiddenIndex) {
    return brainOutputBlockOffset(base, inputCount, layers) + neuron * brainLastHiddenSize(layers) +
           hiddenIndex;
}

VKEXP_BRAIN_FN uint brainOutputBiasIndex(uint base, uint inputCount, uint layers, uint outputCount,
                                         uint neuron) {
    return brainOutputBlockOffset(base, inputCount, layers) +
           brainLastHiddenSize(layers) * outputCount + neuron;
}

// One gene per hidden neuron, numbered across all layers end to end, the same
// way the states are.
VKEXP_BRAIN_FN uint brainTimeConstantGeneIndex(uint base, uint inputCount, uint layers,
                                               uint outputCount, uint neuron) {
    return brainOutputBiasIndex(base, inputCount, layers, outputCount, outputCount) + neuron;
}

VKEXP_BRAIN_FN uint brainGateBlockOffset(uint base, uint inputCount, uint layers,
                                         uint outputCount) {
    return brainTimeConstantGeneIndex(base, inputCount, layers, outputCount,
                                      brainHiddenNeuronCount(layers));
}

VKEXP_BRAIN_FN uint brainGateWeightIndex(uint base, uint inputCount, uint layers, uint outputCount,
                                         uint layer, uint neuron, uint sourceIndex) {
    return brainLayerWeightIndex(brainGateBlockOffset(base, inputCount, layers, outputCount),
                                 inputCount, layers, layer, neuron, sourceIndex);
}

VKEXP_BRAIN_FN uint brainGateBiasIndex(uint base, uint inputCount, uint layers, uint outputCount,
                                       uint layer, uint neuron) {
    return brainLayerBiasIndex(brainGateBlockOffset(base, inputCount, layers, outputCount),
                               inputCount, layers, layer, neuron);
}

// --- active shape packed into one uint for the GPU ---------------------------

// inputs | outputs, packed into the one uint the shader reads beside the layer
// plan. The genome stride used to live here too, in twelve bits; the capacity
// for three layers puts it past 4095, so it travels as its own uint now. Better
// than widening the field: a stride that silently wrapped would address another
// genome's weights and still produce numbers.
const uint BrainInputShift = 0u;
const uint BrainOutputShift = 7u;
const uint BrainCountMask = 0x7fu;
const uint BrainOutputCountMask = 0x1fu;

VKEXP_BRAIN_FN uint brainPackLayout(uint inputCount, uint outputCount) {
    return (inputCount << BrainInputShift) | (outputCount << BrainOutputShift);
}

VKEXP_BRAIN_FN uint brainLayoutInputCount(uint packed) {
    return (packed >> BrainInputShift) & BrainCountMask;
}
VKEXP_BRAIN_FN uint brainLayoutOutputCount(uint packed) {
    return (packed >> BrainOutputShift) & BrainOutputCountMask;
}
