// The network preset: how many sensors feed the brain, how wide it is, what its
// outputs mean, and how a flat genome is addressed.
//
// Compiled twice -- as C++ through BrainKernel.hpp and as GLSL through
// shaders/neuro/brain_kernel.glsl -- so both sides build the same network from
// the same declaration. Adding a sensor changes the counts here and nothing
// else: the construction below is written once and derives every offset.
//
// Same common-subset rules as ScenarioKernel.inl: VKEXP_BRAIN_FN in front of
// every function, only uint/bool across boundaries, no standard library. Note
// that GLSL reserves more words than C++ does -- `input`, `output`, `layout`,
// `filter`, `active` and friends cannot be identifiers here.

// --- preset: change these when the sensor suite or brain width changes -------

const uint BrainLightReceptorCount = 7u;
const uint BrainLightChannels = 4u; // RGB + luminance
const uint BrainTactileSectorCount = 8u;
const uint BrainTactileChannels = 2u;     // wall + agent
const uint BrainAntennaCount = 3u;        // left, centre, right ground feelers
const uint BrainAntennaChannels = 3u;     // RGB of the trail under the tip
const uint BrainSelfInputCount = 4u;      // speed, turn rate, energy, own signal
const uint BrainTaskInputCount = 2u;      // cargo level, seeking-home flag
const uint BrainRecurrentCount = 2u;      // memory cells, fed back as inputs
const uint BrainActuatorOutputCount = 6u; // left, right, R, G, B, intensity
const uint BrainHiddenCapacity = 20u;

// --- derived layout: never edited by hand ------------------------------------

const uint BrainLightBlockSize = BrainLightReceptorCount * BrainLightChannels;
const uint BrainTactileBlockSize = BrainTactileSectorCount * BrainTactileChannels;
const uint BrainAntennaBlockSize = BrainAntennaCount * BrainAntennaChannels;

// The antenna block sits inside the reactive prefix, ahead of the task and
// recurrent blocks that scenarios trim: every scenario can smell the ground.
const uint BrainLightOffset = 0u;
const uint BrainTactileOffset = BrainLightOffset + BrainLightBlockSize;
const uint BrainAntennaOffset = BrainTactileOffset + BrainTactileBlockSize;
const uint BrainSelfOffset = BrainAntennaOffset + BrainAntennaBlockSize;
const uint BrainTaskOffset = BrainSelfOffset + BrainSelfInputCount;
const uint BrainRecurrentInputOffset = BrainTaskOffset + BrainTaskInputCount;
const uint BrainInputCapacity = BrainRecurrentInputOffset + BrainRecurrentCount;

const uint BrainMotorLeftOutput = 0u;
const uint BrainMotorRightOutput = 1u;
const uint BrainSignalColorOutput = 2u; // three consecutive channels
const uint BrainSignalIntensityOutput = 5u;
const uint BrainRecurrentOutputOffset = BrainActuatorOutputCount;
const uint BrainOutputCapacity = BrainActuatorOutputCount + BrainRecurrentCount;

VKEXP_BRAIN_FN uint brainLightChannelIndex(uint receptor, uint channel) {
    return BrainLightOffset + receptor * BrainLightChannels + channel;
}

VKEXP_BRAIN_FN uint brainTactileChannelIndex(uint sector, uint channel) {
    return BrainTactileOffset + sector * BrainTactileChannels + channel;
}

VKEXP_BRAIN_FN uint brainAntennaChannelIndex(uint antenna, uint channel) {
    return BrainAntennaOffset + antenna * BrainAntennaChannels + channel;
}

// --- sensor response model ---------------------------------------------------
//
// What a receptor *is*: how sharply it is tuned, how light falls off with range,
// and how RGB collapses to luminance. These use float math, so they carry the
// VKEXP_BRAIN_MATH_FN marker instead -- the C++ side cannot make them constexpr.

const float BrainReceptorSharpness = 12.0f;
const float BrainLightFalloffWidth = 0.25f;
const float BrainDistanceAttenuation = 2.0f;
const float BrainLuminanceRed = 0.2126f;
const float BrainLuminanceGreen = 0.7152f;
const float BrainLuminanceBlue = 0.0722f;

// Smooth cut-off at the edge of sensor range.
VKEXP_BRAIN_MATH_FN float brainRangeFalloff(float distanceToLight, float range) {
    const float fade =
        clamp((range - distanceToLight) / (range * BrainLightFalloffWidth), 0.0f, 1.0f);
    return fade * fade * (3.0f - 2.0f * fade);
}

// Directional tuning of one receptor; `alignment` is a clamped cosine.
VKEXP_BRAIN_MATH_FN float brainReceptorResponse(float alignment) {
    return pow(alignment, BrainReceptorSharpness);
}

VKEXP_BRAIN_MATH_FN float brainDistanceAttenuation(float normalizedDistanceSquared) {
    return 1.0f / (1.0f + normalizedDistanceSquared * BrainDistanceAttenuation);
}

// Trail deposits are unbounded -- a cell many agents stand on keeps growing --
// so the reading is squashed into [0, 1) rather than clamped, which keeps a
// strong scent distinguishable from an overwhelming one.
VKEXP_BRAIN_MATH_FN float brainTrailResponse(float deposit) { return deposit / (1.0f + deposit); }

// --- antenna geometry --------------------------------------------------------
//
// Where the ground feelers sit. They reach well past the body on purpose: the
// three tips have to land in different trail cells for the reading to carry a
// gradient at all, so their lateral spread is what sets the usable trail
// resolution. At 12 cm and 0.7 rad the outer tips are 15 cm apart, which stays
// legible on the 6 cm trail cells.

const float BrainAntennaLength = 0.12f;    // m from the body centre
const float BrainAntennaHalfSpread = 0.7f; // rad from the heading

// Angle of one antenna relative to the agent's heading.
VKEXP_BRAIN_MATH_FN float brainAntennaAngle(uint antenna) {
    const float fraction =
        BrainAntennaCount > 1u ? float(antenna) / float(BrainAntennaCount - 1u) - 0.5f : 0.0f;
    return fraction * 2.0f * BrainAntennaHalfSpread;
}

VKEXP_BRAIN_MATH_FN float brainLuminance(float red, float green, float blue) {
    return red * BrainLuminanceRed + green * BrainLuminanceGreen + blue * BrainLuminanceBlue;
}

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
const uint NeuronModelCount = 3u;

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

VKEXP_BRAIN_FN uint brainWeightCount(uint inputCount, uint hiddenCount, uint outputCount) {
    return inputCount * hiddenCount + hiddenCount + hiddenCount * outputCount + outputCount +
           hiddenCount + hiddenCount * inputCount + hiddenCount;
}

// Start of the gate block: everything before it, in order.
VKEXP_BRAIN_FN uint brainGateBlockOffset(uint base, uint inputCount, uint hiddenCount,
                                         uint outputCount) {
    return base + inputCount * hiddenCount + hiddenCount + hiddenCount * outputCount + outputCount +
           hiddenCount;
}

VKEXP_BRAIN_FN uint brainHiddenWeightIndex(uint base, uint inputCount, uint neuron,
                                           uint inputIndex) {
    return base + neuron * inputCount + inputIndex;
}

VKEXP_BRAIN_FN uint brainHiddenBiasIndex(uint base, uint inputCount, uint hiddenCount,
                                         uint neuron) {
    return base + inputCount * hiddenCount + neuron;
}

VKEXP_BRAIN_FN uint brainOutputWeightIndex(uint base, uint inputCount, uint hiddenCount,
                                           uint neuron, uint hiddenIndex) {
    return base + inputCount * hiddenCount + hiddenCount + neuron * hiddenCount + hiddenIndex;
}

VKEXP_BRAIN_FN uint brainOutputBiasIndex(uint base, uint inputCount, uint hiddenCount,
                                         uint outputCount, uint neuron) {
    return base + inputCount * hiddenCount + hiddenCount + hiddenCount * outputCount + neuron;
}

VKEXP_BRAIN_FN uint brainTimeConstantGeneIndex(uint base, uint inputCount, uint hiddenCount,
                                               uint outputCount, uint neuron) {
    return base + inputCount * hiddenCount + hiddenCount + hiddenCount * outputCount + outputCount +
           neuron;
}

VKEXP_BRAIN_FN uint brainGateWeightIndex(uint base, uint inputCount, uint hiddenCount,
                                         uint outputCount, uint neuron, uint inputIndex) {
    return brainGateBlockOffset(base, inputCount, hiddenCount, outputCount) + neuron * inputCount +
           inputIndex;
}

VKEXP_BRAIN_FN uint brainGateBiasIndex(uint base, uint inputCount, uint hiddenCount,
                                       uint outputCount, uint neuron) {
    return brainGateBlockOffset(base, inputCount, hiddenCount, outputCount) +
           hiddenCount * inputCount + neuron;
}

// --- active shape packed into one uint for the GPU ---------------------------

// stride | inputs | hidden | outputs, packed into the one uint the shader reads.
// The input and hidden fields are 7 bits rather than 6: the antenna block put the
// capacity at 61, two short of a 6-bit ceiling, and a sensor suite that cannot
// grow is not a preset. 12 + 7 + 7 + 5 = 31 bits, one still spare.
const uint BrainStrideMask = 0xfffu;
const uint BrainInputShift = 12u;
const uint BrainHiddenShift = 19u;
const uint BrainOutputShift = 26u;
const uint BrainCountMask = 0x7fu;
const uint BrainOutputCountMask = 0x1fu;

VKEXP_BRAIN_FN uint brainPackLayout(uint genomeStride, uint inputCount, uint hiddenCount,
                                    uint outputCount) {
    return genomeStride | (inputCount << BrainInputShift) | (hiddenCount << BrainHiddenShift) |
           (outputCount << BrainOutputShift);
}

VKEXP_BRAIN_FN uint brainLayoutStride(uint packed) { return packed & BrainStrideMask; }
VKEXP_BRAIN_FN uint brainLayoutInputCount(uint packed) {
    return (packed >> BrainInputShift) & BrainCountMask;
}
VKEXP_BRAIN_FN uint brainLayoutHiddenCount(uint packed) {
    return (packed >> BrainHiddenShift) & BrainCountMask;
}
VKEXP_BRAIN_FN uint brainLayoutOutputCount(uint packed) {
    return (packed >> BrainOutputShift) & BrainOutputCountMask;
}
