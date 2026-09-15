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

// The front hemisphere, one slot per cell the agent can see. Restated rather
// than included from LatticeKernel.inl because the two kernels compile into
// separate namespaces on the C++ side and GLSL has no namespaces at all;
// testLatticeBrain asserts the two agree, which is the same arrangement the 2D
// build used for the body radius it shared with the scenario kernel.
const uint BrainNeighborCount = 17u;
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
// Two outputs for one turn, and they have to agree: both over the threshold
// turns one way, both under the negative one turns the other, and any
// disagreement -- including one of them sitting in the dead zone -- is no turn.
//
// One output could say the same thing in half the genes, and did. The reason
// for two is what they do to a network nobody has trained yet. A tanh output
// saturates on almost any input, so a single one clears the dead zone on almost
// every tick: a fresh population spent 93% of its ticks pivoting on the spot at
// a threshold of 0.25 and 63% at 0.7, which is a generation spent looking around
// rather than walking. Two outputs that must agree have to saturate the same way
// at the same time, and two fresh outputs are near enough independent that this
// roughly squares the chance.
//
// What it does not do is make turning harder for a policy that wants to turn: a
// network that has learned to steer simply drives both outputs together. It only
// makes turning rare by accident, which is the thing that was wrong.
const uint BrainTurnOutputCount = 2u;
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
// 52 rather than 20, and three layers rather than one. The cost of the headroom
// is paid in the genome stride, which is sized for the widest plan the capacity
// allows -- one 52-wide layer -- so a run using fewer neurons carries weights it
// never reads. That is the same trade the gate block already makes, and for the
// same reason: one buffer size and one genome length means a population stays
// loadable across plans, and comparing two plans stays possible at all.
const uint BrainHiddenNeuronCapacity = 52u;
const uint BrainHiddenLayerCapacity = 3u;

const uint BrainActivationTanh = 0u;
const uint BrainActivationSine = 1u;
// tanh of the sum divided by the square root of how many things it sums. Costs
// no parameters and is the textbook answer to a layer whose pre-activation grows
// with its width -- which is the control this project needed and did not have:
// if a squash that only rescales catches up with sine, then what sine bought was
// scale and not periodicity.
const uint BrainActivationTanhScaled = 2u;
// x / (1 + |x|). Saturates, so a neuron can still hold a decision, but reaches
// its asymptote an order of magnitude more slowly than tanh, so a layer of them
// does not all pile up at the extremes. The middle of the same axis.
const uint BrainActivationSoftsign = 3u;
// max(0, x). The first squash here that is neither odd nor bounded, and it is
// here for one reason: the spiking model beats every saturating one, and the
// difference that stands out in the code is not its timing but what it hands
// the output layer. A spiking hidden unit emits 0 or 1 and is silent most
// ticks, so the output's pre-activation is a sum over the few that fired. A
// layer of tanh hands it thirty-five values near +-1 at once, and the output
// tanh is pinned. Rectifying is the way to ask whether a sparse, one-signed
// code is what buys the climb, without adopting the spiking neuron's dynamics
// with it: the state still integrates the same way, only its reading changes.
const uint BrainActivationRelu = 4u;
// min(max(0, x), 1). The same code with the spike's ceiling put back. Together
// with the one above it these separate two properties the spiking neuron has at
// once -- sparse and one-signed, and bounded -- so a result can say which of
// them mattered instead of naming the pair.
const uint BrainActivationReluUnit = 5u;
// Not a kind: what the kinds are counted by, for a menu and for the bits below.
const uint BrainActivationCount = 6u;

// Measured, and the answer is half of one. Three hundred steps of construction
// on random genomes, one layer of 35, the instrument in scratchpad/activations:
//
//   plan                     |h|   silent   output |z|   walk  build  turn
//   tanh        (reactive)   0.73    4%        2.22      28%    28%    42%
//   relu-unit   (reactive)   0.40   53%        1.28      34%    25%    39%
//   relu-unit-3 (reactive)   ----   ---        0.62      56%    21%    22%
//   spiking                  0.03   97%        0.60      56%    17%    25%
//
// where relu-unit-3 is the same rectified layer with three subtracted from every
// hidden bias, which for a rectifier is a firing threshold. Read down the
// columns: sparsity alone reproduces the spiking model's decisions almost
// exactly. Same output magnitude, same split between walking, building and
// turning, to a point or two. So what the spiking hidden layer sells the output
// layer is silence -- an output whose pre-activation is a sum over the few units
// that fired, rather than over thirty-five that are all shouting -- and a
// rectifier with a threshold buys the same thing.
//
// What it does not buy is churn. Spiking holds a decision 2.58 ticks on average,
// median 1, ninety-ninth percentile 19. The rectified layer at the same sparsity
// holds 4.69 ticks, median 2, ninety-ninth percentile 53 -- and under the
// time-constant model 6.62 and 65. Sparsity makes the output quiet, and a quiet
// output is also a steady one: the two properties pull against each other in
// every model here. The spiking neuron gets both because its state resets on
// firing, which is a source of change inside the neuron rather than in what it
// reads.
//
// Which is worth knowing before reaching for a gate. A GRU reset gate decides
// how much history to carry given the input; it does not make a neuron let go of
// its own accord. If the field runs keep showing what they show -- spiking
// walking 20-25% of ticks where the others record 1-3% -- then the variable to
// chase is how long a refused decision persists, and the reset is the thing in
// the spiking neuron that answers it.

// What a world means when it does not say otherwise: one hidden layer of 35,
// squashed by sine. Separate from the capacity on purpose -- raising how many
// neurons there *may* be must not quietly widen every world's brain, which is
// exactly what sharing one constant would have done.
//
// It was one layer of twenty under tanh for the whole life of this project. What
// moved it is the squash and not the depth: four generations of construction
// under the reactive model, three seeds, on a two-layer 35+15 plan,
//
//   35+15 squash    walk   blocks    median
//   tanh, tanh     10.1%    37-49     45-51
//   sin,  tanh     16.4%    63-84     81-97
//   tanh, sin      20.9%   104-134   125-133
//   sin,  sin      29.9%   136-144   218-234
//
// and the order holds under the time-constant model too, at a third of the
// scores. Sine in the first layer alone is the weakest of the three placements,
// which is worth saying because the obvious guess is the other way round: it is
// the layer feeding the output that gains most from not being nearly binary.
//
// One layer and not the two those numbers were taken on, which is a deliberate
// trade and not what was measured: the second layer doubles the genome for a
// gain nobody has separated from sine's, and a single layer is the plan every
// other measurement in this project was taken on. The width is the first layer's
// from that sweep. A two-layer plan is one flag away -- `--hidden 35,15` -- and
// still carries sine in both.
const uint BrainDefaultHiddenWidth = 35u;
const uint BrainDefaultHiddenSquash = BrainActivationSine;

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
//   * Saturating. A neuron driven hard sits at +/-1 and stops responding, so a
//     rising input cannot talk it back out of its answer; under a periodic
//     activation a further rise changes the answer. This is not about memory --
//     neither activation has any, and what holds a decision here is the neuron's
//     own state, its time constant and the two recurrent cells -- it is about
//     whether a decision, once reached, survives more of the same evidence. Nor
//     is it about the Lipschitz bound, since tanh and sin are both 1-Lipschitz:
//     it is about where the derivative vanishes.
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

// A hidden layer's squash, which a plan may choose per layer. The output layer
// is not offered the choice and never will be: every threshold in the rules
// reads an output as "how far, and which way", and that only means anything
// under something monotone and bounded.
//
// The reservations above are real and none of them are answered by putting the
// choice in the plan; what the plan does is let a run be compared against
// itself. Sine is offered on hidden layers because the objections weigh
// differently in the middle of a network than at its ends: a periodic unit deep
// in a stack is a basis function rather than a decision, which is what CPPNs and
// SIREN use them for, and a layer of them still feeds a tanh output that has to
// commit.
//
// It measures well, which was not the expected answer. Four generations of
// construction under the reactive model on a 35+15 plan, three seeds, against
// the same plan with tanh throughout: three times the blocks and four times the
// median score. Two controls say what that is not, and they are the reason the
// other two kinds below exist:
//
//   squash        walk   placed   blocks    median
//   tanh          10.1%   0.76%    37-49     45-51
//   sin           29.9%   2.95%   136-144   218-234
//   tanh / sqrt(n) 5.2%   0.24%    27-29      8-15
//   softsign       8.8%   0.53%    25-45     32-36
//
// Rescaling alone is the worst of the four, so the gain is not scale. Saturating
// an order of magnitude more slowly does not reproduce it either. Both of those
// were plausible and both are now ruled out.
//
// What scratchpad/activations.cpp measures on real trajectories, which is the
// only place these questions can be asked:
//
//   * The periodicity is genuinely exercised and only just. Roughly 45% of
//     pre-activations exceed pi/2 -- where the response folds -- and 9-20%
//     exceed pi, but under 1% reach a full period. So this is the first fold and
//     not a Fourier basis, which is also why the sine case's CPU/GPU drift sits
//     in the same band as tanh's: nothing here is a large-argument sine.
//   * A sine layer does not collect near zero, which was the first guess and is
//     wrong. It spreads: values above 0.9 in magnitude 29% of the time against
//     tanh's 48-56%, mean magnitude 0.63 against 0.74. A tanh layer under load
//     is very nearly binary; a sine layer uses the middle of its range.
//   * Spreading is not the explanation either. Softsign spreads about as much
//     (mean 0.67, above 0.9 in 19%) and scores like tanh.
//   * Nor is a less saturated output. Sine does cut the sum reaching the output
//     tanh from a mean of 1.94 to 1.27, but tanh/sqrt(n) cuts it to 0.53 and
//     comes last, and sine spends slightly *less* of its time in the band that
//     means "walk" (40% against 45%).
//
// The one property that separates sine from every control is the fold itself,
// and it is exercised on about half the samples. Why a folded unit should be
// worth three times the blocks is not answered here: a plausible reading is that
// it gives each neuron two decision boundaries instead of one, at the same
// parameter count, but that is a hypothesis and not a measurement.
//
// The cost is measurable too, and it is the objection above, stated in ticks:
// an untrained sine population holds a decision 1.86 ticks on average against
// tanh's 2.76, and its 99th percentile run is 18 ticks against 34. A periodic
// activation can re-switch the response as the input grows; how much that costs
// a trained policy is a question about trajectories, and nobody has run one.
//
// Under the spiking model it does nothing at all: a spiking neuron writes 1 or 0
// directly and never reaches a squash. That is not an oversight to fix, it is
// what "integrate and fire" means, but it does mean --hidden-squash and the
// spiking model do not combine.

VKEXP_BRAIN_MATH_FN float brainLayerActivate(uint activation, float value, uint sources) {
    if (activation == BrainActivationSine) {
        return sin(value);
    }
    if (activation == BrainActivationTanhScaled) {
        // The count and not the count minus the bias: one gene out of forty is
        // not worth a second constant, and the scale only has to be the right
        // order.
        return brainActivation(value / sqrt(float(sources < 1u ? 1u : sources)));
    }
    if (activation == BrainActivationSoftsign) {
        return value / (1.0f + abs(value));
    }
    if (activation == BrainActivationRelu) {
        return value > 0.0f ? value : 0.0f;
    }
    if (activation == BrainActivationReluUnit) {
        return clamp(value, 0.0f, 1.0f);
    }
    return brainActivation(value);
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

// What a neuron fires at when nothing has happened yet, and the most an
// adapting threshold can be raised by one discharge. The rest is the spiking
// neuron's constant, so the two models are the same neuron at bump zero.
const float BrainThresholdRest = 1.0f;
const float BrainAdaptationBumpMaximum = 2.0f;

// The bump a discharge adds, from an unbounded gene. Bounded above because a
// threshold that can be raised without limit is a neuron that fires once and
// then never again, which is a dead unit dressed as an adapting one.
VKEXP_BRAIN_MATH_FN float brainAdaptationBump(float gene) {
    return BrainAdaptationBumpMaximum / (1.0f + exp(-gene));
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

// One discharge, shared the way the integrator is. `state` and `excess` are
// read and written: state is the membrane, excess is how far this neuron's
// threshold currently sits above the resting one, and is zero for every model
// but Adaptive. The return value is what the neuron emits, 1 or 0.
//
// The order matters and is the one thing a second implementation could get
// subtly right-looking and wrong: the threshold this tick is the one the last
// tick left behind, the bump lands after the comparison, and the relaxation
// runs last. Bump before compare would make a neuron unable to fire twice in a
// row even at bump zero, which is the spiking model it has to reduce to.
VKEXP_BRAIN_MATH_FN float brainDischarge(VKEXP_BRAIN_INOUT state, VKEXP_BRAIN_INOUT excess,
                                         float bump, float relaxTimeConstant, float deltaTime) {
    float emitted = 0.0f;
    if (state >= BrainThresholdRest + excess) {
        emitted = 1.0f;
        state = 0.0f;
        excess += bump;
    } else if (state < -1.0f) {
        // Inhibition has a floor, so a unit driven hard negative recovers in
        // bounded time instead of being switched off for the rest of the trial.
        state = -1.0f;
    }
    excess = brainIntegrateNeuron(excess, 0.0f, relaxTimeConstant, deltaTime);
    return emitted;
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
// Spiking with a threshold that is not a constant. Every discharge raises it,
// and it relaxes back towards the resting threshold on a time constant of its
// own. A neuron that has just fired is therefore harder to fire again, which is
// a second timescale the neuron owns rather than one its inputs hand it.
//
// Here because of what the rectified layer measured: sparsity reproduced the
// spiking model's decisions and not its churn, and the property left over is
// change that comes from inside the neuron. The reset is one such source; an
// adapting threshold is the same idea given a knob, so a genome can choose how
// long a unit stays quiet after speaking rather than always the same tick.
//
// Strictly generalises Spiking, the way Gated generalises TimeConstant: the
// bump gene runs to zero, and a neuron whose bump is zero has a constant
// threshold of one and is the spiking neuron exactly.
const uint NeuronModelAdaptive = 4u;
const uint NeuronModelCount = 5u;

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

// Which squash a hidden layer uses, three bits each, above the three widths. In
// the same word because the word is what already crosses into GLSL: a layer's
// activation is part of what the network is, and carrying it anywhere else would
// mean a second thing to pass, a second thing to store in a file, and a second
// thing to forget.
//
// Three bits and not two because the four saturating kinds filled two exactly,
// and the rectified pair had nowhere to go. The word still fits: three widths of
// six bits and three activations of three is twenty-seven. Widening it moved
// where layers two and three keep their activation, which is why the archive
// version moved with it -- see genomeArchiveVersion.
const uint BrainLayerActivationBits = 3u;
const uint BrainLayerActivationMask = 0x7u;
const uint BrainLayerActivationShift = 3u * BrainLayerSizeBits;
VKEXP_BRAIN_FN uint brainPackHiddenLayers(uint first, uint second, uint third) {
    return (first & BrainLayerSizeMask) | ((second & BrainLayerSizeMask) << BrainLayerSizeBits) |
           ((third & BrainLayerSizeMask) << (2u * BrainLayerSizeBits));
}

// The activations folded into a packed plan. Separate from the widths so that
// every existing caller keeps meaning what it meant -- zero is tanh, which is
// what a plan that says nothing about activations has always used.
VKEXP_BRAIN_FN uint brainWithLayerActivations(uint layers, uint first, uint second, uint third) {
    return layers | ((first & BrainLayerActivationMask) << BrainLayerActivationShift) |
           ((second & BrainLayerActivationMask)
            << (BrainLayerActivationShift + BrainLayerActivationBits)) |
           ((third & BrainLayerActivationMask)
            << (BrainLayerActivationShift + 2u * BrainLayerActivationBits));
}

VKEXP_BRAIN_FN uint brainLayerActivation(uint layers, uint layer) {
    if (layer >= BrainHiddenLayerCapacity) {
        return BrainActivationTanh;
    }
    return (layers >> (BrainLayerActivationShift + layer * BrainLayerActivationBits)) &
           BrainLayerActivationMask;
}

VKEXP_BRAIN_FN uint brainHiddenLayerSize(uint layers, uint layer) {
    if (layer >= BrainHiddenLayerCapacity) {
        return 0u;
    }
    return (layers >> (layer * BrainLayerSizeBits)) & BrainLayerSizeMask;
}

// Only the widths, with any activation bits dropped. What sizes a genome and
// what names a layer plan: two runs that differ only in a squash read the same
// weights, so they must agree on where every weight is.
VKEXP_BRAIN_FN uint brainLayerWidths(uint layers) {
    return layers & ((1u << BrainLayerActivationShift) - 1u);
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

// Two genes per hidden neuron: how much a discharge raises the threshold, and
// the time constant the raise relaxes on. Carried by every genome whatever model
// is selected, for the same reason the gate block is -- switching a model is a
// parameter change and not a reinterpretation of the population.
const uint BrainAdaptationGeneCount = 2u;
const uint BrainAdaptationBumpGene = 0u;
const uint BrainAdaptationDecayGene = 1u;

VKEXP_BRAIN_FN uint brainWeightCount(uint inputCount, uint layers, uint outputCount) {
    const uint forward = brainForwardBlockSize(inputCount, layers);
    return forward + brainLastHiddenSize(layers) * outputCount + outputCount +
           brainHiddenNeuronCount(layers) + forward +
           brainHiddenNeuronCount(layers) * BrainAdaptationGeneCount;
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

VKEXP_BRAIN_FN uint brainAdaptationBlockOffset(uint base, uint inputCount, uint layers,
                                              uint outputCount) {
    return brainGateBlockOffset(base, inputCount, layers, outputCount) +
           brainForwardBlockSize(inputCount, layers);
}

// Neurons numbered across all layers end to end, the same way the states and
// the time constants are.
VKEXP_BRAIN_FN uint brainAdaptationGeneIndex(uint base, uint inputCount, uint layers,
                                             uint outputCount, uint neuron, uint gene) {
    return brainAdaptationBlockOffset(base, inputCount, layers, outputCount) +
           neuron * BrainAdaptationGeneCount + gene;
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
