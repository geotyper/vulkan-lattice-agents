#include "vkexp/compute/ComputeResources.hpp"
#include "vkexp/simulation/LatticeBindings.hpp"
#include "vkexp/graphics/TransparencyKernel.hpp"
#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/evolution/GenomeArchive.hpp"
#include "vkexp/lattice/LatticeKernel.hpp"
#include "vkexp/lattice/LatticeWorld.hpp"
#include "vkexp/neuro/BrainDescription.hpp"
#include "vkexp/neuro/BrainKernel.hpp"
#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/profiling/CpuProfiler.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"
#include "vkexp/simulation/CpuLattice.hpp"
#include "vkexp/simulation/ExperimentSweep.hpp"
#include "vkexp/simulation/LatticeSensors.hpp"
#include "vkexp/simulation/RunSnapshot.hpp"
#include "vkexp/simulation/SimulationState.hpp"
#include "vkexp/simulation/StepParameters.hpp"
#include "vkexp/simulation/StructureShape.hpp"
#include "vkexp/simulation/Units.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <set>
#include <iostream>
#include <iterator>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

bool closeTo(const float left, const float right, const float tolerance = 0.0001F) {
    return std::abs(left - right) < tolerance;
}

void testTimingSeries() {
    vkexp::TimingSeries series;
    series.add(1.0);
    series.add(2.0);
    series.add(3.0);
    series.add(4.0);

    const auto statistics = series.statistics();
    check(statistics.sampleCount == 4, "TimingSeries sample count");
    check(closeTo(statistics.currentMs, 4.0F), "TimingSeries current value");
    check(closeTo(statistics.averageMs, 2.5F), "TimingSeries average");
    check(closeTo(statistics.minimumMs, 1.0F), "TimingSeries minimum");
    check(closeTo(statistics.maximumMs, 4.0F), "TimingSeries maximum");
    check(closeTo(statistics.percentile95Ms, 4.0F), "TimingSeries p95");
}

void testCpuProfiler() {
    constexpr vkexp::ProfileMetricId frameMetric = 0;
    constexpr vkexp::ProfileMetricId cpuWorkMetric = 1;
    constexpr vkexp::ProfileMetricId customMetric = 2;
    vkexp::CpuProfiler profiler;
    profiler.beginFrame();
    profiler.addDuration(customMetric, 1.25);
    const vkexp::CpuProfiler::FrameSample sample = profiler.endFrame(frameMetric, cpuWorkMetric, 3);

    check(sample.wallMilliseconds >= 0.0, "CPU profiler wall time");
    check(sample.cpuMilliseconds >= 0.0, "CPU profiler process time");
    check(profiler.series(frameMetric).statistics().sampleCount == 1, "CPU frame sample");
    check(profiler.series(cpuWorkMetric).statistics().sampleCount == 1, "CPU work sample");
    check(closeTo(profiler.series(customMetric).statistics().currentMs, 1.25F),
          "CPU custom duration");
}

void testDispatchSize() {
    check(vkexp::divideRoundUp(17, 8) == 3, "Rounded-up integer division");
    check(vkexp::divideRoundUp(16, 8) == 2, "Exact integer division");

    const vkexp::DispatchSize groups = vkexp::dispatchSize({1921, 1081, 1}, {8, 8, 1});
    check(groups.x == 241, "Dispatch width");
    check(groups.y == 136, "Dispatch height");
    check(groups.z == 1, "Dispatch depth");

    bool rejectedZero = false;
    try {
        static_cast<void>(vkexp::dispatchSize({1, 1, 1}, {0, 1, 1}));
    } catch (const std::exception&) {
        rejectedZero = true;
    }
    check(rejectedZero, "Zero local size rejection");

    VkPhysicalDeviceLimits limits{};
    limits.maxComputeWorkGroupCount[0] = 1024;
    limits.maxComputeWorkGroupCount[1] = 1024;
    limits.maxComputeWorkGroupCount[2] = 64;
    limits.maxComputeWorkGroupSize[0] = 1024;
    limits.maxComputeWorkGroupSize[1] = 1024;
    limits.maxComputeWorkGroupSize[2] = 64;
    limits.maxComputeWorkGroupInvocations = 1024;
    limits.maxPushConstantsSize = 128;
    limits.maxStorageBufferRange = 4096;
    const std::array<VkDeviceSize, 2> validRanges{1024, 2048};
    vkexp::validateComputeLimits(limits, groups, {8, 8, 1}, 16, validRanges);

    bool rejectedGroupCount = false;
    try {
        vkexp::validateComputeLimits(limits, {1025, 1, 1}, {8, 8, 1});
    } catch (const std::exception&) {
        rejectedGroupCount = true;
    }
    check(rejectedGroupCount, "Dispatch group limit rejection");

    bool rejectedInvocations = false;
    try {
        vkexp::validateComputeLimits(limits, {1, 1, 1}, {64, 64, 1});
    } catch (const std::exception&) {
        rejectedInvocations = true;
    }
    check(rejectedInvocations, "Local invocation limit rejection");

    bool rejectedPushConstants = false;
    try {
        vkexp::validateComputeLimits(limits, {1, 1, 1}, {8, 8, 1}, 132);
    } catch (const std::exception&) {
        rejectedPushConstants = true;
    }
    check(rejectedPushConstants, "Push constant limit rejection");

    const std::array<VkDeviceSize, 1> oversizedRange{8192};
    bool rejectedStorageRange = false;
    try {
        vkexp::validateComputeLimits(limits, {1, 1, 1}, {8, 8, 1}, 0, oversizedRange);
    } catch (const std::exception&) {
        rejectedStorageRange = true;
    }
    check(rejectedStorageRange, "Storage buffer range limit rejection");
}

void testComputeResourceValidation() {
    check(vkexp::tightlyPackedImageSize(VK_FORMAT_R8G8B8A8_UNORM, {4, 4}) == 64,
          "RGBA8 tightly-packed image size");
    check(vkexp::tightlyPackedImageSize(VK_FORMAT_R32G32_SFLOAT, {3, 2}) == 48,
          "RG32 tightly-packed image size");

    bool rejectedUnsupportedFormat = false;
    try {
        static_cast<void>(vkexp::tightlyPackedImageSize(VK_FORMAT_D32_SFLOAT, {4, 4}));
    } catch (const std::exception&) {
        rejectedUnsupportedFormat = true;
    }
    check(rejectedUnsupportedFormat, "Unsupported image transfer format rejection");

    vkexp::ComputePipelineBuilder builder{VK_NULL_HANDLE, VK_NULL_HANDLE};
    builder.specializationConstant(7, std::uint32_t{42});
    bool rejectedDuplicateConstant = false;
    try {
        builder.specializationConstant(7, std::uint32_t{43});
    } catch (const std::exception&) {
        rejectedDuplicateConstant = true;
    }
    check(rejectedDuplicateConstant, "Duplicate specialization constant rejection");

    bool rejectedUnavailableDescriptorSet = false;
    try {
        static_cast<void>(vkexp::PingPongDescriptorSets{}.forReadIndex(0));
    } catch (const std::exception&) {
        rejectedUnavailableDescriptorSet = true;
    }
    check(rejectedUnavailableDescriptorSet, "Unavailable ping-pong descriptor rejection");
}

void testLogicalWorldPartition() {
    constexpr std::uint32_t genomes = 25;
    constexpr std::uint32_t agentsPerWorld = 10;
    constexpr std::uint32_t trials = 4;
    check(vkexp::minimumAgentsPerWorld == 1,
          "World partition permits an agent to have its own world");
    check(vkexp::clampAgentsPerWorld(genomes, 0) == 1,
          "World partition clamps an empty group size to one");
    check(vkexp::clampAgentsPerWorld(genomes, 100) == genomes,
          "World partition supports all agents in one group");
    check(vkexp::worldGroupCount(genomes, 1) == genomes,
          "One-agent worlds create one group per genome");
    check(vkexp::logicalWorldCount(genomes, 1, trials) == genomes * trials,
          "One-agent groups preserve every evaluation trial");
    check(vkexp::agentsInLogicalWorld(genomes, 1, trials, genomes * trials - 1) == 1,
          "The last one-agent trial contains exactly one agent");
    check(vkexp::worldGroupCount(genomes, agentsPerWorld) == 3,
          "World partition rounds up the group count");
    check(vkexp::logicalWorldCount(genomes, agentsPerWorld, trials) == 12,
          "World partition creates one world per group and trial");
    check(vkexp::logicalWorldForAgent(39, agentsPerWorld, trials) == 3,
          "Last agent in the first group stays in its trial world");
    check(vkexp::logicalWorldForAgent(40, agentsPerWorld, trials) == 4,
          "First agent in the second group enters the next set of worlds");
    check(vkexp::logicalWorldForAgent(98, agentsPerWorld, trials) == 10,
          "Partial final group maps to the expected trial world");
    check(vkexp::agentsInLogicalWorld(genomes, agentsPerWorld, trials, 0) == 10,
          "Full logical world reports its agent count");
    check(vkexp::agentsInLogicalWorld(genomes, agentsPerWorld, trials, 8) == 5,
          "Partial logical world reports its agent count");
    check(vkexp::logicalWorldCount(genomes, genomes, trials) == trials,
          "All-agent mode preserves only the evaluation trial worlds");
}

void testPingPongState() {
    vkexp::PingPongBuffer buffers;
    check(buffers.readIndex() == 0 && buffers.writeIndex() == 1, "Initial ping-pong indices");
    buffers.swap();
    check(buffers.readIndex() == 1 && buffers.writeIndex() == 0, "Swapped ping-pong indices");
    buffers.swap();
    check(buffers.readIndex() == 0 && buffers.writeIndex() == 1, "Restored ping-pong indices");
}

void testNeuralNetworkContract() {
    vkexp::neuro::Weights weights = vkexp::neuro::makeWeights(vkexp::neuro::maximumBrainShape);
    vkexp::neuro::Inputs inputs{};
    inputs.fill(1.0F);
    const vkexp::neuro::Outputs outputs = vkexp::neuro::evaluate(weights, inputs);
    for (const float output : outputs) {
        check(closeTo(output, 0.0F), "Zero neural network output");
    }
    // Derived from the preset rather than pinned to a snapshot: adding a sensor
    // is meant to be a one-line edit in BrainKernel.inl, not a test rewrite.
    namespace kernel = vkexp::neuro::kernel;
    check(vkexp::neuro::Topology::inputCount ==
              kernel::BrainNeighborCount * kernel::BrainNeighborChannels +
                  kernel::BrainBeaconInputCount + kernel::BrainSelfInputCount +
                  kernel::BrainRecurrentCount,
          "Input capacity is the sum of the declared sensor blocks");
    check(vkexp::neuro::Topology::outputCount ==
              kernel::BrainActuatorOutputCount + kernel::BrainRecurrentCount,
          "Output capacity is actuators plus recurrent cells");
    check(kernel::BrainActuatorOutputCount ==
              kernel::BrainTurnOutputCount + kernel::BrainActionOutputCount +
                  kernel::BrainSignalOutputCount,
          "Actuators are the turn, the action and the broadcast");
    check(vkexp::neuro::Topology::maximumWeightCount ==
              vkexp::neuro::maximumBrainShape.weightCount(),
          "Genome capacity matches the widest brain shape");

    // The brain's neighbourhood block and the lattice's neighbourhood are
    // separate constants in separate kernels, because one compiles into
    // vkexp::neuro::kernel and the other into vkexp::lattice::kernel while GLSL
    // has neither namespace. This is where they are held to each other -- the
    // same arrangement the 2D build used for the body radius it shared with the
    // scenario kernel.
    check(kernel::BrainNeighborCount == vkexp::lattice::kernel::LatticeNeighborCount,
          "The input vector has one slot per cell of the lattice neighbourhood");
    check(kernel::BrainNeighborChannels == 4,
          "A neighbour reads as occupied, a wall, a block and broadcasting");
    check(kernel::BrainTurnOutputCount == 2 && kernel::BrainActionOutputCount == 1,
          "Turning is two signed outputs that must agree, acting is one thresholded output");

    // The sensor blocks must tile the input vector without gaps or overlaps.
    check(kernel::BrainNeighborOffset == 0, "The neighbourhood block starts the input vector");
    check(kernel::brainNeighborChannelIndex(kernel::BrainNeighborCount - 1,
                                            kernel::BrainNeighborChannels - 1) +
                  1 ==
              kernel::BrainBeaconOffset,
          "The beacon block follows the neighbourhood block");
    check(kernel::BrainBeaconOffset + kernel::BrainBeaconInputCount == kernel::BrainSelfOffset,
          "The self block follows the beacon block");
    check(kernel::BrainSelfOffset + kernel::BrainSelfInputCount ==
              kernel::BrainRecurrentInputOffset,
          "Recurrent inputs follow the self block");
    check(kernel::BrainRecurrentInputOffset + kernel::BrainRecurrentCount ==
              kernel::BrainInputCapacity,
          "Recurrent inputs close the input vector");

    // The default plan is what a run gets when the settings say nothing, and it
    // is deliberately not the capacity: raising how many neurons there may be
    // must not widen every run's brain behind its back.
    const vkexp::SimulationStep defaults{};
    const vkexp::neuro::BrainShape plan = vkexp::resolvedBrain(defaults);
    check(plan.inputCount == vkexp::neuro::Topology::inputCount &&
              plan.outputCount == vkexp::neuro::Topology::outputCount,
          "The default plan uses the lattice's own input and output widths");
    check(plan.hiddenCount == vkexp::neuro::Topology::defaultHiddenCount &&
              plan.hiddenLayerCount() == 1,
          "The default plan is one hidden layer of the default width");

    vkexp::SimulationStep deep{};
    deep.hiddenLayers = {12, 8, 8};
    const vkexp::neuro::BrainShape deepPlan = vkexp::resolvedBrain(deep);
    check(deepPlan.hiddenLayerCount() == 3 && deepPlan.hiddenTotal() == 28,
          "A stated hidden plan replaces the default one");

    const std::uint32_t layout = vkexp::neuro::packBrainLayout(plan);
    const std::uint32_t layers = plan.packedLayers();
    const vkexp::neuro::BrainShape unpacked = vkexp::neuro::brainShape(layout, layers);
    check(unpacked.inputCount == plan.inputCount && unpacked.hiddenCount == plan.hiddenCount &&
              unpacked.outputCount == plan.outputCount &&
              unpacked.hiddenLayerCount() == plan.hiddenLayerCount(),
          "The packed GPU layout and layer plan preserve the active shape");
}

void testFitnessWeightsAreParameters() {
    vkexp::AgentState agent{};
    // best nearness, contacts, effort, refusals
    agent.metrics = {0.5F, 4.0F, 10.0F, 3.0F};

    vkexp::FitnessWeights base{};
    const float reference = vkexp::agentFitness(agent, base);

    vkexp::FitnessWeights doubledBonus = base;
    doubledBonus.objectiveBonus = base.objectiveBonus * 2.0F;
    check(closeTo(vkexp::agentFitness(agent, doubledBonus),
                  reference + base.objectiveBonus * agent.metrics.y),
          "Objective bonus is a parameter, not a literal");

    vkexp::FitnessWeights freeMotors = base;
    freeMotors.motorCostWeight = 0.0F;
    check(closeTo(vkexp::agentFitness(agent, freeMotors),
                  reference + agent.metrics.z * base.motorCostWeight),
          "Motor cost weight is a parameter, not a literal");

    vkexp::FitnessWeights freeRefusals = base;
    freeRefusals.refusalPenalty = 0.0F;
    check(closeTo(vkexp::agentFitness(agent, freeRefusals),
                  reference + agent.metrics.w * base.refusalPenalty),
          "Refusal penalty is a parameter, not a literal");

    // Walking into a neighbour has to cost more than walking around one, or the
    // pressure the lattice is meant to apply points the wrong way.
    check(base.refusalPenalty > base.motorCostWeight,
          "A refused move costs more than a move that worked");
}

void testBrainForwardPass() {
    namespace bk = vkexp::neuro::kernel;

    // Uniform everything. With every weight and bias set to w and every input to
    // x, the whole network collapses to a chain that can be written down:
    //
    //   a0 = w * (1 + n_inputs * x)         h0 = tanh(a0)
    //   ak = w * (1 + n_(k-1) * h_(k-1))    hk = tanh(ak)
    //   y  = tanh(w * (1 + n_last * h_last))
    //
    // The counts in it are exactly the connectivity: a layer reading the wrong
    // number of sources, or reading the input vector when it should read the
    // layer before it, moves the answer.
    const auto uniformExpectation = [](const vkexp::neuro::BrainShape& shape, const float w,
                                       const float x) {
        float signal = static_cast<float>(shape.inputCount) * x;
        for (std::size_t layer = 0; layer < shape.hiddenLayerCount(); ++layer) {
            const float activation = std::tanh(w * (1.0F + signal));
            signal = static_cast<float>(shape.hiddenLayer(layer)) * activation;
        }
        return std::tanh(w * (1.0F + signal));
    };

    struct Case {
        vkexp::neuro::BrainShape shape;
        const char* what;
    };
    // Several topologies, and deliberately not only the shipping ones: a one
    // neuron layer and a widening plan are where an off-by-one in a source count
    // shows up as something other than a rounding difference.
    const std::array<Case, 6> cases{{
        {vkexp::neuro::defaultBrainShape, "the default 78 -> 20 -> 6"},
        {{57, 20, 5}, "a trimmed 57 -> 20 -> 5"},
        {{8, 4, 5}, "a small 8 -> 4 -> 5"},
        {{4, 1, 5}, "a single hidden neuron"},
        {{8, 4, 5, 3, 2}, "three layers narrowing"},
        {{8, 2, 5, 5, 7}, "three layers widening"},
    }};
    for (const Case& item : cases) {
        check(item.shape.fitsCapacity(), std::string{"Test topology fits: "} + item.what);
        // Chosen so nothing saturates: at tanh's flat end every wrong answer
        // rounds to the right one, and the test would pass on a broken sum.
        const float w = 0.5F / (1.0F + static_cast<float>(item.shape.inputCount));
        vkexp::neuro::Weights weights = vkexp::neuro::makeWeights(item.shape);
        std::fill(weights.begin(), weights.end(), w);
        vkexp::neuro::Inputs inputs{};
        inputs.fill(1.0F);

        vkexp::neuro::HiddenState state{};
        const vkexp::neuro::Outputs outputs = vkexp::neuro::evaluate(
            weights, inputs, state, 1.0F, bk::NeuronModelReactive, item.shape);
        const float expected = uniformExpectation(item.shape, w, 1.0F);
        bool everyOutput = true;
        for (std::size_t output = 0; output < item.shape.outputCount; ++output) {
            everyOutput = everyOutput && closeTo(outputs[output], expected);
        }
        check(everyOutput,
              std::string{"Uniform weights give the hand-computed output on "} + item.what);
        // And every output is the same number, because every output neuron sees
        // the same layer through the same weights. One that differed would mean
        // an output row reaching somewhere its neighbours do not.
        check(closeTo(outputs[0], outputs[item.shape.outputCount - 1]),
              std::string{"Every output neuron reads the same last layer on "} + item.what);
    }

    // A matrix that is uniform cannot catch its own transpose, so the second
    // case makes every weight distinct and drives one input at a time. Under the
    // reactive model the state *is* the pre-activation, so what comes back is
    // the single weight that was addressed -- and if rows and columns were
    // swapped it would be a different one.
    const vkexp::neuro::BrainShape wired{6, 3, 5};
    const bk::uint layers = wired.packedLayers();
    const auto sources = static_cast<bk::uint>(wired.inputCount);
    const auto weightFor = [](const bk::uint neuron, const bk::uint source) {
        return 0.1F * static_cast<float>(neuron + 1) + 0.01F * static_cast<float>(source + 1);
    };
    // Written by the test's own arithmetic, not by the kernel's index function.
    // That is the point: filling the genome through the same function that reads
    // it would make a transposed layout invisible, because the test would write
    // and read the same wrong place. The layout being asserted is the documented
    // one -- the first layer starts the genome, one contiguous row per neuron,
    // sources in order, biases after the last row.
    vkexp::neuro::Weights wiring = vkexp::neuro::makeWeights(wired);
    constexpr bk::uint wiredNeurons = 3;
    for (bk::uint neuron = 0; neuron < wiredNeurons; ++neuron) {
        for (bk::uint source = 0; source < sources; ++source) {
            wiring[neuron * sources + source] = weightFor(neuron, source);
        }
    }
    // And the kernel agrees about where that is, which is the other half of the
    // claim: the layout above is the one the shader walks, not a second opinion.
    check(bk::brainLayerWeightIndex(0u, sources, layers, 0u, 2u, 1u) == 2u * sources + 1u &&
              bk::brainLayerBiasIndex(0u, sources, layers, 0u, 1u) == wiredNeurons * sources + 1u,
          "The kernel addresses the first layer row by row, biases after the rows");
    for (bk::uint source = 0; source < sources; ++source) {
        vkexp::neuro::Inputs oneHot{};
        oneHot[source] = 1.0F;
        vkexp::neuro::HiddenState state{};
        (void)vkexp::neuro::evaluate(wiring, oneHot, state, 1.0F, bk::NeuronModelReactive, wired);
        bool addressed = true;
        for (bk::uint neuron = 0; neuron < 3; ++neuron) {
            addressed = addressed && closeTo(state[neuron], weightFor(neuron, source));
        }
        check(addressed, "One input drives exactly the weights that connect it to each neuron");
    }

    // Two inputs at once: the neuron adds them. A layer that took the last
    // source, or the largest, would pass the one-hot case above and fail here.
    {
        vkexp::neuro::Inputs twoHot{};
        twoHot[1] = 1.0F;
        twoHot[4] = 1.0F;
        vkexp::neuro::HiddenState state{};
        (void)vkexp::neuro::evaluate(wiring, twoHot, state, 1.0F, bk::NeuronModelReactive, wired);
        bool summed = true;
        for (bk::uint neuron = 0; neuron < 3; ++neuron) {
            summed =
                summed && closeTo(state[neuron], weightFor(neuron, 1u) + weightFor(neuron, 4u));
        }
        check(summed, "Two live inputs are summed, not chosen between");
    }

    // Scaling: an input of 2 contributes twice what an input of 1 does. Anything
    // treating the input as a flag rather than a value passes everything above.
    {
        vkexp::neuro::Inputs scaled{};
        scaled[2] = 2.0F;
        vkexp::neuro::HiddenState state{};
        (void)vkexp::neuro::evaluate(wiring, scaled, state, 1.0F, bk::NeuronModelReactive, wired);
        check(closeTo(state[0], 2.0F * weightFor(0u, 2u)),
              "An input's value scales its weight rather than switching it on");
    }

    // The bias is added once, and only to its own neuron.
    {
        vkexp::neuro::Weights biased = vkexp::neuro::makeWeights(wired);
        biased[wiredNeurons * sources + 1u] = 0.75F;
        vkexp::neuro::HiddenState state{};
        (void)vkexp::neuro::evaluate(biased, vkexp::neuro::Inputs{}, state, 1.0F,
                                     bk::NeuronModelReactive, wired);
        check(closeTo(state[0], 0.0F) && closeTo(state[1], 0.75F) && closeTo(state[2], 0.0F),
              "A bias reaches its own neuron, once, with no input at all");
    }

    // And the second layer reads the first layer's *activation*, not its state.
    // The two are different numbers whenever the state is outside tanh's linear
    // part, which is exactly where a controller spends its time.
    {
        const vkexp::neuro::BrainShape chain{4, 2, 5, 1, 0};
        const bk::uint chainLayers = chain.packedLayers();
        const auto chainInputs = static_cast<bk::uint>(chain.inputCount);
        vkexp::neuro::Weights weights = vkexp::neuro::makeWeights(chain);
        // Two first-layer neurons driven to clearly different, clearly nonlinear
        // places, and one second-layer neuron summing both with unit weights.
        weights[bk::brainLayerBiasIndex(0u, chainInputs, chainLayers, 0u, 0u)] = 1.4F;
        weights[bk::brainLayerBiasIndex(0u, chainInputs, chainLayers, 0u, 1u)] = -0.9F;
        weights[bk::brainLayerWeightIndex(0u, chainInputs, chainLayers, 1u, 0u, 0u)] = 1.0F;
        weights[bk::brainLayerWeightIndex(0u, chainInputs, chainLayers, 1u, 0u, 1u)] = 1.0F;
        vkexp::neuro::HiddenState state{};
        (void)vkexp::neuro::evaluate(weights, vkexp::neuro::Inputs{}, state, 1.0F,
                                     bk::NeuronModelReactive, chain);
        const float throughActivations = std::tanh(1.4F) + std::tanh(-0.9F);
        const float throughStates = 1.4F - 0.9F;
        check(closeTo(state[2], throughActivations),
              "A deeper layer reads the activations in front of it");
        check(!closeTo(throughActivations, throughStates),
              "and the two readings really are different numbers here");
    }

    // Finally the same uniform chain under the time-constant model, one step from
    // rest: the integrator scales the step by the neuron's own time constant, so
    // this says the genes reach the neurons they belong to as well as that the
    // sums are right.
    {
        const vkexp::neuro::BrainShape shape{8, 4, 5, 3, 0};
        const float w = 0.05F;
        const float step = 1.0F / 60.0F;
        vkexp::neuro::Weights weights = vkexp::neuro::makeWeights(shape);
        std::fill(weights.begin(), weights.end(), w);
        vkexp::neuro::Inputs inputs{};
        inputs.fill(1.0F);
        vkexp::neuro::HiddenState state{};
        (void)vkexp::neuro::evaluate(weights, inputs, state, step, bk::NeuronModelTimeConstant,
                                     shape);
        // Every gene is w, so every neuron runs at the same rate.
        const float rate = std::min(step / bk::brainTimeConstant(w), 1.0F);
        const float firstActivation = w * (1.0F + 8.0F * 1.0F);
        const float firstState = rate * firstActivation;
        const float secondActivation = w * (1.0F + 4.0F * std::tanh(firstState));
        check(closeTo(state[0], firstState) && closeTo(state[4], rate * secondActivation),
              "The integrator scales each layer's own sum by the time constant it was given");
    }
}

void testLayeredBrain() {
    namespace bk = vkexp::neuro::kernel;
    const vkexp::neuro::BrainShape flat{8, 4, 5};
    const vkexp::neuro::BrainShape deep{8, 4, 5, 3, 2};

    check(flat.hiddenLayerCount() == 1 && flat.hiddenTotal() == 4,
          "One width is one layer, and every scenario that wrote three numbers still means that");
    check(deep.hiddenLayerCount() == 3 && deep.hiddenTotal() == 9,
          "Three widths are three layers and their total");

    // A hole is refused rather than closed up: {4, 0, 2} could mean a two-layer
    // plan or a mistake, and guessing between them is worse than saying no.
    const vkexp::neuro::BrainShape holed{8, 4, 5, 0, 2};
    check(!holed.fitsCapacity(), "A plan with a hole in the middle is refused");
    const vkexp::neuro::BrainShape overspent{8, vkexp::neuro::Topology::hiddenNeuronCapacity, 8,
                                             vkexp::neuro::Topology::hiddenNeuronCapacity, 0};
    check(!overspent.fitsCapacity(), "A plan spending more neurons than there are is refused");
    check(deep.fitsCapacity() && flat.fitsCapacity(), "and the plans that do fit are accepted");

    // Every layer's states live end to end in the one block on the agent, so no
    // two neurons may share a slot. A collision would make a deep brain hold one
    // memory where it thinks it holds two.
    const bk::uint layers = deep.packedLayers();
    check(bk::brainHiddenLayerStateOffset(layers, 0u) == 0 &&
              bk::brainHiddenLayerStateOffset(layers, 1u) == 4 &&
              bk::brainHiddenLayerStateOffset(layers, 2u) == 7,
          "Each layer's states begin after the layers before it");
    check(bk::brainLayerSourceCount(8u, layers, 0u) == 8 &&
              bk::brainLayerSourceCount(8u, layers, 1u) == 4 &&
              bk::brainLayerSourceCount(8u, layers, 2u) == 3,
          "A layer reads the inputs first and the layer before it after that");

    // The blocks of a deep plan tile the genome exactly, the same claim
    // testBrainDescription makes for the flat one.
    const vkexp::neuro::BrainDescription description = vkexp::neuro::describeBrain(deep);
    std::uint32_t cursor = 0;
    bool tiles = true;
    for (const vkexp::neuro::BrainBlock& block : description.weights) {
        tiles = tiles && block.offset == cursor && block.count > 0;
        cursor += block.count;
    }
    check(tiles && cursor == description.weightCount,
          "A three-layer genome is tiled by its blocks with no gap and no overlap");
    check(description.hiddenLayers == std::vector<std::uint32_t>{4, 3, 2},
          "and the description says how the neurons are divided");

    // And the arithmetic: a chain of three neurons, one per layer, each reading
    // only the one before it. Under the reactive model the state is the
    // activation outright, so the whole network is a composition of tanh and the
    // expected value can be written down.
    vkexp::neuro::Weights weights = vkexp::neuro::makeWeights(deep);
    const auto inputs8 = static_cast<bk::uint>(deep.inputCount);
    weights[bk::brainLayerWeightIndex(0u, inputs8, layers, 0u, 0u, 0u)] = 1.5F;
    weights[bk::brainLayerWeightIndex(0u, inputs8, layers, 1u, 0u, 0u)] = 1.25F;
    weights[bk::brainLayerWeightIndex(0u, inputs8, layers, 2u, 0u, 0u)] = 1.75F;
    weights[bk::brainOutputWeightIndex(0u, inputs8, layers, 0u, 0u)] = 2.0F;
    vkexp::neuro::Inputs inputs{};
    inputs[0] = 0.8F;

    vkexp::neuro::HiddenState state{};
    const vkexp::neuro::Outputs deepOut =
        vkexp::neuro::evaluate(weights, inputs, state, 1.0F, bk::NeuronModelReactive, deep);
    const float expected =
        std::tanh(2.0F * std::tanh(1.75F * std::tanh(1.25F * std::tanh(1.5F * 0.8F))));
    check(closeTo(deepOut[0], expected),
          "Three layers compose: each one reads the one before it and nothing else");

    // The same weights under a one-layer plan cannot give the same answer, or
    // the depth is being ignored somewhere and every assertion above is about a
    // network nobody is running.
    vkexp::neuro::HiddenState flatState{};
    const vkexp::neuro::Outputs flatOut =
        vkexp::neuro::evaluate(weights, inputs, flatState, 1.0F, bk::NeuronModelReactive, flat);
    check(std::abs(flatOut[0] - deepOut[0]) > 1.0e-3F,
          "and a flat plan on the same weights is a different network, not the same one");

    // Each layer holds its own state, which is what depth is for under the time
    // constant models: a slow layer behind a fast one.
    vkexp::neuro::HiddenState settled{};
    for (int step = 0; step < 4; ++step) {
        (void)vkexp::neuro::evaluate(weights, inputs, settled, 1.0F / 60.0F,
                                     bk::NeuronModelTimeConstant, deep);
    }
    check(std::abs(settled[0]) > 0.0F && std::abs(settled[4]) > 0.0F,
          "Neurons in the second layer carry state of their own");
    // The regression this constant exists to prevent, asserted rather than
    // remembered: raising how many neurons there *may* be must not widen any
    // world's brain behind its back. A run gets twenty hidden neurons unless it
    // is asked for something else, and the capacity is a separate number that
    // happens to be larger.
    check(vkexp::neuro::defaultBrainShape.hiddenTotal() == 20 &&
              vkexp::neuro::Topology::hiddenNeuronCapacity > 20,
          "The default width and the neuron capacity are different numbers");

    // And the genome is as long as the plan reading it, not as long as the
    // widest plan there could be. This is what lets a file say which network it
    // holds instead of every run sharing one length.
    const vkexp::neuro::BrainShape wide{vkexp::neuro::Topology::inputCount,
                                        vkexp::neuro::Topology::hiddenNeuronCapacity,
                                        vkexp::neuro::Topology::outputCount};
    check(deep.weightCount() < vkexp::neuro::defaultBrainShape.weightCount() &&
              vkexp::neuro::defaultBrainShape.weightCount() < wide.weightCount(),
          "A deeper plan is shorter than the flat default, which is shorter than the widest");
    check(vkexp::neuro::makeWeights(deep).size() == deep.weightCount(),
          "A genome is made exactly as long as its own plan");
}

void testBrainDescription() {
    namespace bk = vkexp::neuro::kernel;
    const vkexp::neuro::BrainShape shape = vkexp::neuro::maximumBrainShape;
    const vkexp::neuro::BrainDescription description = vkexp::neuro::describeBrain(shape);

    check(description.inputCount == shape.inputCount &&
              description.hiddenCount == shape.hiddenCount &&
              description.outputCount == shape.outputCount &&
              description.weightCount == shape.weightCount(),
          "The description reports the shape it was asked for");

    // Every block tiles its vector: consecutive, no gap, no overlap, ending
    // exactly at the count. A gap is a slot nothing names -- a sensor that would
    // be silently unreachable -- and an overlap is two names for one number.
    const auto tiles = [](const std::vector<vkexp::neuro::BrainBlock>& blocks,
                          const std::uint32_t total) {
        std::uint32_t cursor = 0;
        for (const vkexp::neuro::BrainBlock& block : blocks) {
            if (block.offset != cursor || block.count == 0) {
                return false;
            }
            cursor += block.count;
        }
        return cursor == total;
    };
    check(tiles(description.inputs, description.inputCount),
          "The sensor blocks tile the input vector exactly");
    check(tiles(description.outputs, description.outputCount),
          "The actuator blocks tile the output vector exactly");
    check(tiles(description.weights, description.weightCount),
          "The weight blocks tile the genome exactly");

    // And a matrix block's shape has to account for its own size, or "20x61"
    // is decoration rather than a claim.
    bool shapesAgree = true;
    for (const vkexp::neuro::BrainBlock& block : description.weights) {
        if (block.isMatrix()) {
            shapesAgree = shapesAgree && block.rows * block.columns == block.count;
        }
    }
    check(shapesAgree, "A matrix block's rows times columns is its own size");

    // The claim that makes the description usable rather than decorative: every
    // index the shader computes lands inside the block that names it. Checked at
    // the corners, which is where an off-by-one lands.
    const auto inputs = static_cast<bk::uint>(shape.inputCount);
    const auto hidden = static_cast<bk::uint>(shape.hiddenCount);
    const auto outputs = static_cast<bk::uint>(shape.outputCount);
    const bk::uint layers = shape.packedLayers();
    const auto inside = [&](const char* name, const bk::uint index) {
        const vkexp::neuro::BrainBlock* const block = description.block(name);
        return block != nullptr && index >= block->offset && index < block->offset + block->count;
    };
    check(inside("hidden0_weights", bk::brainLayerWeightIndex(0u, inputs, layers, 0u, 0u, 0u)) &&
              inside("hidden0_weights",
                     bk::brainLayerWeightIndex(0u, inputs, layers, 0u, hidden - 1u, inputs - 1u)),
          "Both corners of the input-to-hidden matrix fall in its block");
    check(inside("hidden0_bias", bk::brainLayerBiasIndex(0u, inputs, layers, 0u, 0u)) &&
              inside("hidden0_bias", bk::brainLayerBiasIndex(0u, inputs, layers, 0u, hidden - 1u)),
          "Both ends of the hidden bias fall in its block");
    check(inside("output_weights", bk::brainOutputWeightIndex(0u, inputs, layers, 0u, 0u)) &&
              inside("output_weights",
                     bk::brainOutputWeightIndex(0u, inputs, layers, outputs - 1u, hidden - 1u)),
          "Both corners of the hidden-to-output matrix fall in its block");
    check(inside("output_bias", bk::brainOutputBiasIndex(0u, inputs, layers, outputs, 0u)) &&
              inside("output_bias",
                     bk::brainOutputBiasIndex(0u, inputs, layers, outputs, outputs - 1u)),
          "Both ends of the output bias fall in its block");
    check(
        inside("time_constants", bk::brainTimeConstantGeneIndex(0u, inputs, layers, outputs, 0u)) &&
            inside("time_constants",
                   bk::brainTimeConstantGeneIndex(0u, inputs, layers, outputs, hidden - 1u)),
        "Both ends of the time constants fall in their block");
    check(inside("gate0_weights",
                 bk::brainGateWeightIndex(0u, inputs, layers, outputs, 0u, 0u, 0u)) &&
              inside("gate0_weights", bk::brainGateWeightIndex(0u, inputs, layers, outputs, 0u,
                                                               hidden - 1u, inputs - 1u)),
          "Both corners of the gate matrix fall in its block");
    check(inside("gate0_bias", bk::brainGateBiasIndex(0u, inputs, layers, outputs, 0u, 0u)) &&
              inside("gate0_bias",
                     bk::brainGateBiasIndex(0u, inputs, layers, outputs, 0u, hidden - 1u)),
          "Both ends of the gate bias fall in their block");

    // And the sensor blocks against the sensor index functions, which is the
    // half a weight-block check cannot reach.
    check(
        inside("neighbourhood", bk::brainNeighborChannelIndex(0u, 0u)) &&
            inside("neighbourhood", bk::brainNeighborChannelIndex(bk::BrainNeighborCount - 1u,
                                                                  bk::BrainNeighborChannels - 1u)),
        "The neighbourhood block covers every cell channel");
    check(inside("task", bk::brainBeaconInputIndex(0u)) &&
              inside("task", bk::brainBeaconInputIndex(bk::BrainBeaconInputCount - 1u)),
          "The task block covers every task-specific channel");
    check(inside("turn", bk::BrainTurnOutput) && inside("action", bk::BrainActionOutput) &&
              inside("signal", bk::BrainSignalIntensityOutput) &&
              inside("memory_out", bk::BrainRecurrentOutputOffset),
          "Every named output slot falls in the block that claims it");

    // Through JSON and back unchanged. This is what an archive carries, so a
    // round trip that loses a field would lose it silently in every file.
    const std::string json = vkexp::neuro::brainDescriptionToJson(description);
    const vkexp::neuro::BrainDescription parsed = vkexp::neuro::parseBrainDescription(json);
    check(vkexp::neuro::compareBrainDescriptions(description, parsed).empty(),
          "A description survives JSON in both directions");
    check(vkexp::neuro::brainDescriptionToJson(parsed) == json,
          "and writing it again produces the same document");

    // A trimmed scenario describes a smaller network, not a broken one.
    const vkexp::neuro::BrainDescription trimmed = vkexp::neuro::describeBrain({52, 20, 5});
    check(tiles(trimmed.inputs, 52) && tiles(trimmed.weights, trimmed.weightCount),
          "A trimmed shape still tiles both vectors");
    check(!vkexp::neuro::compareBrainDescriptions(description, trimmed).empty(),
          "and is reported as different from the full one");

    // The parser is strict, because a structure file that is quietly half-read
    // describes a network nobody has.
    const auto rejects = [](const std::string& text) {
        try {
            (void)vkexp::neuro::parseBrainDescription(text);
        } catch (const vkexp::neuro::BrainDescriptionError&) {
            return true;
        }
        return false;
    };
    check(rejects("{ \"hidden_count\": 20 }"), "A description missing its counts is rejected");
    check(rejects("{ \"mystery\": 1 }"), "An unknown field is rejected rather than ignored");
    check(rejects(json.substr(0, json.size() / 2)), "A truncated document is rejected");
    check(rejects(json + "{}"), "Trailing content is rejected");

    // And a difference is reported by name, since "block 4 moved" helps nobody.
    vkexp::neuro::BrainDescription moved = description;
    moved.inputs.front().count += 1;
    const std::vector<std::string> differences =
        vkexp::neuro::compareBrainDescriptions(description, moved);
    check(!differences.empty() && differences.front().find("neighbourhood") != std::string::npos,
          "A moved block is reported by its own name");
}

void testGenomeArchiveRoundTrip() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "vkexp_archive_test" / "population.vkng";
    std::vector<vkexp::Genome> genomes(
        3, vkexp::Genome{vkexp::neuro::makeWeights(vkexp::neuro::BrainShape{
               70, 20, vkexp::neuro::Topology::actuatorOutputCount})});
    for (std::size_t index = 0; index < genomes.size(); ++index) {
        for (std::size_t weight = 0; weight < genomes[index].weights.size(); ++weight) {
            genomes[index].weights[weight] =
                std::sin(static_cast<float>(index * 31 + weight) * 0.017F);
        }
    }
    // A trimmed shape rather than the default one, so the file has something to
    // say that the build would not have assumed. Trimmed down to the actuators
    // and no further: an output vector shorter than the actuators names a brain
    // that cannot drive the world, and the loader refuses it.
    const vkexp::neuro::BrainShape archivePlan{70, 20,
                                               vkexp::neuro::Topology::actuatorOutputCount};
    const vkexp::GenomeArchiveMetadata metadata{
        42,
        4,
        0xC0FFEEU,
        1.5F,
        0.25F,
        70,
        20,
        static_cast<std::uint32_t>(vkexp::neuro::Topology::actuatorOutputCount),
        archivePlan.packedLayers()};
    vkexp::saveGenomeArchive(path, genomes, metadata);

    const vkexp::GenomeArchive loaded = vkexp::loadGenomeArchive(path);
    check(loaded.genomes.size() == genomes.size(), "Archive genome count round-trip");
    check(loaded.metadata.generation == 42, "Archive generation round-trip");
    check(loaded.metadata.beaconSeed == 4, "Archive beacon seed round-trip");
    check(loaded.metadata.seed == 0xC0FFEEU, "Archive seed round-trip");
    check(closeTo(loaded.metadata.bestFitness, 1.5F), "Archive best fitness round-trip");
    check(loaded.metadata.brainOutputCount == vkexp::neuro::Topology::actuatorOutputCount,
          "Archive brain shape round-trip");
    bool identical = true;
    for (std::size_t index = 0; index < genomes.size(); ++index) {
        identical = identical && loaded.genomes[index].weights == genomes[index].weights;
    }
    check(identical, "Archive weights round-trip bit-exactly");
    check(loaded.describedStructure, "An archive states the structure its weights are laid out in");
    check(loaded.description.inputCount == 70 && loaded.description.hiddenCount == 20 &&
              loaded.description.outputCount == vkexp::neuro::Topology::actuatorOutputCount,
          "and states it for the shape the run actually used");

    // The whole reason the structure is in the file: a file whose weights mean
    // something else has to fail, and fail by naming what moved. Patched in
    // place and byte for byte -- "neighbourhood" becomes "neighbourhooX", same
    // length, so the header's byte count still matches and nothing but the
    // meaning changes.
    const std::filesystem::path renamed = path.parent_path() / "renamed.vkng";
    std::filesystem::copy_file(path, renamed, std::filesystem::copy_options::overwrite_existing);
    {
        std::fstream stream{renamed, std::ios::binary | std::ios::in | std::ios::out};
        std::string contents{std::istreambuf_iterator<char>{stream},
                             std::istreambuf_iterator<char>{}};
        const std::size_t at = contents.find("neighbourhood");
        check(at != std::string::npos, "The structure block is really in the file as text");
        stream.clear();
        stream.seekp(static_cast<std::streamoff>(at));
        stream.write("neighbourhooX", 13);
    }
    std::string complaint;
    try {
        (void)vkexp::loadGenomeArchive(renamed);
    } catch (const vkexp::GenomeArchiveError& error) {
        complaint = error.what();
    }
    check(complaint.find("neighbourhood") != std::string::npos &&
              complaint.find("neighbourhooX") != std::string::npos,
          "A file describing a different network is refused, and both names are said");

    // Version 1 files predate the structure block and used to load, because the
    // weights were laid out the same way and the file simply did not say so.
    // They are refused now. The lattice changed what every weight addresses --
    // the input vector is a neighbourhood where it was a photoreceptor array --
    // so a file from before it is not an old version of this format, it is a
    // description of a different network that happens to be the same length.
    // Built by surgery on a current file, because there is no writer for the old
    // format any more: version at byte 4, structure length at byte 60, header 64.
    const std::filesystem::path legacy = path.parent_path() / "legacy.vkng";
    {
        std::ifstream input{path, std::ios::binary};
        std::string contents{std::istreambuf_iterator<char>{input},
                             std::istreambuf_iterator<char>{}};
        std::uint32_t structureBytes = 0;
        std::memcpy(&structureBytes, contents.data() + 60, sizeof(structureBytes));
        check(structureBytes > 0, "A current file records how long its structure block is");
        const std::uint32_t one = 1;
        std::memcpy(contents.data() + 4, &one, sizeof(one));
        const std::uint32_t none = 0;
        std::memcpy(contents.data() + 60, &none, sizeof(none));
        contents.erase(64, structureBytes);
        std::ofstream output{legacy, std::ios::binary | std::ios::trunc};
        output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    }
    bool rejectedLegacy = false;
    try {
        (void)vkexp::loadGenomeArchive(legacy);
    } catch (const vkexp::GenomeArchiveError&) {
        rejectedLegacy = true;
    }
    check(rejectedLegacy, "An archive written before the lattice is refused rather than reread");

    // A file holds whatever network it was written under, and says which. This
    // is what replaced one compiled-in genome length: interchangeability now
    // comes from the file describing itself, so an archive of a three-layer
    // brain is a perfectly good file even in a run set up for a flat one.
    const vkexp::neuro::BrainShape deepPlan{64, 12, 5, 8, 8};
    const std::filesystem::path deepPath = path.parent_path() / "deep.vkng";
    std::vector<vkexp::Genome> deepGenomes(2, vkexp::Genome{vkexp::neuro::makeWeights(deepPlan)});
    deepGenomes.front().weights.front() = 0.5F;
    const vkexp::GenomeArchiveMetadata deepMetadata{
        7,
        5,
        1U,
        0.5F,
        0.25F,
        64,
        static_cast<std::uint32_t>(deepPlan.hiddenTotal()),
        static_cast<std::uint32_t>(deepPlan.outputCount),
        deepPlan.packedLayers()};
    vkexp::saveGenomeArchive(deepPath, deepGenomes, deepMetadata);
    const vkexp::GenomeArchive deepLoaded = vkexp::loadGenomeArchive(deepPath);
    check(deepLoaded.genomes.front().weights.size() == deepPlan.weightCount(),
          "An archive of a three-layer brain comes back at that brain's length");
    check(deepLoaded.description.hiddenLayers == std::vector<std::uint32_t>{12, 8, 8},
          "and says which three layers they were");
    check(deepLoaded.genomes.front().weights.front() == 0.5F, "and the weights survive it");

    // A corrupted magic must fail loudly rather than load noise as a population.
    const std::filesystem::path corrupted = path.parent_path() / "corrupted.vkng";
    std::filesystem::copy_file(path, corrupted, std::filesystem::copy_options::overwrite_existing);
    {
        std::fstream stream{corrupted, std::ios::binary | std::ios::in | std::ios::out};
        stream.seekp(0);
        stream.write("XXXX", 4);
    }
    bool rejectedMagic = false;
    try {
        (void)vkexp::loadGenomeArchive(corrupted);
    } catch (const vkexp::GenomeArchiveError&) {
        rejectedMagic = true;
    }
    check(rejectedMagic, "Archive rejects a foreign file");

    // Truncation must not yield a half-filled population either.
    const std::filesystem::path truncated = path.parent_path() / "truncated.vkng";
    std::filesystem::copy_file(path, truncated, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) - 16);
    bool rejectedTruncation = false;
    try {
        (void)vkexp::loadGenomeArchive(truncated);
    } catch (const vkexp::GenomeArchiveError&) {
        rejectedTruncation = true;
    }
    check(rejectedTruncation, "Archive rejects a truncated file");

    std::error_code cleanupError;
    std::filesystem::remove_all(path.parent_path(), cleanupError);
}

void testGroupFitnessSharing() {
    // Two worlds of three, deliberately lopsided: one strong genome beside two
    // weak ones, and a flat group that must come back untouched at any setting.
    const std::array<float, 6> individual{9.0F, 0.0F, 0.0F, 2.0F, 2.0F, 2.0F};

    const std::vector<float> off = vkexp::shareFitnessWithinGroups(individual, 3, 0.0F);
    check(std::equal(off.begin(), off.end(), individual.begin()),
          "Sharing at zero returns the scores unchanged");

    const std::vector<float> full = vkexp::shareFitnessWithinGroups(individual, 3, 1.0F);
    check(closeTo(full[0], 3.0F) && closeTo(full[1], 3.0F) && closeTo(full[2], 3.0F),
          "Full sharing scores a whole world together");
    check(closeTo(full[3], 2.0F) && closeTo(full[4], 2.0F) && closeTo(full[5], 2.0F),
          "Full sharing leaves an already uniform world alone");

    const std::vector<float> half = vkexp::shareFitnessWithinGroups(individual, 3, 0.5F);
    check(closeTo(half[0], 6.0F) && closeTo(half[1], 1.5F),
          "Half sharing sits midway between the genome and its world");

    // The mean of a group is what sharing must not move: a blend cannot invent
    // or destroy fitness, only redistribute it inside a world. Selection
    // pressure between worlds therefore survives at every setting.
    for (const float share : {0.0F, 0.25F, 0.5F, 1.0F}) {
        const std::vector<float> blended = vkexp::shareFitnessWithinGroups(individual, 3, share);
        const float before = std::accumulate(individual.begin(), individual.begin() + 3, 0.0F);
        const float after = std::accumulate(blended.begin(), blended.begin() + 3, 0.0F);
        check(closeTo(before, after), "Sharing conserves a world's total fitness");
    }

    // A population that does not divide evenly by the group size leaves a short
    // last world, which must be averaged over its real members rather than
    // reading past the end or diluting against absent ones.
    const std::array<float, 5> ragged{4.0F, 0.0F, 0.0F, 6.0F, 0.0F};
    const std::vector<float> raggedShared = vkexp::shareFitnessWithinGroups(ragged, 3, 1.0F);
    check(closeTo(raggedShared[0], 4.0F / 3.0F) && closeTo(raggedShared[3], 3.0F) &&
              closeTo(raggedShared[4], 3.0F),
          "A short last world averages over its own members");

    check(vkexp::shareFitnessWithinGroups(individual, 0, 1.0F)[0] > 8.0F,
          "A zero group size cannot divide by zero");
}

void testRunSnapshotRoundTrip() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "vkexp_run_snapshot_test.vklr";
    std::error_code removeError;
    std::filesystem::remove(path, removeError);

    vkexp::RunSnapshot snapshot;
    // Every field set to something other than its default, so a field that is
    // saved but not loaded -- or loaded into the wrong slot -- shows up as a
    // value that did not come back rather than as a default that happened to
    // match.
    snapshot.settings.deltaTime = 1.0F / 90.0F;
    snapshot.settings.latticeWidth = 24;
    snapshot.settings.latticeHeight = 20;
    snapshot.settings.latticeDepth = 12;
    snapshot.settings.turnThreshold = 0.4F;
    snapshot.settings.beaconContactRadius = 3;
    snapshot.settings.beaconSeed = 0xFACEU;
    snapshot.settings.neighborhood = vkexp::Neighborhood::Faces;
    snapshot.settings.hiddenLayers = {12, 8, 4};
    snapshot.settings.neuronModel = vkexp::NeuronModel::Gated;
    snapshot.settings.fitness.trackingReward = 0.9F;
    snapshot.settings.fitness.objectiveBonus = 0.11F;
    snapshot.settings.fitness.motorCostWeight = 0.013F;
    snapshot.settings.fitness.refusalPenalty = 0.017F;
    snapshot.settings.fitness.signalCostFactor = 0.31F;
    snapshot.settings.fitness.groupSharing = 0.7F;
    snapshot.generation = 91;
    snapshot.step = 37;
    snapshot.stepsPerGeneration = 600;
    snapshot.requestedAgentsPerWorld = 11;
    snapshot.trialsPerGenome = 3;
    snapshot.seed = 0xC0FFEEU;

    const vkexp::neuro::BrainShape shape = vkexp::resolvedBrain(snapshot.settings);
    snapshot.genomes.assign(4, vkexp::Genome{vkexp::neuro::makeWeights(shape)});
    snapshot.genomes[2].weights[5] = -1.75F;
    snapshot.agents.resize(12);
    snapshot.agents[7].cell = {5, 6, 7, 13};
    snapshot.agents[7].intent = {5, 6, 7, 1};
    snapshot.agents[7].beacon = {1, 2, 3, 4};
    snapshot.agents[7].metrics = {0.5F, 2.0F, 9.0F, 4.0F};
    snapshot.agents[7].memory = {0.25F, -0.5F, 0.0F, 0.0F};
    snapshot.agents[7].hidden[1].z = 0.125F;

    vkexp::saveRunSnapshot(path, snapshot);
    const vkexp::RunSnapshot loaded = vkexp::loadRunSnapshot(path);

    check(loaded.generation == snapshot.generation && loaded.step == snapshot.step &&
              loaded.stepsPerGeneration == snapshot.stepsPerGeneration &&
              loaded.requestedAgentsPerWorld == snapshot.requestedAgentsPerWorld &&
              loaded.trialsPerGenome == snapshot.trialsPerGenome && loaded.seed == snapshot.seed,
          "A run snapshot round-trips its run header");
    check(closeTo(loaded.settings.deltaTime, snapshot.settings.deltaTime) &&
              closeTo(loaded.settings.turnThreshold, snapshot.settings.turnThreshold),
          "A run snapshot round-trips its float settings");
    check(loaded.settings.latticeWidth == 24 && loaded.settings.latticeHeight == 20 &&
              loaded.settings.latticeDepth == 12,
          "A run snapshot round-trips the lattice it ran in");
    check(loaded.settings.beaconContactRadius == 3 && loaded.settings.beaconSeed == 0xFACEU &&
              loaded.settings.neighborhood == vkexp::Neighborhood::Faces &&
              loaded.settings.neuronModel == vkexp::NeuronModel::Gated,
          "A run snapshot round-trips its integer settings");
    check(loaded.settings.hiddenLayers == snapshot.settings.hiddenLayers,
          "A run snapshot round-trips the brain plan");
    check(closeTo(loaded.settings.fitness.trackingReward, 0.9F) &&
              closeTo(loaded.settings.fitness.objectiveBonus, 0.11F) &&
              closeTo(loaded.settings.fitness.motorCostWeight, 0.013F) &&
              closeTo(loaded.settings.fitness.refusalPenalty, 0.017F) &&
              closeTo(loaded.settings.fitness.signalCostFactor, 0.31F) &&
              closeTo(loaded.settings.fitness.groupSharing, 0.7F),
          "A run snapshot round-trips every fitness weight");
    check(loaded.genomes.size() == snapshot.genomes.size() &&
              loaded.genomes[2].weights.size() == shape.weightCount() &&
              closeTo(loaded.genomes[2].weights[5], -1.75F),
          "A run snapshot round-trips the population");
    check(loaded.agents.size() == snapshot.agents.size() &&
              std::memcmp(&loaded.agents[7], &snapshot.agents[7], sizeof(vkexp::AgentState)) == 0,
          "A run snapshot round-trips every byte of an agent record");

    // The occupancy grid is deliberately absent from the file, because it is a
    // function of where everybody stands. This is the claim that lets it be
    // left out: rebuilding it from the agents gives the grid back exactly.
    vkexp::SimulationStep small{};
    small.latticeWidth = 6;
    small.latticeHeight = 6;
    small.latticeDepth = 4;
    const vkexp::lattice::PopulationLayout layout{12, 10, 2};
    const std::vector<vkexp::AgentState> spawned = vkexp::lattice::makeInitialAgents(small, layout);
    std::vector<std::int32_t> first(static_cast<std::size_t>(vkexp::latticeCellsPerWorld(small)) *
                                    layout.worldCount());
    std::vector<std::int32_t> second(first.size());
    vkexp::lattice::buildOccupancy(spawned, small, layout, first);
    vkexp::lattice::buildOccupancy(spawned, small, layout, second);
    check(first == second &&
              std::count(first.begin(), first.end(), vkexp::lattice::kernel::LatticeNoOccupant) ==
                  static_cast<std::ptrdiff_t>(first.size() - spawned.size()),
          "The occupancy grid is exactly recoverable from the agents, so it need not be saved");

    // A file written for another agent layout is refused rather than
    // reinterpreted: every offset in the record would otherwise shift silently.
    {
        std::fstream stream{path, std::ios::binary | std::ios::in | std::ios::out};
        stream.seekp(20); // agentStateBytes, the sixth uint32 of the header
        const std::uint32_t wrong = sizeof(vkexp::AgentState) + 16;
        stream.write(reinterpret_cast<const char*>(&wrong), sizeof(wrong));
    }
    bool rejected = false;
    try {
        (void)vkexp::loadRunSnapshot(path);
    } catch (const vkexp::RunSnapshotError&) {
        rejected = true;
    }
    check(rejected, "A run snapshot written for another agent layout is refused");

    std::filesystem::remove(path, removeError);
}

void testLayerActivation() {
    namespace bk = vkexp::neuro::kernel;

    // The squash rides in the same word as the widths, so the first thing to
    // establish is that it does not disturb them. A plan whose widths shifted
    // when a layer changed its squash would be a genome laid out differently for
    // two networks that must read the same weights.
    const vkexp::neuro::BrainShape plain{40, 12, 6, 8, 0};
    vkexp::neuro::BrainShape sine = plain;
    sine.hiddenActivation = {bk::BrainActivationTanh, bk::BrainActivationSine,
                             bk::BrainActivationTanh};
    check(plain.packedLayers() != sine.packedLayers(),
          "A layer's squash is part of the packed plan");
    check(plain.packedWidths() == sine.packedWidths(),
          "and it is not part of the widths, which is what lays a genome out");
    check(plain.weightCount() == sine.weightCount(),
          "so two plans that differ only in a squash are the same length");
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        check(bk::brainHiddenLayerSize(sine.packedLayers(), layer) ==
                  bk::brainHiddenLayerSize(plain.packedLayers(), layer),
              "and every layer is still as wide as it was");
    }
    check(bk::brainLayerActivation(sine.packedLayers(), 0) == bk::BrainActivationTanh &&
              bk::brainLayerActivation(sine.packedLayers(), 1) == bk::BrainActivationSine &&
              bk::brainLayerActivation(sine.packedLayers(), 2) == bk::BrainActivationTanh,
          "The squash reads back out of the plan, layer by layer");
    check(bk::brainLayerActivation(plain.packedLayers(), 1) == bk::BrainActivationTanh,
          "A plan that says nothing about squashes means tanh, which is what every "
          "plan written before they existed meant");

    // And the one model that does not reach a squash at all does not record one.
    // A spiking run that carried "sin" in its plan would write an archive
    // claiming a network it was never trained as, and then refuse to load into
    // the run that produced it.
    vkexp::SimulationStep settings{};
    settings.hiddenActivation = {bk::BrainActivationSine, bk::BrainActivationSine,
                                 bk::BrainActivationSine};
    settings.neuronModel = vkexp::NeuronModel::Reactive;
    check(vkexp::resolvedBrain(settings).hiddenActivation[0] == bk::BrainActivationSine,
          "A run that reaches a squash keeps the one it asked for");
    settings.neuronModel = vkexp::NeuronModel::Spiking;
    check(vkexp::resolvedBrain(settings).hiddenActivation[0] == bk::BrainActivationTanh,
          "and a spiking run records no squash, because it never reaches one");

    // And the squash itself. Sine is offered on hidden layers only; the check
    // that matters about it is that it is not tanh, at a value where the two
    // would otherwise be easy to confuse.
    check(closeTo(bk::brainLayerActivate(bk::BrainActivationTanh, 0.5F, 20U), std::tanh(0.5F)),
          "Tanh is what it always was, and does not look at the fan-in");
    check(closeTo(bk::brainLayerActivate(bk::BrainActivationSine, 0.5F, 20U), std::sin(0.5F)),
          "and sine is sine");
    check(bk::brainLayerActivate(bk::BrainActivationSine, 3.0F, 20U) <
              bk::brainLayerActivate(bk::BrainActivationSine, 1.0F, 20U),
          "Sine is not monotone, which is the whole objection to it and the reason it "
          "is offered rather than imposed");

    // The two controls. Scaled tanh divides the sum by the square root of what
    // it summed, which is the same answer the initialisation gives at zero
    // parameters; softsign saturates like tanh but reaches its asymptote an
    // order of magnitude later.
    check(closeTo(bk::brainLayerActivate(bk::BrainActivationTanhScaled, 4.0F, 16U),
                  std::tanh(1.0F)),
          "Scaled tanh divides its sum by the square root of the fan-in");
    check(bk::brainLayerActivate(bk::BrainActivationTanhScaled, 4.0F, 16U) <
              bk::brainLayerActivate(bk::BrainActivationTanh, 4.0F, 16U),
          "so a wide layer is squashed less hard than an unscaled one");
    check(closeTo(bk::brainLayerActivate(bk::BrainActivationSoftsign, 3.0F, 20U), 0.75F),
          "Softsign is x over one plus its magnitude");
    check(bk::brainLayerActivate(bk::BrainActivationSoftsign, 3.0F, 20U) <
              bk::brainLayerActivate(bk::BrainActivationTanh, 3.0F, 20U),
          "which at the same input is further from saturated than tanh");
    for (const float value : {-6.0F, -1.0F, 0.0F, 1.0F, 6.0F}) {
        check(std::abs(bk::brainLayerActivate(bk::BrainActivationSoftsign, value, 20U)) < 1.0F,
              "and still bounded, which is what keeps it a decision a neuron can hold");
    }
}

void testRandomWeights() {
    namespace bk = vkexp::neuro::kernel;
    const vkexp::neuro::BrainShape shape{40, 12, 6, 8, 0};
    std::mt19937 random{0x51EEDU};
    constexpr float spread = 0.55F;
    const vkexp::neuro::Weights weights = vkexp::neuro::randomWeights(shape, random, true, spread);

    check(weights.size() == shape.weightCount(), "A fresh genome is as long as its plan");
    // Every gene was written. Nothing here draws an exact zero with any
    // probability worth naming, so a zero is a block the walk missed -- which is
    // the failure this guards, since a missed block is a network with a dead
    // layer that still runs and still scores.
    check(std::count(weights.begin(), weights.end(), 0.0F) == 0,
          "Every gene of a fresh genome was drawn, so no block was skipped");

    const auto inputs = static_cast<bk::uint>(shape.inputCount);
    const auto outputs = static_cast<bk::uint>(shape.outputCount);
    const bk::uint layers = shape.packedLayers();
    const auto deviation = [&](const std::vector<std::size_t>& indices) {
        double sum = 0.0;
        for (const std::size_t at : indices) {
            sum += static_cast<double>(weights[at]) * static_cast<double>(weights[at]);
        }
        return std::sqrt(sum / static_cast<double>(indices.size()));
    };

    // Each block is drawn at a width that follows its own fan-in, which is the
    // whole point: one width for every gene saturates a wide layer and leaves a
    // narrow one timid. Measured rather than asserted from the constant, because
    // what matters is what came out.
    std::vector<std::size_t> firstLayer;
    for (bk::uint neuron = 0; neuron < 12; ++neuron) {
        for (bk::uint source = 0; source < inputs; ++source) {
            firstLayer.push_back(bk::brainLayerWeightIndex(0U, inputs, layers, 0U, neuron, source));
        }
    }
    std::vector<std::size_t> outputLayer;
    for (bk::uint neuron = 0; neuron < outputs; ++neuron) {
        for (bk::uint hidden = 0; hidden < 8; ++hidden) {
            outputLayer.push_back(bk::brainOutputWeightIndex(0U, inputs, layers, neuron, hidden));
        }
    }
    const double firstExpected = spread / std::sqrt(40.0);
    const double outputExpected = spread / std::sqrt(8.0);
    check(std::abs(deviation(firstLayer) - firstExpected) < 0.25 * firstExpected,
          "The first layer is drawn at a width that follows its input count");
    check(std::abs(deviation(outputLayer) - outputExpected) < 0.25 * outputExpected,
          "and the output layer at one that follows the last hidden width");
    check(deviation(outputLayer) > 1.5 * deviation(firstLayer),
          "so a narrow layer is drawn wider than a wide one, per weight");

    // The time constants are not weights. Each is read through a sigmoid onto a
    // rate, so narrowing them would pull every neuron toward the same middle
    // instead of spreading them over the range the model offers.
    std::vector<std::size_t> rates;
    for (bk::uint neuron = 0; neuron < shape.hiddenTotal(); ++neuron) {
        rates.push_back(bk::brainTimeConstantGeneIndex(0U, inputs, layers, outputs, neuron));
    }
    check(std::abs(deviation(rates) - spread) < 0.35 * spread,
          "The time-constant genes keep the width they are read at");

    // And the other policy, which is what the thresholds this project ships were
    // tuned against: one width everywhere, so a wide layer saturates. Held to the
    // same coverage rule, because a mode that skipped a block would be a mode
    // that quietly runs a network with a dead layer.
    std::mt19937 flatRandom{0x51EEDU};
    const vkexp::neuro::Weights flat =
        vkexp::neuro::randomWeights(shape, flatRandom, false, spread);
    check(std::count(flat.begin(), flat.end(), 0.0F) == 0,
          "The flat policy draws every gene too");
    std::vector<float> flatFirst;
    for (const std::size_t at : firstLayer) {
        flatFirst.push_back(flat[at]);
    }
    double flatSum = 0.0;
    for (const float weight : flatFirst) {
        flatSum += static_cast<double>(weight) * static_cast<double>(weight);
    }
    const double flatDeviation = std::sqrt(flatSum / static_cast<double>(flatFirst.size()));
    check(std::abs(flatDeviation - spread) < 0.25 * spread,
          "Flat means what it says: the widest layer is drawn at the same width as the rest");
    check(flatDeviation > 3.0 * deviation(firstLayer),
          "which for forty inputs is several times wider per weight");
}

void testPopulationReload() {
    const vkexp::EvolutionSettings settings{
        8, 2, 3, 0.5F, 0.1F, 0.2F, 42U, vkexp::neuro::defaultBrainShape};
    vkexp::GeneticAlgorithm evolution{settings};
    std::vector<vkexp::Genome> replacement(
        settings.populationSize,
        vkexp::Genome{vkexp::neuro::makeWeights(vkexp::neuro::defaultBrainShape)});
    replacement.front().weights[0] = 3.25F;
    evolution.setPopulation(replacement, 17);
    check(evolution.generation() == 17, "Loaded population restores the generation counter");
    check(closeTo(evolution.population().front().weights[0], 3.25F),
          "Loaded population replaces the weights");

    bool rejectedMismatch = false;
    try {
        const std::vector<vkexp::Genome> wrongSize(
            settings.populationSize - 1,
            vkexp::Genome{vkexp::neuro::makeWeights(vkexp::neuro::defaultBrainShape)});
        evolution.setPopulation(wrongSize, 0);
    } catch (const std::invalid_argument&) {
        rejectedMismatch = true;
    }
    check(rejectedMismatch, "Loaded population size mismatch rejection");
}

void testStepParameterPacking() {
    // The GPU step parameters outgrew the 128 bytes Vulkan guarantees for push
    // constants, which is why they travel in a storage buffer. What matters now
    // is only that the block stays a whole number of 16-byte vectors: that is
    // the alignment both languages round it to, and a stride the two disagree
    // about is invisible at step zero and nonsense at every step after it.
    check(sizeof(vkexp::GpuStepParameters) % 16 == 0,
          "The step parameter block is a whole number of 16-byte vectors");

    vkexp::SimulationStep settings{};
    settings.latticeWidth = 20;
    settings.latticeHeight = 16;
    settings.latticeDepth = 8;
    settings.neighborhood = vkexp::Neighborhood::Faces;
    settings.turnThreshold = 0.4F;
    settings.beaconContactRadius = 2;
    settings.neuronModel = vkexp::NeuronModel::Spiking;
    settings.hiddenLayers = {12, 8, 0};
    settings.fitness.signalCostFactor = 0.31F;

    const vkexp::StepParameterLayout layout{
        .agentCount = 96, .trialsPerGenome = 4, .agentsPerWorld = 12, .worldCount = 32};
    const vkexp::GpuStepParameters packed = vkexp::packStepParameters(settings, layout);

    check(packed.agentCount == 96 && packed.trialsPerGenome == 4 && packed.agentsPerWorld == 12 &&
              packed.worldCount == 32,
          "The population layout reaches the shader");
    check(packed.latticeWidth == 20 && packed.latticeHeight == 16 && packed.latticeDepth == 8 &&
              packed.cellsPerWorld == 20 * 16 * 8,
          "The lattice extents and their product reach the shader");
    check(packed.neighborhood == static_cast<std::uint32_t>(vkexp::Neighborhood::Faces),
          "The neighbourhood reaches the shader");
    // Derived on the host so every invocation of every pass does not recompute
    // it. Under a faces-only neighbourhood the axes advance one at a time, so
    // the longest journey is the sum of the spans rather than the largest.
    check(packed.maximumDistance == (20 - 1) + (16 - 1) + (8 - 1),
          "The longest journey is packed as a Manhattan distance under faces");
    check(closeTo(packed.turnThreshold, 0.4F) && packed.beaconContactRadius == 2 &&
              packed.neuronModel == static_cast<std::uint32_t>(vkexp::NeuronModel::Spiking),
          "The movement and neuron settings reach the shader");
    check(closeTo(packed.fitness.signalCostFactor, 0.31F),
          "The step-time fitness weights reach the shader");

    const vkexp::neuro::BrainShape shape = vkexp::resolvedBrain(settings);
    check(packed.brainLayout == vkexp::neuro::packBrainLayout(shape) &&
              packed.brainHiddenLayers == shape.packedLayers() &&
              packed.brainGenomeStride == shape.weightCount(),
          "The brain plan and the genome stride reach the shader");

    // Under Moore the three axes advance together, so the same box is a shorter
    // journey. Getting this wrong scales the beacon shaping by up to three.
    settings.neighborhood = vkexp::Neighborhood::Moore;
    check(vkexp::packStepParameters(settings, layout).maximumDistance == 20 - 1,
          "The longest journey is packed as a Chebyshev distance under Moore");
}

void testNeuronTimeConstants() {
    namespace kernel = vkexp::neuro::kernel;

    // The gene enters a bounded, logarithmic range. Bounded because an unbounded
    // time constant is either a step or an eternity and neither is a neuron;
    // logarithmic because what a memory is worth is its order of magnitude.
    check(kernel::brainTimeConstant(-40.0F) >= kernel::BrainTimeConstantMinimum * 0.999F,
          "A very negative gene bottoms out at the shortest time constant");
    check(kernel::brainTimeConstant(40.0F) <= kernel::BrainTimeConstantMaximum * 1.001F,
          "A very positive gene tops out at the longest time constant");
    check(kernel::brainTimeConstant(-1.0F) < kernel::brainTimeConstant(0.0F) &&
              kernel::brainTimeConstant(0.0F) < kernel::brainTimeConstant(1.0F),
          "The time constant grows with the gene");
    check(closeTo(kernel::brainTimeConstant(0.0F),
                  std::sqrt(kernel::BrainTimeConstantMinimum * kernel::BrainTimeConstantMaximum),
                  1.0e-4F),
          "A gene of zero lands on the geometric middle of the range");

    // The identity the whole ablation rests on: a time constant of one step
    // makes the update an assignment, so memory off is the memoryless network
    // exactly rather than an approximation of it.
    const float step = vkexp::units::fixedTimeStep;
    check(kernel::brainIntegrateNeuron(0.37F, -0.85F, step, step) == -0.85F,
          "A one-step time constant assigns the activation outright");
    check(kernel::brainIntegrateNeuron(0.37F, -0.85F, step * 0.5F, step) == -0.85F,
          "A time constant shorter than the step cannot overshoot");

    // And the claim that makes a time constant mean something: after tau
    // seconds a neuron has closed 1 - 1/e of the gap to its input, whatever the
    // step rate. That is rule 3e for the brain -- a memory measured in seconds.
    for (const float rate : {30.0F, 60.0F, 240.0F}) {
        const float deltaTime = 1.0F / rate;
        const float timeConstant = 0.5F;
        float state = 0.0F;
        const auto steps = static_cast<int>(timeConstant * rate);
        for (int index = 0; index < steps; ++index) {
            state = kernel::brainIntegrateNeuron(state, 1.0F, timeConstant, deltaTime);
        }
        check(std::abs(state - (1.0F - std::exp(-1.0F))) < 0.02F,
              "One time constant of stepping closes 1 - 1/e of the gap at any rate");
    }

    // The evaluator honours both, and the stateless overload is the memory-off
    // one rather than a second network.
    vkexp::neuro::Weights weights = vkexp::neuro::makeWeights(vkexp::neuro::maximumBrainShape);
    constexpr auto inputCount = static_cast<kernel::uint>(vkexp::neuro::Topology::inputCount);
    constexpr auto hiddenCount =
        static_cast<kernel::uint>(vkexp::neuro::Topology::hiddenNeuronCapacity);
    constexpr kernel::uint layers = kernel::brainPackHiddenLayers(hiddenCount, 0u, 0u);
    weights[kernel::brainLayerWeightIndex(0u, inputCount, layers, 0u, 0u, 0u)] = 3.0F;
    weights[kernel::brainOutputWeightIndex(0u, inputCount, layers, 0u, 0u)] = 3.0F;
    vkexp::neuro::Inputs inputs{};
    inputs[0] = 1.0F;

    vkexp::neuro::HiddenState state{};
    const vkexp::neuro::Outputs memoryless =
        vkexp::neuro::evaluate(weights, inputs, state, step, kernel::NeuronModelReactive);
    check(std::equal(memoryless.begin(), memoryless.end(),
                     vkexp::neuro::evaluate(weights, inputs).begin()),
          "The reactive model is exactly what the stateless evaluator computes");

    // A gene of zero is a quarter-second neuron, so one step must move it a
    // fraction of the way and repeated steps must converge -- a neuron that
    // reached its input immediately would not be holding anything.
    vkexp::neuro::HiddenState remembering{};
    const vkexp::neuro::Outputs firstStep =
        vkexp::neuro::evaluate(weights, inputs, remembering, step, kernel::NeuronModelTimeConstant);
    // Checked on the state and not on the output: two tanh layers compress the
    // difference until a genuinely sluggish neuron still drives the output most
    // of the way, so the output is the wrong place to read a time constant.
    check(closeTo(remembering[0], 3.0F * step / kernel::brainTimeConstant(0.0F), 1.0e-4F),
          "One step moves a remembering neuron exactly dt/tau of the way");
    check(std::abs(firstStep[0]) < std::abs(memoryless[0]),
          "A remembering neuron drives its output less hard on the first step");
    for (int index = 0; index < 400; ++index) {
        (void)vkexp::neuro::evaluate(weights, inputs, remembering, step,
                                     kernel::NeuronModelTimeConstant);
    }
    check(closeTo(remembering[0], 3.0F, 1.0e-3F),
          "Held on a constant input, the neuron converges on its activation");
}

void testGatedNeurons() {
    namespace kernel = vkexp::neuro::kernel;
    constexpr auto inputCount = static_cast<kernel::uint>(vkexp::neuro::Topology::inputCount);
    constexpr auto hiddenCount =
        static_cast<kernel::uint>(vkexp::neuro::Topology::hiddenNeuronCapacity);
    constexpr auto outputCount = static_cast<kernel::uint>(vkexp::neuro::Topology::outputCount);
    constexpr kernel::uint layers = kernel::brainPackHiddenLayers(hiddenCount, 0u, 0u);
    const float step = vkexp::units::fixedTimeStep;

    vkexp::neuro::Weights weights = vkexp::neuro::makeWeights(vkexp::neuro::maximumBrainShape);
    weights[kernel::brainLayerWeightIndex(0u, inputCount, layers, 0u, 0u, 0u)] = 3.0F;
    weights[kernel::brainOutputWeightIndex(0u, inputCount, layers, 0u, 0u)] = 3.0F;
    vkexp::neuro::Inputs inputs{};
    inputs[0] = 1.0F;

    // The claim that makes gated a generalisation rather than a third network:
    // a gate that does not listen to anything is the fixed-time-constant neuron,
    // exactly. Both genes go through the same mapping, so setting the gate bias
    // and the time-constant gene to the same number has to give the same state.
    for (const float gene : {-2.0F, 0.0F, 1.5F}) {
        vkexp::neuro::Weights fixed = weights;
        vkexp::neuro::Weights gated = weights;
        fixed[kernel::brainTimeConstantGeneIndex(0u, inputCount, layers, outputCount, 0u)] = gene;
        gated[kernel::brainGateBiasIndex(0u, inputCount, layers, outputCount, 0u, 0u)] = gene;

        vkexp::neuro::HiddenState fixedState{};
        vkexp::neuro::HiddenState gatedState{};
        for (int index = 0; index < 20; ++index) {
            (void)vkexp::neuro::evaluate(fixed, inputs, fixedState, step,
                                         kernel::NeuronModelTimeConstant);
            (void)vkexp::neuro::evaluate(gated, inputs, gatedState, step, kernel::NeuronModelGated);
        }
        check(closeTo(fixedState[0], gatedState[0], 1.0e-6F),
              "A gate that ignores its inputs is the fixed-time-constant neuron");
    }

    // And the point of the thing: with a weight on it, the same neuron runs at
    // different rates depending on what it is being shown. The gate asks for a
    // time constant, so driving it up makes the neuron hold and leaving it low
    // makes the neuron follow -- the opposite of a GRU update gate, and worth
    // pinning down here because the sign is the easy thing to get backwards.
    vkexp::neuro::Weights listening = weights;
    listening[kernel::brainGateWeightIndex(0u, inputCount, layers, outputCount, 0u, 0u, 1u)] = 8.0F;
    vkexp::neuro::Inputs holding = inputs;
    holding[1] = 1.0F; // drives the gate up, so the neuron should barely move
    vkexp::neuro::HiddenState held{};
    vkexp::neuro::HiddenState following{};
    for (int index = 0; index < 20; ++index) {
        (void)vkexp::neuro::evaluate(listening, holding, held, step, kernel::NeuronModelGated);
        (void)vkexp::neuro::evaluate(listening, inputs, following, step, kernel::NeuronModelGated);
    }
    check(held[0] < following[0] * 0.25F,
          "A gate driven up holds while the same neuron left alone follows");

    // The gate block is real genome, not a reinterpretation of existing weights:
    // it sits after everything else and the count has room for it.
    // The gate block is real genome, not a reinterpretation of existing weights.
    // Asserted against the widest plan, which is what the genome is sized for:
    // under a narrower plan the genome has a tail nothing reads, by design.
    constexpr kernel::uint widest = kernel::brainPackHiddenLayers(
        static_cast<kernel::uint>(vkexp::neuro::Topology::hiddenNeuronCapacity), 0u, 0u);
    constexpr auto capacityInputs = static_cast<kernel::uint>(vkexp::neuro::Topology::inputCount);
    constexpr auto capacityOutputs = static_cast<kernel::uint>(vkexp::neuro::Topology::outputCount);
    check(kernel::brainGateBiasIndex(
              0u, capacityInputs, widest, capacityOutputs, 0u,
              static_cast<kernel::uint>(vkexp::neuro::Topology::hiddenNeuronCapacity) - 1u) ==
              vkexp::neuro::Topology::maximumWeightCount - 1u,
          "The gate block ends exactly at the end of the genome");
    check(kernel::brainGateWeightIndex(0u, inputCount, layers, outputCount, 0u, 0u, 0u) >
              kernel::brainTimeConstantGeneIndex(0u, inputCount, layers, outputCount,
                                                 hiddenCount - 1u),
          "The gate block starts after the time constants");
}

void testSpikingNeuronModel() {
    namespace kernel = vkexp::neuro::kernel;
    constexpr auto inputCount = static_cast<kernel::uint>(vkexp::neuro::Topology::inputCount);
    constexpr auto hiddenCount =
        static_cast<kernel::uint>(vkexp::neuro::Topology::hiddenNeuronCapacity);
    constexpr kernel::uint layers = kernel::brainPackHiddenLayers(hiddenCount, 0u, 0u);
    const float step = vkexp::units::fixedTimeStep;

    vkexp::neuro::Weights weights = vkexp::neuro::makeWeights(vkexp::neuro::maximumBrainShape);
    weights[kernel::brainLayerWeightIndex(0u, inputCount, layers, 0u, 0u, 0u)] = 15.0F;
    weights[kernel::brainOutputWeightIndex(0u, inputCount, layers, 0u, 0u)] = 2.0F;

    vkexp::neuro::Inputs inputs{};
    inputs[0] = 1.0F;

    vkexp::neuro::HiddenState state{};
    bool spiked = false;
    for (int index = 0; index < 50; ++index) {
        (void)vkexp::neuro::evaluate(weights, inputs, state, step, kernel::NeuronModelSpiking);
        if (state[0] == 0.0F && index > 0) {
            spiked = true;
        }
    }
    check(spiked,
          "Spiking LIF neuron accumulates potential, fires spike, and resets membrane potential");
}

// Below this the plateau the straight-line shaping creates has no perceptual way
// out, which is what 450 generations of Two doors demonstrated.
constexpr float minimumTargetVisibility = 0.12F;

void testExperimentSweep() {
    vkexp::SweepState sweep;
    sweep.values = {0.0F, 0.5F, 1.0F};
    sweep.generationsPerStage = 3;
    vkexp::startSweep(sweep);
    check(sweep.running && sweep.stages.size() == 1, "A sweep arms one stage at a time");
    check(closeTo(vkexp::sweepValue(sweep), 0.0F), "A sweep starts at its first value");

    // The boundary is the whole reason this is not a shell loop, so it is what
    // the test pins down: the generation that fills a stage belongs to that
    // stage, and the advance is reported exactly once, on that generation.
    check(!vkexp::recordSweepGeneration(sweep, 1.0F, 0.5F, 0.1F), "No advance mid-stage");
    check(!vkexp::recordSweepGeneration(sweep, 2.0F, 0.6F, 0.2F), "No advance mid-stage");
    check(vkexp::recordSweepGeneration(sweep, 3.0F, 0.7F, 0.3F), "The filling generation advances");
    check(sweep.stages.front().medianFitness.size() == 3,
          "The filling generation is recorded in the stage it filled");
    check(sweep.stages.size() == 2 && closeTo(vkexp::sweepValue(sweep), 0.5F),
          "The next stage is armed at the next value");
    check(sweep.generationsInStage == 0, "A new stage starts empty");
    check(closeTo(sweep.stages.front().arrivalRatio.back(), 0.3F), "Stage curves keep their order");

    for (int generation = 0; generation < 3; ++generation) {
        (void)vkexp::recordSweepGeneration(sweep, 1.0F, 1.0F, 0.5F);
    }
    check(sweep.running && sweep.stages.size() == 3, "The middle stage hands over to the last");

    // The last stage has nothing to hand over to, so it must stop rather than
    // report an advance the caller would act on by restarting a fourth run.
    check(!vkexp::recordSweepGeneration(sweep, 1.0F, 1.0F, 0.5F), "No advance mid-final-stage");
    (void)vkexp::recordSweepGeneration(sweep, 1.0F, 1.0F, 0.5F);
    check(!vkexp::recordSweepGeneration(sweep, 4.0F, 2.0F, 0.9F),
          "The final stage does not advance");
    check(!sweep.running && sweep.stages.size() == 3,
          "A finished sweep stops with every stage kept");
    check(closeTo(sweep.stages.back().bestFitness.back(), 4.0F),
          "The last generation of the last stage is kept");
    check(!vkexp::recordSweepGeneration(sweep, 9.0F, 9.0F, 9.0F),
          "A stopped sweep records nothing further");
    check(sweep.stages.back().bestFitness.size() == 3, "A stopped sweep grows no stage");

    // Results have to outlive the run that made them, or stopping early throws
    // away the comparison the sweep was started for.
    vkexp::startSweep(sweep);
    (void)vkexp::recordSweepGeneration(sweep, 5.0F, 5.0F, 0.5F);
    vkexp::stopSweep(sweep);
    check(!sweep.running && sweep.stages.size() == 1 &&
              closeTo(sweep.stages.front().bestFitness.front(), 5.0F),
          "Stopping a sweep keeps what it measured");

    // A plan that cannot run must not report itself as running, so no caller has
    // to guard against a sweep with no stage in flight.
    vkexp::SweepState empty;
    empty.values.clear();
    vkexp::startSweep(empty);
    check(!empty.running && empty.stages.empty(), "A sweep with no values does not start");
    vkexp::SweepState instant;
    instant.generationsPerStage = 0;
    vkexp::startSweep(instant);
    check(!instant.running, "A sweep with no generations per stage does not start");
    check(!vkexp::recordSweepGeneration(instant, 1.0F, 1.0F, 1.0F),
          "A sweep that never started records nothing");
}

void testGeneticAlgorithm() {
    const vkexp::EvolutionSettings settings{8, 2, 3, 0.5F, 0.1F, 0.2F, 42U};
    vkexp::GeneticAlgorithm evolution{settings};
    const std::vector<vkexp::Genome> original = evolution.population();
    const std::vector<float> fitness{-4.0F, -3.0F, -2.0F, -1.0F, 0.0F, 1.0F, 2.0F, 3.0F};
    const vkexp::GenerationSummary summary = evolution.evolve(fitness);
    check(summary.championIndex == 7, "GA champion selection");
    check(closeTo(summary.bestFitness, 3.0F), "GA best fitness");
    check(evolution.generation() == 1, "GA generation counter");
    check(evolution.population().front().weights == original.back().weights,
          "GA preserves champion as first elite");
}

// --- the lattice ------------------------------------------------------------
//
// LatticeKernel.inl compiles into both languages, and compute_smoke checks that
// the two agree. What it cannot check is whether the shared answer is the right
// one -- two identical implementations of a wrong rule agree perfectly. These
// are the cases that pin the rule itself, and they need no device.

namespace lk = vkexp::lattice::kernel;

// What a compiled shader says it binds, read out of its own SPIR-V.
//
// The binary is a word stream: a five-word header, then instructions whose
// first word carries the length in the top half and the opcode in the bottom.
// Only two decorations are wanted -- DescriptorSet and Binding -- so the walk
// is short, and being able to ask a shader what it declares is worth more than
// the twenty lines it costs.
struct ShaderBindings {
    std::size_t count{};
    bool singleSet{true};
};

[[nodiscard]] ShaderBindings readShaderBindings(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Unable to open the compiled shader " + path.string());
    }
    const std::vector<char> bytes{std::istreambuf_iterator<char>(stream),
                                  std::istreambuf_iterator<char>()};
    if (bytes.size() % sizeof(std::uint32_t) != 0 || bytes.size() < 5 * sizeof(std::uint32_t)) {
        throw std::runtime_error("The compiled shader " + path.string() + " is not a word stream");
    }
    std::vector<std::uint32_t> words(bytes.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), bytes.data(), bytes.size());
    constexpr std::uint32_t spirvMagic = 0x07230203U;
    if (words.front() != spirvMagic) {
        throw std::runtime_error("The compiled shader " + path.string() + " is not SPIR-V");
    }

    constexpr std::uint32_t opDecorate = 71;
    constexpr std::uint32_t decorationDescriptorSet = 34;
    constexpr std::uint32_t decorationBinding = 33;
    std::set<std::uint32_t> bound;
    ShaderBindings result{};
    for (std::size_t index = 5; index < words.size();) {
        const std::uint32_t length = words[index] >> 16U;
        const std::uint32_t opcode = words[index] & 0xffffU;
        if (length == 0 || index + length > words.size()) {
            break;
        }
        if (opcode == opDecorate && length >= 4) {
            if (words[index + 2] == decorationBinding) {
                bound.insert(words[index + 1]);
            } else if (words[index + 2] == decorationDescriptorSet && words[index + 3] != 0) {
                result.singleSet = false;
            }
        }
        index += length;
    }
    result.count = bound.size();
    return result;
}

// Every pass, against the one place the count is written down.
//
// This is the test that would have caught a step shader growing two buffers
// while a descriptor layout stayed at six: the shader is the source, the
// constant is what the driver and the parity harness both build from, and a
// disagreement between them is undefined behaviour rather than an error the
// loader reports.
// The camera's basis, checked against what it is supposed to be.
//
// Sliding is written as closed-form trigonometry rather than as a cross
// product, which is cheaper and easier to get subtly wrong -- a sign that only
// shows up at one heading looks like a control that "feels odd" rather than
// like a bug. So the two claims that make a slide a slide are asserted
// directly: the movement is perpendicular to where the camera is looking, and
// its length is what was asked for.
// The shape of the transparency weight, not its values.
//
// Weighted blended transparency resolves a pixel as a weighted mean, so the
// weight is the method. When it is wrong the pass still runs, still blends and
// still produces a picture -- an evenly averaged one, in which moving any
// single fragment's opacity barely moves the result. There is no error to
// report and nothing to see but a control that does nothing, which is exactly
// what the first version of this did: window depth across the box came out as
// 2.2e5 against 1.3e5, both of which clamped to the same ceiling, so the near
// face of the lattice and the far face weighed the same.
//
// So the assertions are about spread and monotonicity, and the one that matters
// is that the front of the box outweighs the back of it by a wide margin at
// every distance the camera can be at.
void testTransparencyWeight() {
    namespace weight = vkexp::graphics::kernel;

    // The camera lives between these two, in half-diagonals; the box is one
    // half-diagonal in every direction from what it looks at. Framing the whole
    // lattice and pressing against one corner of it have to weigh their
    // fragments the same way -- a spread that collapsed as the camera pulled
    // back would fail precisely when seeing into a crowd is worth most.
    constexpr float nearestCamera = 0.35F;
    constexpr float furthestCamera = 12.0F;

    for (const float distance : {nearestCamera, 1.0F, 2.3F, 6.0F, furthestCamera}) {
        const float front = weight::latticeTransparencyWeight(0.5F, distance - 1.0F, distance);
        const float middle = weight::latticeTransparencyWeight(0.5F, distance, distance);
        const float back = weight::latticeTransparencyWeight(0.5F, distance + 1.0F, distance);
        check(front > middle && middle > back,
              "A nearer fragment always weighs more than a further one");
        // Ten to one is the difference between a surface and a smear. Below
        // that the nearest voxel stops reading as the thing in front.
        check(front > back * 10.0F,
              "The near face of the box outweighs the far face by at least ten to one");
    }

    // Opacity has to keep reaching the weight across its whole range. The
    // published form runs it through min(1, alpha * 10), which is flat for
    // everything above 0.1 -- so the one control a viewer has stops working
    // exactly where it starts being useful.
    float previous = 0.0F;
    for (const float alpha : {0.02F, 0.1F, 0.34F, 0.7F, 1.0F}) {
        const float value = weight::latticeTransparencyWeight(alpha, 2.3F, 2.3F);
        check(value > previous, "A more opaque fragment weighs more, at every opacity");
        previous = value;
    }
    check(closeTo(weight::latticeTransparencyWeight(0.0F, 2.3F, 2.3F), 0.0F),
          "A fully transparent fragment contributes nothing");

    // And the accumulation buffer is half precision, so a crowded pixel has to
    // stay well inside its range. The weight is per fragment; a few hundred of
    // them at the nearest the camera goes is the worst case a viewer can build.
    const float worst =
        weight::latticeTransparencyWeight(1.0F, nearestCamera - 1.0F, nearestCamera) * 512.0F;
    check(worst < 60000.0F, "A crowded pixel cannot overflow a half-float accumulation");
}

void testLatticeCameraSlide() {
    const auto eyeDirection = [](const vkexp::LatticeCamera& camera) {
        return std::array<float, 3>{std::cos(camera.pitch) * std::sin(camera.yaw),
                                    std::sin(camera.pitch),
                                    std::cos(camera.pitch) * std::cos(camera.yaw)};
    };
    const auto dot = [](const std::array<float, 3>& left, const std::array<float, 3>& right) {
        return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    };

    // Straight on, the basis is the screen's own: right is +x and up is +y.
    vkexp::LatticeCamera straight{};
    straight.yaw = 0.0F;
    straight.pitch = 0.0F;
    straight.slide(2.0F, 3.0F);
    check(closeTo(straight.targetX, 2.0F) && closeTo(straight.targetY, 3.0F) &&
              closeTo(straight.targetZ, 0.0F),
          "Looking down the z axis, a slide moves the target across x and y");

    for (const float yaw : {0.0F, 0.7F, 2.1F, -1.3F}) {
        for (const float pitch : {0.0F, 0.6F, -0.9F}) {
            vkexp::LatticeCamera camera{};
            camera.yaw = yaw;
            camera.pitch = pitch;
            const std::array<float, 3> direction = eyeDirection(camera);

            camera.slide(1.0F, 0.0F);
            const std::array<float, 3> right{camera.targetX, camera.targetY, camera.targetZ};
            check(closeTo(std::sqrt(dot(right, right)), 1.0F, 1.0e-4F),
                  "A slide of one moves the target by one");
            check(std::abs(dot(right, direction)) < 1.0e-4F,
                  "Sliding right moves across the view, never along it");
            // Sideways is sideways: a right slide never changes the height,
            // which is what makes the two axes independent controls.
            check(closeTo(right[1], 0.0F, 1.0e-4F), "Sliding right does not change the height");

            vkexp::LatticeCamera raised{};
            raised.yaw = yaw;
            raised.pitch = pitch;
            raised.slide(0.0F, 1.0F);
            const std::array<float, 3> up{raised.targetX, raised.targetY, raised.targetZ};
            check(closeTo(std::sqrt(dot(up, up)), 1.0F, 1.0e-4F),
                  "A slide of one upward moves the target by one");
            check(std::abs(dot(up, direction)) < 1.0e-4F,
                  "Sliding up moves across the view, never along it");
            check(std::abs(dot(up, right)) < 1.0e-4F, "The two slide axes are perpendicular");
            // Up is up. Pitching the camera changes how much of the movement is
            // vertical, never whether it is.
            check(up[1] > 0.0F, "Sliding up raises the target whatever the pitch");
        }
    }
}

void testShaderBindingContract() {
    struct Case {
        const char* shader;
        std::uint32_t bindings;
    };
    const std::array cases{
        Case{"lattice_step.comp.spv", vkexp::latticeStepBindings},
        Case{"lattice_resolve.comp.spv", vkexp::latticeResolveBindings},
        Case{"lattice_clear.comp.spv", vkexp::latticeClearBindings},
        Case{"trail_capture.comp.spv", vkexp::latticeTrailCaptureBindings},
        Case{"layout_echo.comp.spv", vkexp::latticeLayoutEchoBindings},
    };
    for (const Case& probe : cases) {
        const ShaderBindings declared =
            readShaderBindings(std::filesystem::path{VKEXP_SHADER_DIR} / probe.shader);
        check(declared.count == probe.bindings,
              std::string{probe.shader} + " declares " + std::to_string(probe.bindings) +
                  " bound buffers, not " + std::to_string(declared.count));
        // Every layout in the driver is one set. A shader that started using a
        // second one would need a second layout, and counting bindings across
        // both would stop meaning anything.
        check(declared.singleSet, std::string{probe.shader} + " binds everything in set 0");
    }
}

// How many agents a world can be given a cell of its own, and what happens at
// exactly that number.
//
// The slider goes to the whole population, so this is reachable: ask for more
// agents than the lattice has cells and the surplus keep their default corner,
// standing inside each other. That breaks the invariant the arbitration rests
// on -- a cell holds one agent -- before the first step runs.
void testLatticeSpawnCapacity() {
    vkexp::SimulationStep settings{};
    settings.latticeWidth = 4;
    settings.latticeHeight = 4;
    settings.latticeDepth = 4;
    check(vkexp::latticeSpawnCapacity(settings) == 63,
          "A beacon world holds one agent fewer than it has cells, because of the beacon");
    settings.worldMode = vkexp::WorldMode::Construction;
    check(vkexp::latticeSpawnCapacity(settings) == 16,
          "A construction world holds one course, because everybody starts on the floor");
    settings.worldMode = vkexp::WorldMode::Beacon;

    // At exactly the capacity every agent gets a cell of its own, and none of
    // them gets the beacon.
    const std::uint32_t capacity = vkexp::latticeSpawnCapacity(settings);
    const vkexp::lattice::PopulationLayout layout{capacity, capacity, 1};
    const std::vector<vkexp::AgentState> agents =
        vkexp::lattice::makeInitialAgents(settings, layout);
    check(agents.size() == capacity, "A full world spawns every agent it was given");

    const vkexp::Int4 beacon = vkexp::lattice::beaconCell(settings, 0);
    std::set<std::array<std::int32_t, 3>> cells;
    for (const vkexp::AgentState& agent : agents) {
        check(vkexp::lattice::kernel::latticeInBounds(agent.cell.x, agent.cell.y, agent.cell.z,
                                                      settings.latticeWidth,
                                                      settings.latticeHeight,
                                                      settings.latticeDepth),
              "Every spawned agent is inside the lattice");
        check(!(agent.cell.x == beacon.x && agent.cell.y == beacon.y && agent.cell.z == beacon.z),
              "Nobody spawns on the beacon");
        cells.insert({agent.cell.x, agent.cell.y, agent.cell.z});
    }
    check(cells.size() == agents.size(), "A full world stands every agent in a cell of its own");

    // And the occupancy built from them agrees: one owner per cell, as many
    // owners as agents.
    std::vector<std::int32_t> occupancy(
        static_cast<std::size_t>(vkexp::latticeCellsPerWorld(settings)) * layout.worldCount());
    vkexp::lattice::buildOccupancy(agents, settings, layout, occupancy);
    const auto occupied = static_cast<std::size_t>(
        std::count_if(occupancy.begin(), occupancy.end(), [](const std::int32_t owner) {
            return owner != vkexp::lattice::kernel::LatticeNoOccupant;
        }));
    check(occupied == agents.size(), "The occupancy grid holds exactly one cell per agent");
}

void testStructureShape() {
    vkexp::SimulationStep settings{};
    settings.worldMode = vkexp::WorldMode::Construction;
    settings.latticeWidth = 4;
    settings.latticeHeight = 4;
    settings.latticeDepth = 4;
    const std::uint32_t cells = vkexp::latticeCellsPerWorld(settings);

    const auto place = [&](std::vector<std::int32_t>& field, const int x, const int y,
                           const int z) {
        field[vkexp::lattice::kernel::latticeCellIndex(x, y, z, settings.latticeWidth,
                                                       settings.latticeHeight)] = 1;
    };

    // A field of the wrong length is a layout mistake, not a smaller world.
    std::vector<std::int32_t> truncated(cells - 1, 0);
    check(vkexp::measureStructureShape(truncated, settings).blocks == 0,
          "A field that is not one world long measures as nothing");

    std::vector<std::int32_t> empty(cells, 0);
    const vkexp::StructureShape nothing = vkexp::measureStructureShape(empty, settings);
    check(nothing.blocks == 0 && nothing.footprint == 0 && nothing.peak == 0,
          "An empty world has built nothing");
    check(nothing.compactness == 0.0F && nothing.heightSpread == 0.0F,
          "Nothing has no shape either, rather than a divide by zero");

    // One four-high column in a corner: a spire.
    std::vector<std::int32_t> spire(cells, 0);
    for (int y = 0; y < 4; ++y) {
        place(spire, 1, y, 2);
    }
    const vkexp::StructureShape tall = vkexp::measureStructureShape(spire, settings);
    check(tall.blocks == 4 && tall.footprint == 1 && tall.peak == 4,
          "A single column is four blocks standing on one square");
    check(tall.meanHeight == 4.0F && tall.heightSpread == 0.0F,
          "One column is its own mean and has no spread");
    check(tall.compactness == 1.0F, "One column fills its own bounding rectangle");
    check(tall.overhangs == 0 && tall.enclosed == 0, "A solid column floats nothing and roofs nothing");

    // The whole floor: the same block count as four spires, a different shape.
    std::vector<std::int32_t> slab(cells, 0);
    for (int z = 0; z < 4; ++z) {
        for (int x = 0; x < 4; ++x) {
            place(slab, x, 0, z);
        }
    }
    const vkexp::StructureShape flat = vkexp::measureStructureShape(slab, settings);
    check(flat.blocks == 16 && flat.footprint == 16 && flat.peak == 1,
          "A full course is one block on every square");
    check(flat.meanHeight == 1.0F && flat.heightSpread == 0.0F && flat.compactness == 1.0F,
          "A full course is flat, even and solid in plan");

    // Four corner piers with a lintel across one pair: an overhang and a room.
    std::vector<std::int32_t> gate(cells, 0);
    place(gate, 0, 0, 0);
    place(gate, 2, 0, 0);
    place(gate, 0, 1, 0);
    place(gate, 2, 1, 0);
    place(gate, 1, 1, 0); // nothing beneath it
    const vkexp::StructureShape arch = vkexp::measureStructureShape(gate, settings);
    check(arch.blocks == 5 && arch.footprint == 3, "The arch stands on three squares");
    check(arch.overhangs == 1, "Exactly one block of the arch stands on empty space");
    check(arch.enclosed == 1, "And exactly one empty cell is roofed by it");
    check(arch.peak == 2 && arch.meanHeight == 2.0F,
          "Every column of the arch reaches the same height");
    check(std::abs(arch.compactness - 1.0F) < 1.0e-6F,
          "Three squares in a row of three fill their bounding rectangle");

    // Shape is not mass: the spire and a four-square course have the same
    // block count and must not measure the same.
    std::vector<std::int32_t> spread(cells, 0);
    for (int x = 0; x < 4; ++x) {
        place(spread, x, 0, 3);
    }
    const vkexp::StructureShape row = vkexp::measureStructureShape(spread, settings);
    check(row.blocks == tall.blocks, "The row and the spire cost the same number of blocks");
    check(row.peak != tall.peak && row.footprint != tall.footprint,
          "But they are not the same shape, which is the whole point of measuring");
    check(std::abs(row.compactness - 1.0F) < 1.0e-6F,
          "A row of four is its own bounding rectangle, not a quarter of the floor");
}

void testBodyFrame() {
    namespace lk = vkexp::lattice::kernel;

    // Four facings, and turning is a cycle in both directions. A turn that did
    // not come back where it started after four would be a frame that slowly
    // drifts away from the world, which no amount of parity would catch: both
    // sides would drift together.
    for (std::uint32_t facing = 0; facing < lk::LatticeFacingCount; ++facing) {
        std::uint32_t right = facing;
        for (std::uint32_t turn = 0; turn < lk::LatticeFacingCount; ++turn) {
            right = lk::latticeTurn(right, true);
        }
        check(right == facing, "Four turns one way come back to where they started");
        check(lk::latticeTurn(lk::latticeTurn(facing, true), false) == facing,
              "and a turn each way cancels");
        check(lk::latticeTurn(facing, true) != lk::latticeTurn(facing, false),
              "The sign of the output is which way the agent goes");
    }

    // The forward vector is the body's +x turned into the world, and it is a
    // unit cardinal step for every facing -- never a diagonal and never zero.
    for (std::uint32_t facing = 0; facing < lk::LatticeFacingCount; ++facing) {
        const int forwardX = lk::latticeFacingX(facing);
        const int forwardZ = lk::latticeFacingZ(facing);
        check(std::abs(forwardX) + std::abs(forwardZ) == 1,
              "Forward is one cardinal step, whichever way the agent looks");
        check(forwardX == lk::latticeRotatedX(facing, 1, 0) &&
                  forwardZ == lk::latticeRotatedZ(facing, 1, 0),
              "and it is the body's own +x, turned");
    }

    // Rotating and rotating back is the identity. This is the pair the sensors
    // rely on: the neighbourhood is read body-to-world and the objective is read
    // world-to-body, so an inverse that was not one would put the task somewhere
    // the neighbourhood is not.
    for (std::uint32_t facing = 0; facing < lk::LatticeFacingCount; ++facing) {
        const std::uint32_t inverse = (lk::LatticeFacingCount - facing) % lk::LatticeFacingCount;
        for (int x = -2; x <= 2; ++x) {
            for (int z = -2; z <= 2; ++z) {
                const int worldX = lk::latticeRotatedX(facing, x, z);
                const int worldZ = lk::latticeRotatedZ(facing, x, z);
                check(lk::latticeRotatedX(inverse, worldX, worldZ) == x &&
                          lk::latticeRotatedZ(inverse, worldX, worldZ) == z,
                      "Turning a vector into the body frame and back leaves it alone");
            }
        }
    }

    // The turn needs both outputs to agree, which is the whole reason there are
    // two of them. Tested at the corners rather than sampled: the interesting
    // cases are the disagreements, and there are only a few of them.
    const float threshold = 0.25F;
    check(lk::latticeTurnStep(0.9F, 0.9F, threshold) == 1 &&
              lk::latticeTurnStep(-0.9F, -0.9F, threshold) == -1,
          "Two outputs that agree turn the agent the way they agree on");
    check(lk::latticeTurnStep(0.9F, -0.9F, threshold) == 0 &&
              lk::latticeTurnStep(-0.9F, 0.9F, threshold) == 0,
          "Two that point opposite ways cancel");
    check(lk::latticeTurnStep(0.9F, 0.0F, threshold) == 0 &&
              lk::latticeTurnStep(0.0F, -0.9F, threshold) == 0,
          "and one that is sure while the other is undecided is not agreement either");
    check(lk::latticeTurnStep(0.0F, 0.0F, threshold) == 0, "Two silences are a silence");
    check(lk::latticeTurnStep(threshold, threshold, threshold) == 0,
          "Exactly at the threshold is inside the dead zone, for both of them");

    // A rotation and not a reflection: turning preserves lengths, and the left
    // of the agent stays on its left.
    check(lk::latticeRotatedX(1u, 1, 0) == -lk::latticeRotatedX(3u, 1, 0) &&
              lk::latticeRotatedZ(1u, 1, 0) == -lk::latticeRotatedZ(3u, 1, 0),
          "Opposite turns give opposite directions");
}

void testLatticeStillness() {
    namespace lk = vkexp::lattice::kernel;
    // A ramp and not a flag. The point of the input is that it is never twice
    // the same while an agent is stuck: a constant would move the fixed point a
    // deterministic policy settles into rather than give it a way out.
    check(lk::latticeStillness(0.0F) == 0.0F, "An agent that just moved reads nothing");
    const float oneTick = lk::latticeStillness(1.0F);
    const float twoTicks = lk::latticeStillness(2.0F);
    check(oneTick > 0.0F && twoTicks > oneTick,
          "Standing still reads higher every tick, which is what a flag could not do");
    check(lk::latticeStillness(lk::LatticeStillnessSpan) == 1.0F,
          "And saturates at one after the full span");
    check(lk::latticeStillness(lk::LatticeStillnessSpan * 10.0F) == 1.0F,
          "Standing still far longer than that is still one, not more");
}

void testHarvestResource() {
    vkexp::SimulationStep settings{};
    settings.worldMode = vkexp::WorldMode::Harvest;
    settings.latticeWidth = 12;
    settings.latticeHeight = 10;
    settings.latticeDepth = 9;
    settings.resourceHeightLow = 4;
    settings.resourceHeightHigh = 4;
    settings.beaconSeed = 0x5EEDU;

    // Inside the box, at the height it was asked for, and the same answer every
    // time it is asked: the device works this out for itself from the same
    // functions, so a placement that drifted would put the shader's resource
    // somewhere the host never draws.
    std::set<std::array<std::int32_t, 3>> placements;
    for (std::uint32_t world = 0; world < 64; ++world) {
        const vkexp::Int4 cell = vkexp::lattice::resourceCell(settings, world);
        check(vkexp::lattice::kernel::latticeInBounds(cell.x, cell.y, cell.z, settings.latticeWidth,
                                                      settings.latticeHeight,
                                                      settings.latticeDepth),
              "Every world's resource is inside the lattice");
        check(cell.y == 4, "and at the height the setting asked for");
        check(cell.w == static_cast<std::int32_t>(world), "and knows which world it belongs to");
        check(vkexp::lattice::resourceCell(settings, world).x == cell.x,
              "Placement is a function of the world index, not a generator");
        placements.insert({cell.x, cell.y, cell.z});
    }
    check(placements.size() > 16,
          "Sixty-four worlds do not all put the resource in the same column");

    // A resource asked for above the ceiling is clamped rather than wrapped.
    // Wrapping would put it near the floor and make the world quietly easy,
    // which is the opposite of saying the setting was wrong.
    settings.resourceHeightLow = 99;
    settings.resourceHeightHigh = 99;
    check(vkexp::lattice::resourceCell(settings, 0).y ==
              static_cast<std::int32_t>(settings.latticeHeight) - 1,
          "A resource above the ceiling sits on the ceiling, not back near the floor");

    // It never shares a cell with the beacon of the same seed and world: the two
    // are hashed against different constants precisely so that a run switched
    // from one world to the other is a different problem and not the same one.
    settings.resourceHeightLow = 4;
    settings.resourceHeightHigh = 4;
    std::size_t collisions = 0;
    for (std::uint32_t world = 0; world < 64; ++world) {
        const vkexp::Int4 resource = vkexp::lattice::resourceCell(settings, world);
        const vkexp::Int4 beacon = vkexp::lattice::beaconCell(settings, world);
        collisions += resource.x == beacon.x && resource.z == beacon.z ? 1 : 0;
    }
    check(collisions < 16, "The resource and the beacon are not the same placement");

    // Harvest builds, so it spawns on the floor and keeps the construction
    // spawn capacity rather than the beacon one.
    check(vkexp::worldBuilds(vkexp::WorldMode::Harvest) &&
              vkexp::worldBuilds(vkexp::WorldMode::Construction) &&
              !vkexp::worldBuilds(vkexp::WorldMode::Beacon),
          "Harvest is a building world and beacon is not");
    check(vkexp::latticeSpawnCapacity(settings) == settings.latticeWidth * settings.latticeDepth,
          "A harvest world stands its group on the floor, like a construction world");

    // The chasm is the same resource placed under one extra rule: it may only
    // hang over the half nobody can walk to. The split runs along x, so that is
    // a range of columns -- and a resource that strayed back over solid ground
    // would turn the world into a climb, which is the one thing it is not.
    vkexp::SimulationStep chasm = settings;
    chasm.worldMode = vkexp::WorldMode::Chasm;
    chasm.latticeWidth = 24;
    chasm.latticeDepth = 20;
    chasm.chasmGroundWidth = 0; // half
    const std::uint32_t ground = vkexp::latticeGroundWidth(chasm);
    check(ground == 12, "A chasm asking for zero ground gets half the width");
    std::set<std::int32_t> chasmColumns;
    std::set<std::int32_t> chasmRows;
    for (std::uint32_t world = 0; world < 64; ++world) {
        const vkexp::Int4 cell = vkexp::lattice::resourceCell(chasm, world);
        check(cell.x >= static_cast<std::int32_t>(ground),
              "The chasm resource hangs over the open half, never over the ground");
        check(cell.x < static_cast<std::int32_t>(chasm.latticeWidth) && cell.z >= 0 &&
                  cell.z < static_cast<std::int32_t>(chasm.latticeDepth),
              "and still inside the box");
        chasmColumns.insert(cell.x);
        chasmRows.insert(cell.z);
    }
    check(chasmColumns.size() > 4 && chasmRows.size() > 4,
          "The open half is used across both of its axes, not one line of it");

    // Ground is measured along x now, so the spawn plan is ground columns by
    // full depth. Getting this wrong stands agents over the void, which the
    // arbitration cannot represent.
    check(vkexp::latticeSpawnCapacity(chasm) == ground * chasm.latticeDepth,
          "A chasm spawns only on the columns that have bedrock under them");
    check(vkexp::lattice::kernel::latticeGroundColumn(0, ground) &&
              vkexp::lattice::kernel::latticeGroundColumn(
                  static_cast<int>(ground) - 1, ground) &&
              !vkexp::lattice::kernel::latticeGroundColumn(static_cast<int>(ground), ground),
          "Ground runs from x=0 up to the split and stops there");

    // And the terrain agrees with all of it: bedrock under every ground column
    // of every row, nothing over the chasm, and only on the bottom course.
    const std::vector<std::int32_t> terrain = vkexp::lattice::makeTerrain(chasm, 2);
    const std::uint32_t cells = vkexp::latticeCellsPerWorld(chasm);
    std::size_t bedrock = 0;
    for (std::uint32_t world = 0; world < 2; ++world) {
        for (std::uint32_t z = 0; z < chasm.latticeDepth; ++z) {
            for (std::uint32_t x = 0; x < chasm.latticeWidth; ++x) {
                const std::size_t index =
                    static_cast<std::size_t>(world) * cells +
                    vkexp::lattice::kernel::latticeCellIndex(static_cast<int>(x), 0,
                                                             static_cast<int>(z),
                                                             chasm.latticeWidth,
                                                             chasm.latticeHeight);
                const bool solid = terrain[index] == vkexp::lattice::kernel::LatticeBedrock;
                check(solid == (x < ground), "Bedrock covers the ground columns and only those");
                bedrock += solid ? 1 : 0;
            }
        }
    }
    check(bedrock == static_cast<std::size_t>(ground) * chasm.latticeDepth * 2,
          "and every world gets its own course");
}

void testLatticeAddressing() {
    constexpr std::uint32_t width = 7;
    constexpr std::uint32_t height = 5;
    constexpr std::uint32_t depth = 3;
    check(lk::latticeCellCount(width, height, depth) == 105, "The cell count is the box's volume");

    // Every cell gets its own index, and the indices fill [0, count) exactly.
    // The occupancy grid is addressed by this and nothing else, so a collision
    // here is two agents sharing a slot and a gap is a cell nobody can stand in.
    std::vector<int> seen(lk::latticeCellCount(width, height, depth), 0);
    for (int z = 0; z < static_cast<int>(depth); ++z) {
        for (int y = 0; y < static_cast<int>(height); ++y) {
            for (int x = 0; x < static_cast<int>(width); ++x) {
                const std::uint32_t index = lk::latticeCellIndex(x, y, z, width, height);
                check(index < seen.size(), "A cell index stays inside the grid");
                ++seen[index];
            }
        }
    }
    check(std::count(seen.begin(), seen.end(), 1) == static_cast<std::ptrdiff_t>(seen.size()),
          "Cell indexing is a bijection onto the grid");
    // x fastest, then y, then z: the shader's cell walk and the host's spawn
    // both assume this order, and a transposed index would place every agent
    // somewhere else without failing anything.
    check(lk::latticeCellIndex(1, 0, 0, width, height) == 1 &&
              lk::latticeCellIndex(0, 1, 0, width, height) == width &&
              lk::latticeCellIndex(0, 0, 1, width, height) == width * height,
          "Cells are laid out x fastest and z slowest");

    check(!lk::latticeInBounds(-1, 0, 0, width, height, depth) &&
              !lk::latticeInBounds(0, 0, static_cast<int>(depth), width, height, depth) &&
              lk::latticeInBounds(static_cast<int>(width) - 1, static_cast<int>(height) - 1,
                                  static_cast<int>(depth) - 1, width, height, depth),
          "The bounds test is half-open on every axis");
}

void testLatticeNeighbourhood() {
    check(lk::LatticeNeighborCount == 17 && lk::LatticeFaceNeighborCount == 5,
          "The sensed neighbourhood is the front half of the block, centre removed");

    // Every neighbour is a distinct non-zero offset in front of the agent's own
    // plane, and the numbering round trips through latticeNeighborIndex. Counted
    // as well as checked: a numbering that skipped a cell and repeated another
    // would pass every test in the loop and still be missing a direction.
    std::vector<std::array<int, 3>> offsets;
    std::uint32_t faces = 0;
    for (std::uint32_t neighbor = 0; neighbor < lk::LatticeNeighborCount; ++neighbor) {
        const int x = lk::latticeNeighborX(neighbor);
        const int y = lk::latticeNeighborY(neighbor);
        const int z = lk::latticeNeighborZ(neighbor);
        check(x != 0 || y != 0 || z != 0, "No neighbour is the centre cell");
        check(x >= 0, "Nothing behind the agent's own plane is sensed");
        check(std::abs(x) <= 1 && std::abs(y) <= 1 && std::abs(z) <= 1,
              "Every neighbour is one step away on each axis");
        check(lk::latticeNeighborIndex(x, y, z) == neighbor,
              "The neighbour numbering round-trips through its inverse");
        faces += std::abs(x) + std::abs(y) + std::abs(z) == 1 ? 1U : 0U;
        offsets.push_back({x, y, z});
    }
    std::sort(offsets.begin(), offsets.end());
    check(std::adjacent_find(offsets.begin(), offsets.end()) == offsets.end(),
          "No two neighbours share an offset");
    check(offsets.size() == 17, "Seventeen cells: two 3x3 planes less the agent's own");
    check(faces == lk::LatticeFaceNeighborCount,
          "Five neighbours share a face -- the sixth is the one behind");

    // The three the rules themselves reach for, by name rather than by number,
    // because these are the slots a policy has to be able to read for the
    // climbing rule to be learnable at all.
    check(lk::latticeNeighborX(lk::latticeNeighborIndex(0, -1, 0)) == 0 &&
              lk::latticeNeighborY(lk::latticeNeighborIndex(0, -1, 0)) == -1,
          "The cell underfoot is sensed");
    check(lk::latticeNeighborIndex(1, 0, 0) < lk::LatticeNeighborCount,
          "and so is the cell a block would go in");
    check(lk::latticeNeighborIndex(1, -1, 0) < lk::LatticeNeighborCount,
          "and the wall in front of the feet, which is what a climber holds");

    // Distance is counted in moves, so it follows the neighbourhood. A corner of
    // a 4x4x4 box is three Moore steps away and nine Manhattan ones.
    check(lk::latticeStepDistance(lk::LatticeNeighborhoodMoore, 3, 3, 3) == 3 &&
              lk::latticeStepDistance(lk::LatticeNeighborhoodFaces, 3, 3, 3) == 9,
          "Distance is Chebyshev under Moore and Manhattan under faces");
    check(lk::latticeStepDistance(lk::LatticeNeighborhoodMoore, -4, 1, 0) == 4,
          "Distance ignores the sign of a displacement");
    check(lk::latticeMaximumDistance(lk::LatticeNeighborhoodMoore, 20, 16, 8) == 19 &&
              lk::latticeMaximumDistance(lk::LatticeNeighborhoodFaces, 20, 16, 8) == 19 + 15 + 7,
          "The longest journey follows the neighbourhood too");

    // Nearness is what the brain reads and what the shaping banks: 1 on the
    // beacon, 0 at the far corner, and never outside that range.
    check(closeTo(lk::latticeNearness(0, 10), 1.0F) && closeTo(lk::latticeNearness(10, 10), 0.0F) &&
              closeTo(lk::latticeNearness(5, 10), 0.5F) &&
              closeTo(lk::latticeNearness(99, 10), 0.0F),
          "Nearness runs from 1 on the beacon to 0 at the far corner");
}

void testLatticeMoveRule() {
    // The dead zone is the whole of the decision to stand still, so it is tested
    // at its edges: a drive exactly on the threshold does not move.
    check(lk::latticeAxisStep(0.3F, 0.25F) == 1 && lk::latticeAxisStep(-0.3F, 0.25F) == -1 &&
              lk::latticeAxisStep(0.25F, 0.25F) == 0 && lk::latticeAxisStep(-0.25F, 0.25F) == 0,
          "A drive has to clear the dead zone strictly to become a step");
    check(lk::latticeAxisStep(0.001F, 0.0F) == 1,
          "A dead zone of zero means an agent moves on every step whatever it thinks");

    // The same dead zone reads every command the agent has: a turn is its sign,
    // and so is the vertical in a world with no gravity. One rule and one
    // parameter, which is why there is only one place for the two sides to
    // disagree about it.
    check(lk::latticeAxisStep(-0.9F, 0.25F) == -1 && lk::latticeAxisStep(0.9F, 0.25F) == 1,
          "A saturated output commits in the direction of its sign");

    check(lk::latticeCellEnterable(lk::LatticeNoOccupant) && !lk::latticeCellEnterable(0) &&
              !lk::latticeCellEnterable(7),
          "A cell may be entered only when it is empty");
    // The arbitration itself: a minimum over agent indices, so the winner does
    // not depend on the order the bids arrive in.
    check(lk::latticeBetterClaim(lk::LatticeNoClaim, 4) == 4 && lk::latticeBetterClaim(4, 9) == 4 &&
              lk::latticeBetterClaim(9, 4) == 4,
          "A contested cell goes to the lowest agent index, whichever bid first");
}

void testLatticeSpawn() {
    vkexp::SimulationStep settings{};
    settings.latticeWidth = 6;
    settings.latticeHeight = 6;
    settings.latticeDepth = 4;
    const vkexp::lattice::PopulationLayout layout{12, 10, 2};
    const std::vector<vkexp::AgentState> agents =
        vkexp::lattice::makeInitialAgents(settings, layout);
    check(agents.size() == layout.agentCount(), "The spawn produces one agent per trial");

    const std::uint32_t cells = vkexp::latticeCellsPerWorld(settings);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> occupied;
    for (std::uint32_t index = 0; index < agents.size(); ++index) {
        const vkexp::AgentState& agent = agents[index];
        const std::uint32_t world =
            vkexp::logicalWorldForAgent(index, layout.groupSize(), layout.trialsPerGenome);
        check(static_cast<std::uint32_t>(agent.beacon.w) == world,
              "An agent is stamped with the world it lives in");
        check(lk::latticeInBounds(agent.cell.x, agent.cell.y, agent.cell.z, settings.latticeWidth,
                                  settings.latticeHeight, settings.latticeDepth),
              "An agent spawns inside the lattice");
        check(agent.cell.x != agent.beacon.x || agent.cell.y != agent.beacon.y ||
                  agent.cell.z != agent.beacon.z,
              "Nobody spawns on the beacon, which would solve the world before the first step");
        check(static_cast<std::uint32_t>(agent.cell.w) < lk::LatticeFacingCount,
              "A fresh agent is already looking somewhere: there is no unfacing state");
        occupied.emplace_back(world,
                              lk::latticeCellIndex(agent.cell.x, agent.cell.y, agent.cell.z,
                                                   settings.latticeWidth, settings.latticeHeight));
    }
    std::sort(occupied.begin(), occupied.end());
    check(std::adjacent_find(occupied.begin(), occupied.end()) == occupied.end(),
          "No two agents spawn in one cell");
    check(cells > 0, "A lattice has cells to spawn into");

    // Placement is a pure function of the seed and the world index, not a draw
    // from a generator: that is what lets the parity test place one world
    // without simulating the ones ahead of it, and a snapshot resume rebuild the
    // same lattice from four numbers.
    const vkexp::Int4 again = vkexp::lattice::beaconCell(settings, 5);
    check(again.x == vkexp::lattice::beaconCell(settings, 5).x &&
              again.y == vkexp::lattice::beaconCell(settings, 5).y &&
              again.z == vkexp::lattice::beaconCell(settings, 5).z,
          "The beacon of a world is a function of the world, not of what was asked before it");
    // And it moves with the seed, or a run would score every generation against
    // one placement a policy could memorise.
    vkexp::SimulationStep moved = settings;
    moved.beaconSeed = settings.beaconSeed + 1;
    std::uint32_t different = 0;
    for (std::uint32_t world = 0; world < 16; ++world) {
        const vkexp::Int4 first = vkexp::lattice::beaconCell(settings, world);
        const vkexp::Int4 second = vkexp::lattice::beaconCell(moved, world);
        different += (first.x != second.x || first.y != second.y || first.z != second.z) ? 1U : 0U;
    }
    check(different >= 14, "A seed one apart produces a different set of beacons");
}

void testLatticeSensing() {
    vkexp::SimulationStep settings{};
    settings.latticeWidth = 5;
    settings.latticeHeight = 5;
    settings.latticeDepth = 5;
    namespace bk = vkexp::neuro::kernel;

    const std::uint32_t cells = vkexp::latticeCellsPerWorld(settings);
    std::vector<std::int32_t> occupancy(cells, lk::LatticeNoOccupant);
    std::vector<float> signals{0.0F, 0.75F};

    vkexp::AgentState agent{};
    agent.cell = {2, 2, 2, 0}; // facing +x
    agent.beacon = {4, 2, 2, 0};
    agent.intent = {2, 2, 2, 0};

    // Agent 1 stands one cell along +x, broadcasting.
    const std::uint32_t plusX =
        lk::latticeCellIndex(3, 2, 2, settings.latticeWidth, settings.latticeHeight);
    occupancy[plusX] = 1;

    const vkexp::neuro::Inputs middle =
        vkexp::sampleAgentInputs(agent, signals, occupancy, settings);
    const std::uint32_t neighborPlusX = lk::latticeNeighborIndex(1, 0, 0);
    check(
        closeTo(middle[bk::brainNeighborChannelIndex(neighborPlusX, lk::LatticeNeighborOccupied)],
                1.0F) &&
            closeTo(
                middle[bk::brainNeighborChannelIndex(neighborPlusX, lk::LatticeNeighborEdge)],
                0.0F) &&
            closeTo(middle[bk::brainNeighborChannelIndex(neighborPlusX, lk::LatticeNeighborSignal)],
                    0.75F),
        "An occupied neighbour reads as occupied, not a wall, and broadcasting what it emits");
    // An empty cell beside the agent. Beside and not behind: nothing behind the
    // agent's own plane has a slot at all, which is the point of the hemisphere.
    const std::uint32_t bodyLeft = lk::latticeNeighborIndex(0, 0, 1);
    check(
        closeTo(middle[bk::brainNeighborChannelIndex(bodyLeft, lk::LatticeNeighborOccupied)], 0.0F)
            && closeTo(middle[bk::brainNeighborChannelIndex(bodyLeft, lk::LatticeNeighborEdge)],
                       0.0F),
        "An empty neighbour inside the lattice reads as neither occupied nor a wall");

    // The direction to the beacon is a unit vector, and the nearness is what the
    // shaping banks. Two cells along +x in a 5-wide box under Moore is 2 of a
    // longest journey of 4.
    check(closeTo(middle[bk::brainBeaconInputIndex(0)], 1.0F) &&
              closeTo(middle[bk::brainBeaconInputIndex(1)], 0.0F) &&
              closeTo(middle[bk::brainBeaconInputIndex(2)], 0.0F) &&
              closeTo(middle[bk::brainBeaconInputIndex(3)], 0.5F),
          "The beacon reads as a unit direction and a nearness");

    // The edge of the lattice reads as a wall. There is no boundary geometry and
    // no push-out: a lattice ends, and this is the one place that says so. Read
    // in front, since that is where an agent meets one: at the far wall, facing
    // it, the cell it would step into is off the lattice.
    agent.cell = {static_cast<std::int32_t>(settings.latticeWidth) - 1, 2, 2, 0};
    const vkexp::neuro::Inputs edge = vkexp::sampleAgentInputs(agent, signals, occupancy, settings);
    check(closeTo(edge[bk::brainNeighborChannelIndex(neighborPlusX, lk::LatticeNeighborEdge)],
                  1.0F) &&
              closeTo(
                  edge[bk::brainNeighborChannelIndex(neighborPlusX, lk::LatticeNeighborOccupied)],
                  0.0F),
          "A neighbour outside the lattice reads as a wall rather than empty");

    // The self block is two numbers now: whether the last move was refused and
    // how long it has been standing. There is no heading channel, because in a
    // body frame the agent faces forward by definition and its absolute
    // orientation could only ever have carried a constant.
    agent.intent.w = 1;
    const vkexp::neuro::Inputs refused =
        vkexp::sampleAgentInputs(agent, signals, occupancy, settings);
    check(closeTo(edge[bk::BrainSelfOffset], 0.0F) && closeTo(refused[bk::BrainSelfOffset], 1.0F),
          "The self block opens with the refusal flag");

    // And the whole point of the frame: turn the agent and the same world reads
    // out of different slots. Facing +z, the agent that was in front is now off
    // to one side, and so is the beacon -- the network sees one arrangement for
    // what used to be four.
    agent.cell = {2, 2, 2, 1}; // facing +z
    agent.intent.w = 0;
    const vkexp::neuro::Inputs turned =
        vkexp::sampleAgentInputs(agent, signals, occupancy, settings);
    const std::uint32_t bodyRight = lk::latticeNeighborIndex(0, 0, -1);
    check(closeTo(turned[bk::brainNeighborChannelIndex(neighborPlusX,
                                                       lk::LatticeNeighborOccupied)],
                  0.0F) &&
              closeTo(turned[bk::brainNeighborChannelIndex(bodyRight,
                                                           lk::LatticeNeighborOccupied)],
                      1.0F),
          "Turning moves the neighbour from the slot in front to the slot beside");
    check(closeTo(turned[bk::brainBeaconInputIndex(0)], 0.0F) &&
              closeTo(turned[bk::brainBeaconInputIndex(2)], -1.0F),
          "The task direction turns with the agent rather than with the world");
}

// Two agents asking for one cell, run through the reference. The device side of
// this is in compute_smoke; what is checked here is that the shared rule gives
// the answer it claims to -- the lowest index wins, the loser stays and is
// charged, and nobody ever enters a cell that was occupied when the step began.
void testLatticeContention() {
    vkexp::SimulationStep settings{};
    settings.latticeWidth = 5;
    settings.latticeHeight = 5;
    settings.latticeDepth = 5;
    settings.neighborhood = vkexp::Neighborhood::Faces;
    settings.neuronModel = vkexp::NeuronModel::Reactive;

    const vkexp::lattice::PopulationLayout layout{2, 2, 1};
    const vkexp::neuro::BrainShape brain = vkexp::resolvedBrain(settings);
    const auto stride = static_cast<std::uint32_t>(brain.weightCount());
    std::vector<float> weights(static_cast<std::size_t>(stride) * layout.genomeCount, 0.0F);
    // No output bias at all: with every weight zero the turn stays under the
    // threshold and the action lands in the band that means "walk", so each
    // agent simply steps the way it is pointed. Which way that is, is the
    // facing in cell.w -- towards each other.
    std::vector<vkexp::AgentState> agents(2);
    for (vkexp::AgentState& agent : agents) {
        agent.beacon = {4, 4, 4, 0};
    }
    agents[0].cell = {1, 2, 2, 0}; // facing +x
    agents[1].cell = {3, 2, 2, 2}; // facing -x
    for (vkexp::AgentState& agent : agents) {
        agent.intent = {agent.cell.x, agent.cell.y, agent.cell.z, 0};
    }

    const std::uint32_t cells = vkexp::latticeCellsPerWorld(settings);
    std::vector<std::int32_t> occupancy(static_cast<std::size_t>(cells) * layout.worldCount());
    vkexp::lattice::buildOccupancy(agents, settings, layout, occupancy);
    std::vector<std::int32_t> claims(occupancy.size());

    vkexp::stepLatticeCpu(
        {agents, occupancy, claims, weights, stride, layout.groupSize(), layout.trialsPerGenome},
        settings);
    check(agents[0].cell.x == 2 && agents[0].cell.y == 2 && agents[0].cell.z == 2,
          "The lower-numbered agent takes the contested cell");
    check(agents[1].cell.x == 3, "The higher-numbered agent stays where it was");
    check(closeTo(agents[1].metrics.w, 1.0F) && closeTo(agents[0].metrics.w, 0.0F),
          "Losing a contested cell is charged as a refusal and winning one is not");
    check(closeTo(agents[0].metrics.z, 1.0F), "A move that worked is charged as one move");
    check(agents[0].cell.w == 0 && agents[1].cell.w == 2,
          "A move leaves the facing alone: only a turn changes where an agent looks");
    check(agents[0].intent.w == 0 && agents[1].intent.w == 1,
          "The refusal flag is what the next step reads back as a self input");

    // The occupancy grid follows: the winner's old cell is empty and its new one
    // names it. Two agents sharing a cell is the one thing the grid cannot say.
    check(occupancy[lk::latticeCellIndex(1, 2, 2, settings.latticeWidth, settings.latticeHeight)] ==
                  lk::LatticeNoOccupant &&
              occupancy[lk::latticeCellIndex(2, 2, 2, settings.latticeWidth,
                                             settings.latticeHeight)] == 0 &&
              occupancy[lk::latticeCellIndex(3, 2, 2, settings.latticeWidth,
                                             settings.latticeHeight)] == 1,
          "The occupancy grid follows the move that actually happened");

    // Now the loser is asked to walk into the winner, which it cannot: a cell may
    // only be entered if it was empty when the step began.
    vkexp::stepLatticeCpu(
        {agents, occupancy, claims, weights, stride, layout.groupSize(), layout.trialsPerGenome},
        settings);
    check(agents[1].cell.x == 3 && closeTo(agents[1].metrics.w, 2.0F),
          "Walking into an occupied cell is refused and charged");
}

// The foundation rule on its own, away from any agent. It is restored with a
// fill of half rather than the quarter it first ran at: at a quarter one block
// underneath already filled a small window, so the foundation was the level
// immediately below and the five-level lead was never spent -- five refusals in
// two and a half million attempts. Half asks for a mass.
void testConstructionLocalFoundation() {
    vkexp::SimulationStep settings{};
    settings.worldMode = vkexp::WorldMode::Construction;
    settings.latticeWidth = 8;
    settings.latticeHeight = 8;
    settings.latticeDepth = 8;
    settings.constructionCourseFill = 0.5F;
    const std::uint32_t cells = vkexp::latticeCellsPerWorld(settings);
    std::vector<std::int32_t> field(cells, 0);
    const auto place = [&](const int x, const int y, const int z) {
        field[vkexp::lattice::kernel::latticeCellIndex(x, y, z, settings.latticeWidth,
                                                       settings.latticeHeight)] = 1;
    };

    settings.constructionSupportRadius = 1;
    check(vkexp::constructionLocalFoundation(field, settings, 4, 3, 4) == 0,
          "Nothing under a site is no foundation at all");
    check(vkexp::constructionLocalFoundation(field, settings, 4, 0, 4) == 0,
          "A site on the floor has nothing below it to scan");

    // A solid 3x3 directly under the site: the whole window, so dense at any
    // fill, and the foundation is the level above it.
    for (int z = 3; z <= 5; ++z) {
        for (int x = 3; x <= 5; ++x) {
            place(x, 2, z);
        }
    }
    check(vkexp::constructionLocalFoundation(field, settings, 4, 3, 4) == 3,
          "A full window one level down is a foundation of that level plus one");
    check(vkexp::constructionLocalFoundation(field, settings, 4, 6, 4) == 3,
          "The scan finds the same platform from higher up: levels need not be consecutive");

    // The radius is what decides. Standing two cells away, a radius of one
    // cannot see the platform at all and a radius of three can.
    check(vkexp::constructionLocalFoundation(field, settings, 7, 3, 7) == 0,
          "A narrow window sees nothing two cells away from the platform");
    settings.constructionSupportRadius = 3;
    check(vkexp::constructionLocalFoundation(field, settings, 7, 3, 7) == 0,
          "A wider window reaches the platform, but four cells of sixteen is under half");
    settings.constructionCourseFill = 0.25F;
    check(vkexp::constructionLocalFoundation(field, settings, 7, 3, 7) == 3,
          "At a lower fill the same reach is enough, which is the trade the two sliders make");

    // Clipping at the wall shrinks the question rather than failing it: a corner
    // sees fewer cells and needs proportionally fewer of them.
    settings.constructionSupportRadius = 1;
    settings.constructionCourseFill = 1.0F;
    std::fill(field.begin(), field.end(), 0);
    place(0, 0, 0);
    place(1, 0, 0);
    place(0, 0, 1);
    place(1, 0, 1);
    check(vkexp::constructionLocalFoundation(field, settings, 0, 1, 0) == 1,
          "A corner window is four cells, and four blocks fill it completely");
    check(vkexp::constructionLocalFoundation(field, settings, 2, 1, 2) == 0,
          "One cell short of full is not full, whatever the window size");

    // A fill of zero still needs one block: an empty level is never something
    // to stand on, however forgiving the setting.
    settings.constructionCourseFill = 0.0F;
    check(vkexp::constructionLocalFoundation(field, settings, 6, 2, 6) == 0,
          "An empty window is never a foundation, even at zero fill");
    check(vkexp::constructionLocalFoundation(field, settings, 2, 2, 2) == 1,
          "At zero fill a single block within reach is enough");
}


// Walking off the edge of a chasm. The rule under test is that a column with no
// bottom cannot be entered -- which is not a rule about chasms at all, but the
// consequence of there being no implicit floor any more. It needs its own test
// because the parity probe cannot reach it: parity says the two implementations
// agree, and while the landing search stopped at height zero both of them agreed
// on the same wrong answer, which was that the bottom of the hole is a floor.
//
// What is not a bug, and is worth stating because it looks like one: the first
// column of the void is enterable at any height where the cliff is beside it.
// Support has always included a vertical face -- that is the climbing rule, and
// it is what lets an agent go up a wall at all -- so an agent may hang on the
// cliff and move along it. It may not leave it, which is the difference between
// hugging an edge and walking across a hole.
void testChasmEdge() {
    vkexp::SimulationStep settings{};
    settings.worldMode = vkexp::WorldMode::Chasm;
    settings.latticeWidth = 8;
    settings.latticeHeight = 6;
    settings.latticeDepth = 5;
    settings.neighborhood = vkexp::Neighborhood::Faces;
    settings.neuronModel = vkexp::NeuronModel::Reactive;
    settings.chasmGroundWidth = 3; // ground at x 0..2, open air at x 3..7
    // Nothing is built in this test: a tanh output cannot exceed one, so the
    // gate never opens and what is measured is walking alone.
    settings.buildThreshold = 2.0F;
    const auto ground = static_cast<std::int32_t>(vkexp::latticeGroundWidth(settings));
    check(ground == 3, "The fixture's ground is where it says it is");

    const vkexp::lattice::PopulationLayout layout{1, 1, 1};
    const vkexp::neuro::BrainShape brain = vkexp::resolvedBrain(settings);
    const auto stride = static_cast<std::uint32_t>(brain.weightCount());
    std::vector<float> weights(static_cast<std::size_t>(stride) * layout.genomeCount, 0.0F);
    // Every weight zero: the turn stays under its threshold and the action lands
    // in the band that means "walk", so the agent spends every tick stepping the
    // way it is pointed, which is east at the cliff.
    std::vector<vkexp::AgentState> agents(1);
    agents[0].cell = {ground - 1, 1, 2, 0}; // facing +x
    agents[0].beacon = {-1, -1, -1, 0};
    agents[0].intent = {agents[0].cell.x, agents[0].cell.y, agents[0].cell.z, 0};

    std::vector<std::int32_t> structures =
        vkexp::lattice::makeTerrain(settings, layout.worldCount());
    const std::uint32_t cells = vkexp::latticeCellsPerWorld(settings);
    std::vector<std::int32_t> occupancy(static_cast<std::size_t>(cells) * layout.worldCount());
    vkexp::lattice::buildOccupancy(agents, settings, layout, occupancy);
    std::vector<std::int32_t> claims(occupancy.size());
    std::vector<std::uint32_t> outcomes(static_cast<std::size_t>(layout.worldCount()) *
                                        lk::LatticeBuildOutcomeCount);

    const auto step = [&] {
        std::fill(outcomes.begin(), outcomes.end(), 0U);
        vkexp::stepLatticeCpu({agents, occupancy, claims, weights, stride, layout.groupSize(),
                               layout.trialsPerGenome, structures, outcomes},
                              settings);
    };

    // Twenty steps of walking east as hard as the output can ask. Without the
    // rule the agent reaches the far wall: every cell of the hole was a landing,
    // so the hole was a road.
    for (std::uint32_t tick = 0; tick < 20; ++tick) {
        step();
        check(agents[0].cell.x <= ground,
              "Nobody walks past the cliff face into open air, however hard they push");
    }
    check(agents[0].cell.x == ground,
          "The cliff face itself is reachable -- hanging on a wall is the climbing rule");
    check(agents[0].intent.w == 1,
          "and the step beyond it is charged as a refusal, like the edge of the lattice");

    // Not a special case for the void, and not a wall: give the next column a
    // bottom and the very same walk reaches it. Without this the test would pass
    // just as well if stepping east had been forbidden outright.
    //
    // It takes two ticks, and that is the climbing rule rather than an accident:
    // the first step meets the block's face and lifts the agent a level in its
    // own column, holding that face with its feet; the second carries it over
    // the top. Arriving on top in one step is what the old rule did, and it is
    // what made building upward impossible -- from the top of your own block
    // there is nothing in front to build on.
    const int ledgeX = ground + 1;
    const int lane = agents[0].cell.z;
    structures[lk::latticeCellIndex(ledgeX, 0, lane, settings.latticeWidth,
                                    settings.latticeHeight)] = 1;
    step();
    check(agents[0].cell.x == ground && agents[0].cell.y == 1,
          "A block in front is climbed rather than stepped onto: the agent rises beside it");
    step();
    check(agents[0].cell.x == ledgeX && agents[0].cell.y == 1,
          "and the step after that carries it over the top");
    check(structures[lk::latticeCellIndex(ledgeX, 0, agents[0].cell.z, settings.latticeWidth,
                                          settings.latticeHeight)] == 1 &&
              agents[0].cell.y == 1,
          "and the agent is standing on that block, one level up from nothing");
}

// The two turning views, and the difference between them. Both are tested from
// a dragged-off-centre view, because with the view centred they are the same
// motion and a test taken there would pass on either implementation.
void testCameraSpin() {
    constexpr float radius = 10.0F;
    constexpr float turn = 0.37F;
    const auto close = [](const float left, const float right) {
        return std::abs(left - right) < 1.0e-4F;
    };

    vkexp::LatticeCamera camera{};
    camera.yaw = 0.8F;
    camera.pitch = 0.3F;
    camera.targetX = 4.0F;
    camera.targetZ = -3.0F;

    // Orbit: the look-at point does not move, so the eye circles it and its
    // distance from it is all that is preserved.
    vkexp::LatticeCamera orbit = camera;
    const std::array<float, 3> before = camera.eye(radius);
    orbit.spinBy(turn, false);
    const std::array<float, 3> orbited = orbit.eye(radius);
    check(close(orbit.targetX, camera.targetX) && close(orbit.targetZ, camera.targetZ),
          "Orbit leaves the look-at point where it was");
    const auto span = [](const std::array<float, 3>& point, const vkexp::LatticeCamera& from) {
        const float dx = point[0] - from.targetX;
        const float dz = point[2] - from.targetZ;
        return std::sqrt(dx * dx + dz * dz);
    };
    check(close(span(before, camera), span(orbited, orbit)),
          "and keeps the eye the same distance from it");
    // And the box centre does not hold still under it: that is the behaviour the
    // turntable exists to replace, so if this ever stops being true the two
    // modes have collapsed into one.
    const float centredBefore = std::sqrt(before[0] * before[0] + before[2] * before[2]);
    const float centredAfter = std::sqrt(orbited[0] * orbited[0] + orbited[2] * orbited[2]);
    check(!close(centredBefore, centredAfter),
          "Orbiting an off-centre view changes how far the eye is from the box");

    // Turntable: the whole camera frame turns about the origin, which is where
    // the box is centred. Both the eye and the look-at point come out rotated by
    // exactly the same angle, which is what makes the box appear to spin in
    // place rather than travel across the frame.
    vkexp::LatticeCamera table = camera;
    table.spinBy(turn, true);
    const std::array<float, 3> turned = table.eye(radius);
    const float cosTurn = std::cos(turn);
    const float sinTurn = std::sin(turn);
    check(close(turned[0], before[0] * cosTurn + before[2] * sinTurn) &&
              close(turned[2], -before[0] * sinTurn + before[2] * cosTurn) &&
              close(turned[1], before[1]),
          "The turntable rotates the eye about the origin, not about the look-at point");
    check(close(table.targetX, camera.targetX * cosTurn + camera.targetZ * sinTurn) &&
              close(table.targetZ, -camera.targetX * sinTurn + camera.targetZ * cosTurn),
          "and carries the look-at point round by the same angle");
    check(close(std::sqrt(turned[0] * turned[0] + turned[2] * turned[2]), centredBefore),
          "so the eye stays exactly as far from the box as it was");

    // Centred, the two are one motion. This is why the bug was invisible: the
    // default view is centred, and the modes only part once the view is dragged.
    vkexp::LatticeCamera centred{};
    centred.yaw = 0.8F;
    centred.pitch = 0.3F;
    vkexp::LatticeCamera centredTable = centred;
    centred.spinBy(turn, false);
    centredTable.spinBy(turn, true);
    const std::array<float, 3> one = centred.eye(radius);
    const std::array<float, 3> other = centredTable.eye(radius);
    check(close(one[0], other[0]) && close(one[1], other[1]) && close(one[2], other[2]),
          "With nothing dragged, orbit and turntable are the same motion");
}

void testLatticeFitness() {
    // The four counters and what each is worth. Written out rather than folded
    // into agentFitness so that changing a weight and changing the arithmetic
    // are distinguishable failures.
    const vkexp::FitnessWeights weights{};
    vkexp::AgentState reached{};
    reached.metrics = {1.0F, 10.0F, 20.0F, 0.0F};
    vkexp::AgentState stuck{};
    stuck.metrics = {0.2F, 0.0F, 20.0F, 60.0F};
    check(vkexp::agentFitness(reached, weights) > vkexp::agentFitness(stuck, weights),
          "Reaching the beacon beats jamming against a wall");

    // Nearness is clamped, so a metric that somehow exceeded one cannot buy an
    // unbounded score.
    vkexp::AgentState absurd{};
    absurd.metrics = {5.0F, 0.0F, 0.0F, 0.0F};
    vkexp::AgentState perfect{};
    perfect.metrics = {1.0F, 0.0F, 0.0F, 0.0F};
    check(closeTo(vkexp::agentFitness(absurd, weights), vkexp::agentFitness(perfect, weights)),
          "Nearness is clamped, so it cannot buy an unbounded score");

    // Standing near the beacon is scored per step, not per arrival: a cell holds
    // one agent, so "got there" is a race that eleven of twelve lose whatever
    // they did, and "stayed near" is not.
    vkexp::AgentState brief = reached;
    brief.metrics.y = 1.0F;
    check(vkexp::agentFitness(reached, weights) > vkexp::agentFitness(brief, weights),
          "Time spent within the contact radius is what the objective bonus pays for");
}

} // namespace

int main() {
    testTimingSeries();
    testCpuProfiler();
    testDispatchSize();
    testComputeResourceValidation();
    testLogicalWorldPartition();
    testPingPongState();
    testShaderBindingContract();
    testLatticeCameraSlide();
    testTransparencyWeight();
    testLatticeSpawnCapacity();
    testStructureShape();
    testBodyFrame();
    testLatticeStillness();
    testHarvestResource();
    testLatticeAddressing();
    testLatticeNeighbourhood();
    testLatticeMoveRule();
    testCameraSpin();
    testConstructionLocalFoundation();
    testChasmEdge();
    testLatticeSpawn();
    testLatticeSensing();
    testLatticeContention();
    testLatticeFitness();
    testNeuralNetworkContract();
    testGeneticAlgorithm();
    testExperimentSweep();
    testNeuronTimeConstants();
    testGatedNeurons();
    testSpikingNeuronModel();
    testFitnessWeightsAreParameters();
    testBrainForwardPass();
    testLayeredBrain();
    testBrainDescription();
    testGenomeArchiveRoundTrip();
    testGroupFitnessSharing();
    testRunSnapshotRoundTrip();
    testLayerActivation();
    testRandomWeights();
    testPopulationReload();
    testStepParameterPacking();
    if (failures == 0) {
        std::cout << "All unit tests passed\n";
        return 0;
    }
    std::cerr << failures << " unit test check(s) failed\n";
    return 1;
}
