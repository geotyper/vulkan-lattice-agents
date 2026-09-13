#include "vkexp/compute/ComputeResources.hpp"
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
#include "vkexp/simulation/Units.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
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
    check(kernel::BrainActuatorOutputCount == kernel::BrainMoveOutputCount +
                                                  kernel::BrainSignalOutputCount +
                                                  kernel::BrainBuildOutputCount,
          "Actuators are the three move drives, broadcast and build impulse");
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
    check(kernel::BrainNeighborChannels == 3,
          "A neighbour reads as occupied, blocked and broadcasting");
    check(kernel::BrainMoveOutputCount == 3, "One move drive per lattice axis");

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
        {vkexp::neuro::defaultBrainShape, "the default 61 -> 20 -> 8"},
        {{57, 20, 6}, "a trimmed 57 -> 20 -> 6"},
        {{8, 4, 6}, "a small 8 -> 4 -> 6"},
        {{4, 1, 6}, "a single hidden neuron"},
        {{8, 4, 6, 3, 2}, "three layers narrowing"},
        {{8, 2, 6, 5, 7}, "three layers widening"},
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
    const vkexp::neuro::BrainShape wired{6, 3, 6};
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
        const vkexp::neuro::BrainShape chain{4, 2, 6, 1, 0};
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
        const vkexp::neuro::BrainShape shape{8, 4, 6, 3, 0};
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
    const vkexp::neuro::BrainShape flat{8, 4, 6};
    const vkexp::neuro::BrainShape deep{8, 4, 6, 3, 2};

    check(flat.hiddenLayerCount() == 1 && flat.hiddenTotal() == 4,
          "One width is one layer, and every scenario that wrote three numbers still means that");
    check(deep.hiddenLayerCount() == 3 && deep.hiddenTotal() == 9,
          "Three widths are three layers and their total");

    // A hole is refused rather than closed up: {4, 0, 2} could mean a two-layer
    // plan or a mistake, and guessing between them is worse than saying no.
    const vkexp::neuro::BrainShape holed{8, 4, 6, 0, 2};
    check(!holed.fitsCapacity(), "A plan with a hole in the middle is refused");
    const vkexp::neuro::BrainShape overspent{8, vkexp::neuro::Topology::hiddenNeuronCapacity, 6,
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
    check(inside("move", bk::BrainMoveOutput) &&
              inside("move", bk::BrainMoveOutput + bk::BrainMoveOutputCount - 1u) &&
              inside("signal", bk::BrainSignalIntensityOutput) &&
              inside("build", bk::BrainBuildOutput) &&
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
    const vkexp::neuro::BrainDescription trimmed = vkexp::neuro::describeBrain({52, 20, 8});
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
        3, vkexp::Genome{vkexp::neuro::makeWeights(vkexp::neuro::BrainShape{70, 20, 5})});
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
    const vkexp::neuro::BrainShape deepPlan{88, 12, 6, 8, 8};
    const std::filesystem::path deepPath = path.parent_path() / "deep.vkng";
    std::vector<vkexp::Genome> deepGenomes(2, vkexp::Genome{vkexp::neuro::makeWeights(deepPlan)});
    deepGenomes.front().weights.front() = 0.5F;
    const vkexp::GenomeArchiveMetadata deepMetadata{
        7,
        5,
        1U,
        0.5F,
        0.25F,
        88,
        static_cast<std::uint32_t>(deepPlan.hiddenTotal()),
        6,
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
    snapshot.settings.moveThreshold = 0.4F;
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
              closeTo(loaded.settings.moveThreshold, snapshot.settings.moveThreshold),
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

void testPopulationReload() {
    const vkexp::EvolutionSettings settings{
        8, 2, 3, 0.5F, 0.1F, 0.2F, 42U, vkexp::neuro::defaultBrainShape.weightCount()};
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
    // constants, which is why they travel in a storage buffer.
    check(sizeof(vkexp::GpuStepParameters) <= 128,
          "Step parameters fit a cache line pair, which is what indexing them per "
          "step is worth doing for");

    vkexp::SimulationStep settings{};
    settings.latticeWidth = 20;
    settings.latticeHeight = 16;
    settings.latticeDepth = 8;
    settings.neighborhood = vkexp::Neighborhood::Faces;
    settings.moveThreshold = 0.4F;
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
    check(closeTo(packed.moveThreshold, 0.4F) && packed.beaconContactRadius == 2 &&
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
    check(lk::LatticeNeighborCount == 26 && lk::LatticeFaceNeighborCount == 6,
          "The neighbourhood is the 3x3x3 block less its centre");

    // Every neighbour is a distinct non-zero offset, and the numbering round
    // trips through latticeNeighborIndex. The inverse is what turns a move back
    // into a heading, so an error here is an agent that reports facing somewhere
    // it did not go.
    std::vector<std::array<int, 3>> offsets;
    std::uint32_t faces = 0;
    for (std::uint32_t neighbor = 0; neighbor < lk::LatticeNeighborCount; ++neighbor) {
        const int x = lk::latticeNeighborX(neighbor);
        const int y = lk::latticeNeighborY(neighbor);
        const int z = lk::latticeNeighborZ(neighbor);
        check(x != 0 || y != 0 || z != 0, "No neighbour is the centre cell");
        check(std::abs(x) <= 1 && std::abs(y) <= 1 && std::abs(z) <= 1,
              "Every neighbour is one step away on each axis");
        check(lk::latticeNeighborIndex(x, y, z) == neighbor,
              "The neighbour numbering round-trips through its inverse");
        faces += lk::latticeIsFaceNeighbor(neighbor) ? 1U : 0U;
        offsets.push_back({x, y, z});
    }
    std::sort(offsets.begin(), offsets.end());
    check(std::adjacent_find(offsets.begin(), offsets.end()) == offsets.end(),
          "No two neighbours share an offset");
    check(faces == lk::LatticeFaceNeighborCount, "Exactly six neighbours share a face");

    // Sensing reads all 26 under both settings; only walking is restricted. That
    // is what lets a population trained on one setting load into the other.
    std::uint32_t walkable = 0;
    for (std::uint32_t neighbor = 0; neighbor < lk::LatticeNeighborCount; ++neighbor) {
        walkable += lk::latticeNeighborWalkable(lk::LatticeNeighborhoodFaces, neighbor) ? 1U : 0U;
        check(lk::latticeNeighborWalkable(lk::LatticeNeighborhoodMoore, neighbor),
              "Every neighbour is walkable under Moore");
    }
    check(walkable == lk::LatticeFaceNeighborCount, "Only the faces are walkable under faces");

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

    // Under Moore all three axes commit at once, which is what makes a diagonal
    // one step rather than three.
    const float drives[3] = {0.9F, -0.8F, 0.4F};
    for (std::uint32_t axis = 0; axis < 3; ++axis) {
        check(lk::latticeMoveComponent(lk::LatticeNeighborhoodMoore, axis, drives[0], drives[1],
                                       drives[2], 0.25F) != 0,
              "Every axis that clears the dead zone moves under Moore");
    }
    // Under faces only the loudest does, so the result is always a face step.
    check(lk::latticeMoveComponent(lk::LatticeNeighborhoodFaces, 0, drives[0], drives[1], drives[2],
                                   0.25F) == 1 &&
              lk::latticeMoveComponent(lk::LatticeNeighborhoodFaces, 1, drives[0], drives[1],
                                       drives[2], 0.25F) == 0 &&
              lk::latticeMoveComponent(lk::LatticeNeighborhoodFaces, 2, drives[0], drives[1],
                                       drives[2], 0.25F) == 0,
          "Only the loudest axis commits under a faces-only neighbourhood");
    // Ties broken x then y then z, and fixed rather than arbitrary: an arbitrary
    // tie-break is a divergence between the two implementations that no numeric
    // tolerance would forgive.
    check(lk::latticeDominantAxis(0.5F, 0.5F, 0.5F) == 0 &&
              lk::latticeDominantAxis(0.4F, 0.5F, 0.5F) == 1 &&
              lk::latticeDominantAxis(0.4F, 0.4F, 0.5F) == 2,
          "Equal drives break toward x, then y, then z");

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
        check(static_cast<std::uint32_t>(agent.cell.w) == lk::LatticeNeighborCount,
              "A fresh agent has no heading, which is not the same as heading at neighbour zero");
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
    agent.cell = {2, 2, 2, static_cast<std::int32_t>(lk::LatticeNeighborCount)};
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
                middle[bk::brainNeighborChannelIndex(neighborPlusX, lk::LatticeNeighborBlocked)],
                0.0F) &&
            closeTo(middle[bk::brainNeighborChannelIndex(neighborPlusX, lk::LatticeNeighborSignal)],
                    0.75F),
        "An occupied neighbour reads as occupied, unblocked, and broadcasting what it emits");
    const std::uint32_t neighborMinusX = lk::latticeNeighborIndex(-1, 0, 0);
    check(
        closeTo(middle[bk::brainNeighborChannelIndex(neighborMinusX, lk::LatticeNeighborOccupied)],
                0.0F) &&
            closeTo(
                middle[bk::brainNeighborChannelIndex(neighborMinusX, lk::LatticeNeighborBlocked)],
                0.0F),
        "An empty neighbour inside the lattice reads as neither occupied nor blocked");

    // The direction to the beacon is a unit vector, and the nearness is what the
    // shaping banks. Two cells along +x in a 5-wide box under Moore is 2 of a
    // longest journey of 4.
    check(closeTo(middle[bk::brainBeaconInputIndex(0)], 1.0F) &&
              closeTo(middle[bk::brainBeaconInputIndex(1)], 0.0F) &&
              closeTo(middle[bk::brainBeaconInputIndex(2)], 0.0F) &&
              closeTo(middle[bk::brainBeaconInputIndex(3)], 0.5F),
          "The beacon reads as a unit direction and a nearness");

    // The edge of the lattice reads as a wall. There is no boundary geometry and
    // no push-out: a lattice ends, and this is the one place that says so.
    agent.cell = {0, 2, 2, agent.cell.w};
    const vkexp::neuro::Inputs edge = vkexp::sampleAgentInputs(agent, signals, occupancy, settings);
    check(closeTo(edge[bk::brainNeighborChannelIndex(neighborMinusX, lk::LatticeNeighborBlocked)],
                  1.0F) &&
              closeTo(
                  edge[bk::brainNeighborChannelIndex(neighborMinusX, lk::LatticeNeighborOccupied)],
                  0.0F),
          "A neighbour outside the lattice reads as blocked rather than empty");

    // An agent that has not moved reads zero on all three heading channels,
    // which is a distinguishable state rather than a direction.
    check(closeTo(edge[bk::BrainSelfOffset], 0.0F) &&
              closeTo(edge[bk::BrainSelfOffset + 1], 0.0F) &&
              closeTo(edge[bk::BrainSelfOffset + 2], 0.0F),
          "An agent that has never moved reports no heading");
    agent.cell.w = static_cast<std::int32_t>(lk::latticeNeighborIndex(0, 0, 1));
    agent.intent.w = 1;
    const vkexp::neuro::Inputs headed =
        vkexp::sampleAgentInputs(agent, signals, occupancy, settings);
    check(closeTo(headed[bk::BrainSelfOffset + 2], 1.0F) &&
              closeTo(headed[bk::BrainSelfOffset + 3], 1.0F),
          "A heading reads back as the unit step it was, beside the refusal flag");
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
    namespace bk = vkexp::neuro::kernel;
    const std::size_t moveBias = bk::brainOutputBiasIndex(
        0U, static_cast<std::uint32_t>(brain.inputCount), brain.packedLayers(),
        static_cast<std::uint32_t>(brain.outputCount), bk::BrainMoveOutput);
    weights[moveBias] = 8.0F;           // genome 0 drives +x
    weights[stride + moveBias] = -8.0F; // genome 1 drives -x

    std::vector<vkexp::AgentState> agents(2);
    for (vkexp::AgentState& agent : agents) {
        agent.cell.w = static_cast<std::int32_t>(lk::LatticeNeighborCount);
        agent.beacon = {4, 4, 4, 0};
    }
    agents[0].cell = {1, 2, 2, agents[0].cell.w};
    agents[1].cell = {3, 2, 2, agents[1].cell.w};
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
    check(static_cast<std::uint32_t>(agents[0].cell.w) == lk::latticeNeighborIndex(1, 0, 0),
          "A move sets the heading to the step it took");
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
    testLatticeAddressing();
    testLatticeNeighbourhood();
    testLatticeMoveRule();
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
    testPopulationReload();
    testStepParameterPacking();
    if (failures == 0) {
        std::cout << "All unit tests passed\n";
        return 0;
    }
    std::cerr << failures << " unit test check(s) failed\n";
    return 1;
}
