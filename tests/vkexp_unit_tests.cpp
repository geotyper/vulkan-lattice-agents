#include "vkexp/compute/ComputeResources.hpp"
#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/evolution/GenomeArchive.hpp"
#include "vkexp/neuro/BrainDescription.hpp"
#include "vkexp/neuro/BrainKernel.hpp"
#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/profiling/CpuProfiler.hpp"
#include "vkexp/profiling/ProfilerTypes.hpp"
#include "vkexp/simulation/CpuSimulation.hpp"
#include "vkexp/simulation/ExperimentSweep.hpp"
#include "vkexp/simulation/Locomotion.hpp"
#include "vkexp/simulation/Sensors.hpp"
#include "vkexp/simulation/PuckKernel.hpp"
#include "vkexp/worlds/scenarios/GatePlateScenario.hpp"
#include "vkexp/simulation/SimulationState.hpp"
#include "vkexp/simulation/TrailKernel.hpp"
#include "vkexp/simulation/Units.hpp"
#include "vkexp/simulation/WorldSnapshot.hpp"
#include "vkexp/worlds/ScenarioKernel.hpp"
#include "vkexp/worlds/ScenarioMath.hpp"
#include "vkexp/worlds/WorldScenario.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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
    check(vkexp::clampAgentsPerWorld(genomes, 0) == vkexp::minimumAgentsPerWorld,
          "World partition clamps an empty group size");
    check(vkexp::clampAgentsPerWorld(genomes, 100) == genomes,
          "World partition supports all agents in one group");
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
    vkexp::neuro::Weights weights{};
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
              kernel::BrainLightReceptorCount * kernel::BrainLightChannels +
                  kernel::BrainTactileSectorCount * kernel::BrainTactileChannels +
                  kernel::BrainAntennaCount * kernel::BrainAntennaChannels +
                  kernel::BrainSelfInputCount + kernel::BrainTaskInputCount +
                  kernel::BrainRecurrentCount,
          "Input capacity is the sum of the declared sensor blocks");
    check(vkexp::neuro::Topology::outputCount ==
              kernel::BrainActuatorOutputCount + kernel::BrainRecurrentCount,
          "Output capacity is actuators plus recurrent cells");
    check(vkexp::neuro::Topology::weightCount == vkexp::neuro::maximumBrainShape.weightCount(),
          "Genome capacity matches the widest brain shape");

    // The sensor blocks must tile the input vector without gaps or overlaps.
    check(kernel::BrainLightOffset == 0, "Light block starts the input vector");
    check(kernel::brainLightChannelIndex(kernel::BrainLightReceptorCount - 1,
                                         kernel::BrainLightChannels - 1) +
                  1 ==
              kernel::BrainTactileOffset,
          "Tactile block follows the light block");
    check(kernel::brainTactileChannelIndex(kernel::BrainTactileSectorCount - 1,
                                           kernel::BrainTactileChannels - 1) +
                  1 ==
              kernel::BrainAntennaOffset,
          "Antenna block follows the tactile block");
    check(kernel::brainAntennaChannelIndex(kernel::BrainAntennaCount - 1,
                                           kernel::BrainAntennaChannels - 1) +
                  1 ==
              kernel::BrainSelfOffset,
          "Self block follows the antenna block");
    // The antennae have to reach into different trail cells or the three
    // readings collapse into one number and carry no gradient.
    const float antennaSpread =
        2.0F * kernel::BrainAntennaLength * std::sin(kernel::BrainAntennaHalfSpread);
    // Strictly more than one cell apart is the guarantee that matters: two points
    // further apart than a cell is wide cannot share a cell, whatever the phase.
    // At the coarsest 8 cm setting the 15.5 cm spread leaves 1.9 cells.
    check(antennaSpread > vkexp::trailCellSizeForBodyFraction(vkexp::trailCellFractionCoarsest),
          "Outer antenna tips cannot share a cell at the coarsest trail resolution");
    check(kernel::BrainSelfOffset + kernel::BrainSelfInputCount == kernel::BrainTaskOffset,
          "Task block follows the self block");
    check(kernel::BrainTaskOffset + kernel::BrainTaskInputCount ==
              kernel::BrainRecurrentInputOffset,
          "Recurrent inputs follow the task block");
    check(kernel::BrainRecurrentInputOffset + kernel::BrainRecurrentCount ==
              kernel::BrainInputCapacity,
          "Recurrent inputs close the input vector");

    const auto& stationary = vkexp::scenarioDefinition(vkexp::BeaconScenario::Stationary);
    const auto& forage = vkexp::scenarioDefinition(vkexp::BeaconScenario::ForageHome);
    check(stationary.brain.inputCount == vkexp::neuro::Topology::inputCount -
                                             vkexp::neuro::Topology::taskInputCount -
                                             vkexp::neuro::Topology::recurrentMemoryCount &&
              stationary.brain.outputCount == vkexp::neuro::Topology::actuatorOutputCount,
          "Stationary scenario owns a reactive brain shape without task or memory");
    check(forage.brain.inputCount == vkexp::neuro::Topology::inputCount &&
              forage.brain.outputCount == vkexp::neuro::Topology::outputCount,
          "Forage scenario owns the full recurrent brain shape");
    check(std::string_view(stationary.name) == "Stationary" &&
              std::string_view(forage.name) == "Forage + home",
          "Scenario definitions own their display names");
    const std::uint32_t layout = vkexp::neuro::packBrainLayout(stationary.brain);
    const vkexp::neuro::BrainShape unpacked = vkexp::neuro::brainShape(layout);
    check(vkexp::neuro::brainGenomeStride(layout) == vkexp::neuro::Topology::weightCount &&
              unpacked.inputCount == stationary.brain.inputCount &&
              unpacked.hiddenCount == stationary.brain.hiddenCount &&
              unpacked.outputCount == stationary.brain.outputCount,
          "Packed GPU brain layout preserves capacity and active shape");
}

void testMultimodalSensors() {
    vkexp::AgentState agent{};
    agent.pose = {0.0F, 0.0F, 0.0F, vkexp::agentBodyRadius};
    agent.motion.w = 1.0F;
    agent.target = {1.0F, 0.0F, 0.0F, 0.0F};
    agent.internal = {0.7F, 1.0F, -0.4F, 0.6F};
    agent.wallTouch0.x = 0.7F;
    agent.agentTouch1.w = 0.8F;
    const vkexp::SimulationStep settings{};
    const vkexp::neuro::Inputs inputs = vkexp::sampleAgentInputs(agent, settings);
    // Offsets come from the preset, not from arithmetic repeated here: the point
    // of testNeuralNetworkContract is that the preset derives them correctly, so
    // restating the sums would only test the restatement.
    namespace topology = vkexp::neuro;
    constexpr std::size_t center = 3 * topology::Topology::lightChannelsPerReceptor;
    check(inputs[center] > 0.0F && inputs[center + 1] > inputs[center],
          "RGB photoreceptor observes cyan beacon");
    check(inputs[center + 3] > 0.0F, "Photoreceptor luminance channel");
    constexpr std::size_t tactile = topology::Topology::tactileOffset;
    check(closeTo(inputs[tactile], 0.7F), "Wall tactile sector mapping");
    check(closeTo(inputs[tactile + 7 * 2 + 1], 0.8F), "Agent tactile sector mapping");
    constexpr std::size_t task = topology::Topology::taskOffset;
    check(closeTo(inputs[task], 0.7F) && closeTo(inputs[task + 1], 1.0F),
          "Cargo and task-state input mapping");
    constexpr std::size_t recurrent = topology::Topology::recurrentInputOffset;
    check(closeTo(inputs[recurrent], -0.4F) && closeTo(inputs[recurrent + 1], 0.6F),
          "Recurrent memory input mapping");
}

void testWorldAndBeaconScenarios() {
    check(closeTo(vkexp::worldRadiusForSize(vkexp::WorldSize::Small), 1.84F), "Small world radius");
    check(closeTo(vkexp::worldRadiusForSize(vkexp::WorldSize::Medium), 1.84F * 1.5F),
          "Medium world radius");
    check(closeTo(vkexp::worldRadiusForSize(vkexp::WorldSize::Large), 1.84F * 3.0F),
          "Large world radius");
    vkexp::SimulationStep arrivalSettings{};
    arrivalSettings.arrivalRadiusMultiplier = 0.1F;
    check(closeTo(vkexp::beaconArrivalRadius(arrivalSettings), vkexp::beaconVisualRadius * 0.1F),
          "Minimum arrival multiplier scales the beacon radius");
    arrivalSettings.arrivalRadiusMultiplier = 5.0F;
    check(closeTo(vkexp::beaconArrivalRadius(arrivalSettings), vkexp::beaconVisualRadius * 5.0F),
          "Maximum arrival multiplier scales the beacon radius");

    vkexp::AgentState agent{};
    agent.target = {1.0F, 0.5F, 2.0F, 0.0F};
    vkexp::SimulationStep settings{};
    const vkexp::ActiveBeacons stationary = vkexp::activeBeacons(agent, settings);
    check(stationary.count == 1 && closeTo(stationary.values[0].position.x, 1.0F) &&
              closeTo(stationary.values[0].position.y, 0.5F),
          "Stationary beacon uses the trial target");

    settings.beaconScenario = vkexp::BeaconScenario::AlternatingDiagonals;
    settings.beaconPhase = 0;
    const vkexp::ActiveBeacons firstDiagonal = vkexp::activeBeacons(agent, settings);
    settings.beaconPhase = 1;
    const vkexp::ActiveBeacons secondDiagonal = vkexp::activeBeacons(agent, settings);
    check(firstDiagonal.count == 2 &&
              firstDiagonal.values[0].position.x * firstDiagonal.values[0].position.y > 0.0F &&
              firstDiagonal.values[1].position.x * firstDiagonal.values[1].position.y > 0.0F,
          "First beacon pair occupies one diagonal");
    check(secondDiagonal.count == 2 &&
              secondDiagonal.values[0].position.x * secondDiagonal.values[0].position.y < 0.0F &&
              secondDiagonal.values[1].position.x * secondDiagonal.values[1].position.y < 0.0F,
          "Second beacon pair occupies the opposite diagonal");
    check(vkexp::beaconPhaseForStep(vkexp::BeaconScenario::AlternatingDiagonals, 449, 900) == 0 &&
              vkexp::beaconPhaseForStep(vkexp::BeaconScenario::AlternatingDiagonals, 450, 900) == 1,
          "Beacon diagonal changes at the generation midpoint");

    settings.beaconScenario = vkexp::BeaconScenario::Rotating;
    settings.beaconRadiusRatio = std::hypot(1.0F, 0.5F) / settings.worldRadius;
    settings.beaconRotationAngle = 1.57079632679F;
    const vkexp::ActiveBeacons rotating = vkexp::activeBeacons(agent, settings);
    check(rotating.count == 1 && closeTo(rotating.values[0].position.x, -0.5F) &&
              closeTo(rotating.values[0].position.y, 1.0F),
          "Rotating beacon orbits around the world center");
    check(closeTo(vkexp::beaconRotationAngleForStep(0.5F, 0.1F, 20), 1.0F),
          "Beacon angle follows simulation time and angular speed");

    settings.beaconScenario = vkexp::BeaconScenario::RandomMovement;
    settings.beaconRadiusRatio = 0.65F;
    settings.beaconTeleportProbability = 0.25F;
    settings.beaconMotionSeed = 42U;
    settings.beaconMotionTime = 0.0F;
    const vkexp::ActiveBeacons randomStart = vkexp::activeBeacons(agent, settings);
    settings.beaconRandomSpeed = 0.0F;
    settings.beaconMotionTime = 1.5F;
    const vkexp::ActiveBeacons randomStopped = vkexp::activeBeacons(agent, settings);
    settings.beaconRandomSpeed = 0.18F;
    const vkexp::ActiveBeacons randomMiddle = vkexp::activeBeacons(agent, settings);
    const float maximumRoamRadius = settings.worldRadius * settings.beaconRadiusRatio;
    check(randomStart.count == 1 &&
              std::hypot(randomStart.values[0].position.x, randomStart.values[0].position.y) <=
                  maximumRoamRadius &&
              std::hypot(randomMiddle.values[0].position.x, randomMiddle.values[0].position.y) <=
                  maximumRoamRadius,
          "Random beacon remains inside its configured roaming radius");
    check(closeTo(randomStart.values[0].position.x, randomStopped.values[0].position.x) &&
              closeTo(randomStart.values[0].position.y, randomStopped.values[0].position.y),
          "Zero wander speed stops continuous random movement");
    check(!closeTo(randomStart.values[0].position.x, randomMiddle.values[0].position.x) ||
              !closeTo(randomStart.values[0].position.y, randomMiddle.values[0].position.y),
          "Random beacon moves between deterministic waypoints");

    settings.beaconScenario = vkexp::BeaconScenario::ForageHome;
    settings.beaconRotationAngle = 0.0F;
    settings.beaconMotionSeed = 42U;
    settings.beaconMotionTime = 0.0F;
    const vkexp::ActiveBeacons forage = vkexp::activeBeacons(agent, settings);
    check(forage.count == 2 && forage.values[0].color.x > forage.values[0].color.z &&
              forage.values[1].color.z > forage.values[1].color.x,
          "Forage scenario exposes orange resource and blue home beacons");
    agent.pose = forage.values[0].position;
    agent.internal.y = 0.0F;
    check(closeTo(vkexp::nearestBeaconDistance(agent, settings), 0.0F),
          "Forage task targets the resource while empty");
    agent.pose = forage.values[1].position;
    agent.internal.y = 1.0F;
    check(closeTo(vkexp::nearestBeaconDistance(agent, settings), 0.0F),
          "Forage task targets home while carrying cargo");
    settings.beaconMotionTime = vkexp::forageHomeRelocationSeconds - 0.01F;
    const vkexp::Float4 homeBeforeRelocation = vkexp::homeBeaconPosition(agent, settings);
    settings.beaconMotionTime = vkexp::forageHomeRelocationSeconds;
    const vkexp::Float4 homeAfterRelocation = vkexp::homeBeaconPosition(agent, settings);
    check(!closeTo(homeBeforeRelocation.x, homeAfterRelocation.x) ||
              !closeTo(homeBeforeRelocation.y, homeAfterRelocation.y),
          "Forage home deterministically relocates at its configured interval");
    check(vkexp::homeBeaconRelocated(settings),
          "Forage scenario reports the exact home relocation step");
    settings.beaconMotionTime += settings.deltaTime;
    check(!vkexp::homeBeaconRelocated(settings),
          "Forage scenario reports relocation for only one simulation step");
}

void testForageCycleAndMemory() {
    vkexp::AgentState agent{};
    agent.pose.w = vkexp::agentBodyRadius;
    agent.motion.w = 1.0F;
    agent.target = {1.0F, 0.0F, 0.0F, 0.0F};
    vkexp::SimulationStep settings{};
    settings.beaconScenario = vkexp::BeaconScenario::ForageHome;
    settings.beaconRotationAngle = 0.0F;
    const vkexp::ActiveBeacons beacons = vkexp::activeBeacons(agent, settings);
    agent.pose.x = beacons.values[0].position.x;
    agent.pose.y = beacons.values[0].position.y;
    agent.metrics = {};

    vkexp::neuro::Weights weights{};
    constexpr std::size_t outputBias =
        vkexp::neuro::Topology::inputCount * vkexp::neuro::Topology::hiddenCount +
        vkexp::neuro::Topology::hiddenCount +
        vkexp::neuro::Topology::hiddenCount * vkexp::neuro::Topology::outputCount;
    weights[outputBias + 6] = 0.5F;
    weights[outputBias + 7] = -0.75F;

    vkexp::stepAgentCpu(agent, weights, settings);
    check(closeTo(agent.internal.x, 1.0F) && closeTo(agent.internal.y, 1.0F),
          "Resource pickup fills cargo and switches the task to home");
    check(closeTo(agent.internal.z, std::tanh(0.5F)) &&
              closeTo(agent.internal.w, std::tanh(-0.75F)),
          "Neural memory outputs persist in agent state");

    agent.pose.x = beacons.values[1].position.x;
    agent.pose.y = beacons.values[1].position.y;
    agent.motion.x = 0.0F;
    agent.motion.y = 0.0F;
    vkexp::stepAgentCpu(agent, weights, settings);
    check(closeTo(agent.internal.x, 0.0F) && closeTo(agent.internal.y, 0.0F),
          "Home delivery empties cargo and switches the task back to resource");
    check(vkexp::completedForageCycles(agent) == 1, "Home delivery completes one forage cycle");
    check(vkexp::agentFitness(agent, settings.beaconScenario) > 2.0F,
          "Completed forage cycle produces positive fitness");

    vkexp::AgentState radiusProbe{};
    radiusProbe.pose = {beacons.values[0].position.x + vkexp::beaconVisualRadius * 2.0F,
                        beacons.values[0].position.y, 0.0F, vkexp::agentBodyRadius};
    radiusProbe.motion.w = 1.0F;
    radiusProbe.target = {1.0F, 0.0F, 0.0F, 0.0F};
    radiusProbe.metrics = {vkexp::beaconVisualRadius * 2.0F, vkexp::beaconVisualRadius * 2.0F, 0.0F,
                           0.0F};
    vkexp::neuro::Weights zeroWeights{};
    settings.arrivalRadiusMultiplier = 1.0F;
    vkexp::stepAgentCpu(radiusProbe, zeroWeights, settings);
    check(radiusProbe.internal.y < 0.5F,
          "Default arrival radius requires entering the beacon circle");
    settings.arrivalRadiusMultiplier = 2.1F;
    vkexp::stepAgentCpu(radiusProbe, zeroWeights, settings);
    check(radiusProbe.internal.y >= 0.5F,
          "Expanded arrival radius permits pickup outside the visible circle");
}

void testWallCollisionPenalty() {
    vkexp::AgentState agent{};
    agent.pose = {1.817F, 0.0F, 0.0F, vkexp::agentBodyRadius};
    agent.motion = {0.55F, 0.0F, 0.0F, 1.0F};
    agent.target = {0.0F, 0.0F, 0.0F, 0.0F};
    agent.metrics = {1.817F, 1.817F, 0.0F, 0.0F};
    vkexp::SimulationStep settings{};
    settings.worldShape = vkexp::WorldShape::Square;
    settings.wallCollisionPenalty = 0.1F;
    const vkexp::neuro::Weights weights{};

    vkexp::stepAgentCpu(agent, weights, settings);

    const float touch = agent.wallTouch0.x + agent.wallTouch0.y + agent.wallTouch0.z +
                        agent.wallTouch0.w + agent.wallTouch1.x + agent.wallTouch1.y +
                        agent.wallTouch1.z + agent.wallTouch1.w;
    check(touch > 0.0F, "World boundary produces tactile contact");
    check(agent.penalties.x > 0.0F, "World boundary accumulates a fitness penalty");
    const vkexp::FitnessWeights fitnessWeights{};
    check(vkexp::agentFitness(agent, settings.beaconScenario, fitnessWeights) <
              agent.metrics.w + (agent.metrics.x - agent.metrics.y) -
                  agent.metrics.z * fitnessWeights.motorCostWeight,
          "Wall collision penalty lowers fitness");
}

// The step used to mix two time bases: velocities, drags and motor costs were
// integrated per second, while the wall penalty and the contact solver
// accumulated per step. That made deltaTime a fitness parameter in disguise --
// the same trial at 240 Hz charged four times the wall penalty of one at 60 Hz,
// which is why deltaTime was pinned at 1/60 and never exposed.
//
// Driving an agent into a wall and holding it there for two simulated seconds
// is what makes the difference visible. Charging per second leaves a residual
// spread of about 1.4x across a 16x change in step rate, all of it from contact
// detection in a bouncing model: the impact speed feeding the contact strength
// shrinks with the step, and the chatter duty cycle drifts from 95% to 88%.
// Charging per step would instead scale straight with the step count.
void testFixedStepIndependence() {
    const vkexp::neuro::BrainShape brain =
        vkexp::scenarioDefinition(vkexp::BeaconScenario::Stationary).brain;
    const auto inputCount = static_cast<vkexp::neuro::kernel::uint>(brain.inputCount);
    const auto hiddenCount = static_cast<vkexp::neuro::kernel::uint>(brain.hiddenCount);
    const auto outputCount = static_cast<vkexp::neuro::kernel::uint>(brain.outputCount);
    vkexp::neuro::Weights drivingWeights{};
    for (const vkexp::neuro::kernel::uint motor : {vkexp::neuro::kernel::BrainMotorLeftOutput,
                                                   vkexp::neuro::kernel::BrainMotorRightOutput}) {
        drivingWeights[vkexp::neuro::kernel::brainOutputBiasIndex(0, inputCount, hiddenCount,
                                                                  outputCount, motor)] = 3.0F;
    }

    const auto penaltyForTwoSeconds = [&drivingWeights](const float rateHz) {
        vkexp::SimulationStep settings{};
        settings.deltaTime = 1.0F / rateHz;
        settings.worldShape = vkexp::WorldShape::Square;
        settings.wallCollisionPenalty = 0.6F;
        vkexp::AgentState agent{};
        agent.pose = {1.81F, 0.0F, 0.0F, vkexp::agentBodyRadius};
        agent.motion = {0.0F, 0.0F, 0.0F, 1.0F};
        agent.metrics = {1.81F, 1.81F, 0.0F, 0.0F};
        const auto steps = static_cast<std::uint32_t>(rateHz * 2.0F);
        for (std::uint32_t step = 0; step < steps; ++step) {
            vkexp::stepAgentCpu(agent, drivingWeights, settings);
        }
        return agent.penalties.x;
    };

    const float baseline = penaltyForTwoSeconds(vkexp::units::simulationRateHz);
    check(baseline > 0.0F, "Driving into a wall for two seconds accumulates a penalty");
    for (const float rateHz : {30.0F, 120.0F, 240.0F, 480.0F}) {
        const float ratio = penaltyForTwoSeconds(rateHz) / baseline;
        const float stepCountRatio = rateHz / vkexp::units::simulationRateHz;
        check(ratio > 0.75F && ratio < 1.25F,
              "Wall penalty over two simulated seconds barely moves with the step rate");
        // The discriminating half: per-step accumulation would put the ratio at
        // the step-count ratio instead, which is 0.5x to 8x here.
        check(std::abs(ratio - 1.0F) < std::abs(stepCountRatio - 1.0F) * 0.5F,
              "Wall penalty tracks simulated time rather than step count");
    }

    check(std::abs(vkexp::units::secondsForSteps(900, vkexp::units::fixedTimeStep) - 15.0F) < 1e-5F,
          "900 steps at the fixed rate is 15 seconds");
    check(std::abs(vkexp::units::secondsForSteps(3600, 1.0F / 240.0F) - 15.0F) < 1e-5F,
          "3600 steps at 240 Hz covers the same 15 seconds");
}

void testScenarioRegistryContract() {
    const std::span<const vkexp::ScenarioDefinition* const> registry = vkexp::scenarioRegistry();
    check(registry.size() == vkexp::beaconScenarioCount, "Registry covers every BeaconScenario");
    for (std::size_t index = 0; index < registry.size(); ++index) {
        const vkexp::ScenarioDefinition& scenario = *registry[index];
        const std::string label{scenario.name};
        check(scenario.id == static_cast<vkexp::BeaconScenario>(index),
              label + ": registry order matches the enum");
        check(&vkexp::scenarioDefinition(scenario.id) == &scenario,
              label + ": lookup returns the registered definition");
        check(scenario.key != nullptr && scenario.key[0] != '\0', label + ": has a CLI key");
        check(scenario.brain.fitsCapacity(), label + ": brain fits the genome capacity");

        // beaconCount is declared separately because the renderer needs it
        // without an agent; it must still agree with what beacons() reports.
        vkexp::SimulationStep settings{};
        settings.beaconScenario = scenario.id;
        vkexp::AgentState agent{};
        agent.target = {settings.worldRadius * 0.7F, 0.0F, 0.0F, 0.0F};
        check(scenario.beacons(agent, settings).count == scenario.beaconCount,
              label + ": declared beacon count matches the beacons it reports");

        // Every scenario must be steppable without the caller knowing which it is.
        const vkexp::neuro::Weights zeroWeights{};
        vkexp::AgentState stepped = agent;
        stepped.pose.w = vkexp::agentBodyRadius;
        vkexp::stepAgentCpu(stepped, zeroWeights, settings);
        check(std::isfinite(stepped.pose.x) && std::isfinite(stepped.metrics.w),
              label + ": one step through the hooks stays finite");

        // Every scenario says how long a trial has to be for its objectives to
        // be reachable, because a world run in too short a trial reports a ratio
        // that cannot reach one and nothing about the picture says so. The UI
        // offers the number and the headless runner defaults to it, so a
        // scenario that left it at zero would silently ask for no time at all.
        check(scenario.nominalStepsPerGeneration >= 120U,
              label + ": declares a trial length its objectives can be reached in");
    }
}

void testFitnessWeightsAreParameters() {
    vkexp::AgentState agent{};
    agent.metrics = {2.0F, 1.0F, 10.0F, 3.0F};
    agent.target.w = 1.0F; // one completed phase
    const vkexp::BeaconScenario scenario = vkexp::BeaconScenario::Stationary;

    vkexp::FitnessWeights base{};
    const float reference = vkexp::agentFitness(agent, scenario, base);

    vkexp::FitnessWeights doubledBonus = base;
    doubledBonus.objectiveBonus = base.objectiveBonus * 2.0F;
    check(closeTo(vkexp::agentFitness(agent, scenario, doubledBonus),
                  reference + base.objectiveBonus),
          "Objective bonus is a parameter, not a literal");

    vkexp::FitnessWeights freeMotors = base;
    freeMotors.motorCostWeight = 0.0F;
    check(closeTo(vkexp::agentFitness(agent, scenario, freeMotors),
                  reference + agent.metrics.z * base.motorCostWeight),
          "Motor cost weight is a parameter, not a literal");
}

void testSharedScenarioKernel() {
    namespace kernel = vkexp::worlds::kernel;
    // The shaders compile these same functions from ScenarioKernel.inl, and the
    // CPU/GPU parity tests compare the results; these checks pin the C++ side so
    // a change to the shared source cannot pass unnoticed without a GPU.
    check(kernel::scenarioHash(0U) == 0U, "Hash of zero is zero");
    const float sample = kernel::scenarioRandom01(12345U);
    check(sample >= 0.0F && sample <= 1.0F, "Scenario random is normalised");
    check(closeTo(kernel::scenarioRandom01(12345U), sample), "Scenario random is deterministic");

    check(kernel::forageHomeEpoch(0.0F) == 0U, "Forage epoch starts at zero");
    check(kernel::forageHomeEpoch(kernel::ForageHomeRelocationSeconds + 0.1F) == 1U,
          "Forage epoch advances at the relocation period");
    check(kernel::forageHomeRelocated(kernel::ForageHomeRelocationSeconds, 1.0F / 60.0F),
          "Forage relocation is reported on the epoch boundary");
    check(!kernel::forageHomeRelocated(1.0F, 1.0F / 60.0F),
          "Forage relocation is not reported mid-epoch");

    const kernel::vec2 home = kernel::forageHomeOffset(kernel::forageHomeKey(7U, 1U, 2U));
    const float homeRatio = kernel::length(home);
    check(homeRatio >= kernel::ForageHomeMinimumRadiusRatio - 0.0001F &&
              homeRatio <=
                  kernel::ForageHomeMinimumRadiusRatio + kernel::ForageHomeRadiusRange + 0.0001F,
          "Forage home stays inside its configured annulus");

    // A quarter turn maps +x to +y.
    const kernel::vec2 rotated =
        kernel::rotatingOrbitOffset({1.0F, 0.0F}, kernel::ScenarioTau * 0.25F);
    check(closeTo(rotated.x, 0.0F) && closeTo(rotated.y, 1.0F), "Orbit offset rotates correctly");

    const kernel::vec2 wander =
        kernel::randomWanderOffset(kernel::randomWanderKey(3U, 0U, 0U), 5.0F);
    check(kernel::length(wander) <= 1.0F, "Wander offset stays within the roam radius");

    check(kernel::alternatingDiagonalOffset(0U, 0U).x < 0.0F &&
              kernel::alternatingDiagonalOffset(1U, 0U).x > 0.0F,
          "Alternating beacons sit on opposite sides");
}

// The network written down as structure, and whether that writing-down is
// trustworthy. A description that merely looks right is worse than none: the
// whole point of putting it in a file is that a loader can act on it.
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
    const auto inside = [&](const char* name, const bk::uint index) {
        const vkexp::neuro::BrainBlock* const block = description.block(name);
        return block != nullptr && index >= block->offset && index < block->offset + block->count;
    };
    check(inside("hidden_weights", bk::brainHiddenWeightIndex(0u, inputs, 0u, 0u)) &&
              inside("hidden_weights",
                     bk::brainHiddenWeightIndex(0u, inputs, hidden - 1u, inputs - 1u)),
          "Both corners of the input-to-hidden matrix fall in its block");
    check(inside("hidden_bias", bk::brainHiddenBiasIndex(0u, inputs, hidden, 0u)) &&
              inside("hidden_bias", bk::brainHiddenBiasIndex(0u, inputs, hidden, hidden - 1u)),
          "Both ends of the hidden bias fall in its block");
    check(inside("output_weights", bk::brainOutputWeightIndex(0u, inputs, hidden, 0u, 0u)) &&
              inside("output_weights",
                     bk::brainOutputWeightIndex(0u, inputs, hidden, outputs - 1u, hidden - 1u)),
          "Both corners of the hidden-to-output matrix fall in its block");
    check(inside("output_bias", bk::brainOutputBiasIndex(0u, inputs, hidden, outputs, 0u)) &&
              inside("output_bias",
                     bk::brainOutputBiasIndex(0u, inputs, hidden, outputs, outputs - 1u)),
          "Both ends of the output bias fall in its block");
    check(inside("time_constants",
                 bk::brainTimeConstantGeneIndex(0u, inputs, hidden, outputs, 0u)) &&
              inside("time_constants",
                     bk::brainTimeConstantGeneIndex(0u, inputs, hidden, outputs, hidden - 1u)),
          "Both ends of the time constants fall in their block");
    check(inside("gate_weights", bk::brainGateWeightIndex(0u, inputs, hidden, outputs, 0u, 0u)) &&
              inside("gate_weights", bk::brainGateWeightIndex(0u, inputs, hidden, outputs,
                                                              hidden - 1u, inputs - 1u)),
          "Both corners of the gate matrix fall in its block");
    check(inside("gate_bias", bk::brainGateBiasIndex(0u, inputs, hidden, outputs, 0u)) &&
              inside("gate_bias",
                     bk::brainGateBiasIndex(0u, inputs, hidden, outputs, hidden - 1u)),
          "Both ends of the gate bias fall in their block");

    // And the sensor blocks against the sensor index functions, which is the
    // half a weight-block check cannot reach.
    check(inside("light", bk::brainLightChannelIndex(0u, 0u)) &&
              inside("light", bk::brainLightChannelIndex(bk::BrainLightReceptorCount - 1u,
                                                         bk::BrainLightChannels - 1u)),
          "The light block covers every receptor channel");
    check(inside("tactile", bk::brainTactileChannelIndex(0u, 0u)) &&
              inside("tactile", bk::brainTactileChannelIndex(bk::BrainTactileSectorCount - 1u,
                                                             bk::BrainTactileChannels - 1u)),
          "The tactile block covers every sector channel");
    check(inside("antennae", bk::brainAntennaChannelIndex(0u, 0u)) &&
              inside("antennae", bk::brainAntennaChannelIndex(bk::BrainAntennaCount - 1u,
                                                              bk::BrainAntennaChannels - 1u)),
          "The antenna block covers every tip channel");
    check(inside("motor_left", bk::BrainMotorLeftOutput) &&
              inside("motor_right", bk::BrainMotorRightOutput) &&
              inside("signal_color", bk::BrainSignalColorOutput) &&
              inside("signal_intensity", bk::BrainSignalIntensityOutput) &&
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
    const vkexp::neuro::BrainDescription trimmed =
        vkexp::neuro::describeBrain({52, 20, 8});
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
    check(!differences.empty() && differences.front().find("light") != std::string::npos,
          "A moved block is reported by its own name");
}

void testGenomeArchiveRoundTrip() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "vkexp_archive_test" / "population.vkng";
    std::vector<vkexp::Genome> genomes(3);
    for (std::size_t index = 0; index < genomes.size(); ++index) {
        for (std::size_t weight = 0; weight < genomes[index].weights.size(); ++weight) {
            genomes[index].weights[weight] =
                std::sin(static_cast<float>(index * 31 + weight) * 0.017F);
        }
    }
    const vkexp::GenomeArchiveMetadata metadata{42, 4, 0xC0FFEEU, 1.5F, 0.25F, 52, 20, 8};
    vkexp::saveGenomeArchive(path, genomes, metadata);

    const vkexp::GenomeArchive loaded = vkexp::loadGenomeArchive(path);
    check(loaded.genomes.size() == genomes.size(), "Archive genome count round-trip");
    check(loaded.metadata.generation == 42, "Archive generation round-trip");
    check(loaded.metadata.scenario == 4, "Archive scenario round-trip");
    check(loaded.metadata.seed == 0xC0FFEEU, "Archive seed round-trip");
    check(closeTo(loaded.metadata.bestFitness, 1.5F), "Archive best fitness round-trip");
    check(loaded.metadata.brainOutputCount == 8, "Archive brain shape round-trip");
    bool identical = true;
    for (std::size_t index = 0; index < genomes.size(); ++index) {
        identical = identical && loaded.genomes[index].weights == genomes[index].weights;
    }
    check(identical, "Archive weights round-trip bit-exactly");
    check(loaded.describedStructure, "An archive states the structure its weights are laid out in");
    check(loaded.description.inputCount == 52 && loaded.description.hiddenCount == 20 &&
              loaded.description.outputCount == 8,
          "and states it for the shape the run actually used");

    // The whole reason the structure is in the file: a file whose weights mean
    // something else has to fail, and fail by naming what moved. Patched in
    // place and byte for byte -- "antennae" becomes "antennaX", same length, so
    // the header's byte count still matches and nothing but the meaning changes.
    const std::filesystem::path renamed = path.parent_path() / "renamed.vkng";
    std::filesystem::copy_file(path, renamed, std::filesystem::copy_options::overwrite_existing);
    {
        std::fstream stream{renamed, std::ios::binary | std::ios::in | std::ios::out};
        std::string contents{std::istreambuf_iterator<char>{stream},
                             std::istreambuf_iterator<char>{}};
        const std::size_t at = contents.find("antennae");
        check(at != std::string::npos, "The structure block is really in the file as text");
        stream.clear();
        stream.seekp(static_cast<std::streamoff>(at));
        stream.write("antennaX", 8);
    }
    std::string complaint;
    try {
        (void)vkexp::loadGenomeArchive(renamed);
    } catch (const vkexp::GenomeArchiveError& error) {
        complaint = error.what();
    }
    check(complaint.find("antennae") != std::string::npos &&
              complaint.find("antennaX") != std::string::npos,
          "A file describing a different network is refused, and both names are said");

    // Version 1 files predate the structure block and still load: the weights
    // were laid out the same way, the file simply does not say so. Built by
    // surgery on a version 2 file, because there is no writer for the old format
    // any more -- version at byte 4, structure length at byte 52, header 56.
    const std::filesystem::path legacy = path.parent_path() / "legacy.vkng";
    {
        std::ifstream input{path, std::ios::binary};
        std::string contents{std::istreambuf_iterator<char>{input},
                             std::istreambuf_iterator<char>{}};
        std::uint32_t structureBytes = 0;
        std::memcpy(&structureBytes, contents.data() + 52, sizeof(structureBytes));
        check(structureBytes > 0, "A version 2 file records how long its structure block is");
        const std::uint32_t one = 1;
        std::memcpy(contents.data() + 4, &one, sizeof(one));
        const std::uint32_t none = 0;
        std::memcpy(contents.data() + 52, &none, sizeof(none));
        contents.erase(56, structureBytes);
        std::ofstream output{legacy, std::ios::binary | std::ios::trunc};
        output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    }
    const vkexp::GenomeArchive old = vkexp::loadGenomeArchive(legacy);
    check(old.genomes.size() == genomes.size() &&
              old.genomes.front().weights == genomes.front().weights,
          "A version 1 archive still loads its weights");
    check(!old.describedStructure,
          "and says plainly that nothing about its structure was checked");

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

void testWorldSnapshotRoundTrip() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "vkexp_snapshot_test" / "world.vknw";

    vkexp::WorldSnapshot snapshot;
    // Values chosen to be distinguishable from each other and from any default,
    // and named here by hand rather than through the saver's own field list, so
    // a field dropped from that list shows up as a wrong value rather than
    // agreeing with itself.
    snapshot.physics.worldRadius = 3.75F;
    snapshot.physics.maximumSpeed = 0.91F;
    snapshot.physics.trailHalfLife = 11.5F;
    snapshot.physics.trailCellSize = 0.031F;
    snapshot.physics.beaconAngularSpeed = 0.77F;
    snapshot.physics.fitness.trackingReward = 0.42F;
    snapshot.physics.fitness.energyDrain = 0.0013F;
    snapshot.physics.worldSize = vkexp::WorldSize::Large;
    snapshot.physics.worldShape = vkexp::WorldShape::Square;
    snapshot.physics.beaconScenario = vkexp::BeaconScenario::ScentRelay;
    snapshot.physics.beaconMotionSeed = 987654U;
    snapshot.physics.beaconPhase = 3U;
    // The middle setting on purpose: it is the one a saver still shaped like a
    // bool would silently collapse, and it is neither the default nor the value
    // the inverted pass below uses.
    snapshot.physics.trailMode = vkexp::TrailMode::Visual;
    snapshot.physics.agentLightEnabled = false;
    snapshot.physics.agentCollisionsEnabled = true;
    // Named here rather than left at its default: a setting the saver forgets
    // still round-trips its own default, so only a value nothing defaults to
    // proves the field made the journey.
    snapshot.physics.neuronModel = vkexp::NeuronModel::Gated;
    // The three world options are bools that default to false and landed in
    // padding, so the size assertion beside the saver's field lists did not move
    // when they were added. This test is their only guard.
    snapshot.physics.swapDeliveryEnds = true;
    snapshot.physics.uniformBeaconColor = true;
    snapshot.physics.blockedDoorPerGeneration = true;
    snapshot.physics.gateLatchSeconds = 2.75F;
    snapshot.physics.puckBreakawayPushes = 2.5F;
    snapshot.physics.puckRandomStart = true;

    snapshot.genomes.resize(4);
    for (std::size_t index = 0; index < snapshot.genomes.size(); ++index) {
        for (std::size_t weight = 0; weight < snapshot.genomes[index].weights.size(); ++weight) {
            snapshot.genomes[index].weights[weight] =
                std::cos(static_cast<float>(index * 17 + weight) * 0.023F);
        }
    }
    snapshot.agents.resize(8);
    for (std::size_t index = 0; index < snapshot.agents.size(); ++index) {
        const auto value = static_cast<float>(index) * 0.125F;
        snapshot.agents[index].pose = {value, -value, value * 2.0F, vkexp::agentBodyRadius};
        snapshot.agents[index].motion = {value * 3.0F, value, 0.5F, 1.0F - value * 0.05F};
        snapshot.agents[index].signal = {value, 1.0F - value, 0.25F, value * 0.5F};
        snapshot.agents[index].metrics = {value, value + 1.0F, value * 4.0F, 2.0F};
    }
    snapshot.generation = 137;
    snapshot.step = 451;
    snapshot.stepsPerGeneration = 900;
    snapshot.requestedAgentsPerWorld = 12;
    snapshot.trialsPerGenome = 4;
    snapshot.seed = 0xBADF00DU;

    vkexp::saveWorldSnapshot(path, snapshot);
    const vkexp::WorldSnapshot loaded = vkexp::loadWorldSnapshot(path);

    check(loaded.generation == 137 && loaded.step == 451, "Snapshot resumes on the saved step");
    check(loaded.stepsPerGeneration == 900 && loaded.requestedAgentsPerWorld == 12 &&
              loaded.trialsPerGenome == 4 && loaded.seed == 0xBADF00DU,
          "Snapshot layout round-trip");
    check(closeTo(loaded.physics.worldRadius, 3.75F) &&
              closeTo(loaded.physics.maximumSpeed, 0.91F) &&
              closeTo(loaded.physics.trailHalfLife, 11.5F) &&
              closeTo(loaded.physics.trailCellSize, 0.031F) &&
              closeTo(loaded.physics.beaconAngularSpeed, 0.77F),
          "Snapshot physics floats round-trip");
    check(closeTo(loaded.physics.fitness.trackingReward, 0.42F) &&
              closeTo(loaded.physics.fitness.energyDrain, 0.0013F),
          "Snapshot fitness weights round-trip");
    check(loaded.physics.worldSize == vkexp::WorldSize::Large &&
              loaded.physics.worldShape == vkexp::WorldShape::Square &&
              loaded.physics.beaconScenario == vkexp::BeaconScenario::ScentRelay &&
              loaded.physics.beaconMotionSeed == 987654U && loaded.physics.beaconPhase == 3U,
          "Snapshot world identity round-trip");
    check(loaded.physics.trailMode == vkexp::TrailMode::Visual &&
              !loaded.physics.agentLightEnabled &&
              loaded.physics.agentCollisionsEnabled,
          "Snapshot ablation flags round-trip");
    check(loaded.physics.neuronModel == vkexp::NeuronModel::Gated,
          "Snapshot keeps the neuron model");
    check(loaded.physics.swapDeliveryEnds && loaded.physics.uniformBeaconColor &&
              loaded.physics.blockedDoorPerGeneration,
          "Snapshot keeps the world options");
    check(closeTo(loaded.physics.gateLatchSeconds, 2.75F),
          "Snapshot keeps the gate latch, which is a whole world's difficulty");
    check(closeTo(loaded.physics.puckBreakawayPushes, 2.5F),
          "Snapshot keeps the puck's friction floor, which decides whether one agent can move it");
    check(loaded.physics.puckRandomStart, "Snapshot keeps where the puck is placed");

    // Once more with every flag inverted, because one polarity proves nothing
    // about a bool. All four default to true, so a flag the loader forgets to
    // assign keeps its default and passes a test that only ever asks for true,
    // while a flag the saver forgets to write reads back as false and passes a
    // test that only ever asks for false. Only both directions catch both, and
    // this is the check that has to: the size assertion beside the saver's field
    // lists does not move when a bool is added into padding the trailing bools
    // already carry.
    {
        vkexp::WorldSnapshot inverted = snapshot;
        inverted.physics.trailMode = vkexp::TrailMode::Off;
        inverted.physics.agentLightEnabled = true;
        inverted.physics.agentCollisionsEnabled = false;
        inverted.physics.beaconPhaseChanged = true;
        inverted.physics.neuronModel = vkexp::NeuronModel::Reactive;
        inverted.physics.swapDeliveryEnds = false;
        inverted.physics.uniformBeaconColor = false;
        inverted.physics.blockedDoorPerGeneration = false;
        inverted.physics.puckRandomStart = false;
        const std::filesystem::path invertedPath = path.parent_path() / "inverted.vknw";
        vkexp::saveWorldSnapshot(invertedPath, inverted);
        const vkexp::WorldSnapshot back = vkexp::loadWorldSnapshot(invertedPath);
        check(back.physics.trailMode == vkexp::TrailMode::Off &&
                  back.physics.agentLightEnabled &&
                  !back.physics.agentCollisionsEnabled && back.physics.beaconPhaseChanged &&
                  back.physics.neuronModel == vkexp::NeuronModel::Reactive &&
                  !back.physics.swapDeliveryEnds && !back.physics.uniformBeaconColor &&
                  !back.physics.blockedDoorPerGeneration && !back.physics.puckRandomStart,
              "Snapshot ablation flags round-trip in both directions");
    }

    bool weightsIdentical = loaded.genomes.size() == snapshot.genomes.size();
    for (std::size_t index = 0; weightsIdentical && index < snapshot.genomes.size(); ++index) {
        weightsIdentical = loaded.genomes[index].weights == snapshot.genomes[index].weights;
    }
    check(weightsIdentical, "Snapshot weights round-trip bit-exactly");

    // The point of a snapshot over an archive: the agents come back where they
    // stood, not respawned.
    bool posesIdentical = loaded.agents.size() == snapshot.agents.size();
    for (std::size_t index = 0; posesIdentical && index < snapshot.agents.size(); ++index) {
        const vkexp::AgentState& want = snapshot.agents[index];
        const vkexp::AgentState& got = loaded.agents[index];
        posesIdentical = closeTo(got.pose.x, want.pose.x) && closeTo(got.pose.y, want.pose.y) &&
                         closeTo(got.pose.z, want.pose.z) &&
                         closeTo(got.motion.x, want.motion.x) &&
                         closeTo(got.motion.w, want.motion.w) &&
                         closeTo(got.signal.w, want.signal.w) &&
                         closeTo(got.metrics.z, want.metrics.z);
    }
    check(posesIdentical, "Snapshot agent state round-trip");

    const std::filesystem::path corrupted = path.parent_path() / "corrupted.vknw";
    std::filesystem::copy_file(path, corrupted, std::filesystem::copy_options::overwrite_existing);
    {
        std::fstream stream{corrupted, std::ios::binary | std::ios::in | std::ios::out};
        stream.seekp(0);
        stream.write("VKNG", 4);
    }
    bool rejectedArchive = false;
    try {
        (void)vkexp::loadWorldSnapshot(corrupted);
    } catch (const vkexp::WorldSnapshotError&) {
        rejectedArchive = true;
    }
    // A genome archive and a world snapshot both end in binary weights; loading
    // one as the other has to fail on the magic rather than half-work.
    check(rejectedArchive, "Snapshot rejects a genome archive");

    const std::filesystem::path truncated = path.parent_path() / "truncated.vknw";
    std::filesystem::copy_file(path, truncated, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) - 24);
    bool rejectedTruncation = false;
    try {
        (void)vkexp::loadWorldSnapshot(truncated);
    } catch (const vkexp::WorldSnapshotError&) {
        rejectedTruncation = true;
    }
    check(rejectedTruncation, "Snapshot rejects a truncated file");

    std::error_code cleanupError;
    std::filesystem::remove_all(path.parent_path(), cleanupError);
}

void testPopulationReload() {
    const vkexp::EvolutionSettings settings{8, 2, 3, 0.5F, 0.1F, 0.2F, 42U};
    vkexp::GeneticAlgorithm evolution{settings};
    std::vector<vkexp::Genome> replacement(settings.populationSize);
    replacement.front().weights[0] = 3.25F;
    evolution.setPopulation(replacement, 17);
    check(evolution.generation() == 17, "Loaded population restores the generation counter");
    check(closeTo(evolution.population().front().weights[0], 3.25F),
          "Loaded population replaces the weights");

    bool rejectedMismatch = false;
    try {
        const std::vector<vkexp::Genome> wrongSize(settings.populationSize - 1);
        evolution.setPopulation(wrongSize, 0);
    } catch (const std::invalid_argument&) {
        rejectedMismatch = true;
    }
    check(rejectedMismatch, "Loaded population size mismatch rejection");
}

void testStepParameterPacking() {
    // The GPU step parameters outgrew the 128 bytes Vulkan guarantees for push
    // constants, which is why they now travel in a storage buffer.
    check(sizeof(vkexp::GpuStepParameters) > 128,
          "Step parameters exceed the guaranteed push constant size");
    check(sizeof(vkexp::ScenarioParameterBlock) == 48, "Scenario block size");

    vkexp::SimulationStep settings{};
    settings.beaconScenario = vkexp::BeaconScenario::Rotating;
    settings.beaconRotationAngle = 0.75F;
    settings.beaconRadiusRatio = 0.6F;
    const vkexp::ScenarioParameterBlock rotating =
        vkexp::scenarioDefinition(settings.beaconScenario).gpuParameters(settings);
    check(closeTo(rotating.floats0.x, 0.75F), "Rotating scenario packs its rotation angle");
    check(closeTo(rotating.floats0.y, 0.6F), "Rotating scenario packs its radius ratio");

    settings.beaconScenario = vkexp::BeaconScenario::ForageHome;
    settings.beaconMotionSeed = 0xABCDU;
    settings.forageCargoDecayRate = 0.11F;
    const vkexp::ScenarioParameterBlock forage =
        vkexp::scenarioDefinition(settings.beaconScenario).gpuParameters(settings);
    check(forage.integers[0] == 0xABCDU, "Forage scenario packs its motion seed");
    check(closeTo(forage.floats0.w, 0.11F), "Forage scenario packs its cargo decay rate");

    // Stationary beacons live in the agent, so its block must stay empty.
    settings.beaconScenario = vkexp::BeaconScenario::Stationary;
    const vkexp::ScenarioParameterBlock stationary =
        vkexp::scenarioDefinition(settings.beaconScenario).gpuParameters(settings);
    check(stationary.floats0.x == 0.0F && stationary.integers[0] == 0U,
          "Stationary scenario sends no scenario parameters");
}

void testResolvedStepSettings() {
    vkexp::SimulationStep base{};
    base.beaconScenario = vkexp::BeaconScenario::AlternatingDiagonals;
    base.beaconAngularSpeed = 1.0F;
    constexpr std::uint32_t steps = 100;
    const vkexp::SimulationStep before = vkexp::resolveStepSettings(base, 10, steps);
    const vkexp::SimulationStep atFlip = vkexp::resolveStepSettings(base, steps / 2, steps);
    const vkexp::SimulationStep after = vkexp::resolveStepSettings(base, steps / 2 + 1, steps);
    check(before.beaconPhase == 0 && !before.beaconPhaseChanged, "Phase before the flip");
    check(atFlip.beaconPhase == 1 && atFlip.beaconPhaseChanged, "Phase change reported once");
    check(after.beaconPhase == 1 && !after.beaconPhaseChanged, "Phase after the flip");
    check(closeTo(atFlip.beaconMotionTime, base.deltaTime * static_cast<float>(steps / 2)),
          "Resolved motion time follows the step index");
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
    vkexp::neuro::Weights weights{};
    constexpr auto inputCount = static_cast<kernel::uint>(vkexp::neuro::Topology::inputCount);
    constexpr auto hiddenCount = static_cast<kernel::uint>(vkexp::neuro::Topology::hiddenCount);
    weights[kernel::brainHiddenWeightIndex(0u, inputCount, 0u, 0u)] = 3.0F;
    weights[kernel::brainOutputWeightIndex(0u, inputCount, hiddenCount, 0u, 0u)] = 3.0F;
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
    constexpr auto hiddenCount = static_cast<kernel::uint>(vkexp::neuro::Topology::hiddenCount);
    constexpr auto outputCount = static_cast<kernel::uint>(vkexp::neuro::Topology::outputCount);
    const float step = vkexp::units::fixedTimeStep;

    vkexp::neuro::Weights weights{};
    weights[kernel::brainHiddenWeightIndex(0u, inputCount, 0u, 0u)] = 3.0F;
    weights[kernel::brainOutputWeightIndex(0u, inputCount, hiddenCount, 0u, 0u)] = 3.0F;
    vkexp::neuro::Inputs inputs{};
    inputs[0] = 1.0F;

    // The claim that makes gated a generalisation rather than a third network:
    // a gate that does not listen to anything is the fixed-time-constant neuron,
    // exactly. Both genes go through the same mapping, so setting the gate bias
    // and the time-constant gene to the same number has to give the same state.
    for (const float gene : {-2.0F, 0.0F, 1.5F}) {
        vkexp::neuro::Weights fixed = weights;
        vkexp::neuro::Weights gated = weights;
        fixed[kernel::brainTimeConstantGeneIndex(0u, inputCount, hiddenCount, outputCount, 0u)] =
            gene;
        gated[kernel::brainGateBiasIndex(0u, inputCount, hiddenCount, outputCount, 0u)] = gene;

        vkexp::neuro::HiddenState fixedState{};
        vkexp::neuro::HiddenState gatedState{};
        for (int index = 0; index < 20; ++index) {
            (void)vkexp::neuro::evaluate(fixed, inputs, fixedState, step,
                                         kernel::NeuronModelTimeConstant);
            (void)vkexp::neuro::evaluate(gated, inputs, gatedState, step,
                                         kernel::NeuronModelGated);
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
    listening[kernel::brainGateWeightIndex(0u, inputCount, hiddenCount, outputCount, 0u, 1u)] =
        8.0F;
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
    check(kernel::brainGateBiasIndex(0u, inputCount, hiddenCount, outputCount, hiddenCount - 1u) ==
              vkexp::neuro::Topology::weightCount - 1u,
          "The gate block ends exactly at the end of the genome");
    check(kernel::brainGateWeightIndex(0u, inputCount, hiddenCount, outputCount, 0u, 0u) >
              kernel::brainTimeConstantGeneIndex(0u, inputCount, hiddenCount, outputCount,
                                                 hiddenCount - 1u),
          "The gate block starts after the time constants");
    check(vkexp::neuro::Topology::weightCount <= kernel::BrainStrideMask,
          "The genome still fits the packed brain layout");
}

// What fraction of the far side of the wall can see `target` at all: in light
// range and with no box in the way. This is the number that separates a world
// that is learned from one that is not, so it belongs in the build rather than
// in a notebook. Two doors sat at 5.7% and was not solved in 450 generations;
// Two gaps sits at 19% and is solved. The floor below is set between them.
//
// Swept over the area rather than probed at points: the question is how much
// room there is to pick the signal up, and no single point answers that.
template <typename BoxAt>
float visibleFractionOfFarSide(const float radius, const vkexp::worlds::kernel::vec2 target,
                               const std::uint32_t boxCount, BoxAt&& boxAt,
                               const float sideSign = 1.0F) {
    namespace kernel = vkexp::worlds::kernel;
    vkexp::SimulationStep settings;
    settings.worldRadius = radius;
    const float range = vkexp::lightRangeForWorld(settings);
    const float body = vkexp::agentBodyRadius;
    constexpr int samples = 200;
    int standable = 0;
    int visible = 0;
    for (int ix = 0; ix < samples; ++ix) {
        for (int iy = 0; iy < samples; ++iy) {
            const float x = -radius + 2.0F * radius * (static_cast<float>(ix) + 0.5F) /
                                          static_cast<float>(samples);
            // The far side is the half the target is not on. `sideSign` flips
            // which half that is, for a world whose target sits on the other one.
            const float y = sideSign * radius * (static_cast<float>(iy) + 0.5F) /
                            static_cast<float>(samples);
            if (std::hypot(x, y) > radius) {
                continue;
            }
            bool insideAWall = false;
            for (std::uint32_t index = 0; index < boxCount; ++index) {
                const auto [centre, halfExtent] = boxAt(index);
                if (std::abs(x - centre.x) <= halfExtent.x + body &&
                    std::abs(y - centre.y) <= halfExtent.y + body) {
                    insideAWall = true;
                }
            }
            if (insideAWall) {
                continue;
            }
            ++standable;
            if (std::hypot(x - target.x, y - target.y) >= range) {
                continue;
            }
            bool blocked = false;
            for (std::uint32_t index = 0; index < boxCount; ++index) {
                const auto [centre, halfExtent] = boxAt(index);
                if (kernel::segmentHitsBox({x, y}, target, centre, halfExtent)) {
                    blocked = true;
                }
            }
            if (!blocked) {
                ++visible;
            }
        }
    }
    return standable == 0 ? 0.0F : static_cast<float>(visible) / static_cast<float>(standable);
}

// Below this the plateau the straight-line shaping creates has no perceptual way
// out, which is what 450 generations of Two doors demonstrated.
constexpr float minimumTargetVisibility = 0.12F;

void testTwoDoorsGeometry() {
    namespace kernel = vkexp::worlds::kernel;
    constexpr float radius = 1.84F;
    // The top of the UI's maximum-speed slider, which is the fastest an agent
    // can be configured to move.
    constexpr float maximumConfigurableSpeed = 1.50F;

    const auto insideAnyBox = [&](const float x, const float y, const std::uint32_t blocked) {
        for (kernel::uint index = 0; index < kernel::TwoDoorsBoxCount; ++index) {
            const kernel::vec2 centre = kernel::twoDoorsBoxCentre(index, radius, blocked);
            const kernel::vec2 halfExtent = kernel::twoDoorsBoxHalfExtent(index, radius);
            if (std::abs(x - centre.x) <= halfExtent.x && std::abs(y - centre.y) <= halfExtent.y) {
                return true;
            }
        }
        return false;
    };

    // Whether the world is a task at all comes down to the wall having exactly
    // two gaps in it, so that is checked as a wall rather than as six boxes: the
    // line y = 0 must be solid everywhere except at the two door centres.
    for (const std::uint32_t blocked : {0U, 1U}) {
        const float leftDoor = kernel::twoDoorsDoorCentre(0U, radius);
        const float rightDoor = kernel::twoDoorsDoorCentre(1U, radius);
        check(!insideAnyBox(leftDoor, 0.0F, blocked) && !insideAnyBox(rightDoor, 0.0F, blocked),
              "Both doors are open at the wall");
        check(insideAnyBox(0.0F, 0.0F, blocked) && insideAnyBox(-radius * 0.9F, 0.0F, blocked) &&
                  insideAnyBox(radius * 0.9F, 0.0F, blocked),
              "The wall is solid between and outside the doors");

        // Sweeping the line is what catches a gap the three segments leave by
        // arithmetic rather than by design -- a seam at a segment join would
        // pass every point check above and let every agent through.
        bool solidExceptAtDoors = true;
        for (int sample = -400; sample <= 400; ++sample) {
            const float x = static_cast<float>(sample) / 400.0F * radius;
            const float doorHalf = kernel::TwoDoorsDoorHalfWidth * radius;
            // Just outside the doors, not just inside them: the segments meet
            // the door edges exactly, so a sample on an edge belongs to neither
            // and would report a seam that is not there.
            const bool inADoor = std::abs(x - leftDoor) < doorHalf * 1.01F ||
                                 std::abs(x - rightDoor) < doorHalf * 1.01F;
            if (!inADoor && !insideAnyBox(x, 0.0F, blocked)) {
                solidExceptAtDoors = false;
            }
        }
        check(solidExceptAtDoors, "The wall has no seam between its segments");

        // The dead end has to be a dead end. From inside the pocket the cap and
        // both sides are solid; if any of the three were missing the agent could
        // walk around and the world would be two open doors with extra scenery.
        const float blockedX = kernel::twoDoorsDoorCentre(blocked, radius);
        const float depth = kernel::TwoDoorsPocketDepth * radius;
        const float doorHalf = kernel::TwoDoorsDoorHalfWidth * radius;
        const float side = doorHalf + kernel::TwoDoorsWallHalfThickness;
        // Swept, not sampled at a point: a cap shrunk to zero width still
        // contains its own centre, so a single probe there proves nothing about
        // whether the pocket is closed.
        bool capSolid = true;
        bool sidesSolid = true;
        for (int sample = 0; sample <= 100; ++sample) {
            const float across =
                blockedX + (static_cast<float>(sample) / 50.0F - 1.0F) * doorHalf;
            if (!insideAnyBox(across, depth, blocked)) {
                capSolid = false;
            }
            const float up = static_cast<float>(sample) / 100.0F * depth;
            if (!insideAnyBox(blockedX - side, up, blocked) ||
                !insideAnyBox(blockedX + side, up, blocked)) {
                sidesSolid = false;
            }
        }
        check(capSolid, "The pocket is capped across its full width");
        check(sidesSolid, "The pocket is closed up both sides, from wall to cap");

        // And the other door is genuinely a way through: the whole column above
        // it, to past the depth the pocket reaches, is clear.
        const float openX = kernel::twoDoorsDoorCentre(1U - blocked, radius);
        bool openColumnClear = true;
        for (int sample = 0; sample <= 200; ++sample) {
            const float y = -depth + static_cast<float>(sample) / 200.0F * (depth * 3.0F);
            if (insideAnyBox(openX, y, blocked)) {
                openColumnClear = false;
            }
        }
        check(openColumnClear, "The open door leads all the way through");
    }

    // No barrier may be thin enough to be stepped clean over between two contact
    // tests, or the wall is decoration. The condition is on the box as the
    // contact test sees it -- inflated by the body radius on both sides -- and
    // against the fastest the speed slider goes, not the default: a wall that
    // holds only at default speed is a wall that fails when the experiment is
    // turned up. An earlier version of this check compared the bare half extent
    // against one step, which is far stricter than the physics and would have
    // argued against a wall this thin for no reason.
    const float fastestStep = maximumConfigurableSpeed * vkexp::units::fixedTimeStep;
    bool everyBoxStopsAnAgent = true;
    for (kernel::uint index = 0; index < kernel::TwoDoorsBoxCount; ++index) {
        const kernel::vec2 halfExtent = kernel::twoDoorsBoxHalfExtent(index, radius);
        const float acrossX = 2.0F * (halfExtent.x + vkexp::agentBodyRadius);
        const float acrossY = 2.0F * (halfExtent.y + vkexp::agentBodyRadius);
        if (acrossX <= fastestStep || acrossY <= fastestStep) {
            everyBoxStopsAnAgent = false;
        }
    }
    check(everyBoxStopsAnAgent, "No barrier can be stepped over between two contact tests");

    // What this world was retuned for, and the assertion Two gaps carries too.
    // Fitness shapes on the best straight-line approach, so a wall between the
    // two ends makes the blindest spot the highest-scoring one; the only way out
    // of that plateau is seeing the target, which needs it to be in range at all.
    // At the original 0.72 the ends sat 1.10x the light range apart -- the same
    // ratio at every world size, since the range is a fraction of the arena
    // radius -- so an agent on one end perceived nothing whatever of the other.
    for (const vkexp::WorldSize size :
         {vkexp::WorldSize::Small, vkexp::WorldSize::Medium, vkexp::WorldSize::Large}) {
        vkexp::SimulationStep settings;
        settings.worldRadius = vkexp::worldRadiusForSize(size);
        const kernel::vec2 resource = kernel::twoDoorsResourcePosition(settings.worldRadius);
        const kernel::vec2 home = kernel::twoDoorsHomePosition(settings.worldRadius);
        check(std::hypot(resource.x - home.x, resource.y - home.y) <
                  vkexp::lightRangeForWorld(settings),
              "Each end is inside light range of the other, in every world size");
    }

    // The other half of the retune, and the one that mattered most. Widening the
    // door was a stronger lever on visibility than moving the beacons: from the
    // home side only one opening transmits light, and how much of the far side
    // it lights is what decides whether the plateau can be escaped.
    for (const kernel::uint blocked : {0U, 1U}) {
        const float visible = visibleFractionOfFarSide(
            radius, kernel::twoDoorsHomePosition(radius), kernel::TwoDoorsBoxCount,
            [&](const kernel::uint index) {
                return std::pair{kernel::twoDoorsBoxCentre(index, radius, blocked),
                                 kernel::twoDoorsBoxHalfExtent(index, radius)};
            });
        check(visible > minimumTargetVisibility, "Home is visible from enough of the far side");
    }

    // And the divider has to stay a divider: with a door wider than the wall
    // between the two, the pair reads as one opening with a post in it and there
    // is no choice left for the world to be about.
    const float doorWidth = 2.0F * kernel::TwoDoorsDoorHalfWidth * radius;
    const float divider =
        2.0F * (kernel::TwoDoorsDoorOffset - kernel::TwoDoorsDoorHalfWidth) * radius;
    check(divider > doorWidth, "The wall between the doors is wider than either door");

    // The body radius is written in the shared kernel and in AgentTypes.hpp and
    // neither can reference the other, so this is what keeps them equal.
    check(closeTo(kernel::ScenarioAgentBodyRadius, vkexp::agentBodyRadius),
          "The kernel and the C++ side agree on the body radius");

    // And the wall is sized by the agent rather than by the room, so it stays a
    // divider instead of becoming masonry when the arena grows.
    for (const float arena : {1.84F, 2.76F, 5.52F}) {
        check(closeTo(kernel::twoDoorsBoxHalfExtent(1U, arena).y, vkexp::agentBodyRadius),
              "Wall thickness does not scale with the arena");
    }

    // Occlusion, stated as the world rather than as the slab test: standing at
    // home, the resource is hidden by the wall; standing in a doorway, it is
    // not. If the wall stopped bodies but not light the gradient would pull
    // every agent straight at the one place it cannot go.
    for (const std::uint32_t blocked : {0U, 1U}) {
        const kernel::vec2 home = kernel::twoDoorsHomePosition(radius);
        const kernel::vec2 resource = kernel::twoDoorsResourcePosition(radius);
        const auto blockedBetween = [&](const kernel::vec2 from, const kernel::vec2 to) {
            for (kernel::uint index = 0; index < kernel::TwoDoorsBoxCount; ++index) {
                if (kernel::segmentHitsBox(from, to,
                                           kernel::twoDoorsBoxCentre(index, radius, blocked),
                                           kernel::twoDoorsBoxHalfExtent(index, radius))) {
                    return true;
                }
            }
            return false;
        };
        check(blockedBetween(home, resource), "The wall hides the resource from home");
        const float openX = kernel::twoDoorsDoorCentre(1U - blocked, radius);
        check(!blockedBetween({openX, 0.0F}, resource),
              "From the open doorway the resource is in sight");
        check(!blockedBetween({openX, 0.0F}, home),
              "From the open doorway home is still in sight");
        // Inside the dead end the agent is walled in on light as well as on
        // movement, which is what makes the mistake cost something to discover.
        const float deadEndX = kernel::twoDoorsDoorCentre(blocked, radius);
        const float inside = kernel::TwoDoorsPocketDepth * radius * 0.5F;
        check(blockedBetween({deadEndX, inside}, resource),
              "The dead end hides the resource too");
        check(!blockedBetween(home, {0.0F, kernel::TwoDoorsHomeY * radius * 0.2F}),
              "Nothing occludes a line that never reaches the wall");
    }

    // Both clocks the dead end can run on, and each has to ignore the other's
    // input entirely -- a version that mixed them would still alternate and
    // would still pass a check that only looked at one argument.
    for (const kernel::uint generation : {0U, 1U, 2U, 7U}) {
        check(kernel::twoDoorsBlockedDoor(0U, generation, false) !=
                      kernel::twoDoorsBlockedDoor(1U, generation, false) &&
                  kernel::twoDoorsBlockedDoor(0U, generation, false) ==
                      kernel::twoDoorsBlockedDoor(2U, generation, false),
              "By trial, the dead end swaps every trial whatever the generation");
    }
    for (const kernel::uint trial : {0U, 1U, 2U, 3U}) {
        check(kernel::twoDoorsBlockedDoor(trial, 0U, true) !=
                      kernel::twoDoorsBlockedDoor(trial, 1U, true) &&
                  kernel::twoDoorsBlockedDoor(trial, 0U, true) ==
                      kernel::twoDoorsBlockedDoor(trial, 2U, true),
              "By generation, the dead end swaps every generation whatever the trial");
    }
    // The whole point of the option: by generation, every trial in a generation
    // meets the same layout, so a population is scored on one door at a time.
    check(kernel::twoDoorsBlockedDoor(0U, 3U, true) == kernel::twoDoorsBlockedDoor(1U, 3U, true) &&
              kernel::twoDoorsBlockedDoor(1U, 3U, true) ==
                  kernel::twoDoorsBlockedDoor(2U, 3U, true),
          "By generation, one generation is one layout for every trial");

    // Push-out separates and does not teleport: an agent overlapping a wall ends
    // up outside it, on the side it came from.
    const kernel::vec2 centre = kernel::twoDoorsBoxCentre(1U, radius, 0U);
    const kernel::vec2 halfExtent = kernel::twoDoorsBoxHalfExtent(1U, radius);
    const float body = vkexp::agentBodyRadius;
    const kernel::vec2 fromBelow =
        kernel::boxPushOut({0.0F, centre.y - halfExtent.y - body * 0.5F}, body, centre, halfExtent);
    check(fromBelow.y < 0.0F && closeTo(fromBelow.x, 0.0F),
          "A wall pushes an agent back the way it came");
    check(closeTo(kernel::boxPushOut({0.0F, centre.y - halfExtent.y - body * 2.0F}, body, centre,
                                     halfExtent)
                      .y,
                  0.0F),
          "A clear agent is not pushed at all");
    const float resolved = centre.y - halfExtent.y - body * 0.5F + fromBelow.y;
    check(std::abs(resolved - centre.y) >= halfExtent.y + body - 1.0e-5F,
          "The push-out fully separates the circle from the box");
}

void testShuttleGeometry() {
    namespace kernel = vkexp::worlds::kernel;
    constexpr float radius = 1.84F;
    constexpr float maximumConfigurableSpeed = 1.50F;

    const kernel::vec2 centre = kernel::shuttleBoxCentre();
    const kernel::vec2 halfExtent = kernel::shuttleBoxHalfExtent(radius);
    const kernel::vec2 resource = kernel::shuttleResourcePosition(radius);
    const kernel::vec2 home = kernel::shuttleHomePosition(radius);

    // The whole point of the wall: the straight line between the two beacons is
    // closed, so going directly is not an option and the shortest route is
    // around an end.
    check(kernel::segmentHitsBox(home, resource, centre, halfExtent),
          "The wall closes the straight line between the beacons");

    // And it is a detour, not a maze: the arena is not divided, so rounding
    // either end gets there. The route is two legs via a turning point past the
    // end -- a straight line to that point still crosses the wall near the
    // middle, which is what makes this a detour worth taking rather than a
    // slightly angled approach.
    const float pastEnd = halfExtent.x + vkexp::agentBodyRadius * 2.0F;
    for (const float side : {-1.0F, 1.0F}) {
        const kernel::vec2 turn{side * pastEnd, 0.0F};
        check(!kernel::segmentHitsBox(home, turn, centre, halfExtent) &&
                  !kernel::segmentHitsBox(turn, resource, centre, halfExtent),
              "Rounding either end of the wall gets there");
    }
    check(halfExtent.x < radius, "The wall is shorter than the arena is wide");

    // Sized by the agent across and by the arena along: how far the detour is
    // should scale with the room, how solid the wall is should not.
    for (const float arena : {1.84F, 2.76F, 5.52F}) {
        check(closeTo(kernel::shuttleBoxHalfExtent(arena).y, vkexp::agentBodyRadius),
              "Wall thickness does not scale with the arena");
        check(kernel::shuttleBoxHalfExtent(arena).x > kernel::shuttleBoxHalfExtent(1.0F).x,
              "Wall length does scale with the arena");
    }

    const float fastestStep = maximumConfigurableSpeed * vkexp::units::fixedTimeStep;
    check(2.0F * (halfExtent.y + vkexp::agentBodyRadius) > fastestStep,
          "The wall cannot be stepped over between two contact tests");

    // A round trip has to fit the default trial more than once, or "keep
    // shuttling until time runs out" is a single trip with a wait at the end.
    // Measured along the two-leg detour round the end, at the speed limit.
    const float oneWay = 2.0F * std::hypot(pastEnd, resource.y);
    const float roundTripSeconds = 2.0F * oneWay / vkexp::SimulationStep{}.maximumSpeed;
    const float trialSeconds =
        vkexp::units::secondsForSteps(vkexp::SimulationControls{}.stepsPerGeneration,
                                      vkexp::units::fixedTimeStep);
    check(roundTripSeconds * 2.0F < trialSeconds,
          "The default trial has room for at least two round trips");
}

void testTwoGapsGeometry() {
    namespace kernel = vkexp::worlds::kernel;
    constexpr float radius = 1.84F;
    constexpr float maximumConfigurableSpeed = 1.50F;
    const vkexp::SimulationStep defaults;

    const kernel::vec2 resource = kernel::twoGapsResourcePosition(radius, false);
    const kernel::vec2 home = kernel::twoGapsHomePosition(radius, false);

    const auto crossesWall = [&](const kernel::vec2 from, const kernel::vec2 to) {
        for (std::uint32_t index = 0; index < kernel::TwoGapsBoxCount; ++index) {
            if (kernel::segmentHitsBox(from, to, kernel::twoGapsBoxCentre(index, radius),
                                       kernel::twoGapsBoxHalfExtent(index, radius))) {
                return true;
            }
        }
        return false;
    };

    // The wall divides the arena: the straight line between the ends is closed,
    // and so is every line that does not pass through a gap. A point probe would
    // pass a seam between two boxes, so this sweeps the whole span instead.
    check(crossesWall(home, resource), "The wall closes the straight line between the ends");
    const float gap = kernel::TwoGapsGapOffset * radius;
    const float gapHalf = kernel::TwoGapsGapHalfWidth * radius;
    constexpr int samples = 400;
    int openColumns = 0;
    for (int step = 0; step <= samples; ++step) {
        const float x = -radius + 2.0F * radius * static_cast<float>(step) /
                                      static_cast<float>(samples);
        const bool open = !crossesWall({x, -radius}, {x, radius});
        if (open) {
            ++openColumns;
            check(std::abs(std::abs(x) - gap) <= gapHalf,
                  "The wall is open only inside one of the two gaps");
        }
    }
    check(openColumns > 0, "Both gaps are actually open");

    // Neither gap is a dead end -- that is what separates this world from Two
    // doors. Through either one and on to the far end, in two legs.
    for (const float side : {-1.0F, 1.0F}) {
        const kernel::vec2 mouth{side * gap, 0.0F};
        check(!crossesWall(home, mouth) && !crossesWall(mouth, resource),
              "Either gap leads all the way through");
    }

    // Wide enough to steer through rather than to squeeze through, and thin
    // enough not to be stepped over between two contact tests at the top of the
    // speed slider.
    check(gapHalf > vkexp::agentBodyRadius * 2.0F, "A gap is wider than the body that uses it");
    const float fastestStep = maximumConfigurableSpeed * vkexp::units::fixedTimeStep;
    check(2.0F * (kernel::TwoGapsWallHalfThickness + vkexp::agentBodyRadius) > fastestStep,
          "The wall cannot be stepped over between two contact tests");
    for (const float arena : {1.84F, 2.76F, 5.52F}) {
        check(closeTo(kernel::twoGapsBoxHalfExtent(1u, arena).y, vkexp::agentBodyRadius),
              "Wall thickness does not scale with the arena");
    }

    // The number the geometry was actually chosen for. Fitness shapes on the
    // best straight-line approach, so the wall makes a plateau; the way out of
    // it is seeing the far end, which is only possible if it is in range at all.
    // Two doors puts the ends 1.10x the light range apart -- at every world size,
    // since the range is a fraction of the arena radius -- so an agent standing
    // on one end perceives nothing whatever of the other, and no amount of
    // generations turns that into a gradient. This asserts the fix.
    const float separation = std::hypot(resource.x - home.x, resource.y - home.y);
    for (const vkexp::WorldSize size :
         {vkexp::WorldSize::Small, vkexp::WorldSize::Medium, vkexp::WorldSize::Large}) {
        vkexp::SimulationStep settings = defaults;
        settings.worldRadius = vkexp::worldRadiusForSize(size);
        const float range = vkexp::lightRangeForWorld(settings);
        const kernel::vec2 far = kernel::twoGapsResourcePosition(settings.worldRadius, false);
        const kernel::vec2 near = kernel::twoGapsHomePosition(settings.worldRadius, false);
        check(std::hypot(far.x - near.x, far.y - near.y) < range,
              "Each end is inside light range of the other, in every world size");
    }
    check(separation < vkexp::lightRangeForWorld(defaults), "The ends are mutually visible");

    // In range is necessary and not sufficient: the wall still has to leave room
    // to pick the signal up. The same floor the retuned Two doors is held to.
    const float visible = visibleFractionOfFarSide(
        radius, kernel::twoGapsHomePosition(radius, false), kernel::TwoGapsBoxCount,
        [&](const kernel::uint index) {
            return std::pair{kernel::twoGapsBoxCentre(index, radius),
                             kernel::twoGapsBoxHalfExtent(index, radius)};
        });
    check(visible > minimumTargetVisibility, "Home is visible from enough of the far side");

    // The swap. Off, it never fires whatever the generation; on, it alternates,
    // and it exchanges the two ends rather than moving either one somewhere new.
    for (std::uint32_t generation = 0; generation < 4; ++generation) {
        check(!kernel::twoGapsEndsSwapped(generation, false), "Off, the ends never trade places");
        check(kernel::twoGapsEndsSwapped(generation, true) == (generation % 2 == 1),
              "On, the ends trade places on odd generations");
    }
    const kernel::vec2 swappedResource = kernel::twoGapsResourcePosition(radius, true);
    const kernel::vec2 swappedHome = kernel::twoGapsHomePosition(radius, true);
    check(closeTo(swappedResource.y, home.y) && closeTo(swappedHome.y, resource.y),
          "Swapping exchanges the two ends");
    // The wall is the same wall either way, so a swapped generation is the same
    // world seen the other way up and not a second geometry to get right.
    for (std::uint32_t index = 0; index < kernel::TwoGapsBoxCount; ++index) {
        const kernel::vec2 centre = kernel::twoGapsBoxCentre(index, radius);
        check(closeTo(centre.y, 0.0F), "Every wall segment sits on the axis the ends swap across");
    }
}

void testBeaconColorAblation() {
    vkexp::AgentState agent{};
    agent.pose = {0.0F, 0.0F, 0.0F, vkexp::agentBodyRadius};

    vkexp::SimulationStep settings;
    settings.beaconScenario = vkexp::BeaconScenario::TwoGaps;
    check(!settings.uniformBeaconColor, "The colour cue is present unless it is ablated");

    const vkexp::ActiveBeacons lit = vkexp::activeBeacons(agent, settings);
    check(lit.count == 2, "Two gaps has two ends");
    const vkexp::Float4 first = lit.values[0].color;
    const vkexp::Float4 second = lit.values[1].color;
    check(!closeTo(first.x, second.x) || !closeTo(first.y, second.y) ||
              !closeTo(first.z, second.z),
          "The two ends are told apart by colour to begin with");

    settings.uniformBeaconColor = true;
    const vkexp::ActiveBeacons ablated = vkexp::activeBeacons(agent, settings);
    check(closeTo(ablated.values[0].color.x, ablated.values[1].color.x) &&
              closeTo(ablated.values[0].color.y, ablated.values[1].color.y) &&
              closeTo(ablated.values[0].color.z, ablated.values[1].color.z),
          "Ablated, the two ends are the same colour");

    // The point of averaging rather than copying one colour onto the other: the
    // information goes and the light stays. If the ends emitted more or less
    // light than before, a control run would be answering two questions at once.
    check(closeTo(ablated.values[0].color.x + ablated.values[1].color.x, first.x + second.x) &&
              closeTo(ablated.values[0].color.y + ablated.values[1].color.y, first.y + second.y) &&
              closeTo(ablated.values[0].color.z + ablated.values[1].color.z, first.z + second.z),
          "Ablating the cue leaves the total emitted light unchanged");

    // Positions are not a colour cue and must survive untouched, or the control
    // would be moving the world as well as recolouring it.
    check(closeTo(ablated.values[0].position.y, lit.values[0].position.y) &&
              closeTo(ablated.values[1].position.y, lit.values[1].position.y),
          "Ablating the cue moves neither end");

    // And it reaches the receptors, which is the only place it matters. Standing
    // on the wall itself would prove nothing -- both ends are occluded from
    // there and every channel reads zero either way -- so this stands clear of
    // it on the resource side, facing the resource.
    vkexp::SimulationStep sensing = settings;
    sensing.uniformBeaconColor = false;
    sensing.neuronModel = vkexp::NeuronModel::Reactive;
    agent.pose.y = settings.worldRadius * 0.25F;
    agent.pose.z = 1.5707963F; // +y, straight at the resource
    const vkexp::neuro::Inputs plain = vkexp::sampleAgentInputs(agent, sensing);
    sensing.uniformBeaconColor = true;
    const vkexp::neuro::Inputs ablatedInputs = vkexp::sampleAgentInputs(agent, sensing);
    bool anyChannelMoved = false;
    for (std::size_t index = 0; index < plain.size(); ++index) {
        if (!closeTo(plain[index], ablatedInputs[index])) {
            anyChannelMoved = true;
        }
    }
    check(anyChannelMoved, "The ablation reaches the receptors, not only the beacon record");
}

// The rule a delivery world lives or dies by: a beacon may not be scored again
// until the opposite one has been reached. Without it the cheapest policy is to
// sit on the resource and collect the pickup reward every step, and no amount of
// world design would matter.
//
// It is worth a test rather than a reading of the code, because the guard is not
// where it looks like it is. Nothing counts arrivals or remembers which beacon
// was last touched: the early return compares against the distance to the
// *current* target, and picking up switches that target to the far end, so the
// second visit is not inside any arrival radius to begin with. That is a subtle
// thing to preserve by accident.
void testDeliveryCannotBeScoredTwice() {
    vkexp::SimulationStep settings;
    settings.beaconScenario = vkexp::BeaconScenario::TwoGaps;

    vkexp::AgentState agent{};
    agent.pose.w = vkexp::agentBodyRadius;
    const vkexp::ActiveBeacons ends = vkexp::activeBeacons(agent, settings);
    const vkexp::Float4 resource = ends.values[0].position;
    const vkexp::Float4 home = ends.values[1].position;

    const auto standAt = [&](const vkexp::Float4 place, const int steps) {
        agent.pose.x = place.x;
        agent.pose.y = place.y;
        for (int step = 0; step < steps; ++step) {
            vkexp::worlds::deliveryCycleAfterStep(agent, settings,
                                                  vkexp::nearestBeaconDistance(agent, settings));
        }
    };

    agent.metrics.x = vkexp::nearestBeaconDistance(agent, settings);
    agent.metrics.y = agent.metrics.x;

    // Loitering on the resource: one pickup, and then nothing however long it
    // stays. The second step is the one that matters -- it is already carrying,
    // so the target has moved to the far end and the arrival test cannot fire.
    standAt(resource, 1);
    const float afterFirstPickup = agent.metrics.w;
    check(agent.internal.y >= 0.5F, "Reaching the resource picks up");
    check(vkexp::completedForageCycles(agent) == 0, "A pickup is not a completed cycle");
    standAt(resource, 200);
    check(closeTo(agent.metrics.w, afterFirstPickup),
          "Sitting on the resource scores exactly once, not once per step");

    // Leaving and coming back is no different: still carrying, still nothing.
    standAt({resource.x, resource.y * 0.4F, 0.0F, 0.0F}, 1);
    standAt(resource, 1);
    check(closeTo(agent.metrics.w, afterFirstPickup),
          "Returning to the resource while carrying scores nothing");

    // Only the opposite end releases the cycle, and it too counts once.
    standAt(home, 1);
    check(vkexp::completedForageCycles(agent) == 1, "Reaching home completes one cycle");
    check(agent.internal.y < 0.5F, "Delivering drops the cargo");
    const float afterFirstDelivery = agent.metrics.w;
    standAt(home, 200);
    check(closeTo(agent.metrics.w, afterFirstDelivery) &&
              vkexp::completedForageCycles(agent) == 1,
          "Sitting on home delivers exactly once, not once per step");

    // And the next cycle is allowed, or the world would be one trip long.
    standAt(resource, 1);
    check(agent.internal.y >= 0.5F && agent.metrics.w > afterFirstDelivery,
          "The resource is available again after a delivery");
}

void testPuckWorld() {
    namespace puck = vkexp::puck::kernel;
    const vkexp::ScenarioDefinition& scenario =
        vkexp::scenarioDefinition(vkexp::BeaconScenario::PuckPush);
    vkexp::SimulationStep settings;
    settings.beaconScenario = vkexp::BeaconScenario::PuckPush;
    const float radius = settings.worldRadius;
    const float puckSize = puck::puckRadius(radius, settings.puckRadiusRatio);

    // Which side the puck starts on alternates by trial, so one genome meets
    // both and pushing always the same way cannot stand in for perceiving where
    // the puck is.
    check(puck::puckStartSide(0) == -puck::puckStartSide(1) &&
              puck::puckStartSide(0) == puck::puckStartSide(2),
          "The puck starts on alternating sides by trial");
    for (const std::uint32_t trial : {0U, 1U, 2U, 3U}) {
        const auto start = puck::puckStartPosition(radius, trial);
        check(closeTo(start.x, 0.0F), "The puck starts on the arena's axis");
        check(std::abs(start.y) < radius - puckSize,
              "The puck starts inside the arena, clear of the rim");
    }

    const float target = puck::puckTargetRadius(radius, settings.puckTargetRadiusRatio);
    check(target > puckSize * 2.0F, "The target disc is wider than the puck");
    check(puckSize > vkexp::agentBodyRadius * 2.0F,
          "The puck is wider than the agents pushing it, so a group can share its contact arc");

    // The puck has to be perceivable, not merely present. It was not at first:
    // the receptors see beacons and other agents' light and nothing else, so a
    // puck that was neither could only be discovered by walking into it, and a
    // fitness that paid for approaching it was paying for something no agent had
    // a sense of. It is a beacon now, at the position every agent mirrors.
    vkexp::AgentState lookout{};
    lookout.penalties.y = 0.4F;
    lookout.penalties.z = -0.3F;
    const vkexp::ActiveBeacons lit = scenario.beacons(lookout, settings);
    check(lit.count == 2, "The puck world lights both the middle and the puck");
    bool puckIsLit = false;
    for (std::size_t index = 0; index < lit.count; ++index) {
        if (closeTo(lit.values[index].position.x, lookout.penalties.y) &&
            closeTo(lit.values[index].position.y, lookout.penalties.z)) {
            puckIsLit = true;
        }
    }
    check(puckIsLit, "One of the lights is the puck, wherever the puck is");
    check(target < radius * 0.5F, "The target disc is a target and not most of the arena");
    check(puck::puckInsideTarget({0.0F, 0.0F}, target), "The middle is inside the target");
    check(!puck::puckInsideTarget({target * 1.01F, 0.0F}, target),
          "Just outside the target is outside it");
    // The ladder. It replaced two named goals -- cross the middle line, then
    // reach the disc -- which the geometry would not put in that order: the disc
    // is centred on the line and the puck comes from outside, so it enters the
    // disc after 58 per cent of the journey and reaches the line only at the end.
    // The minimum was the harder of the two and never fired first, so a world
    // scored nothing until it scored everything, and the reported curve could
    // only move in whole worlds. These assertions are what the ladder replaced it
    // with: rungs on one journey, ordered by construction.
    const float startDistance =
        std::hypot(puck::puckStartPosition(radius, 0).x, puck::puckStartPosition(radius, 0).y);
    const auto rungAt = [&](const float distance) {
        return puck::puckLevelForJourney(puck::puckJourneyFraction(distance, startDistance, target));
    };
    check(rungAt(startDistance) == puck::PuckLevelNone, "A puck that has not moved is on no rung");
    check(rungAt(target) == puck::PuckLevelCount, "A puck inside the disc is on the top rung");
    check(rungAt(target * 0.5F) == puck::PuckLevelCount,
          "And it stays on the top rung deeper inside, rather than climbing past it");
    check(rungAt(startDistance * 1.5F) == puck::PuckLevelNone,
          "A puck shoved backwards reports no rung rather than a negative one");

    // Strictly ordered and strictly reachable: every rung needs the puck closer
    // than the one below it, and no rung is skipped on the way in. This is the
    // property the two named goals did not have.
    float previous = startDistance;
    for (std::uint32_t rung = 1; rung <= puck::PuckLevelCount; ++rung) {
        const float span = startDistance - target;
        const float reached =
            target + span * (1.0F - static_cast<float>(rung) / static_cast<float>(
                                                              puck::PuckLevelCount));
        check(reached < previous, "Each rung asks the puck to come further in than the last");
        check(rungAt(reached) == rung, "Reaching a rung's distance reports exactly that rung");
        check(rungAt(reached + span * 0.01F) == rung - 1,
              "And a hair short of it reports the rung below");
        previous = reached;
    }

    // The ladder follows the target slider rather than a number written beside
    // it: widen the disc and the same puck is further along its journey.
    const float wideTarget = puck::puckTargetRadius(radius, settings.puckTargetRadiusRatio * 1.5F);
    check(puck::puckJourneyFraction(startDistance * 0.6F, startDistance, wideTarget) >
              puck::puckJourneyFraction(startDistance * 0.6F, startDistance, target),
          "A wider target disc makes the same position further along the journey");

    // The top rung and the disc are the same statement, which is what keeps the
    // maximum the world was specified with intact.
    check(puck::puckInsideTarget({target * 0.99F, 0.0F}, target) &&
              rungAt(target * 0.99F) == puck::PuckLevelCount,
          "Inside the disc and on the top rung are the same claim");
    check(!puck::puckInsideTarget({target * 1.05F, 0.0F}, target) &&
              rungAt(target * 1.05F) < puck::PuckLevelCount,
          "And outside it is neither");

    // The shaping opens against the puck's own starting distance. This is the
    // assertion that catches a spawn which forgets to seed the mirror: with the
    // mirror at zero the trial opens believing the puck is already in the
    // middle, every genome banks the same nothing, and the world scores as a
    // hard task rather than as a broken one. No device test sees it, because the
    // puck still moves exactly as before.
    check(scenario.spawn != nullptr && scenario.targetDistance != nullptr,
          "The puck world places its own agents and measures its own target");
    for (const std::uint32_t trial : {0U, 1U}) {
        vkexp::AgentState agent{};
        agent.pose = {0.3F, 0.2F, 0.0F, vkexp::agentBodyRadius};
        agent.target.z = static_cast<float>(trial);
        scenario.spawn(agent, settings);
        const auto start = puck::puckStartPosition(radius, trial);
        check(closeTo(scenario.targetDistance(agent, settings), std::hypot(start.x, start.y)),
              "A trial opens measuring the puck's distance to the middle, not zero");
        // And the agents are put on the puck's side, so the first thing they
        // have to do is reach it rather than already be behind it.
        check(agent.pose.y * puck::puckStartSide(trial) > 0.0F,
              "Agents spawn on the side the puck starts on");
    }

    // Scoring reads the level the puck pass latched, and is capped at the two
    // levels the world has: a number above that would report more than complete.
    for (float level = 0.0F; level <= 4.0F; level += 1.0F) {
        vkexp::AgentState agent{};
        agent.target.w = level;
        check(scenario.achievedObjectives(agent) ==
                  std::min(static_cast<std::uint32_t>(level), puck::PuckLevelCount),
              "Objectives are the puck's level, capped at the levels that exist");
    }
    check(scenario.objectivesPerAgent == puck::PuckLevelCount,
          "The reported ratio is read against both levels");

    // The journey has to outweigh loitering *on its own*, before any objective
    // completes. This is the other half of why nothing was learned at first, and
    // the arithmetic is worth stating exactly, because the obvious version of
    // the claim is wrong: raw progress in metres did beat loitering once the
    // objective bonus landed, 8.85 against 3.75.
    //
    // What it did not do was leave a gradient to get there. The whole journey to
    // the middle is 1.1 m, so moving the puck a hand's width was worth 0.1
    // against the 3.75 an agent collects by parking beside it for the trial --
    // under three per cent. Evolution improves by increments, and there was no
    // increment: only the completion, which nothing was going to stumble into.
    // Normalising the journey to a fraction and weighting it puts that same
    // push at 29 per cent instead.
    //
    // So the assertion is on the progress term alone, with the objective bonus
    // deliberately withheld from the deliverer, and the loiterer given the same
    // proximity reward it could not really have earned while moving. Both make
    // the check pessimistic.
    const float trialSeconds =
        vkexp::units::secondsForSteps(vkexp::SimulationControls{}.stepsPerGeneration,
                                      vkexp::units::fixedTimeStep);
    const float parked =
        trialSeconds * settings.fitness.trackingReward * puck::PuckProximityShare;

    vkexp::AgentState loiterer{};
    loiterer.metrics = {startDistance, startDistance, 0.0F, parked};
    loiterer.target.w = 0.0F;

    vkexp::AgentState deliverer{};
    deliverer.metrics = {startDistance, 0.0F, 0.0F, parked};
    deliverer.target.w = 0.0F; // no objective bonus: the journey has to carry it

    const float loiteringScore = scenario.fitness(loiterer, settings.fitness);
    const float deliveringScore = scenario.fitness(deliverer, settings.fitness);
    check(deliveringScore > loiteringScore * 2.0F,
          "Moving the puck home outweighs parking beside it before any objective lands");

    // And the increment is what matters, not the endpoint: a small push has to
    // be worth a real fraction of what standing still pays, or there is no path
    // from one behaviour to the other for selection to walk.
    vkexp::AgentState nudged{};
    nudged.metrics = {startDistance, startDistance * 0.9F, 0.0F, parked};
    const float nudge = scenario.fitness(nudged, settings.fitness) - loiteringScore;
    check(nudge > parked * 0.2F,
          "A tenth of the journey is worth a fifth of a whole trial's loitering");
    check(scenario.puck, "The puck world says it has a puck, which is what runs the puck pass");
}

// Who gets paid for moving the puck, which is the one thing in this world that
// is not shared. Every other term in the score is derived from the puck's
// position, so it is the same number for every agent in the world -- and with
// only those terms a population settled on leaning against the near face of the
// puck and blocking it, which collected the proximity reward and cost nothing.
void testPuckPushCredit() {
    namespace puck = vkexp::puck::kernel;
    vkexp::SimulationStep settings;
    settings.beaconScenario = vkexp::BeaconScenario::PuckPush;
    const float radius = puck::puckRadius(settings.worldRadius, settings.puckRadiusRatio);
    const float contact = radius + vkexp::agentBodyRadius;
    // A puck on the axis, above the middle, so "toward the middle" is straight
    // down and the two sides of it are unambiguous.
    const puck::vec2 puckAt{0.0F, 0.6F};

    // Behind it, driving straight at the middle: the whole press is useful.
    const puck::vec2 down{0.0F, -1.0F};
    const puck::vec2 up{0.0F, 1.0F};
    const float behind = puck::puckPushContribution({0.0F, puckAt.y + contact}, down, 1.0F,
                                                    vkexp::agentBodyRadius, puckAt, radius);
    check(closeTo(behind, 1.0F), "An agent driving straight toward the middle presses a whole one");

    // The reason the push is a pressure and not an approach speed. This agent is
    // standing still, wedged and going nowhere, and it presses exactly as hard as
    // one at a run -- which is what lets a crowd behave like tugboats instead of
    // being outdone by a single battering ram. Under the old model it counted for
    // nothing at all.
    check(closeTo(puck::puckPushContribution({0.0F, puckAt.y + contact}, down, 1.0F,
                                             vkexp::agentBodyRadius, puckAt, radius),
                  behind),
          "A motionless agent leaning at full drive presses as hard as a moving one");
    check(closeTo(puck::puckPushContribution({0.0F, puckAt.y + contact}, down, 0.0F,
                                             vkexp::agentBodyRadius, puckAt, radius),
                  0.0F),
          "And one with its motors off presses nothing, however close it stands");
    check(puck::puckPushContribution({0.0F, puckAt.y + contact}, down, 0.4F,
                                     vkexp::agentBodyRadius, puckAt, radius) < behind,
          "Half throttle presses less than full");

    // In the way, driving with exactly the same effort. It is in contact, it is
    // pressing, and it moves the puck the wrong way -- so it earns nothing.
    // Nothing here names a correct side; the projection does the work.
    //
    // Zero and not negative: blocking stops being paid for, it does not become
    // a thing to avoid. An agent taught to keep clear of the puck is worse than
    // one that leans on it.
    const float blocking = puck::puckPushContribution({0.0F, puckAt.y - contact}, up, 1.0F,
                                                      vkexp::agentBodyRadius, puckAt, radius);
    check(closeTo(blocking, 0.0F), "An agent wedged between the puck and the middle earns nothing");

    // Sideways: in contact and pressing, but the push is perpendicular to the
    // journey, so it is worth nothing without being wrong.
    const float sideways = puck::puckPushContribution({contact, puckAt.y}, {-1.0F, 0.0F}, 1.0F,
                                                      vkexp::agentBodyRadius, puckAt, radius);
    check(closeTo(sideways, 0.0F), "A push across the puck's path is worth nothing");

    // Half a turn off the line: paid, but less. This is the part that makes it a
    // gradient rather than a switch -- getting further round the puck pays more.
    const float diagonal = puck::puckPushContribution(
        {contact * 0.7071F, puckAt.y + contact * 0.7071F}, {-0.7071F, -0.7071F}, 1.0F,
        vkexp::agentBodyRadius, puckAt, radius);
    check(diagonal > 0.0F && diagonal < behind,
          "Pushing at an angle pays, and pays less than pushing straight");

    // Facing away, and near but not touching: neither is work.
    check(closeTo(puck::puckPushContribution({0.0F, puckAt.y + contact}, up, 1.0F,
                                             vkexp::agentBodyRadius, puckAt, radius),
                  0.0F),
          "An agent with its back to the puck is not pushing it");
    check(closeTo(puck::puckPushContribution({0.0F, puckAt.y + contact * 3.0F}, down, 1.0F,
                                             vkexp::agentBodyRadius, puckAt, radius),
                  0.0F),
          "An agent that has not reached the puck is not moving it");

    // And the balance: a delivery's worth of pushing has to beat a whole trial
    // of leaning on the puck, or the behaviour that is cheaper still wins. Two
    // agents press the puck along at force over drag, and each is credited its
    // own press for as long as the journey takes.
    const float trialSeconds =
        vkexp::units::secondsForSteps(vkexp::SimulationControls{}.stepsPerGeneration,
                                      vkexp::units::fixedTimeStep);
    const float parked = trialSeconds * settings.fitness.trackingReward * puck::PuckProximityShare;
    const float pairSpeed = 2.0F * puck::PuckPushAcceleration / puck::PuckDrag;
    const float journey = std::hypot(puck::puckStartPosition(settings.worldRadius, 0).x,
                                     puck::puckStartPosition(settings.worldRadius, 0).y);
    const float pushed = (journey / pairSpeed) * puck::PuckWorkReward;
    check(pushed > parked * 2.0F, "Pushing the puck home outearns a whole trial of leaning on it");

    // The equilibrium the acceleration was chosen against, which is the whole
    // shape of the world: one agent moves it slowly, a pair twice as fast, and
    // three reach the agents' own speed limit, at which point pushing harder
    // stops helping because the puck cannot outrun the things pushing it.
    const float soloSpeed = puck::PuckPushAcceleration / puck::PuckDrag;
    check(soloSpeed > 0.05F && soloSpeed < settings.maximumSpeed * 0.5F,
          "One agent alone moves the puck, and slowly");
    check(closeTo(pairSpeed, soloSpeed * 2.0F), "Two press it along twice as fast");
    check(3.0F * soloSpeed >= settings.maximumSpeed,
          "And three reach the speed the puck is capped at, so more is no longer better");

    // The friction floor, which is what turns the world from one that permits a
    // group into one that requires it. Without it a single agent moves the puck
    // on its own, so cooperation is a convenience and the question the world
    // exists to ask -- can selection produce agents that push together -- is one
    // it never puts. Counted in agents, and compared against a pressure that is
    // also counted in agents, so the slider means exactly what it says.
    const float breakaway = puck::puckBreakawayPush(settings.puckBreakawayPushes);
    check(breakaway > 1.0F, "By default one agent pressing at full drive cannot start the puck");
    check(closeTo(puck::puckFrictionFraction(1.0F, breakaway), 0.0F),
          "So its whole press is absorbed");
    check(puck::puckFrictionFraction(2.0F, breakaway) > 0.0F,
          "And two pressing the same way get through");

    // Presses are summed as vectors before the floor is measured, so two agents
    // on opposite faces cancel and move nothing however hard they try. This is
    // what makes "two agents" mean two agents pushing the same way rather than
    // two agents touching.
    check(closeTo(puck::puckFrictionFraction(0.0F, breakaway), 0.0F),
          "Two agents on opposite faces cancel before the floor is measured");

    // Subtracted, not switched: a pair that barely clears the floor moves the
    // puck slowly rather than the world flipping between nothing and everything.
    // Selection needs an increment here for the same reason the journey is a
    // fraction rather than a completion.
    const float justOver = puck::puckFrictionFraction(breakaway * 1.02F, breakaway);
    const float wellOver = puck::puckFrictionFraction(breakaway * 4.0F, breakaway);
    check(justOver > 0.0F && justOver < 0.1F, "Just over the floor almost nothing gets through");
    check(wellOver > justOver && wellOver < 1.0F, "And more press gets more through, never all");

    // At zero the floor is gone and the world is the one it was before, which is
    // what makes this a knob rather than a change of task.
    check(closeTo(puck::puckBreakawayPush(0.0F), 0.0F) &&
              closeTo(puck::puckFrictionFraction(1.0F, 0.0F), 1.0F),
          "At a breakaway of zero one agent moves the puck exactly as before");

    // And the work reward has to follow the puck rather than the pushing, or a
    // lone agent leaning on a puck it cannot start collects all trial for moving
    // nothing -- teaching the futile pushing the floor exists to rule out.
    vkexp::AgentState pusher{};
    // Facing the puck, at full throttle, standing still: the case the pressure
    // model exists for.
    pusher.pose = {0.0F, puckAt.y + contact, -std::numbers::pi_v<float> / 2.0F,
                   vkexp::agentBodyRadius};
    pusher.internal.x = 1.0F;
    pusher.penalties.y = puckAt.x;
    pusher.penalties.z = puckAt.y;
    pusher.target = {0.0F, 0.0F, 0.0F, 0.0F}; // the puck is stuck
    vkexp::worlds::rewardPuckWork(pusher, settings);
    check(closeTo(pusher.metrics.w, 0.0F), "Pushing a puck that is not moving earns nothing");
    pusher.target.y = -settings.maximumSpeed * puck::PuckWorkMovingSpeed;
    vkexp::worlds::rewardPuckWork(pusher, settings);
    check(pusher.metrics.w > 0.0F, "And pushing one that is under way earns the work");

    // Where the puck is placed. Off, the axis, agents on its side. On, scattered
    // through a ring, and the point is that no one layout can be memorised: the
    // same world gets a different puck every generation, and different worlds get
    // different pucks within one.
    const float arena = settings.worldRadius;
    const std::uint32_t seed = 7U;
    const puck::vec2 axis = puck::puckStartPositionFor(arena, 0U, 3U, seed, false);
    check(closeTo(axis.x, puck::puckStartPosition(arena, 0U).x) &&
              closeTo(axis.y, puck::puckStartPosition(arena, 0U).y),
          "With the option off the puck is placed exactly where it always was");

    bool differsBetweenWorlds = false;
    bool differsBetweenGenerations = false;
    for (std::uint32_t world = 0; world < 32U; ++world) {
        const puck::vec2 here = puck::puckStartPositionFor(arena, 0U, world, seed, true);
        const float span = std::hypot(here.x, here.y);
        check(span > puck::PuckScatterInner * arena * 0.999F &&
                  span < puck::PuckScatterOuter * arena * 1.001F,
              "A scattered puck lands in the ring, clear of both the rim and the middle");
        const puck::vec2 neighbour = puck::puckStartPositionFor(arena, 0U, world + 1U, seed, true);
        const puck::vec2 later = puck::puckStartPositionFor(arena, 0U, world, seed + 1U, true);
        if (!closeTo(here.x, neighbour.x) || !closeTo(here.y, neighbour.y)) {
            differsBetweenWorlds = true;
        }
        if (!closeTo(here.x, later.x) || !closeTo(here.y, later.y)) {
            differsBetweenGenerations = true;
        }
        const puck::vec2 again = puck::puckStartPositionFor(arena, 0U, world, seed, true);
        check(closeTo(here.x, again.x) && closeTo(here.y, again.y),
              "And it is the same puck every time the same generation is asked for");
    }
    check(differsBetweenWorlds, "Worlds in one generation get different pucks");
    check(differsBetweenGenerations, "And one world gets a different puck the next generation");

    // The whole point of the option is that the shaping opens against wherever
    // the puck actually is. A spawn that placed the agents from one answer and
    // the driver that placed the puck from another would score a journey that
    // was never travelled -- which is the fault the axis version already had once.
    const vkexp::ScenarioDefinition& puckScenario =
        vkexp::scenarioDefinition(vkexp::BeaconScenario::PuckPush);
    vkexp::SimulationStep scattered = settings;
    scattered.puckRandomStart = true;
    scattered.beaconMotionSeed = seed;
    for (const std::uint32_t world : {0U, 5U, 11U}) {
        vkexp::AgentState agent{};
        agent.pose = {0.3F, 0.2F, 0.0F, vkexp::agentBodyRadius};
        agent.penalties.w = static_cast<float>(world);
        puckScenario.spawn(agent, scattered);
        const puck::vec2 placed =
            puck::puckStartPositionFor(arena, 0U, world, seed, true);
        check(closeTo(puckScenario.targetDistance(agent, scattered),
                      std::hypot(placed.x, placed.y)),
              "A scattered trial opens measuring the puck the driver actually placed");
    }
}

// The gate world. What has to hold is not that it is solvable -- that is what a
// run answers -- but that the two legs it is made of are real: that the plate is
// somewhere other than the doorway, that a shut gate actually hides what is
// behind it, and that the latch does what its one number says.
// The locomotion ladder. Presets are only five points in a space the sliders
// already reach, so what is worth pinning is not the numbers themselves but the
// three claims made about them in the window: that the middle rung is the
// simulation's own defaults, that the ladder is ordered, and that a rung changes
// how fast a body answers without changing what it can ultimately do.
void testLocomotionPresets() {
    const vkexp::SimulationStep defaults{};
    for (std::size_t index = 0; index < vkexp::locomotionStyleCount; ++index) {
        const vkexp::LocomotionPreset& preset = vkexp::locomotionPresets[index];
        check(static_cast<std::size_t>(preset.style) == index,
              "the preset table is in LocomotionStyle order");
        check(preset.key != nullptr && *preset.key != '\0' && preset.name != nullptr,
              "every preset has a name and a command-line key");
        check(vkexp::locomotionPresetForKey(preset.key) == &preset,
              "a preset is reachable by its own key");
    }
    check(vkexp::locomotionPresetForKey("nonesuch") == nullptr, "an unknown key finds nothing");

    // The middle rung is the defaults value for value, not an approximation of
    // them: selecting it has to be a return to the body every scenario was tuned
    // against, or a run before touching this control and a run after it differ
    // by an amount nobody wrote down.
    vkexp::SimulationStep applied = defaults;
    vkexp::applyLocomotionPreset(applied, vkexp::LocomotionStyle::TableRobot);
    check(applied.thrust == defaults.thrust && applied.turnAcceleration == defaults.turnAcceleration &&
              applied.linearDrag == defaults.linearDrag &&
              applied.angularDrag == defaults.angularDrag,
          "the Table robot preset is the simulation's own defaults");

    // Applying a preset touches the four locomotion numbers and nothing else --
    // in particular not the two caps, which is what keeps the styles comparable.
    vkexp::SimulationStep fish = defaults;
    vkexp::applyLocomotionPreset(fish, vkexp::LocomotionStyle::Fish);
    check(fish.maximumSpeed == defaults.maximumSpeed &&
              fish.maximumAngularSpeed == defaults.maximumAngularSpeed,
          "a preset never moves the speed caps");
    check(fish.deltaTime == defaults.deltaTime && fish.worldRadius == defaults.worldRadius,
          "a preset touches nothing outside locomotion");

    // What the combo reads back. A label that could only be written would keep
    // saying "Fish" over sliders that had since been dragged elsewhere.
    const vkexp::LocomotionPreset* found = vkexp::currentLocomotionPreset(fish);
    check(found != nullptr && found->style == vkexp::LocomotionStyle::Fish,
          "the sliders report the preset they were set from");
    fish.linearDrag *= 1.5F;
    check(vkexp::currentLocomotionPreset(fish) == nullptr,
          "and report nothing once one of them is dragged away");

    // The ladder, in the quantities a body is actually judged by rather than in
    // the raw sliders: how long it takes to reach speed and how far it carries
    // once the motors stop. Ordered strictly, so no two rungs are the same body
    // under different names.
    float previousCoast = 0.0F;
    float previousSpin = 0.0F;
    float previousTime = 0.0F;
    for (const vkexp::LocomotionPreset& preset : vkexp::locomotionPresets) {
        vkexp::SimulationStep settings = defaults;
        vkexp::applyLocomotionPreset(settings, preset.style);
        const vkexp::LocomotionResponse response = vkexp::locomotionResponse(settings);
        check(response.coastDistance > previousCoast, "each rung coasts further than the last");
        check(response.spinCoast > previousSpin, "each rung carries its turn further");
        check(response.timeToTopSpeed > previousTime, "each rung takes longer to reach speed");
        previousCoast = response.coastDistance;
        previousSpin = response.spinCoast;
        previousTime = response.timeToTopSpeed;

        // The claim the whole ladder rests on: every style tops out at the same
        // speed. A rung whose thrust could not hold the cap would be slower as
        // well as heavier, and a comparison between two rungs would no longer be
        // a comparison of inertia.
        check(response.speedCapBinds && std::abs(response.topSpeed - settings.maximumSpeed) < 1.0e-6F,
              "every locomotion style reaches the same top speed");
    }

    // The turn is where the claim does not hold, and it is worth failing loudly
    // if that ever silently changes. Four of the rungs reach the turn cap; the
    // defaults do not -- 5.0 rad/s^2 against 2.4/s holds 2.08 rad/s, so at the
    // default settings the maximum turn speed slider has nothing to do. The
    // window says so next to the slider. This asserts the fact rather than the
    // preference, so aligning the defaults will fail here and be noticed.
    vkexp::SimulationStep table = defaults;
    vkexp::applyLocomotionPreset(table, vkexp::LocomotionStyle::TableRobot);
    const vkexp::LocomotionResponse tableResponse = vkexp::locomotionResponse(table);
    check(!tableResponse.turnCapBinds && tableResponse.topTurnRate < table.maximumAngularSpeed,
          "the default body never reaches its own turn cap");
    for (const vkexp::LocomotionPreset& preset : vkexp::locomotionPresets) {
        if (preset.style == vkexp::LocomotionStyle::TableRobot) {
            continue;
        }
        vkexp::SimulationStep settings = defaults;
        vkexp::applyLocomotionPreset(settings, preset.style);
        const vkexp::LocomotionResponse response = vkexp::locomotionResponse(settings);
        check(response.turnCapBinds &&
                  std::abs(response.topTurnRate - settings.maximumAngularSpeed) < 1.0e-6F,
              "every other style does reach the same top turn rate");
    }

    // And the response numbers are read off the sliders, not off the table, so
    // a hand-tuned body is described as honestly as a named one.
    vkexp::SimulationStep byHand = defaults;
    byHand.linearDrag = 4.0F;
    byHand.thrust = 4.0F;
    const vkexp::LocomotionResponse handResponse = vkexp::locomotionResponse(byHand);
    check(std::abs(handResponse.coastDistance - byHand.maximumSpeed / 4.0F) < 1.0e-6F,
          "coast is derived from the sliders in front of the user");
}

void testGateWorld() {
    namespace kernel = vkexp::worlds::kernel;
    const auto completedTrips = [](const vkexp::AgentState& agent) {
        return static_cast<std::uint32_t>(std::max(agent.target.w, 0.0F));
    };
    const vkexp::ScenarioDefinition& scenario =
        vkexp::scenarioDefinition(vkexp::BeaconScenario::GatePlate);
    vkexp::SimulationStep settings;
    settings.beaconScenario = vkexp::BeaconScenario::GatePlate;
    const float radius = settings.worldRadius;
    const kernel::vec2 plate = kernel::gatePlatePosition(radius);
    const kernel::vec2 resource = kernel::gateResourcePosition(radius);

    // The two things to do are in different places. If the plate sat in the
    // doorway the task would collapse into one leg -- walk through, pressing on
    // the way -- and nothing about it would need holding in mind.
    check(plate.y < -kernel::GateWallHalfThickness && resource.y > kernel::GateWallHalfThickness,
          "The plate is in front of the wall and the resource behind it");
    const float plateToOpening =
        std::hypot(plate.x - kernel::GateOpeningOffset * radius, plate.y);
    check(plateToOpening > kernel::gatePlateRadius(radius) + kernel::GateOpeningHalfWidth * radius,
          "Standing on the plate is not standing in the doorway");
    check(kernel::gateOnPlate({plate.x, plate.y}, radius),
          "The plate's own centre is on the plate");
    check(!kernel::gateOnPlate({plate.x + kernel::gatePlateRadius(radius) * 1.05F, plate.y}, radius),
          "And just outside its rim is not");
    check(kernel::gatePlateRadius(radius) > vkexp::agentBodyRadius * 4.0F,
          "The plate is wide enough for several agents to be standing on it at once");
    // Which is also why it is drawn as its own disc rather than as the beacon
    // that lights it: a beacon is drawn at one fixed visual radius, and at that
    // radius the picture would show a dot a third the size of the floor the
    // press test actually reads.
    check(kernel::gatePlateRadius(radius) > vkexp::beaconVisualRadius * 2.0F,
          "And wider than the beacon marking it, so the two cannot be drawn as one thing");

    // A shut gate has to hide the resource, and an open one has to show it. That
    // is the whole of what this world tells an agent it has done: press, and the
    // far light appears. Measured the way the two-door and two-gap walls were --
    // swept over the side the agents stand on rather than probed at a point --
    // and asserted in both directions, because a wall that hides nothing and a
    // wall with no way through are both wrong, and the same test has to fail for
    // both.
    const auto visibleWith = [&](const bool open) {
        return visibleFractionOfFarSide(
            radius, resource, kernel::GatePlateBoxCount,
            [&](const kernel::uint index) {
                return std::pair{kernel::gateBoxCentre(index, radius, open),
                                 kernel::gateBoxHalfExtent(index, radius)};
            },
            -1.0F);
    };
    check(visibleWith(false) < 0.01F,
          "A shut gate leaves the resource invisible from the side the agents start on");
    check(visibleWith(true) > minimumTargetVisibility,
          "An open one shows it from as much of that side as a gap that was learned");

    // The latch, which is the difficulty of this world in one number. Pressed
    // reloads it; released runs it down; and at zero it is open exactly while
    // pressed -- one step and no more, which is the setting that needs a second
    // agent.
    const float dt = settings.deltaTime;
    check(kernel::gateIsOpen(kernel::gateRemaining(0.0F, true, 4.0F, dt)),
          "Pressing the plate opens the gate");
    check(closeTo(kernel::gateRemaining(0.0F, true, 4.0F, dt), 4.0F),
          "And reloads the latch to its full length");
    check(closeTo(kernel::gateRemaining(4.0F, false, 4.0F, dt), 4.0F - dt),
          "Letting go runs the latch down by one step");
    check(!kernel::gateIsOpen(kernel::gateRemaining(dt * 0.5F, false, 4.0F, dt)),
          "And it stops at zero rather than going negative");
    check(kernel::gateIsOpen(kernel::gateRemaining(0.0F, true, 0.0F, dt)),
          "At a latch of zero the gate is still open during the step it is pressed");
    check(!kernel::gateIsOpen(kernel::gateRemaining(dt, false, 0.0F, dt)),
          "And shut the step after it is released, so somebody has to stay");

    // The gate leaf is geometry, and when it is open it must stop nothing. Parked
    // outside the arena rather than resized, because the extent is asked for
    // without an agent to ask about.
    const kernel::vec2 shutLeaf = kernel::gateBoxCentre(2U, radius, false);
    const kernel::vec2 openLeaf = kernel::gateBoxCentre(2U, radius, true);
    check(std::hypot(shutLeaf.x, shutLeaf.y) < radius,
          "A shut gate leaf stands in the arena, in the opening");
    check(std::hypot(openLeaf.x, openLeaf.y) > radius * 4.0F,
          "An open one is parked far enough out to stop nothing and block no light");
    check(scenario.obstacleCount == kernel::GatePlateBoxCount && scenario.obstacle != nullptr,
          "The gate world reports its three boxes and hands them out");

    // Which leg is being shaped. Shut, the plate; open, the resource. This is
    // read through the same internal.y convention the delivery worlds use, so a
    // scenario that set it the other way round would be shaped toward the wrong
    // beacon while every other assertion here still passed.
    vkexp::AgentState agent{};
    agent.pose = {plate.x, plate.y - 0.4F, 0.0F, vkexp::agentBodyRadius};
    scenario.spawn(agent, settings);
    check(!vkexp::worlds::gate_plate::gateOpen(agent), "A trial opens with the gate shut");
    check(agent.pose.y < 0.0F, "And with every agent on the near side of the wall");
    agent.pose = {plate.x, plate.y, 0.0F, vkexp::agentBodyRadius};
    agent.internal.y = 1.0F;
    check(closeTo(scenario.targetDistance(agent, settings), 0.0F),
          "With the gate shut the shaping measures the way to the plate");
    agent.target.x = 2.0F;
    agent.internal.y = 0.0F;
    check(closeTo(scenario.targetDistance(agent, settings),
                  std::hypot(resource.x - agent.pose.x, resource.y - agent.pose.y)),
          "With it open the shaping measures the way to the resource");

    // The cycle. Crossing is half a task -- an agent that is through is done, and
    // the door being held is worth something exactly once. Coming back makes the
    // gate a thing that has to be open twice, so whoever is holding it matters
    // for as long as anybody is still out.
    check(scenario.objectivesPerAgent > 1, "The world asks for round trips, not one crossing");
    vkexp::AgentState busy{};
    busy.target.w = static_cast<float>(scenario.objectivesPerAgent) + 2.0F;
    check(scenario.achievedObjectives(busy) == scenario.objectivesPerAgent,
          "The reported ratio is capped at what the trial has room for");
    check(scenario.fitness(busy, settings.fitness) >
              scenario.fitness([&] {
                  vkexp::AgentState slower{};
                  slower.target.w = static_cast<float>(scenario.objectivesPerAgent);
                  return slower;
              }(),
                               settings.fitness),
          "But the score is not, so extra trips still pay");

    // What the trial has to be long enough for. A leg is the straight line from
    // the plate to the resource; at the speed limit a round trip is about 840
    // steps, so two of them want roughly 1800 -- twice the default. Asserted
    // rather than noted, because the geometry is what would quietly break it: a
    // plate moved further from the door makes the nominal unreachable and the
    // reported ratio would flatten near half without anything looking wrong.
    const float legSeconds =
        std::hypot(resource.x - plate.x, resource.y - plate.y) / settings.maximumSpeed;
    const float nominalSeconds =
        static_cast<float>(scenario.objectivesPerAgent) * 2.0F * legSeconds * 1.8F;
    check(nominalSeconds <= vkexp::units::secondsForSteps(scenario.nominalStepsPerGeneration,
                                                          vkexp::units::fixedTimeStep),
          "The nominal number of round trips fits in the trial the world asks for");
    check(nominalSeconds > vkexp::units::secondsForSteps(900U, vkexp::units::fixedTimeStep) &&
              scenario.nominalStepsPerGeneration > 900U,
          "And does not fit in the usual 900, which is why the world asks out loud");

    // Walking the cycle by hand, because the order is the whole task and every
    // step of it is a place the flags can be crossed. The distance handed to the
    // hook is the distance to whatever the agent was heading for, which is what
    // the step computes, so the walk has to recompute it the same way.
    vkexp::AgentState walker{};
    walker.pose = {plate.x, plate.y, 0.0F, vkexp::agentBodyRadius};
    scenario.spawn(walker, settings);
    walker.pose.x = plate.x;
    walker.pose.y = plate.y;
    const auto step = [&](const float x, const float y, const float latch) {
        walker.pose.x = x;
        walker.pose.y = y;
        walker.target.x = latch;
        scenario.afterStep(walker, settings, scenario.targetDistance(walker, settings));
    };
    // Standing on the plate opens the gate but closes no trip.
    step(plate.x, plate.y, 2.0F);
    check(completedTrips(walker) == 0 && walker.internal.x < 0.5F,
          "Standing on the plate is not an arrival and starts no cargo");
    check(walker.internal.y < 0.5F, "With the gate running the target becomes the resource");
    // Out to the resource: that is the pickup.
    step(resource.x, resource.y, 2.0F);
    check(walker.internal.x >= 0.5F, "Reaching the resource picks it up");
    check(walker.internal.y >= 0.5F, "And turns the agent back toward the plate");
    check(completedTrips(walker) == 0, "Which is not yet a round trip");
    // Sitting on the resource does not collect it twice.
    step(resource.x, resource.y, 2.0F);
    check(completedTrips(walker) == 0 && walker.internal.x >= 0.5F,
          "Lingering on the resource collects it once");
    // And home again.
    step(plate.x, plate.y, 2.0F);
    check(completedTrips(walker) == 1 && walker.internal.x < 0.5F,
          "Coming back to the plate closes the round trip and empties the agent");

    // The gate shutting mid-cycle sends an outbound agent back to the plate and
    // leaves a carrying one where it was going, because a carrying agent was
    // already heading there.
    vkexp::AgentState outbound{};
    outbound.pose = {0.0F, plate.y, 0.0F, vkexp::agentBodyRadius};
    outbound.internal.y = 0.0F;
    outbound.target.x = 0.0F; // the gate has just shut
    scenario.afterStep(outbound, settings, scenario.targetDistance(outbound, settings));
    check(outbound.internal.y >= 0.5F,
          "A shut gate sends an agent that is not carrying back to the plate");
}

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
    check(!vkexp::recordSweepGeneration(sweep, 4.0F, 2.0F, 0.9F), "The final stage does not advance");
    check(!sweep.running && sweep.stages.size() == 3, "A finished sweep stops with every stage kept");
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

} // namespace

int main() {
    testTimingSeries();
    testCpuProfiler();
    testDispatchSize();
    testComputeResourceValidation();
    testLogicalWorldPartition();
    testPingPongState();
    testNeuralNetworkContract();
    testFixedStepIndependence();
    testMultimodalSensors();
    testWorldAndBeaconScenarios();
    testForageCycleAndMemory();
    testWallCollisionPenalty();
    testGeneticAlgorithm();
    testExperimentSweep();
    testNeuronTimeConstants();
    testGatedNeurons();
    testTwoDoorsGeometry();
    testShuttleGeometry();
    testTwoGapsGeometry();
    testBeaconColorAblation();
    testPuckWorld();
    testPuckPushCredit();
    testGateWorld();
    testLocomotionPresets();
    testDeliveryCannotBeScoredTwice();
    testScenarioRegistryContract();
    testFitnessWeightsAreParameters();
    testSharedScenarioKernel();
    testBrainDescription();
    testGenomeArchiveRoundTrip();
    testGroupFitnessSharing();
    testWorldSnapshotRoundTrip();
    testPopulationReload();
    testStepParameterPacking();
    testResolvedStepSettings();
    return failures == 0 ? 0 : 1;
}
