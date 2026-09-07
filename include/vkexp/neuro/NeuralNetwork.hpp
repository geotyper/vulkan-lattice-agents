#pragma once

#include "vkexp/neuro/BrainKernel.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexp::neuro {

// C++ view of the network preset in BrainKernel.inl, which the shaders compile
// from the same source. Nothing here restates a number: change the preset and
// both sides follow.
struct Topology {
    static constexpr std::size_t lightReceptorCount = kernel::BrainLightReceptorCount;
    static constexpr std::size_t lightChannelsPerReceptor = kernel::BrainLightChannels;
    static constexpr std::size_t tactileSectorCount = kernel::BrainTactileSectorCount;
    static constexpr std::size_t tactileChannelsPerSector = kernel::BrainTactileChannels;
    static constexpr std::size_t antennaCount = kernel::BrainAntennaCount;
    static constexpr std::size_t antennaChannelsPerTip = kernel::BrainAntennaChannels;
    static constexpr std::size_t selfInputCount = kernel::BrainSelfInputCount;
    static constexpr std::size_t taskInputCount = kernel::BrainTaskInputCount;
    static constexpr std::size_t recurrentMemoryCount = kernel::BrainRecurrentCount;
    static constexpr std::size_t inputCount = kernel::BrainInputCapacity;
    static constexpr std::size_t hiddenCount = kernel::BrainHiddenCapacity;
    static constexpr std::size_t actuatorOutputCount = kernel::BrainActuatorOutputCount;
    static constexpr std::size_t outputCount = kernel::BrainOutputCapacity;
    static constexpr std::size_t weightCount = kernel::brainWeightCount(
        kernel::BrainInputCapacity, kernel::BrainHiddenCapacity, kernel::BrainOutputCapacity);

    // Offsets into the input vector, shared with the shader's sensor pass.
    static constexpr std::size_t tactileOffset = kernel::BrainTactileOffset;
    static constexpr std::size_t antennaOffset = kernel::BrainAntennaOffset;
    static constexpr std::size_t selfOffset = kernel::BrainSelfOffset;
    static constexpr std::size_t taskOffset = kernel::BrainTaskOffset;
    static constexpr std::size_t recurrentInputOffset = kernel::BrainRecurrentInputOffset;
    static constexpr std::size_t recurrentOutputOffset = kernel::BrainRecurrentOutputOffset;
};

// A scenario selects an active dense-network shape inside the fixed-capacity genome.
// Keeping capacity separate from shape lets GPU buffers and the GA stay reusable.
struct BrainShape {
    std::size_t inputCount{};
    std::size_t hiddenCount{};
    std::size_t outputCount{};

    [[nodiscard]] constexpr std::size_t weightCount() const {
        return kernel::brainWeightCount(static_cast<kernel::uint>(inputCount),
                                        static_cast<kernel::uint>(hiddenCount),
                                        static_cast<kernel::uint>(outputCount));
    }

    [[nodiscard]] constexpr bool fitsCapacity() const {
        return inputCount > 0 && inputCount <= Topology::inputCount && hiddenCount > 0 &&
               hiddenCount <= Topology::hiddenCount &&
               outputCount >= Topology::actuatorOutputCount &&
               outputCount <= Topology::outputCount && weightCount() <= Topology::weightCount;
    }
};

inline constexpr BrainShape maximumBrainShape{Topology::inputCount, Topology::hiddenCount,
                                              Topology::outputCount};

[[nodiscard]] constexpr std::uint32_t
packBrainLayout(const BrainShape shape, const std::size_t genomeStride = Topology::weightCount) {
    return kernel::brainPackLayout(
        static_cast<kernel::uint>(genomeStride), static_cast<kernel::uint>(shape.inputCount),
        static_cast<kernel::uint>(shape.hiddenCount), static_cast<kernel::uint>(shape.outputCount));
}

[[nodiscard]] constexpr std::size_t brainGenomeStride(const std::uint32_t layout) {
    return kernel::brainLayoutStride(layout);
}

[[nodiscard]] constexpr BrainShape brainShape(const std::uint32_t layout) {
    return {kernel::brainLayoutInputCount(layout), kernel::brainLayoutHiddenCount(layout),
            kernel::brainLayoutOutputCount(layout)};
}

static_assert(maximumBrainShape.fitsCapacity());
// The preset has to stay expressible in the packed layout the GPU receives.
static_assert(Topology::weightCount <= kernel::BrainStrideMask,
              "Genome stride no longer fits the packed brain layout");
static_assert(Topology::inputCount <= kernel::BrainCountMask,
              "Input count no longer fits the packed brain layout");
static_assert(Topology::hiddenCount <= kernel::BrainCountMask,
              "Hidden count no longer fits the packed brain layout");
static_assert(Topology::outputCount <= kernel::BrainOutputCountMask,
              "Output count no longer fits the packed brain layout");

using Inputs = std::array<float, Topology::inputCount>;
using Outputs = std::array<float, Topology::outputCount>;
using Weights = std::array<float, Topology::weightCount>;

// The continuous-time state of one brain's hidden layer, carried between steps.
using HiddenState = std::array<float, Topology::hiddenCount>;

// Single-network evaluator for tests, inspection and champion replay. It builds
// the network from the same genome addressing the shader uses, so it is a way to
// look inside one brain rather than a second implementation of the layout.
//
// Each hidden neuron is integrated toward its activation at a time constant the
// selected model decides, and the new state is left in `state`. The model only
// chooses where the time constant comes from; the integrator is the same one in
// every case, which is what makes switching models an ablation rather than a
// swap between two networks. `model` is a kernel::NeuronModel* value.
[[nodiscard]] Outputs evaluate(std::span<const float, Topology::weightCount> weights,
                               const Inputs& inputs, HiddenState& state, float deltaTime,
                               kernel::uint model, BrainShape shape = maximumBrainShape);

// Stateless convenience for the tests and inspections that ask what a brain does
// to one input vector with no history. Defined in terms of the above with a
// fresh state and the reactive model, so there is one evaluator and not two.
[[nodiscard]] Outputs evaluate(std::span<const float, Topology::weightCount> weights,
                               const Inputs& inputs, BrainShape shape = maximumBrainShape);

} // namespace vkexp::neuro
