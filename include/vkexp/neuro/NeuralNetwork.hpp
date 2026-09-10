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
    static constexpr std::size_t hiddenCount = kernel::BrainHiddenNeuronCapacity;
    static constexpr std::size_t hiddenLayerCount = kernel::BrainHiddenLayerCapacity;
    static constexpr std::size_t actuatorOutputCount = kernel::BrainActuatorOutputCount;
    static constexpr std::size_t outputCount = kernel::BrainOutputCapacity;
    // The genome stride: sized for the widest plan the capacity allows, which
    // is one layer using every neuron. A plan that spends them differently uses
    // fewer, and the tail simply goes unread -- one length means one buffer and
    // one loadable population across every plan.
    static constexpr std::size_t weightCount = kernel::brainWeightCount(
        kernel::BrainInputCapacity,
        kernel::brainPackHiddenLayers(kernel::BrainHiddenNeuronCapacity, 0u, 0u),
        kernel::BrainOutputCapacity);

    // Offsets into the input vector, shared with the shader's sensor pass.
    static constexpr std::size_t tactileOffset = kernel::BrainTactileOffset;
    static constexpr std::size_t antennaOffset = kernel::BrainAntennaOffset;
    static constexpr std::size_t selfOffset = kernel::BrainSelfOffset;
    static constexpr std::size_t taskOffset = kernel::BrainTaskOffset;
    static constexpr std::size_t recurrentInputOffset = kernel::BrainRecurrentInputOffset;
    static constexpr std::size_t recurrentOutputOffset = kernel::BrainRecurrentOutputOffset;
};

// A scenario selects an active network inside the fixed-capacity genome. Keeping
// capacity separate from shape lets the GPU buffers and the GA stay reusable,
// and it is what lets a plan be chosen at runtime rather than compiled in.
//
// The first three fields are the network this project had for its whole life, so
// `{61, 20, 8}` still means one 20-wide hidden layer and every scenario that
// wrote that keeps its meaning. The two after it are the second and third hidden
// layers; zero means the layer is not there.
struct BrainShape {
    std::size_t inputCount{};
    std::size_t hiddenCount{}; // width of the first hidden layer
    std::size_t outputCount{};
    std::size_t secondHiddenCount{};
    std::size_t thirdHiddenCount{};

    [[nodiscard]] constexpr std::size_t hiddenLayer(const std::size_t layer) const {
        if (layer == 0) {
            return hiddenCount;
        }
        if (layer == 1) {
            return secondHiddenCount;
        }
        return layer == 2 ? thirdHiddenCount : 0;
    }

    // Layers are dense from the front: a gap would make "two layers" ambiguous
    // about which two, and every offset below walks them in order.
    [[nodiscard]] constexpr std::size_t hiddenLayerCount() const {
        std::size_t count = 0;
        for (std::size_t layer = 0; layer < Topology::hiddenLayerCount; ++layer) {
            if (hiddenLayer(layer) == 0) {
                break;
            }
            count = layer + 1;
        }
        return count;
    }

    [[nodiscard]] constexpr std::size_t hiddenTotal() const {
        std::size_t total = 0;
        for (std::size_t layer = 0; layer < hiddenLayerCount(); ++layer) {
            total += hiddenLayer(layer);
        }
        return total;
    }

    [[nodiscard]] constexpr std::uint32_t packedLayers() const {
        return kernel::brainPackHiddenLayers(static_cast<kernel::uint>(hiddenCount),
                                             static_cast<kernel::uint>(secondHiddenCount),
                                             static_cast<kernel::uint>(thirdHiddenCount));
    }

    [[nodiscard]] constexpr std::size_t weightCount() const {
        return kernel::brainWeightCount(static_cast<kernel::uint>(inputCount), packedLayers(),
                                        static_cast<kernel::uint>(outputCount));
    }

    [[nodiscard]] constexpr bool fitsCapacity() const {
        if (inputCount == 0 || inputCount > Topology::inputCount) {
            return false;
        }
        if (outputCount < Topology::actuatorOutputCount || outputCount > Topology::outputCount) {
            return false;
        }
        // A hole in the middle is rejected rather than silently closed up: a plan
        // that says {20, 0, 8} means something the packing cannot express, and
        // guessing which of the two readings was meant is worse than refusing.
        bool ended = false;
        for (std::size_t layer = 0; layer < Topology::hiddenLayerCount; ++layer) {
            const std::size_t width = hiddenLayer(layer);
            if (width == 0) {
                ended = true;
                continue;
            }
            if (ended || width > Topology::hiddenCount) {
                return false;
            }
        }
        return hiddenLayerCount() > 0 && hiddenTotal() <= Topology::hiddenCount &&
               weightCount() <= Topology::weightCount;
    }
};

// The widest plan: every neuron in one layer. Also what sizes the genome.
inline constexpr BrainShape maximumBrainShape{Topology::inputCount, Topology::hiddenCount,
                                              Topology::outputCount};

// What every scenario ran before plans existed, and still the default: one
// hidden layer of twenty. Named so a scenario can say it means this rather than
// happening to write the same number.
inline constexpr std::size_t defaultHiddenWidth = 20;

[[nodiscard]] constexpr std::uint32_t packBrainLayout(const BrainShape shape) {
    return kernel::brainPackLayout(static_cast<kernel::uint>(shape.inputCount),
                                   static_cast<kernel::uint>(shape.outputCount));
}

[[nodiscard]] constexpr BrainShape brainShape(const std::uint32_t layout,
                                              const std::uint32_t layers) {
    return {kernel::brainLayoutInputCount(layout), kernel::brainHiddenLayerSize(layers, 0u),
            kernel::brainLayoutOutputCount(layout), kernel::brainHiddenLayerSize(layers, 1u),
            kernel::brainHiddenLayerSize(layers, 2u)};
}

static_assert(maximumBrainShape.fitsCapacity());
// The preset has to stay expressible in the packed layout the GPU receives.
static_assert(Topology::inputCount <= kernel::BrainCountMask,
              "Input count no longer fits the packed brain layout");
static_assert(Topology::hiddenCount <= kernel::BrainLayerSizeMask,
              "A hidden layer no longer fits its field in the packed layer plan");
static_assert(Topology::hiddenLayerCount == kernel::BrainHiddenLayerCapacity,
              "The C++ view and the shared kernel disagree on how many layers there may be");
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
