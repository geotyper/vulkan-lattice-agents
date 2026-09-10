#pragma once

#include "vkexp/neuro/NeuralNetwork.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// The network written down as structure rather than as arithmetic.
//
// Every offset in this project is computed: the input vector is addressed by
// brainLightChannelIndex and friends, the genome by brainHiddenWeightIndex and
// friends, and both languages compile the same functions so they cannot
// disagree. What none of that gives is a *statement* of the layout -- something
// a person can read, a file can carry, and a loader can compare against. Six
// numbers in an archive header ("2668 weights, 61 inputs") cannot say which
// sensor moved when a file stops loading.
//
// So this is a description, and deliberately a derived one: every block below
// asks the shared kernel where it starts rather than restating a number, which
// is rule 3c applied to the layout itself. If the arithmetic changes, the
// description changes with it -- and testBrainDescription pins that the blocks
// still tile the vectors exactly, with no gap and no overlap.
//
// What it is not, yet: a way to *choose* the structure. GLSL sizes its arrays
// with compile-time constants, so capacity stays in BrainKernel.inl. What a file
// can already carry is the active shape inside that capacity, which reaches the
// GPU as the packed brain layout.
namespace vkexp::neuro {

class BrainDescriptionError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// One named span of a vector: a sensor block in the input vector, an actuator
// slot in the output vector, or one weight block in the genome.
struct BrainBlock {
    std::string name;
    std::uint32_t offset{};
    std::uint32_t count{};
    // Where a weight block connects from and to, by the name of another block or
    // of a layer. Empty for input and output blocks, which connect to nothing --
    // they are the ends.
    std::string from;
    std::string to;
    // A weight block's shape, so a matrix reads as a matrix. Both zero when the
    // block is a plain vector, which is what every bias is.
    std::uint32_t rows{};
    std::uint32_t columns{};

    [[nodiscard]] bool isMatrix() const { return rows != 0 && columns != 0; }
};

struct BrainDescription {
    std::uint32_t inputCount{};
    std::uint32_t hiddenCount{};
    std::uint32_t outputCount{};
    std::uint32_t weightCount{};
    // Which integrator the weights are read under. The gate block is carried by
    // every genome whatever is selected, so this changes what is evaluated and
    // not what is stored -- but a file that does not say it leaves a reader
    // guessing why the same weights behave differently.
    std::string neuronModel{"time"};
    std::vector<BrainBlock> inputs;
    std::vector<BrainBlock> outputs;
    std::vector<BrainBlock> weights; // in genome order

    [[nodiscard]] const BrainBlock* block(std::string_view name) const;
};

// The description of a network of this shape, derived from the shared kernel.
[[nodiscard]] BrainDescription describeBrain(BrainShape shape,
                                             std::string_view neuronModel = "time");

// JSON, both ways. The document is small and fixed in shape, so the parser is a
// strict one written for exactly this document rather than a general library:
// anything it does not recognise is an error rather than a silent default,
// because a structure file that is quietly half-read is worse than none.
[[nodiscard]] std::string brainDescriptionToJson(const BrainDescription& description);
[[nodiscard]] BrainDescription parseBrainDescription(std::string_view json);

// What differs, in words, most structural difference first. Empty means the two
// describe the same network. This is what a loader reports instead of "weight
// count differs": which block moved, which one is missing, which one changed
// shape.
[[nodiscard]] std::vector<std::string> compareBrainDescriptions(const BrainDescription& expected,
                                                                const BrainDescription& actual);

} // namespace vkexp::neuro
