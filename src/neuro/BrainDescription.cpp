#include "vkexp/neuro/BrainDescription.hpp"

#include "vkexp/neuro/BrainKernel.hpp"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <string>
#include <utility>

namespace vkexp::neuro {
namespace {

namespace bk = kernel;

using bk::uint;

BrainBlock vectorBlock(std::string name, const uint offset, const uint count, const uint rows = 0,
                       const uint columns = 0) {
    BrainBlock value{};
    value.name = std::move(name);
    value.offset = offset;
    value.count = count;
    value.rows = rows;
    value.columns = columns;
    return value;
}

BrainBlock weightBlock(std::string name, std::string from, std::string to, const uint offset,
                       const uint count, const uint rows = 0, const uint columns = 0) {
    BrainBlock value = vectorBlock(std::move(name), offset, count, rows, columns);
    value.from = std::move(from);
    value.to = std::move(to);
    return value;
}

// --- the smallest JSON writer that produces this document --------------------

void appendEscaped(std::string& out, const std::string_view text) {
    out += '"';
    for (const char character : text) {
        switch (character) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out += character;
            break;
        }
    }
    out += '"';
}

void appendBlock(std::string& out, const BrainBlock& block, const std::string_view indent) {
    out += indent;
    out += "{ \"name\": ";
    appendEscaped(out, block.name);
    out += ", \"offset\": " + std::to_string(block.offset);
    out += ", \"count\": " + std::to_string(block.count);
    if (!block.from.empty()) {
        out += ", \"from\": ";
        appendEscaped(out, block.from);
    }
    if (!block.to.empty()) {
        out += ", \"to\": ";
        appendEscaped(out, block.to);
    }
    if (block.isMatrix()) {
        out += ", \"rows\": " + std::to_string(block.rows);
        out += ", \"columns\": " + std::to_string(block.columns);
    }
    out += " }";
}

void appendBlocks(std::string& out, const std::string_view key,
                  const std::vector<BrainBlock>& blocks) {
    out += "  \"";
    out += key;
    out += "\": [\n";
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        appendBlock(out, blocks[index], "    ");
        out += index + 1 < blocks.size() ? ",\n" : "\n";
    }
    out += "  ]";
}

// --- the smallest JSON reader that accepts it --------------------------------
//
// Strict on purpose. A structure file that is quietly half-read describes a
// network nobody has, and the weights loaded against it would be wrong in a way
// that still produces plausible fitness numbers.

class Reader {
public:
    explicit Reader(const std::string_view text) : text_(text) {}

    void skipSpace() {
        while (position_ < text_.size() && (text_[position_] == ' ' || text_[position_] == '\n' ||
                                            text_[position_] == '\r' || text_[position_] == '\t')) {
            ++position_;
        }
    }

    [[nodiscard]] char peek() {
        skipSpace();
        if (position_ >= text_.size()) {
            fail("unexpected end of document");
        }
        return text_[position_];
    }

    void expect(const char character) {
        if (peek() != character) {
            fail(std::string{"expected '"} + character + "'");
        }
        ++position_;
    }

    [[nodiscard]] bool consume(const char character) {
        if (position_ < text_.size() && peek() == character) {
            ++position_;
            return true;
        }
        return false;
    }

    [[nodiscard]] std::string readString() {
        expect('"');
        std::string value;
        while (position_ < text_.size() && text_[position_] != '"') {
            char character = text_[position_++];
            if (character == '\\') {
                if (position_ >= text_.size()) {
                    fail("unterminated escape");
                }
                const char escaped = text_[position_++];
                switch (escaped) {
                case 'n':
                    character = '\n';
                    break;
                case 't':
                    character = '\t';
                    break;
                case '"':
                case '\\':
                case '/':
                    character = escaped;
                    break;
                default:
                    fail("unsupported escape");
                }
            }
            value += character;
        }
        if (position_ >= text_.size()) {
            fail("unterminated string");
        }
        ++position_;
        return value;
    }

    [[nodiscard]] std::uint32_t readNumber() {
        skipSpace();
        const std::size_t start = position_;
        while (position_ < text_.size() && (text_[position_] >= '0' && text_[position_] <= '9')) {
            ++position_;
        }
        if (position_ == start) {
            fail("expected a non-negative whole number");
        }
        std::uint32_t value{};
        const char* const begin = text_.data() + start;
        const auto [ignored, error] = std::from_chars(begin, text_.data() + position_, value);
        (void)ignored;
        if (error != std::errc{}) {
            fail("number out of range");
        }
        return value;
    }

    [[noreturn]] void fail(const std::string& message) const {
        throw BrainDescriptionError("Brain description at byte " + std::to_string(position_) +
                                    ": " + message);
    }

    void expectEnd() {
        skipSpace();
        if (position_ != text_.size()) {
            fail("trailing content after the document");
        }
    }

private:
    std::string_view text_;
    std::size_t position_{};
};

BrainBlock readBlock(Reader& reader) {
    BrainBlock block{};
    bool haveName = false;
    bool haveOffset = false;
    bool haveCount = false;
    reader.expect('{');
    if (!reader.consume('}')) {
        do {
            const std::string key = reader.readString();
            reader.expect(':');
            if (key == "name") {
                block.name = reader.readString();
                haveName = true;
            } else if (key == "from") {
                block.from = reader.readString();
            } else if (key == "to") {
                block.to = reader.readString();
            } else if (key == "offset") {
                block.offset = reader.readNumber();
                haveOffset = true;
            } else if (key == "count") {
                block.count = reader.readNumber();
                haveCount = true;
            } else if (key == "rows") {
                block.rows = reader.readNumber();
            } else if (key == "columns") {
                block.columns = reader.readNumber();
            } else {
                reader.fail("unknown block field '" + key + "'");
            }
        } while (reader.consume(','));
        reader.expect('}');
    }
    if (!haveName || !haveOffset || !haveCount) {
        reader.fail("a block needs a name, an offset and a count");
    }
    return block;
}

std::vector<BrainBlock> readBlocks(Reader& reader) {
    std::vector<BrainBlock> blocks;
    reader.expect('[');
    if (!reader.consume(']')) {
        do {
            blocks.push_back(readBlock(reader));
        } while (reader.consume(','));
        reader.expect(']');
    }
    return blocks;
}

// Every block of a list, in order, against the same list from the other
// description. Reported by name, because "block 4 moved" helps nobody.
void compareBlockList(const std::string_view what, const std::vector<BrainBlock>& expected,
                      const std::vector<BrainBlock>& actual, std::vector<std::string>& out) {
    const auto find = [](const std::vector<BrainBlock>& blocks, const std::string& name) {
        return std::find_if(blocks.begin(), blocks.end(),
                            [&name](const BrainBlock& block) { return block.name == name; });
    };
    for (const BrainBlock& block : expected) {
        const auto found = find(actual, block.name);
        if (found == actual.end()) {
            out.emplace_back(std::string{what} + " block '" + block.name + "' is missing");
            continue;
        }
        if (found->offset != block.offset || found->count != block.count) {
            out.emplace_back(std::string{what} + " block '" + block.name + "' is " +
                             std::to_string(found->count) + " wide at " +
                             std::to_string(found->offset) + ", expected " +
                             std::to_string(block.count) + " at " + std::to_string(block.offset));
        }
        if (found->rows != block.rows || found->columns != block.columns) {
            out.emplace_back(std::string{what} + " block '" + block.name + "' is shaped " +
                             std::to_string(found->rows) + "x" + std::to_string(found->columns) +
                             ", expected " + std::to_string(block.rows) + "x" +
                             std::to_string(block.columns));
        }
        if (found->from != block.from || found->to != block.to) {
            out.emplace_back(std::string{what} + " block '" + block.name + "' connects " +
                             found->from + " -> " + found->to + ", expected " + block.from +
                             " -> " + block.to);
        }
    }
    for (const BrainBlock& block : actual) {
        if (find(expected, block.name) == expected.end()) {
            out.emplace_back(std::string{what} + " block '" + block.name +
                             "' is not part of this build");
        }
    }
}

} // namespace

const BrainBlock* BrainDescription::block(const std::string_view name) const {
    for (const std::vector<BrainBlock>* list : {&inputs, &outputs, &weights}) {
        for (const BrainBlock& candidate : *list) {
            if (candidate.name == name) {
                return &candidate;
            }
        }
    }
    return nullptr;
}

BrainDescription describeBrain(const BrainShape shape, const std::string_view neuronModel) {
    const auto inputCount = static_cast<uint>(shape.inputCount);
    const auto outputCount = static_cast<uint>(shape.outputCount);

    BrainDescription description{};
    description.inputCount = inputCount;
    description.hiddenCount = static_cast<uint>(shape.hiddenTotal());
    for (std::size_t layer = 0; layer < shape.hiddenLayerCount(); ++layer) {
        description.hiddenLayers.push_back(static_cast<uint>(shape.hiddenLayer(layer)));
        description.hiddenActivations.emplace_back(
            brainActivationName(shape.hiddenActivation[layer]));
    }
    description.outputCount = outputCount;
    description.weightCount = bk::brainWeightCount(inputCount, shape.packedLayers(), outputCount);
    description.neuronModel = std::string{neuronModel};

    // Sensor blocks, each asking the kernel where its first channel lands rather
    // than restating an offset. A plan that trims the input vector keeps a dense
    // prefix, so a block past the active count is simply not there.
    const auto addInput = [&](std::string name, const uint offset, const uint count,
                              const uint rows, const uint columns) {
        if (offset >= inputCount) {
            return;
        }
        description.inputs.push_back(vectorBlock(
            std::move(name), offset, std::min(count, inputCount - offset), rows, columns));
    };
    addInput("neighbourhood", bk::brainNeighborChannelIndex(0u, 0u), bk::BrainNeighborBlockSize,
             bk::BrainNeighborCount, bk::BrainNeighborChannels);
    addInput("task", bk::brainBeaconInputIndex(0u), bk::BrainBeaconInputCount, 0, 0);
    addInput("self", bk::BrainSelfOffset, bk::BrainSelfInputCount, 0, 0);
    addInput("memory_in", bk::BrainRecurrentInputOffset, bk::BrainRecurrentCount, 0, 0);

    const auto addOutput = [&](std::string name, const uint offset, const uint count) {
        if (offset >= outputCount) {
            return;
        }
        description.outputs.push_back(
            vectorBlock(std::move(name), offset, std::min(count, outputCount - offset)));
    };
    addOutput("turn", bk::BrainTurnOutput, bk::BrainTurnOutputCount);
    addOutput("action", bk::BrainActionOutput, bk::BrainActionOutputCount);
    addOutput("signal", bk::BrainSignalIntensityOutput, bk::BrainSignalOutputCount);
    addOutput("memory_out", bk::BrainRecurrentOutputOffset, bk::BrainRecurrentCount);

    // The genome, block by block, in the order it is laid out. Offsets come from
    // the same index functions the shader calls, with a base of zero, so this
    // cannot drift from the arithmetic: it *is* the arithmetic, evaluated once.
    const uint layers = shape.packedLayers();
    const uint layerCount = bk::brainHiddenLayerCount(layers);
    const uint base = 0u;
    const auto layerName = [](const std::string_view prefix, const uint layer,
                              const std::string_view suffix) {
        return std::string{prefix} + std::to_string(layer) + std::string{suffix};
    };
    // One pair per hidden layer, then the output layer, then the genes, then the
    // gate block mirroring the forward one layer for layer.
    for (uint layer = 0; layer < layerCount; ++layer) {
        const uint width = bk::brainHiddenLayerSize(layers, layer);
        const uint sourceCount = bk::brainLayerSourceCount(inputCount, layers, layer);
        const std::string source =
            layer == 0 ? std::string{"inputs"} : layerName("hidden", layer - 1, "");
        const std::string target = layerName("hidden", layer, "");
        description.weights.push_back(
            weightBlock(layerName("hidden", layer, "_weights"), source, target,
                        bk::brainLayerWeightIndex(base, inputCount, layers, layer, 0u, 0u),
                        width * sourceCount, width, sourceCount));
        description.weights.push_back(
            weightBlock(layerName("hidden", layer, "_bias"), "", target,
                        bk::brainLayerBiasIndex(base, inputCount, layers, layer, 0u), width));
    }
    const uint lastHidden = bk::brainLastHiddenSize(layers);
    description.weights.push_back(
        weightBlock("output_weights", layerName("hidden", layerCount - 1u, ""), "outputs",
                    bk::brainOutputWeightIndex(base, inputCount, layers, 0u, 0u),
                    lastHidden * outputCount, outputCount, lastHidden));
    description.weights.push_back(weightBlock(
        "output_bias", "", "outputs",
        bk::brainOutputBiasIndex(base, inputCount, layers, outputCount, 0u), outputCount));
    description.weights.push_back(
        weightBlock("time_constants", "", "hidden",
                    bk::brainTimeConstantGeneIndex(base, inputCount, layers, outputCount, 0u),
                    bk::brainHiddenNeuronCount(layers)));
    for (uint layer = 0; layer < layerCount; ++layer) {
        const uint width = bk::brainHiddenLayerSize(layers, layer);
        const uint sourceCount = bk::brainLayerSourceCount(inputCount, layers, layer);
        const std::string source =
            layer == 0 ? std::string{"inputs"} : layerName("hidden", layer - 1, "");
        const std::string target = layerName("hidden", layer, "_rate");
        description.weights.push_back(weightBlock(
            layerName("gate", layer, "_weights"), source, target,
            bk::brainGateWeightIndex(base, inputCount, layers, outputCount, layer, 0u, 0u),
            width * sourceCount, width, sourceCount));
        description.weights.push_back(weightBlock(
            layerName("gate", layer, "_bias"), "", target,
            bk::brainGateBiasIndex(base, inputCount, layers, outputCount, layer, 0u), width));
    }
    // Interleaved rather than one block per gene, so one neuron's genes are
    // contiguous the way its weights are.
    description.weights.push_back(
        weightBlock("neuron_genes", "", "hidden",
                    bk::brainNeuronGeneIndex(base, inputCount, layers, outputCount, 0u, 0u),
                    bk::brainHiddenNeuronCount(layers) * bk::BrainNeuronGeneCount,
                    bk::brainHiddenNeuronCount(layers), bk::BrainNeuronGeneCount));
    // One square per layer, present whatever the plan's lateral bit says: the
    // block is what a genome is, and the bit is what reads it.
    for (uint layer = 0; layer < layerCount; ++layer) {
        const uint width = bk::brainHiddenLayerSize(layers, layer);
        const std::string target = layerName("hidden", layer, "");
        description.weights.push_back(weightBlock(
            layerName("lateral", layer, ""), target, target,
            bk::brainLateralWeightIndex(base, inputCount, layers, outputCount, layer, 0u, 0u),
            width * width, width, width));
    }
    return description;
}

std::string brainDescriptionToJson(const BrainDescription& description) {
    std::string out;
    out.reserve(2048);
    out += "{\n";
    out += "  \"inputs_count\": " + std::to_string(description.inputCount) + ",\n";
    out += "  \"hidden_count\": " + std::to_string(description.hiddenCount) + ",\n";
    out += "  \"hidden_layers\": [";
    for (std::size_t layer = 0; layer < description.hiddenLayers.size(); ++layer) {
        out += layer == 0 ? " " : ", ";
        out += std::to_string(description.hiddenLayers[layer]);
    }
    out += " ],\n";
    out += "  \"hidden_activations\": [";
    for (std::size_t layer = 0; layer < description.hiddenActivations.size(); ++layer) {
        out += layer == 0 ? " " : ", ";
        appendEscaped(out, description.hiddenActivations[layer]);
    }
    out += " ],\n";
    out += "  \"outputs_count\": " + std::to_string(description.outputCount) + ",\n";
    out += "  \"weight_count\": " + std::to_string(description.weightCount) + ",\n";
    out += "  \"neuron_model\": ";
    appendEscaped(out, description.neuronModel);
    out += ",\n";
    appendBlocks(out, "inputs", description.inputs);
    out += ",\n";
    appendBlocks(out, "outputs", description.outputs);
    out += ",\n";
    appendBlocks(out, "weights", description.weights);
    out += "\n}\n";
    return out;
}

BrainDescription parseBrainDescription(const std::string_view json) {
    Reader reader{json};
    BrainDescription description{};
    reader.expect('{');
    if (!reader.consume('}')) {
        do {
            const std::string key = reader.readString();
            reader.expect(':');
            if (key == "inputs_count") {
                description.inputCount = reader.readNumber();
            } else if (key == "hidden_count") {
                description.hiddenCount = reader.readNumber();
            } else if (key == "hidden_layers") {
                reader.expect('[');
                if (!reader.consume(']')) {
                    do {
                        description.hiddenLayers.push_back(reader.readNumber());
                    } while (reader.consume(','));
                    reader.expect(']');
                }
            } else if (key == "hidden_activations") {
                reader.expect('[');
                if (!reader.consume(']')) {
                    do {
                        description.hiddenActivations.push_back(reader.readString());
                    } while (reader.consume(','));
                    reader.expect(']');
                }
            } else if (key == "outputs_count") {
                description.outputCount = reader.readNumber();
            } else if (key == "weight_count") {
                description.weightCount = reader.readNumber();
            } else if (key == "neuron_model") {
                description.neuronModel = reader.readString();
            } else if (key == "inputs") {
                description.inputs = readBlocks(reader);
            } else if (key == "outputs") {
                description.outputs = readBlocks(reader);
            } else if (key == "weights") {
                description.weights = readBlocks(reader);
            } else {
                reader.fail("unknown field '" + key + "'");
            }
        } while (reader.consume(','));
        reader.expect('}');
    }
    reader.expectEnd();
    if (description.weightCount == 0 || description.inputCount == 0 ||
        description.hiddenCount == 0 || description.outputCount == 0) {
        throw BrainDescriptionError("Brain description is missing one of its four counts");
    }
    return description;
}

std::vector<std::string> compareBrainDescriptions(const BrainDescription& expected,
                                                  const BrainDescription& actual) {
    std::vector<std::string> differences;
    // Shape first: it explains most of what follows, and a reader who stops
    // after one line should get the line that matters.
    const auto compareCount = [&](const std::string_view what, const std::uint32_t left,
                                  const std::uint32_t right) {
        if (left != right) {
            differences.emplace_back(std::string{what} + " is " + std::to_string(right) +
                                     ", this build has " + std::to_string(left));
        }
    };
    compareCount("weight count", expected.weightCount, actual.weightCount);
    compareCount("input count", expected.inputCount, actual.inputCount);
    compareCount("hidden count", expected.hiddenCount, actual.hiddenCount);
    if (expected.hiddenLayers != actual.hiddenLayers) {
        const auto describe = [](const std::vector<std::uint32_t>& layers) {
            std::string text;
            for (const std::uint32_t width : layers) {
                text += text.empty() ? "" : "+";
                text += std::to_string(width);
            }
            return text.empty() ? std::string{"none"} : text;
        };
        differences.emplace_back("hidden layers are " + describe(actual.hiddenLayers) +
                                 ", this build has " + describe(expected.hiddenLayers));
    }
    // A file written with an older description has no activations at all, and
    // that is not a difference: tanh everywhere is what such a file meant.
    if (!actual.hiddenActivations.empty() &&
        expected.hiddenActivations != actual.hiddenActivations) {
        const auto describe = [](const std::vector<std::string>& squashes) {
            std::string text;
            for (const std::string& squash : squashes) {
                text += text.empty() ? "" : "+";
                text += squash;
            }
            return text.empty() ? std::string{"none"} : text;
        };
        differences.emplace_back("hidden activations are " + describe(actual.hiddenActivations) +
                                 ", this build has " + describe(expected.hiddenActivations));
    }
    compareCount("output count", expected.outputCount, actual.outputCount);
    compareBlockList("input", expected.inputs, actual.inputs, differences);
    compareBlockList("output", expected.outputs, actual.outputs, differences);
    compareBlockList("weight", expected.weights, actual.weights, differences);
    return differences;
}

} // namespace vkexp::neuro
