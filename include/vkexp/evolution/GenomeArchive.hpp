#pragma once

#include "vkexp/evolution/GeneticAlgorithm.hpp"
#include "vkexp/neuro/BrainDescription.hpp"
#include "vkexp/neuro/NeuralNetwork.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace vkexp {

class GenomeArchiveError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Provenance stored alongside the weights so a resumed run can be traced back
// to the experiment that produced it.
struct GenomeArchiveMetadata {
    std::uint64_t generation{};
    // Which lattice the weights were trained against, as the beacon seed that
    // placed it. It occupies the slot the scenario number used to, and says the
    // same kind of thing: the one number that distinguishes one world from
    // another now that there is only one kind of world.
    std::uint32_t beaconSeed{};
    std::uint32_t seed{};
    float bestFitness{};
    float meanFitness{};
    std::uint32_t brainInputCount{};
    // Hidden neurons in total, and the layer plan they are divided into, packed
    // six bits per layer the way the shader receives it. The total is kept
    // separate because it is what a reader wants for "how big is this brain",
    // and because a file written before plans existed has only that.
    std::uint32_t brainHiddenCount{};
    std::uint32_t brainOutputCount{};
    std::uint32_t brainHiddenLayers{};
};

struct GenomeArchive {
    GenomeArchiveMetadata metadata;
    std::vector<Genome> genomes;
    // The structure the weights were laid out under, as the file states it. A
    // version 1 archive carries none, and this is then what this build would
    // have written for the recorded shape -- see `describedStructure`.
    neuro::BrainDescription description;
    // Whether that came from the file or was assumed. A file that does not say
    // what its weights mean can still be loaded, but nothing has been checked
    // beyond their number.
    bool describedStructure{};
};

// 2 added the structure block; 3 added the hidden layer plan, because a genome
// is now as long as its own network rather than one length everything shares.
// 4 is the lattice. Nothing about the format changed -- the scenario slot became
// the beacon seed, which is the same width -- but everything the numbers mean
// did: the input vector is a neighbourhood rather than a photoreceptor array,
// and weights from a 2D run would load and steer nothing. So 4 is the oldest
// readable version too, which is the only honest thing to say about a file whose
// every weight now addresses a different sensor.
// 5 is the body frame, and is the oldest readable version for the same reason 4
// was: the output block changed from five world-axis drives to a turn and an
// action, the sensor block lost the three heading channels, and the
// neighbourhood shrank to the seventeen cells in front of the agent -- so a
// weight from an older file addresses a slot that is no longer there. A file that loads and
// steers nothing is worse than one that is refused.
// 6 widened a layer's squash from two bits to three, which moved where layers
// two and three keep theirs. Unlike 4 and 5 this does not invalidate a weight:
// the widths are the low eighteen bits under either encoding, so a version 5
// file still lays out exactly the network it always did, and the reader
// re-encodes the plan word on the way in. 5 therefore stays readable.
// 7 is the adaptation block: two genes per hidden neuron, appended after the
// gate block, carried whatever model is selected. A version 6 file is shorter
// than the network this build lays out for the same plan, and its tail is not
// missing data but data that was never there -- so it is refused, the way 4 and
// 5 were, rather than padded with zeroes that would read as a bump of one.
inline constexpr std::uint32_t genomeArchiveVersion = 7;
inline constexpr std::uint32_t genomeArchiveOldestVersion = 7;

// Writes a versioned little-endian archive, followed by the JSON structure the
// weights are laid out under. Six numbers in a header can say that a file no
// longer fits; only the structure can say which sensor moved.
void saveGenomeArchive(const std::filesystem::path& path, std::span<const Genome> genomes,
                       const GenomeArchiveMetadata& metadata);

[[nodiscard]] GenomeArchive loadGenomeArchive(const std::filesystem::path& path);

} // namespace vkexp
