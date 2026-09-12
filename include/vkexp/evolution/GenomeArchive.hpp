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
inline constexpr std::uint32_t genomeArchiveVersion = 4;
inline constexpr std::uint32_t genomeArchiveOldestVersion = 4;

// Writes a versioned little-endian archive, followed by the JSON structure the
// weights are laid out under. Six numbers in a header can say that a file no
// longer fits; only the structure can say which sensor moved.
void saveGenomeArchive(const std::filesystem::path& path, std::span<const Genome> genomes,
                       const GenomeArchiveMetadata& metadata);

[[nodiscard]] GenomeArchive loadGenomeArchive(const std::filesystem::path& path);

} // namespace vkexp
