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
    std::uint32_t scenario{};
    std::uint32_t seed{};
    float bestFitness{};
    float meanFitness{};
    std::uint32_t brainInputCount{};
    std::uint32_t brainHiddenCount{};
    std::uint32_t brainOutputCount{};
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

// 2 added the structure block. Version 1 files still load: their weights are
// laid out the same way, they simply do not say so.
inline constexpr std::uint32_t genomeArchiveVersion = 2;
inline constexpr std::uint32_t genomeArchiveOldestVersion = 1;

// Writes a versioned little-endian archive, followed by the JSON structure the
// weights are laid out under. Six numbers in a header can say that a file no
// longer fits; only the structure can say which sensor moved.
void saveGenomeArchive(const std::filesystem::path& path, std::span<const Genome> genomes,
                       const GenomeArchiveMetadata& metadata);

[[nodiscard]] GenomeArchive loadGenomeArchive(const std::filesystem::path& path);

} // namespace vkexp
