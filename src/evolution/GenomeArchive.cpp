#include "vkexp/evolution/GenomeArchive.hpp"

#include <array>
#include <numeric>
#include <string>
#include <bit>
#include <cstring>
#include <fstream>

namespace vkexp {
namespace {

// Little-endian only for now; the version field lets a future reader adapt.
static_assert(std::endian::native == std::endian::little,
              "Genome archives are written little-endian");

constexpr std::array<char, 4> archiveMagic{'V', 'K', 'N', 'G'};

struct ArchiveHeader {
    std::array<char, 4> magic{};
    std::uint32_t version{};
    std::uint32_t genomeCount{};
    std::uint32_t weightCount{};
    std::uint32_t brainInputCount{};
    std::uint32_t brainHiddenCount{};
    std::uint32_t brainOutputCount{};
    std::uint32_t scenario{};
    std::uint64_t generation{};
    std::uint32_t seed{};
    float bestFitness{};
    float meanFitness{};
    // Was a reserved word in version 1, which always wrote zero. It is the
    // length of the JSON structure block that follows the header in version 2,
    // so a version 1 file reads as "no structure stated" without a special case.
    std::uint32_t descriptionBytes{};
};

static_assert(sizeof(ArchiveHeader) == 56);

} // namespace

void saveGenomeArchive(const std::filesystem::path& path, const std::span<const Genome> genomes,
                       const GenomeArchiveMetadata& metadata) {
    if (genomes.empty()) {
        throw GenomeArchiveError("Refusing to write an empty genome archive");
    }
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code directoryError;
        std::filesystem::create_directories(parent, directoryError);
    }
    std::ofstream stream{path, std::ios::binary | std::ios::trunc};
    if (!stream) {
        throw GenomeArchiveError("Unable to open genome archive for writing: " + path.string());
    }
    const neuro::BrainDescription description = neuro::describeBrain(
        {metadata.brainInputCount, metadata.brainHiddenCount, metadata.brainOutputCount});
    const std::string structure = neuro::brainDescriptionToJson(description);
    const ArchiveHeader header{archiveMagic,
                               genomeArchiveVersion,
                               static_cast<std::uint32_t>(genomes.size()),
                               static_cast<std::uint32_t>(neuro::Topology::weightCount),
                               metadata.brainInputCount,
                               metadata.brainHiddenCount,
                               metadata.brainOutputCount,
                               metadata.scenario,
                               metadata.generation,
                               metadata.seed,
                               metadata.bestFitness,
                               metadata.meanFitness,
                               static_cast<std::uint32_t>(structure.size())};
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    stream.write(structure.data(), static_cast<std::streamsize>(structure.size()));
    for (const Genome& genome : genomes) {
        stream.write(reinterpret_cast<const char*>(genome.weights.data()),
                     static_cast<std::streamsize>(genome.weights.size() * sizeof(float)));
    }
    stream.flush();
    if (!stream) {
        throw GenomeArchiveError("Failed while writing genome archive: " + path.string());
    }
}

GenomeArchive loadGenomeArchive(const std::filesystem::path& path) {
    std::ifstream stream{path, std::ios::binary};
    if (!stream) {
        throw GenomeArchiveError("Unable to open genome archive: " + path.string());
    }
    ArchiveHeader header{};
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!stream || stream.gcount() != static_cast<std::streamsize>(sizeof(header))) {
        throw GenomeArchiveError("Genome archive is truncated: " + path.string());
    }
    if (header.magic != archiveMagic) {
        throw GenomeArchiveError("Not a genome archive: " + path.string());
    }
    if (header.version < genomeArchiveOldestVersion || header.version > genomeArchiveVersion) {
        throw GenomeArchiveError("Unsupported genome archive version " +
                                 std::to_string(header.version) + " in " + path.string());
    }
    if (header.weightCount != neuro::Topology::weightCount) {
        throw GenomeArchiveError("Genome archive stores " + std::to_string(header.weightCount) +
                                 " weights per genome, this build expects " +
                                 std::to_string(neuro::Topology::weightCount));
    }
    if (header.genomeCount == 0) {
        throw GenomeArchiveError("Genome archive contains no genomes: " + path.string());
    }

    GenomeArchive archive;
    archive.metadata = {header.generation,       header.scenario,        header.seed,
                        header.bestFitness,      header.meanFitness,     header.brainInputCount,
                        header.brainHiddenCount, header.brainOutputCount};

    // What this build would lay out for the shape the file records. Everything
    // below is a comparison against this, so the message can name the block that
    // moved rather than the number that no longer matches.
    const neuro::BrainDescription expected = neuro::describeBrain(
        {header.brainInputCount, header.brainHiddenCount, header.brainOutputCount});
    archive.description = expected;
    archive.describedStructure = header.descriptionBytes != 0;
    if (archive.describedStructure) {
        std::string structure(header.descriptionBytes, '\0');
        stream.read(structure.data(), static_cast<std::streamsize>(structure.size()));
        if (!stream || stream.gcount() != static_cast<std::streamsize>(structure.size())) {
            throw GenomeArchiveError("Genome archive is truncated: " + path.string());
        }
        neuro::BrainDescription stored;
        try {
            stored = neuro::parseBrainDescription(structure);
        } catch (const neuro::BrainDescriptionError& error) {
            throw GenomeArchiveError("Genome archive " + path.string() +
                                     " has an unreadable structure block: " + error.what());
        }
        const std::vector<std::string> differences =
            neuro::compareBrainDescriptions(expected, stored);
        if (!differences.empty()) {
            std::string message = "Genome archive " + path.string() +
                                  " describes a different network than this build:";
            for (const std::string& difference : differences) {
                message += "\n  - " + difference;
            }
            throw GenomeArchiveError(message);
        }
        archive.description = stored;
    }
    archive.genomes.resize(header.genomeCount);
    for (Genome& genome : archive.genomes) {
        stream.read(reinterpret_cast<char*>(genome.weights.data()),
                    static_cast<std::streamsize>(genome.weights.size() * sizeof(float)));
        if (!stream || stream.gcount() !=
                           static_cast<std::streamsize>(genome.weights.size() * sizeof(float))) {
            throw GenomeArchiveError("Genome archive is truncated: " + path.string());
        }
    }
    return archive;
}

} // namespace vkexp
