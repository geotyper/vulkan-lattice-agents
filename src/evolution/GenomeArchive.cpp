#include "vkexp/evolution/GenomeArchive.hpp"

#include <array>
#include <cstddef>
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
    std::uint32_t brainHiddenLayers{};
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

static_assert(sizeof(ArchiveHeader) == 64);
// The version-1 compatibility case in the unit tests reaches into a file by
// byte offset, because there is no writer for the old format any more. Pinning
// the two offsets it uses here means a field inserted above fails the build
// rather than making that test quietly rewrite the wrong four bytes.
static_assert(offsetof(ArchiveHeader, version) == 4);
static_assert(offsetof(ArchiveHeader, descriptionBytes) == 60);

} // namespace

void saveGenomeArchive(const std::filesystem::path& path, const std::span<const Genome> genomes,
                       const GenomeArchiveMetadata& metadata) {
    if (genomes.empty()) {
        throw GenomeArchiveError("Refusing to write an empty genome archive");
    }
    // One length for the file, because the header states one. A population of
    // mixed lengths is not a population under any plan, and writing it would
    // produce a file that reads back as something nobody ran.
    for (const Genome& genome : genomes) {
        if (genome.weights.size() != genomes.front().weights.size()) {
            throw GenomeArchiveError("Refusing to write genomes of different lengths");
        }
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
    const neuro::BrainShape saved =
        neuro::brainShape(neuro::kernel::brainPackLayout(metadata.brainInputCount,
                                                         metadata.brainOutputCount),
                          metadata.brainHiddenLayers);
    const neuro::BrainDescription description = neuro::describeBrain(saved);
    const std::string structure = neuro::brainDescriptionToJson(description);
    const ArchiveHeader header{archiveMagic,
                               genomeArchiveVersion,
                               static_cast<std::uint32_t>(genomes.size()),
                               static_cast<std::uint32_t>(genomes.front().weights.size()),
                               metadata.brainInputCount,
                               metadata.brainHiddenCount,
                               metadata.brainOutputCount,
                               metadata.brainHiddenLayers,
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
    if (header.weightCount == 0 || header.weightCount > neuro::Topology::maximumWeightCount) {
        throw GenomeArchiveError("Genome archive claims " + std::to_string(header.weightCount) +
                                 " weights per genome, which no plan this build can run "
                                 "produces");
    }
    if (header.genomeCount == 0) {
        throw GenomeArchiveError("Genome archive contains no genomes: " + path.string());
    }

    GenomeArchive archive;
    archive.metadata = {header.generation,       header.scenario,        header.seed,
                        header.bestFitness,      header.meanFitness,     header.brainInputCount,
                        header.brainHiddenCount, header.brainOutputCount,
                        header.brainHiddenLayers};

    // What this build would lay out for the network the file records. The layer
    // plan comes from the file's own structure block when it has one, because a
    // file is allowed to hold a brain this run is not currently set up for --
    // that is the whole point of writing the plan down. What is compared is
    // everything else: whether *this build* lays that same network out the same
    // way. A sensor added since the file was written moves a block, and the
    // message names it.
    // A file that names its layers is taken at its word; one that does not is a
    // file from before plans existed, and that is one hidden layer.
    neuro::BrainShape recorded{header.brainInputCount, header.brainHiddenCount,
                               header.brainOutputCount};
    if (header.brainHiddenLayers != 0) {
        recorded = neuro::brainShape(
            neuro::kernel::brainPackLayout(header.brainInputCount, header.brainOutputCount),
            header.brainHiddenLayers);
    }
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
        if (!recorded.fitsCapacity()) {
            throw GenomeArchiveError("Genome archive " + path.string() +
                                     " holds a brain plan this build has no room for");
        }
        const std::vector<std::string> differences =
            neuro::compareBrainDescriptions(neuro::describeBrain(recorded), stored);
        if (!differences.empty()) {
            std::string message = "Genome archive " + path.string() +
                                  " describes a different network than this build:";
            for (const std::string& difference : differences) {
                message += "\n  - " + difference;
            }
            throw GenomeArchiveError(message);
        }
        archive.description = stored;
    } else {
        archive.description = neuro::describeBrain(recorded);
    }
    // And the length has to match the network the file says it holds, or the two
    // halves of the file disagree about what it is.
    if (archive.describedStructure && header.weightCount != recorded.weightCount()) {
        throw GenomeArchiveError("Genome archive " + path.string() + " stores " +
                                 std::to_string(header.weightCount) +
                                 " weights per genome, but the brain it describes needs " +
                                 std::to_string(recorded.weightCount()));
    }
    archive.genomes.assign(header.genomeCount, Genome{neuro::Weights(header.weightCount, 0.0F)});
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
