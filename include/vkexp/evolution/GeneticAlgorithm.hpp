#pragma once

#include "vkexp/neuro/NeuralNetwork.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace vkexp {

struct EvolutionSettings {
    std::size_t populationSize{512};
    std::size_t eliteCount{12};
    std::size_t tournamentSize{5};
    float crossoverProbability{0.65F};
    float mutationProbability{0.08F};
    float mutationStrength{0.18F};
    std::uint32_t seed{0xC0FFEEU};
    // How long one genome is. It follows from the brain plan the run is set up
    // with, so it is settled once at reset and never guessed at afterwards --
    // the population, the GPU buffer and every file all read it from here.
    std::size_t weightCount{neuro::maximumBrainShape.weightCount()};
};

struct Genome {
    neuro::Weights weights{};
};

// Whether two genomes can be exchanged at all. A population is interchangeable
// when it is as long as the plan reading it -- which is a question a file can
// now answer about itself, rather than one settled by everything sharing a
// single compiled-in length.
[[nodiscard]] inline bool genomeFits(const Genome& genome, const std::size_t weightCount) {
    return genome.weights.size() == weightCount;
}

// Blends each genome's score toward the mean of the group it was evaluated with.
// Groups are contiguous blocks of `groupSize` genomes -- the same blocks the
// driver spawns into one logical world -- and the last one may be short.
//
// `share` is 0 for pure individual selection and 1 for scoring a whole world
// together, which is what makes a signal that only helps a neighbour pay its
// sender back. At 0 the input is returned unchanged, so turning the option off
// is not merely equivalent to the old behaviour but literally is it.
[[nodiscard]] std::vector<float> shareFitnessWithinGroups(std::span<const float> fitness,
                                                          std::size_t groupSize, float share);

struct GenerationSummary {
    std::uint64_t generation{};
    float bestFitness{};
    float meanFitness{};
    float medianFitness{};
    std::size_t championIndex{};
};

class GeneticAlgorithm {
public:
    explicit GeneticAlgorithm(EvolutionSettings settings = {});

    void reset();
    [[nodiscard]] GenerationSummary evolve(std::span<const float> fitness);

    // Replaces the population, e.g. when resuming from a genome archive. The
    // count must match populationSize so buffer sizes stay valid.
    void setPopulation(std::span<const Genome> genomes, std::uint64_t generation);

    [[nodiscard]] const std::vector<Genome>& population() const { return population_; }
    [[nodiscard]] std::uint64_t generation() const { return generation_; }
    [[nodiscard]] const EvolutionSettings& settings() const { return settings_; }

private:
    [[nodiscard]] std::size_t tournament(std::span<const float> fitness);

    EvolutionSettings settings_;
    std::mt19937 random_;
    std::vector<Genome> population_;
    std::uint64_t generation_{};
};

} // namespace vkexp
