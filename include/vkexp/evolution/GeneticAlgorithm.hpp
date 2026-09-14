#pragma once

#include "vkexp/neuro/NeuralNetwork.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace vkexp {

// How a fresh genome is drawn, which turns out to decide what a run looks like
// before selection has done anything at all.
//
// Saturating draws every gene at one width, so a neuron summing seventy eight
// inputs lands on tanh's flat part and its output is pinned near +-1. The
// population starts maximally opinionated at random. That is what the thresholds
// this project ships were tuned against -- a build gate at 0.55 is only
// reachable at all by an output that saturates -- and over four generations of
// construction it wins on every model, spectacularly so under spiking, whose
// outputs rest at zero and spike: walk by default, act on a spike.
//
// FanIn draws each block at a width that follows its own fan-in, the standard
// answer, so outputs start spread rather than pinned. It needs both thresholds
// scaled down with it -- at the shipped 0.25 and 0.55 nothing ever turns and
// nothing ever builds, and the whole population beaches itself against the wall
// of the world. Rescaled to 0.05 and 0.05 it draws level with saturating for
// tanh models and loses badly under spiking.
//
// Both are kept because four generations is a measurement of where a run starts
// and not of where it can get to, and the case for FanIn is about the second: a
// saturated weight has to cross the whole flat part before its output moves at
// all, so mutation either does nothing or flips a sign, while an unsaturated one
// moves a little for a little. Which matters more is a long run's question.
enum class WeightInit : std::uint32_t { Saturating = 0, FanIn = 1 };

struct EvolutionSettings {
    std::size_t populationSize{512};
    std::size_t eliteCount{12};
    std::size_t tournamentSize{5};
    float crossoverProbability{0.65F};
    float mutationProbability{0.08F};
    float mutationStrength{0.18F};
    std::uint32_t seed{0xC0FFEEU};
    // The brain plan the run is set up with. The genome length follows from it
    // and is never stored beside it, so the two cannot disagree; the population,
    // the GPU buffer and every file read the length from here. The plan itself
    // is carried rather than only its length because drawing a fresh genome
    // needs to know where each weight block starts and how wide it is.
    neuro::BrainShape brain{neuro::maximumBrainShape};
    WeightInit weightInit{WeightInit::Saturating};

    [[nodiscard]] std::size_t weightCount() const { return brain.weightCount(); }
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
