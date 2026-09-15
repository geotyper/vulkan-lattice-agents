// Does a small change to a genome make a small change to the behaviour?
//
// Evolution needs that and nothing else: if a child's trajectory is unrelated to
// its parent's, fitness is noise with respect to the genome and there is nothing
// to select. A model can be lively, can build, can produce every statistic one
// wants, and still be unselectable for this one reason.
//
// So: one agent per world, so nothing but the genome differs; two populations
// from the same genome, one perturbed; identical starts; step both in lockstep
// and count how long they stay in the same cell facing the same way.
#include "vkexp/lattice/LatticeWorld.hpp"
#include "vkexp/neuro/BrainKernel.hpp"
#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/simulation/CpuLattice.hpp"
#include "vkexp/simulation/LatticeTypes.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace bk = vkexp::neuro::kernel;

namespace {

void run(const char* label, const vkexp::NeuronModel model, const float jitter,
         const std::uint32_t lateral = 0) {
    vkexp::SimulationStep settings{};
    settings.worldMode = vkexp::WorldMode::Construction;
    settings.latticeWidth = 32;
    settings.latticeHeight = 16;
    settings.latticeDepth = 32;
    settings.neuronModel = model;
    settings.lateralRecurrence = lateral;
    settings.hiddenLayers = {35, 0, 0};

    // One agent per world: no neighbours, no contention, nothing to blame but
    // the genome.
    const vkexp::lattice::PopulationLayout layout{32, 1, 1};
    const vkexp::neuro::BrainShape brain = vkexp::resolvedBrain(settings);
    const auto stride = static_cast<std::uint32_t>(brain.weightCount());

    std::mt19937 random{0xC0FFEEU};
    std::vector<float> weights;
    for (std::uint32_t genome = 0; genome < layout.genomeCount; ++genome) {
        const vkexp::neuro::Weights drawn = vkexp::neuro::randomWeights(brain, random, false);
        weights.insert(weights.end(), drawn.begin(), drawn.end());
    }
    std::vector<float> nudged = weights;
    std::normal_distribution<float> wobble{0.0F, jitter};
    for (float& gene : nudged) {
        gene += wobble(random);
    }

    const std::uint32_t cells = vkexp::latticeCellsPerWorld(settings);
    const auto makeSide = [&] {
        struct Side {
            std::vector<vkexp::AgentState> agents;
            std::vector<std::int32_t> structures;
            std::vector<std::int32_t> occupancy;
            std::vector<std::int32_t> claims;
        };
        Side side;
        side.agents = vkexp::lattice::makeInitialAgents(settings, layout);
        side.structures = vkexp::lattice::makeTerrain(settings, layout.worldCount());
        side.occupancy.assign(std::size_t(cells) * layout.worldCount(), 0);
        vkexp::lattice::buildOccupancy(side.agents, settings, layout, side.occupancy);
        side.claims.assign(side.occupancy.size(), 0);
        return side;
    };
    auto left = makeSide();
    auto right = makeSide();

    constexpr std::array<std::uint32_t, 8> marks{1, 5, 10, 25, 50, 100, 200, 300};
    std::size_t mark = 0;
    std::printf("  %-22s", label);
    for (std::uint32_t step = 0; step < 300; ++step) {
        vkexp::stepLatticeCpu({left.agents, left.occupancy, left.claims, weights, stride,
                               layout.agentsPerWorld, layout.trialsPerGenome, left.structures},
                              settings);
        vkexp::stepLatticeCpu({right.agents, right.occupancy, right.claims, nudged, stride,
                               layout.agentsPerWorld, layout.trialsPerGenome, right.structures},
                              settings);
        if (mark < marks.size() && step + 1 == marks[mark]) {
            int together = 0;
            for (std::size_t index = 0; index < left.agents.size(); ++index) {
                const vkexp::AgentState& a = left.agents[index];
                const vkexp::AgentState& b = right.agents[index];
                together += (a.cell.x == b.cell.x && a.cell.y == b.cell.y &&
                             a.cell.z == b.cell.z && a.cell.w == b.cell.w)
                                ? 1
                                : 0;
            }
            std::printf(" %3d%%", 100 * together / int(left.agents.size()));
            ++mark;
        }
    }
    // What each genome built, on its own, in its own world. A population whose
    // genomes all score the same is one selection cannot grade, however well
    // they all score -- so the spread matters as much as the mean.
    std::vector<int> perGenome(layout.worldCount(), 0);
    for (std::size_t cell = 0; cell < left.structures.size(); ++cell) {
        if (left.structures[cell] > 0) {
            ++perGenome[cell / cells];
        }
    }
    double sum = 0.0;
    int low = perGenome.front();
    int high = perGenome.front();
    for (const int built : perGenome) {
        sum += built;
        low = built < low ? built : low;
        high = built > high ? built : high;
    }
    const double mean = sum / double(perGenome.size());
    double variance = 0.0;
    for (const int built : perGenome) {
        variance += (built - mean) * (built - mean);
    }
    variance /= double(perGenome.size());
    const double spread = mean > 0.0 ? std::sqrt(variance) / mean : 0.0;
    std::printf("   | built mean %5.1f  min %3d  max %3d  spread %.2f\n", mean, low, high,
                spread);
}

} // namespace

int main() {
    for (const float jitter : {0.002F, 0.02F}) {
        std::printf("\njitter %.3f per gene (spread is 0.55)\n", double(jitter));
        std::printf("  %-22s %s\n", "model", "  t1   t5  t10  t25  t50 t100 t200 t300");
        run("time constant", vkexp::NeuronModel::TimeConstant, jitter);
        run("gated", vkexp::NeuronModel::Gated, jitter);
        run("spiking", vkexp::NeuronModel::Spiking, jitter);
        run("adaptive", vkexp::NeuronModel::Adaptive, jitter);
        run("oscillator", vkexp::NeuronModel::Oscillator, jitter);
        run("time constant + lateral", vkexp::NeuronModel::TimeConstant, jitter, 1U);
    }
    return 0;
}
