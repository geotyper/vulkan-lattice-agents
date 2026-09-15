#include "vkexp/simulation/StepParameters.hpp"

#include "vkexp/neuro/NeuralNetwork.hpp"

namespace vkexp {

GpuStepParameters packStepParameters(const SimulationStep& settings,
                                     const StepParameterLayout& layout) {
    const neuro::BrainShape brain = resolvedBrain(settings);
    return {settings.deltaTime,
            settings.turnThreshold,
            layout.agentCount,
            neuro::packBrainLayout(brain),
            layout.trialsPerGenome,
            layout.agentsPerWorld,
            layout.worldCount,
            settings.latticeWidth,
            settings.latticeHeight,
            settings.latticeDepth,
            latticeCellsPerWorld(settings),
            static_cast<std::uint32_t>(settings.neighborhood),
            latticeMaximumDistance(settings),
            settings.beaconContactRadius,
            static_cast<std::uint32_t>(settings.neuronModel),
            brain.packedLayers(),
            static_cast<std::uint32_t>(brain.weightCount()),
            static_cast<std::uint32_t>(settings.worldMode),
            settings.buildIntervalTicks,
            settings.wastedBuildTicks,
            settings.buildThreshold,
            settings.constructionCourseFill,
            settings.constructionHeightLead,
            settings.constructionSupportRadius,
            // Forced on in the chasm: without a cantilever the far half cannot
            // be reached at all, and a world whose objective is unreachable is
            // worse than no world.
            settings.worldMode == WorldMode::Chasm ? 1U : settings.allowSideSupportedBlocks,
            settings.resourceHeightLow,
            settings.resourceHeightHigh,
            latticeGroundWidth(settings),
            settings.beaconSeed,
            packFitnessWeights(settings.fitness)};
}

} // namespace vkexp
