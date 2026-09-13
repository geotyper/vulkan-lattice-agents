#include "vkexp/simulation/StepParameters.hpp"

#include "vkexp/neuro/NeuralNetwork.hpp"

namespace vkexp {

GpuStepParameters packStepParameters(const SimulationStep& settings,
                                     const StepParameterLayout& layout) {
    const neuro::BrainShape brain = resolvedBrain(settings);
    return {settings.deltaTime,
            settings.moveThreshold,
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
            settings.buildThreshold,
            settings.constructionCourseFill,
            settings.constructionHeightLead,
            settings.allowSideSupportedBlocks,
            packFitnessWeights(settings.fitness)};
}

} // namespace vkexp
