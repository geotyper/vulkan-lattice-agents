#include "vkexp/simulation/StepParameters.hpp"

#include "vkexp/neuro/NeuralNetwork.hpp"
#include "vkexp/simulation/TrailKernel.hpp"
#include "vkexp/worlds/WorldScenario.hpp"

namespace vkexp {

GpuStepParameters packStepParameters(const SimulationStep& resolved,
                                     const StepParameterLayout& layout) {
    const ScenarioDefinition& scenario = scenarioDefinition(resolved.beaconScenario);
    return {resolved.deltaTime,
            resolved.worldRadius,
            resolved.thrust,
            resolved.turnAcceleration,
            resolved.linearDrag,
            resolved.angularDrag,
            resolved.sensorFieldOfView,
            beaconArrivalRadius(resolved),
            resolved.maximumSpeed,
            resolved.maximumAngularSpeed,
            resolved.lightSensorRange,
            resolved.lightExposure,
            resolved.collisionRestitution,
            resolved.contactStiffness,
            layout.gridCellSize,
            resolved.wallCollisionPenalty,
            layout.agentCount,
            neuro::packBrainLayout(scenario.brain),
            layout.trialsPerGenome,
            static_cast<std::uint32_t>(resolved.worldShape),
            layout.gridWidth,
            layout.gridCellsPerWorld,
            resolved.agentCollisionsEnabled ? 1U : 0U,
            resolved.agentLightEnabled ? 1U : 0U,
            static_cast<std::uint32_t>(resolved.beaconScenario),
            resolved.beaconPhase,
            resolved.beaconPhaseChanged ? 1U : 0U,
            scenario.beaconCount,
            resolved.trailCellSize,
            trail::kernel::trailSurvival(
                trail::kernel::trailDecayRateForHalfLife(resolved.trailHalfLife),
                resolved.deltaTime),
            resolved.trailDepositRate * resolved.deltaTime * trail::kernel::TrailFixedPointScale,
            resolved.beaconTrailDepositRate * resolved.deltaTime *
                trail::kernel::TrailFixedPointScale,
            layout.trailWidth,
            layout.trailCellsPerWorld,
            static_cast<std::uint32_t>(resolved.trailMode),
            layout.agentsPerWorld,
            packFitnessWeights(resolved.fitness),
            scenario.gpuParameters(resolved),
            static_cast<std::uint32_t>(resolved.neuronModel),
            scenario.obstacleCount,
            resolved.uniformBeaconColor ? 1U : 0U,
            layout.worldCount,
            resolved.puckTargetRadiusRatio,
            scenario.puck ? 1U : 0U,
            0U,
            0U};
}

} // namespace vkexp
