#ifndef VKEXP_STEP_PARAMETERS_GLSL
#define VKEXP_STEP_PARAMETERS_GLSL

// std430 mirror of vkexp::GpuStepParameters (224 bytes). Shared because both
// agent_step and trail_deposit index the same per-step block; the C++ side pins
// the size and the offsets with static_assert.

#include "worlds/scenario_params.glsl"

struct FitnessWeights {
    float objectiveBonus;
    float motorCostWeight;
    float signalCostFactor;
    float energyDrain;
    float trackingReward;
    float reserved0;
    float reserved1;
    float reserved2;
};

struct StepParameters {
    float deltaTime;
    float worldRadius;
    float thrust;
    float turnAcceleration;
    float linearDrag;
    float angularDrag;
    float sensorFieldOfView;
    float arrivalRadius;
    float maximumSpeed;
    float maximumAngularSpeed;
    float lightSensorRange;
    float lightExposure;
    float collisionRestitution;
    float contactStiffness;
    float gridCellSize;
    float wallCollisionPenalty;
    uint agentCount;
    uint brainLayout;
    uint trialsPerGenome;
    uint worldShape;
    uint gridWidth;
    uint gridCellsPerWorld;
    uint agentCollisionsEnabled;
    uint agentLightEnabled;
    uint beaconScenario;
    uint beaconPhase;
    uint beaconPhaseChanged;
    uint beaconCount;
    float trailCellSize;
    float trailSurvival;
    float trailDeposit;
    float beaconTrailDeposit;
    uint trailWidth;
    uint trailCellsPerWorld;
    uint trailMode;
    uint agentsPerWorld;
    FitnessWeights fitness;
    ScenarioParameters scenario;
    uint neuronModel;
    uint obstacleCount;
    uint uniformBeaconColor;
    uint worldCount;
    float puckTargetRadiusRatio;
    uint puckEnabled;
    uint reserved3;
    uint reserved4;
};

#endif
