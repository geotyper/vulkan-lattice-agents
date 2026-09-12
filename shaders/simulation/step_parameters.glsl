#ifndef VKEXP_STEP_PARAMETERS_GLSL
#define VKEXP_STEP_PARAMETERS_GLSL

// std430 mirror of vkexp::GpuStepParameters (112 bytes). Shared because all
// three passes of a step index the same per-step block; the C++ side pins the
// size and the offsets with static_assert.

struct FitnessWeights {
    float trackingReward;
    float objectiveBonus;
    float motorCostWeight;
    float refusalPenalty;
    float signalCostFactor;
    float reserved0;
    float reserved1;
    float reserved2;
};

struct StepParameters {
    float deltaTime;
    float moveThreshold;
    uint agentCount;
    uint brainLayout;
    uint trialsPerGenome;
    uint agentsPerWorld;
    uint worldCount;
    uint latticeWidth;
    uint latticeHeight;
    uint latticeDepth;
    uint cellsPerWorld;
    uint neighborhood;
    uint maximumDistance;
    uint beaconContactRadius;
    uint neuronModel;
    // Three hidden layer widths packed six bits each, and the genome stride,
    // which outgrew the twelve bits it used to share with the layout word.
    uint brainHiddenLayers;
    uint brainGenomeStride;
    uint reserved0;
    uint reserved1;
    uint reserved2;
    FitnessWeights fitness;
};

#endif
