#ifndef VKEXP_STEP_PARAMETERS_GLSL
#define VKEXP_STEP_PARAMETERS_GLSL

// std430 mirror of vkexp::GpuStepParameters (144 bytes). Shared because all
// three passes of a step index the same per-step block; the C++ side pins the
// size and the offsets with static_assert.
//
// The block below is two vec4s rather than eight floats, and that is the whole
// of what keeps the two languages agreeing. std430 gives a struct the alignment
// of its widest member, so a struct of eight floats aligns to 4 here while
// `alignas(16)` aligns it to 16 in C++. With twenty-three scalars in front of
// it, those two rules put this block at offset 92 and at offset 96 -- a stride
// of 124 against a stride of 128. Index 0 then still very nearly works, so
// every run that submitted one step per batch looked correct and every run that
// batched read whole structs off the end of themselves. A vec4 aligns to 16 in
// both languages, so the agreement now survives the next field somebody adds.
struct FitnessWeights {
    // tracking reward, objective bonus, motor cost, refusal penalty
    vec4 shaping;
    // signal cost factor, three spare lanes
    vec4 costs;
};

float fitnessSignalCostFactor(FitnessWeights weights) { return weights.costs.x; }

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
    uint worldMode;
    uint buildIntervalTicks;
    uint wastedBuildTicks;
    float buildThreshold;
    float constructionCourseFill;
    uint constructionHeightLead;
    uint constructionSupportRadius;
    uint allowSideSupportedBlocks;
    uint resourceHeightLow;
    uint resourceHeightHigh;
    uint groundWidth;
    uint beaconSeed;
    FitnessWeights fitness;
};

#endif
