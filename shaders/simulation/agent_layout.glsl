#ifndef VKEXP_AGENT_LAYOUT_GLSL
#define VKEXP_AGENT_LAYOUT_GLSL

// The std430 agent record, mirroring vkexp::AgentState.
//
// Four shaders read this layout -- the step, the grid build, the trail deposit
// and the vertex stage -- and each used to declare its own copy. Four copies of
// one contract is exactly what rule 3c exists to prevent: the hidden-state block
// below would have had to be added to all four by hand, and a shader that missed
// it would not fail to compile, it would read the wrong fields.
//
// Needs neuro/brain_kernel.glsl included first, for BrainHiddenNeuronCapacity.

// One vec4 per four neurons. Derived, so the block follows the preset rather
// than being resized by hand when the hidden layer changes width.
const uint AgentHiddenVectorCount = (BrainHiddenNeuronCapacity + 3u) / 4u;

struct Agent {
    vec4 pose;
    vec4 motion;
    vec4 signal;
    vec4 target;
    vec4 metrics;
    vec4 penalties;
    vec4 internal;
    vec4 wallTouch0;
    vec4 wallTouch1;
    vec4 agentTouch0;
    vec4 agentTouch1;
    // Continuous-time state of every hidden neuron, carried between steps. Zero
    // at the start of a generation, which is the whole of the reset semantics:
    // an agent begins each trial with no memory of the last one.
    vec4 hidden[AgentHiddenVectorCount];
};

float agentHiddenState(Agent agent, uint neuron) {
    return agent.hidden[neuron >> 2u][neuron & 3u];
}

#endif
