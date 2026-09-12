#ifndef VKEXP_AGENT_LAYOUT_GLSL
#define VKEXP_AGENT_LAYOUT_GLSL

// The std430 agent record, mirroring vkexp::AgentState.
//
// Three shaders read this layout -- the step, the resolve and, once there is
// one, the 3D view -- and each used to declare its own copy. Copies of one
// contract are exactly what this file exists to prevent: the hidden-state block
// below would have had to be added to all of them by hand, and a shader that
// missed it would not fail to compile, it would read the wrong fields.
//
// Needs neuro/brain_kernel.glsl included first, for BrainHiddenNeuronCapacity.

// One vec4 per four neurons. Derived, so the block follows the preset rather
// than being resized by hand when the hidden layer changes width.
const uint AgentHiddenVectorCount = (BrainHiddenNeuronCapacity + 3u) / 4u;

struct Agent {
    ivec4 cell;   // x, y, z, heading as a neighbour index
    ivec4 intent; // desired x, y, z, and whether the last move was refused
    ivec4 beacon; // this world's beacon x, y, z, and which world that is
    vec4 signal;  // broadcast level in .x, three lanes spare
    vec4 metrics; // best nearness, contacts, effort, refusals
    vec4 memory;  // the two recurrent cells in .xy, two lanes spare
    // Continuous-time state of every hidden neuron, carried between steps. Zero
    // at the start of a generation, which is the whole of the reset semantics:
    // an agent begins each trial with no memory of the last one.
    vec4 hidden[AgentHiddenVectorCount];
};

float agentHiddenState(Agent agent, uint neuron) {
    return agent.hidden[neuron >> 2u][neuron & 3u];
}

#endif
