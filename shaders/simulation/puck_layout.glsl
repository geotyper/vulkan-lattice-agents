#ifndef VKEXP_PUCK_LAYOUT_GLSL
#define VKEXP_PUCK_LAYOUT_GLSL

// One per logical world. Mirrors PuckState in
// include/vkexp/simulation/AgentTypes.hpp (32 bytes, std430). Declared once and
// included by everything that touches a puck, so the layout cannot drift the
// way the agent record's did.
//
// pose:   x, y, radius, how far from the middle it was placed
// motion: vx, vy, the highest rung reached so far, unused
struct Puck {
    vec4 pose;
    vec4 motion;
};

#endif
