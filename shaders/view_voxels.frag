#version 450
#extension GL_GOOGLE_include_directive : require

#include "neuro/brain_kernel.glsl"
#include "simulation/agent_layout.glsl"
#include "lattice/lattice_view.glsl"

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragWorld;
layout(location = 2) in vec4 fragColour;

layout(location = 0) out vec4 outColour;

void main() {
    const uint mode = latticeViewMode();
    // Lines, both of them: there is no face to catch the light, and shading a
    // line by a made-up normal only makes it flicker as the camera turns.
    if (mode == LatticeViewModeBounds || mode == LatticeViewModeGoal) {
        outColour = fragColour;
        return;
    }
    outColour = vec4(latticeShade(fragColour.rgb, fragNormal, fragWorld), 1.0);
}
