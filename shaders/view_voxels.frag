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
    if (latticeViewMode() == LatticeViewModeBounds) {
        outColour = fragColour;
        return;
    }
    outColour = vec4(latticeShade(fragColour.rgb, fragNormal, fragWorld), 1.0);
}
