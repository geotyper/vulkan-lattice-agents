#version 450
#extension GL_GOOGLE_include_directive : require

#include "neuro/brain_kernel.glsl"
#include "simulation/agent_layout.glsl"
#include "lattice/lattice_view.glsl"

// Weighted blended order-independent transparency (McGuire & Bavoil). The
// reason it is here rather than a back-to-front sort is the same reason the
// simulation settles a contested cell with atomicMin: the answer must not
// depend on the order the work happened to be issued in. A sort would mean
// reading the agents back to the host every frame to order them, which is a
// stall for a picture, and it would still be wrong the moment two voxels shared
// a depth. This accumulates a weighted average instead, which every fragment
// contributes to commutatively, and resolves once.
//
// What it costs is exactness: the result is an average weighted by depth and
// alpha, not the true over-operator. For a box of separated unit cubes at one
// alpha that is a very good approximation, and the alternative is a sorted pass
// that would be exactly right about an ordering nobody can observe.

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragWorld;
layout(location = 2) in vec4 fragColour;

layout(location = 0) out vec4 outAccumulation;
layout(location = 1) out float outRevealage;

void main() {
    const vec3 shaded = latticeShade(fragColour.rgb, fragNormal, fragWorld);
    const float alpha = clamp(latticeViewMode() == LatticeViewModeTrail ? fragColour.a
                                                                        : view.tint.x,
                              0.0, 1.0);

    // Nearer fragments weigh more, so a voxel at the front of the box is not
    // washed out by the ones behind it. The constants are the paper's: what
    // matters is that the weight falls off fast with depth and never reaches
    // zero, which is what keeps a deep stack from saturating.
    const float depth = gl_FragCoord.z;
    const float weight =
        clamp(pow(min(1.0, alpha * 10.0) + 0.01, 3.0) * 1e8 * pow(1.0 - depth * 0.9, 3.0),
              1e-2, 3e3);

    outAccumulation = vec4(shaded * alpha, alpha) * weight;
    outRevealage = alpha;
}
