#version 450
#extension GL_GOOGLE_include_directive : require

#include "neuro/brain_kernel.glsl"
#include "simulation/agent_layout.glsl"
#include "lattice/lattice_view.glsl"
#include "lattice/transparency_kernel.glsl"

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
    // washed out by the ones behind it. The distance is measured in
    // half-diagonals of the lattice and in a straight line from the eye --
    // linear, and on the same scale as the camera's own distance control.
    // Window depth would be the cheaper thing to reach for and is what the
    // published form uses; see TransparencyKernel.inl for why it cannot work
    // with a near plane this close to the eye.
    const float boxRadius = max(0.5 * length(latticeExtent()), 1.0e-3);
    // The box is centred on the origin whatever the camera has been dragged to
    // look at, so how far the eye is from the origin is how far it is from the
    // lattice -- which is the scale the weight reads the fragment against.
    const float eyeDistance = length(view.camera.xyz) / boxRadius;
    const float fragmentDistance = length(fragWorld - view.camera.xyz) / boxRadius;
    const float weight = latticeTransparencyWeight(alpha, fragmentDistance, eyeDistance);

    outAccumulation = vec4(shaded * alpha, alpha) * weight;
    outRevealage = alpha;
}
