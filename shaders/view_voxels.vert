#version 450
#extension GL_GOOGLE_include_directive : require

#include "neuro/brain_kernel.glsl"
#include "simulation/agent_layout.glsl"
#include "lattice/lattice_view.glsl"

layout(std430, set = 0, binding = 0) readonly buffer Agents {
    Agent agents[];
};
layout(std430, set = 0, binding = 1) readonly buffer TrailHistory {
    ivec4 trailSamples[];
};

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragWorld;
layout(location = 2) out vec4 fragColour;

// Collapsed to a point, which rasterises nothing. This is how a cell outside
// the visible slab is dropped: the vertex stage is where the instance is known,
// and discarding per fragment would shade it first.
void hide() {
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    fragNormal = vec3(0.0, 1.0, 0.0);
    fragWorld = vec3(0.0);
    fragColour = vec4(0.0);
}

void main() {
    const uint mode = latticeViewMode();

    if (mode == LatticeViewModeBounds) {
        const vec3 corner = latticeBoxEdgeVertex(uint(gl_VertexIndex));
        gl_Position = view.viewProjection * vec4(corner, 1.0);
        fragNormal = vec3(0.0, 1.0, 0.0);
        fragWorld = corner;
        fragColour = vec4(0.42, 0.52, 0.68, 1.0);
        return;
    }

    if (mode == LatticeViewModeTrail) {
        const uint count = uint(max(view.beacon.x, 0));
        const uint newest = uint(max(view.beacon.y, 0));
        const uint capacity = uint(max(view.beacon.z, 0));
        if (count == 0u || capacity == 0u) {
            hide();
            return;
        }
        const uint localAgent = uint(gl_InstanceIndex) / count;
        const uint age = uint(gl_InstanceIndex) % count;
        const uint agentIndex =
            uint(view.beacon.w) + localAgent * latticeViewAgentStride();
        const uint historySlot = (newest + capacity - (age % capacity)) % capacity;
        const ivec4 historySample = trailSamples[agentIndex * capacity + historySlot];
        if (historySample.w == 0 || !latticeCellVisible(historySample.xyz)) {
            hide();
            return;
        }

        vec3 normal;
        const vec3 local = latticeCubeVertex(uint(gl_VertexIndex), normal) * view.camera.w;
        const vec3 world = latticeCellCentre(historySample.xyz) + local;
        const float relativeAge = float(age) / float(max(count - 1u, 1u));
        const float fade = pow(max(1.0 - relativeAge, 0.0), 1.6);
        const uint genome = agentIndex / max(latticeViewAgentStride(), 1u);
        gl_Position = view.viewProjection * vec4(world, 1.0);
        fragNormal = normal;
        fragWorld = world;
        fragColour = vec4(latticeTrailColour(genome), view.tint.x * (0.08 + 0.92 * fade));
        return;
    }

    ivec3 cell;
    vec4 colour;
    if (mode == LatticeViewModeBeacon) {
        cell = view.beacon.xyz;
        // Not on the nearness ramp: the beacon is what nearness is measured
        // against, so giving it a place on that scale would be circular.
        colour = vec4(0.96, 0.34, 0.52, 1.0);
    } else {
        const uint agentIndex =
            uint(view.beacon.w) + uint(gl_InstanceIndex) * latticeViewAgentStride();
        const Agent agent = agents[agentIndex];
        cell = agent.cell.xyz;
        vec3 body = latticeNearnessColour(agent.metrics.x);
        // A refused move is the one thing a still picture cannot show, and it is
        // most of what a crowded lattice is doing, so it gets the loudest
        // channel there is.
        body = mix(body, vec3(0.92, 0.24, 0.18), agent.intent.w != 0 ? 0.55 : 0.0);
        // What it is broadcasting, added rather than mixed: a signal is
        // something an agent spends, so it should read as light coming off it.
        body += vec3(0.55, 0.48, 0.12) * clamp(agent.signal.x, 0.0, 1.0);
        colour = vec4(body, 1.0);
    }

    if (!latticeCellVisible(cell)) {
        hide();
        return;
    }

    vec3 normal;
    const vec3 local = latticeCubeVertex(uint(gl_VertexIndex), normal) * view.camera.w;
    const vec3 world = latticeCellCentre(cell) + local;
    gl_Position = view.viewProjection * vec4(world, 1.0);
    fragNormal = normal;
    fragWorld = world;
    fragColour = colour;
}
