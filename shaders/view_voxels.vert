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
layout(std430, set = 0, binding = 2) readonly buffer Structures {
    int structures[];
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

    if (mode == LatticeViewModeGoal) {
        const vec3 point =
            latticeGoalGuideVertex(uint(gl_VertexIndex), latticeCellCentre(view.beacon.xyz));
        gl_Position = view.viewProjection * vec4(point, 1.0);
        fragNormal = vec3(0.0, 1.0, 0.0);
        fragWorld = point;
        // The objective's own colour, dimmed: the guide has to be findable
        // without competing with the thing it points at.
        fragColour = vec4(0.72, 0.26, 0.40, 1.0);
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
    } else if (mode == LatticeViewModeStructure) {
        const uint local = uint(gl_InstanceIndex);
        const uint cellsPerWorld = uint(view.lattice.x * view.lattice.y * view.lattice.z);
        const int builder = structures[uint(view.beacon.w) * cellsPerWorld + local];
        if (builder == 0) {
            hide();
            return;
        }
        const uint width = uint(view.lattice.x);
        const uint height = uint(view.lattice.y);
        cell = ivec3(int(local % width), int((local / width) % height),
                     int(local / (width * height)));
        if (builder < 0) {
            // Terrain, not work. Grey and flat so that what the group built
            // reads against the ground it was built on rather than merging
            // with it -- and so that a chasm is a hole you can see.
            colour = vec4(0.30, 0.31, 0.33, 1.0);
        } else {
            const float elevation = float(cell.y + 1) / float(max(view.lattice.y, 1));
            const float maker = fract(float(builder) * 0.61803398875);
            const vec3 clay = vec3(0.46, 0.16, 0.07);
            const vec3 sun = vec3(1.00, 0.66, 0.20);
            vec3 block = mix(clay, sun, pow(elevation, 0.55));
            block *= 0.90 + 0.16 * maker;
            colour = vec4(block, 1.0);
        }
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

    // The objective is exempt from the slab, as it is from transparency, and for
    // the same reason: it is the one thing in the box whose position is the
    // question rather than the answer. It also keeps it agreeing with its own
    // guide, which is drawn as lines and was never sliced -- a cube that
    // vanished while its plumb line stayed reads as a bug in the world rather
    // than as a setting on the view.
    if (mode != LatticeViewModeBeacon && !latticeCellVisible(cell)) {
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
