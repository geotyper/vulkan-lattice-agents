#version 450
#extension GL_GOOGLE_include_directive : require

#include "simulation/trail_kernel.glsl"
#include "worlds/scenario_params.glsl"
#include "neuro/brain_kernel.glsl"
#include "simulation/agent_layout.glsl"
#include "simulation/puck_layout.glsl"


layout(std430, set = 0, binding = 0) readonly buffer Agents {
    Agent agents[];
};
layout(std430, set = 0, binding = 2) readonly buffer Pucks {
    Puck pucks[];
};
layout(std430, set = 0, binding = 1) readonly buffer TrailField {
    uint trailValues[];
};

layout(push_constant) uniform DrawParameters {
    float scaleX;
    float scaleY;
    float worldRadius;
    float opacity;
    uint mode; // 0 circular body, 1 beacon, 2 light halo, 3 arena, 4 heading
    uint worldShape;
    uint beaconScenario;
    uint beaconPhase;
    uint selectedWorld;
    uint agentsPerWorld;
    uint trialsPerGenome;
    uint trailWidth;
    float trailRenderWidth;
    float trailCellSize;
    uint trailRoundMarks;
    float backgroundBrightness;
    uint beaconCount;
    uint uniformBeaconColor;
    ScenarioParameters scenario;
} params;

#include "worlds/world_scenarios.glsl"

layout(location = 0) out vec4 color;



vec2 circleVertex(uint vertex, float radius, uint segmentCount) {
    const uint corner = vertex % 3;
    const uint segment = vertex / 3;
    if (corner == 0) {
        return vec2(0.0);
    }
    const float angle = ScenarioTau * float(segment + corner - 1) / float(segmentCount);
    return vec2(cos(angle), sin(angle)) * radius;
}

void main() {
    const uint trial = params.selectedWorld % params.trialsPerGenome;
    const uint group = params.selectedWorld / params.trialsPerGenome;
    const uint firstGenome = group * params.agentsPerWorld;
    const uint agentIndex =
        (firstGenome + gl_InstanceIndex) * params.trialsPerGenome + trial;
    const uint firstAgentIndex = firstGenome * params.trialsPerGenome + trial;
    // Modes that draw one thing per agent index into the agent; the rest read
    // the visible world's first agent, which is what carries the trial the
    // beacons and the obstacles are placed from.
    Agent agent = agents[params.mode == 0u || params.mode == 2u || params.mode == 4u
                             ? agentIndex
                             : firstAgentIndex];
    if (params.mode == 5u) {
        // One quad per trail cell of the visible world. Reading the field in the
        // vertex stage keeps the whole renderer on the procedural-instance path
        // it already uses, with no per-fragment buffer access to add.
        const uint cell = uint(gl_InstanceIndex);
        const uint cellX = cell % params.trailWidth;
        const uint cellY = cell / params.trailWidth;
        // The mark still lives in exactly one cell; the shape and the fill only
        // decide how much of that cell is painted. A disc reads as a track, a
        // square tiles the cell exactly and shows the grid for what it is.
        // Below roughly 1.4 cells of fill neither shape reaches its diagonal
        // neighbours and the path reads as dots -- that is the grid showing
        // through, not missing deposits, and a finer resolution is the answer.
        const float fill = clamp(params.trailRenderWidth, 0.02, 1.6);
        const vec2 cellCentre =
            (vec2(float(cellX), float(cellY)) + vec2(0.5)) * params.trailCellSize -
            vec2(params.worldRadius);
        vec2 markOffset;
        if (params.trailRoundMarks != 0u) {
            markOffset = circleVertex(gl_VertexIndex, fill * params.trailCellSize * 0.5, 8);
        } else {
            const vec2 corners[6] = vec2[](vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(0.5, 0.5),
                                           vec2(-0.5, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5));
            markOffset = corners[gl_VertexIndex] * fill * params.trailCellSize;
        }
        const vec2 cellWorld = cellCentre + markOffset;

        const uint cellsPerWorld = params.trailWidth * params.trailWidth;
        vec3 deposit;
        for (uint channel = 0u; channel < TrailChannels; ++channel) {
            deposit[channel] =
                float(trailValues[trailValueIndex(params.selectedWorld, cellsPerWorld, cell,
                                                  channel)]) /
                TrailFixedPointScale;
        }
        // Same soft saturation the antennae see, so what is drawn is what the
        // network is actually given rather than a prettier version of it.
        const float strength = max(max(deposit.r, deposit.g), deposit.b);
        const vec3 tint = strength > 0.0 ? deposit / strength : vec3(0.0);
        color = vec4(tint, params.opacity * (strength / (1.0 + strength)));
        gl_Position = vec4(cellWorld.x * params.scaleX, cellWorld.y * params.scaleY, 0.0, 1.0);
        return;
    }

    vec2 world;
    if (params.mode == 0) {
        world = agent.pose.xy + circleVertex(gl_VertexIndex, agent.pose.w, 16);
        const float wallContact = max(max(max(agent.wallTouch0.x, agent.wallTouch0.y),
                                          max(agent.wallTouch0.z, agent.wallTouch0.w)),
                                      max(max(agent.wallTouch1.x, agent.wallTouch1.y),
                                          max(agent.wallTouch1.z, agent.wallTouch1.w)));
        const float agentContact = max(max(max(agent.agentTouch0.x, agent.agentTouch0.y),
                                           max(agent.agentTouch0.z, agent.agentTouch0.w)),
                                       max(max(agent.agentTouch1.x, agent.agentTouch1.y),
                                           max(agent.agentTouch1.z, agent.agentTouch1.w)));
        vec3 bodyColor = mix(vec3(0.72, 0.82, 0.92), agent.signal.rgb, 0.42);
        bodyColor = scenarioBodyTint(params.beaconScenario, agent, bodyColor);
        bodyColor = mix(bodyColor, vec3(1.0, 0.35, 0.12), wallContact * 0.75);
        bodyColor = mix(bodyColor, vec3(1.0, 0.95, 0.35), agentContact * 0.85);
        color = vec4(bodyColor, params.opacity);
    } else if (params.mode == 1) {
        world = scenarioBeaconPosition(params.beaconScenario, agent, gl_InstanceIndex,
                                       params.beaconPhase, params.worldRadius, params.scenario) +
                circleVertex(gl_VertexIndex, BeaconVisualRadius, 16);
        color = vec4(scenarioBeaconColorCue(params.beaconScenario, gl_InstanceIndex,
                                            params.beaconPhase, trial, params.beaconCount,
                                            params.uniformBeaconColor),
                     params.opacity);
    } else if (params.mode == 7) {
        // The puck of the world being watched, drawn from the same record the
        // simulation integrates, so what is on screen is what is being pushed.
        const Puck puck = pucks[params.selectedWorld];
        world = puck.pose.xy + circleVertex(gl_VertexIndex, puck.pose.z, 24);
        // Tinted by how far it has got: the level is latched, so the colour is a
        // reading of the score rather than of the position.
        const float level = clamp(puck.motion.z / 2.0, 0.0, 1.0);
        color = vec4(mix(vec3(0.85, 0.85, 0.90), vec3(0.35, 1.00, 0.55), level), params.opacity);
    } else if (params.mode == 8) {
        // The target disc, drawn as ground under everything: it is a region the
        // puck has to reach, not a body anything collides with.
        // The ratio is already in the scenario block -- the push constants are at
        // the 128 bytes Vulkan guarantees and have no room for a copy of it.
        const float targetRadius = params.worldRadius * params.scenario.floats0.x;
        world = circleVertex(gl_VertexIndex, targetRadius, 48);
        color = vec4(0.20, 0.42, 0.28, params.opacity);
    } else if (params.mode == 6) {
        // One quad per obstacle box, placed by the same kernel the step uses, so
        // what is drawn is what an agent will actually walk into.
        const vec2 centre = scenarioObstacleCentre(params.beaconScenario, agent,
                                                   uint(gl_InstanceIndex), params.worldRadius,
                                                   params.scenario);
        const vec2 halfExtent = scenarioObstacleHalfExtent(params.beaconScenario,
                                                           uint(gl_InstanceIndex),
                                                           params.worldRadius);
        const vec2 corners[6] = vec2[](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
                                       vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
        world = centre + corners[gl_VertexIndex] * halfExtent;
        color = vec4(0.20, 0.23, 0.30, params.opacity);
    } else if (params.mode == 2) {
        const float radius = agent.pose.w + agent.signal.a * 0.055;
        world = agent.pose.xy + circleVertex(gl_VertexIndex, radius, 16);
        color = vec4(agent.signal.rgb, params.opacity * agent.signal.a);
    } else if (params.mode == 3) {
        if (params.worldShape == 0) {
            world = circleVertex(gl_VertexIndex, params.worldRadius, 64);
        } else {
            const float r = params.worldRadius;
            const vec2 square[6] = vec2[](vec2(-r, -r), vec2(r, -r), vec2(r, r),
                                          vec2(-r, -r), vec2(r, r), vec2(-r, r));
            world = square[gl_VertexIndex];
        }
        color = vec4(vec3(0.022, 0.034, 0.055) * params.backgroundBrightness, params.opacity);
    } else {
        const vec2 heading[3] = vec2[](vec2(1.42, 0.0), vec2(0.25, 0.38),
                                       vec2(0.25, -0.38));
        const vec2 local = heading[gl_VertexIndex] * agent.pose.w;
        const float c = cos(agent.pose.z);
        const float s = sin(agent.pose.z);
        world = agent.pose.xy + mat2(c, s, -s, c) * local;
        color = vec4(0.92, 0.97, 1.0, params.opacity);
    }
    gl_Position = vec4(world.x * params.scaleX, world.y * params.scaleY, 0.0, 1.0);
}
