#include "worlds/scenario_params.glsl"
#include "worlds/scenario_math.glsl"
#include "worlds/stationary.glsl"
#include "worlds/alternating.glsl"
#include "worlds/rotating.glsl"
#include "worlds/random_movement.glsl"
#include "worlds/forage_home.glsl"
#include "worlds/scent_relay.glsl"
#include "worlds/two_doors.glsl"
#include "worlds/shuttle.glsl"
#include "worlds/two_gaps.glsl"

vec2 scenarioBeaconPosition(uint scenario, Agent agent, uint beaconIndex, uint phase,
                            float worldRadius, ScenarioParameters sp) {
    if (scenario == 0u) {
        return stationaryScenarioPosition(agent);
    }
    if (scenario == 1u) {
        return alternatingScenarioPosition(beaconIndex, phase, worldRadius);
    }
    if (scenario == 2u) {
        return rotatingScenarioPosition(agent, worldRadius, sp);
    }
    if (scenario == 3u) {
        const uint trial = uint(max(agent.target.z, 0.0));
        return randomMovementScenarioPosition(trial, worldRadius, sp);
    }
    if (scenario == 4u) {
        return forageHomeScenarioPosition(agent, beaconIndex, worldRadius, sp);
    }
    if (scenario == 6u) {
        return twoDoorsScenarioPosition(beaconIndex, worldRadius);
    }
    if (scenario == 7u) {
        return shuttleScenarioPosition(beaconIndex, worldRadius);
    }
    if (scenario == 8u) {
        return twoGapsScenarioPosition(beaconIndex, worldRadius, sp);
    }
    return scentRelayScenarioPosition(agent, beaconIndex, worldRadius, sp);
}

vec3 scenarioBeaconColor(uint scenario, uint beaconIndex, uint phase, uint trial) {
    if (scenario == 1u) {
        return alternatingScenarioColor(beaconIndex, phase);
    }
    if (scenario == 4u) {
        return forageHomeScenarioColor(beaconIndex);
    }
    if (scenario == 5u) {
        return scentRelayScenarioColor(beaconIndex);
    }
    if (scenario == 6u) {
        return twoDoorsScenarioColor(beaconIndex);
    }
    if (scenario == 7u) {
        return shuttleScenarioColor(beaconIndex);
    }
    if (scenario == 8u) {
        return twoGapsScenarioColor(beaconIndex);
    }
    return stationaryScenarioColor(trial);
}

// Beacon colour as everything downstream should see it: the scenario's own
// colour, or -- with the cue ablated -- the average of the two ends, so they stay
// lit and stop being distinguishable by hue. Mirrors activeBeacons in
// src/worlds/WorldScenario.cpp.
vec3 scenarioBeaconColorCue(uint scenario, uint beaconIndex, uint phase, uint trial,
                            uint beaconCount, uint uniformColor) {
    if (uniformColor == 0u || beaconCount < 2u) {
        return scenarioBeaconColor(scenario, beaconIndex, phase, trial);
    }
    const vec3 first = scenarioBeaconColor(scenario, 0u, phase, trial);
    const vec3 second = scenarioBeaconColor(scenario, 1u, phase, trial);
    return vec3(scenarioColorCueRemoved(first.x, second.x),
                scenarioColorCueRemoved(first.y, second.y),
                scenarioColorCueRemoved(first.z, second.z));
}

// Scenario 4 scores against the beacon the agent is currently seeking rather than
// the nearest one; the others take the nearest of `beaconCount` beacons. The
// count comes from the scenario definition through the step parameters, so this
// file never enumerates which scenarios have two beacons.
float scenarioTargetDistance(uint scenario, Agent agent, vec2 position, uint phase,
                             float worldRadius, uint beaconCount, ScenarioParameters sp) {
    if (scenario == 4u || scenario == 5u || scenario == 6u || scenario == 7u ||
        scenario == 8u) {
        const uint targetIndex = agent.internal.y >= 0.5 ? 1u : 0u;
        return length(scenarioBeaconPosition(scenario, agent, targetIndex, phase, worldRadius,
                                             sp) - position);
    }
    float nearest = worldRadius * 4.0;
    for (uint index = 0u; index < beaconCount; ++index) {
        nearest = min(nearest,
                      length(scenarioBeaconPosition(scenario, agent, index, phase, worldRadius,
                                                    sp) - position));
    }
    return nearest;
}

// Static geometry, dispatched like everything else here. Only one scenario has
// any, and the count comes from the scenario definition through the step
// parameters, so this returns a degenerate box rather than enumerating who does.
vec2 scenarioObstacleCentre(uint scenario, Agent agent, uint index, float worldRadius) {
    if (scenario == 6u) {
        return twoDoorsBoxCentre(index, worldRadius, twoDoorsScenarioBlockedDoor(agent));
    }
    if (scenario == 7u) {
        return shuttleBoxCentre();
    }
    if (scenario == 8u) {
        return twoGapsBoxCentre(index, worldRadius);
    }
    return vec2(0.0);
}

vec2 scenarioObstacleHalfExtent(uint scenario, uint index, float worldRadius) {
    if (scenario == 6u) {
        return twoDoorsBoxHalfExtent(index, worldRadius);
    }
    if (scenario == 7u) {
        return shuttleBoxHalfExtent(worldRadius);
    }
    if (scenario == 8u) {
        return twoGapsBoxHalfExtent(index, worldRadius);
    }
    return vec2(0.0);
}

// True when the scenario's static geometry stands between the two points.
// `obstacleCount` comes from the step parameters, so a scenario with no
// obstacles costs one comparison and this file never enumerates who has any.
bool scenarioSightBlocked(uint scenario, Agent agent, vec2 start, vec2 finish, float worldRadius,
                          uint obstacleCount) {
    for (uint index = 0u; index < obstacleCount; ++index) {
        if (segmentHitsBox(start, finish,
                           scenarioObstacleCentre(scenario, agent, index, worldRadius),
                           scenarioObstacleHalfExtent(scenario, index, worldRadius))) {
            return true;
        }
    }
    return false;
}

// Optional scenario-specific body colouring for the visualization.
vec3 scenarioBodyTint(uint scenario, Agent agent, vec3 bodyColor) {
    if (scenario == 4u) {
        return forageHomeScenarioBodyTint(agent, bodyColor);
    }
    if (scenario == 5u) {
        return scentRelayScenarioBodyTint(agent, bodyColor);
    }
    if (scenario == 6u) {
        return twoDoorsScenarioBodyTint(agent, bodyColor);
    }
    if (scenario == 7u) {
        return shuttleScenarioBodyTint(agent, bodyColor);
    }
    if (scenario == 8u) {
        return twoGapsScenarioBodyTint(agent, bodyColor);
    }
    return bodyColor;
}
