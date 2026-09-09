// A wall with one opening, shut by a gate that runs only while somebody stands
// on the plate in front of it.
//
// floats0 = {latch seconds, unused, unused, unused}.
// Packed by gpuParameters in src/worlds/scenarios/GatePlateScenario.cpp. The
// latch arrives as the setting rather than as an answer, so both sides run
// gateRemaining from the shared kernel and the rule is the shared thing.

// target.x is how long the gate still has to run, written during the step by the
// only pass that can see a whole world at once. Everything downstream -- the
// geometry, the light occlusion, the shaping -- reads it from here.
bool gatePlateScenarioOpen(Agent agent) {
    return gateIsOpen(agent.target.x);
}

vec2 gatePlateScenarioPosition(uint beaconIndex, float worldRadius) {
    return beaconIndex == 0u ? gateResourcePosition(worldRadius)
                             : gatePlatePosition(worldRadius);
}

vec3 gatePlateScenarioColor(uint beaconIndex) {
    return beaconIndex == 0u ? vec3(0.35, 1.00, 0.55) : vec3(0.95, 0.55, 0.15);
}

// Tinted by which leg the agent is on, so a population that has learned the
// order is visible as two colours moving in two directions rather than as a
// number.
vec3 gatePlateScenarioBodyTint(Agent agent, vec3 bodyColor) {
    if (agent.internal.y >= 0.5) {
        return bodyColor;
    }
    return mix(bodyColor, vec3(0.35, 1.00, 0.55), 0.45);
}
