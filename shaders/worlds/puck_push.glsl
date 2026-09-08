// A shared puck and a lit disc in the middle to push it into. The puck itself
// is not a beacon: it is state in its own buffer, integrated by puck_step.comp,
// and what this file describes is only the marker standing on the target.
//
// floats0 = {target radius ratio, unused, unused, unused}.
// Packed by gpuParameters in src/worlds/scenarios/PuckPushScenario.cpp.

// Beacon 0 is the target disc; beacon 1 is the puck, read from the mirror the
// agent already carries. The puck emits because otherwise it is invisible to
// every receptor, and a population cannot climb a fitness gradient it has no
// sense of.
vec2 puckPushScenarioPosition(uint beaconIndex, Agent agent) {
    return beaconIndex == 0u ? vec2(0.0) : agent.penalties.yz;
}

vec3 puckPushScenarioColor(uint beaconIndex) {
    return beaconIndex == 0u ? vec3(0.35, 1.00, 0.55) : vec3(1.00, 0.92, 0.70);
}
