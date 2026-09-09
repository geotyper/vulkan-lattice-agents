#include "worlds/steps/shared.glsl"
#include "worlds/steps/alternating.glsl"
#include "worlds/steps/tracking.glsl"
#include "worlds/steps/forage_home.glsl"
#include "worlds/steps/scent_relay.glsl"
#include "worlds/steps/two_doors.glsl"
#include "worlds/steps/shuttle.glsl"
#include "worlds/steps/two_gaps.glsl"
#include "worlds/steps/puck_push.glsl"

// GLSL has no function pointers, so the C++ side's function-pointer hooks become
// one dispatcher. Keeping it here is the point: agent_step.comp stays free of
// scenario identities, and adding a scenario touches this file and its own
// steps/<scenario>.glsl only.
void scenarioBeforeStep(inout Agent agent) {
    if (params.beaconScenario == 1u) {
        alternatingScenarioBeforeStep(agent);
    } else if (params.beaconScenario == 4u) {
        forageHomeScenarioBeforeStep(agent);
    }
}

void scenarioAfterStep(inout Agent agent, float distance) {
    if (params.beaconScenario == 4u) {
        forageHomeScenarioAfterStep(agent, distance);
    } else if (params.beaconScenario == 5u) {
        scentRelayScenarioAfterStep(agent, distance);
    } else if (params.beaconScenario == 6u) {
        twoDoorsScenarioAfterStep(agent, distance);
    } else if (params.beaconScenario == 7u) {
        shuttleScenarioAfterStep(agent, distance);
    } else if (params.beaconScenario == 8u) {
        twoGapsScenarioAfterStep(agent, distance);
    } else if (params.beaconScenario == 9u) {
        puckPushScenarioAfterStep(agent);
    } else if (params.beaconScenario == 2u || params.beaconScenario == 3u) {
        trackingScenarioAfterStep(agent, distance);
    } else {
        scenarioRecordPhaseArrival(agent, distance);
    }
}
