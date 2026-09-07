// A wall with two gaps, one of them a dead end. The geometry comes from the
// arena radius through the shared kernel; what is packed is only the clock the
// dead end runs on.
//
// floats0 = {unused, unused, unused, cargo decay rate},
// floats1 = {pickup reward, delivery reward, unused, unused},
// integers = {generation, layout keyed to the generation, unused, unused}.
// Packed by gpuParameters in src/worlds/scenarios/TwoDoorsScenario.cpp; the
// float slots are the ones scenarioDeliveryCycleAfterStep reads for every
// scenario running the collect-and-deliver cycle.

uint twoDoorsScenarioBlockedDoor(Agent agent, ScenarioParameters sp) {
    return twoDoorsBlockedDoor(uint(max(agent.target.z, 0.0)), sp.integers.x,
                               sp.integers.y != 0u);
}

vec2 twoDoorsScenarioPosition(uint beaconIndex, float worldRadius) {
    return beaconIndex == 0u ? twoDoorsResourcePosition(worldRadius)
                             : twoDoorsHomePosition(worldRadius);
}

// The resource is lit like any goal; home is lit too, because the question this
// world asks is which gap leads through and an invisible home would stack a
// second one on top of it.
vec3 twoDoorsScenarioColor(uint beaconIndex) {
    return beaconIndex == 0u ? vec3(1.00, 0.82, 0.20) : vec3(0.20, 0.55, 1.00);
}

vec3 twoDoorsScenarioBodyTint(Agent agent, vec3 bodyColor) {
    if (agent.internal.y < 0.5) {
        return bodyColor;
    }
    return mix(bodyColor, vec3(1.00, 0.82, 0.20), 0.35 + 0.45 * clamp(agent.internal.x, 0.0, 1.0));
}
