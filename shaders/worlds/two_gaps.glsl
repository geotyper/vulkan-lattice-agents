// A wall across the arena with two ways through, and two ends that optionally
// trade places from generation to generation.
//
// floats0 = {unused, unused, unused, cargo decay rate},
// floats1 = {pickup reward, delivery reward, unused, unused},
// integers = {generation, swap enabled, unused, unused}.
// Packed by gpuParameters in src/worlds/scenarios/TwoGapsScenario.cpp. The swap
// arrives unresolved so both sides answer it with twoGapsEndsSwapped.

bool twoGapsScenarioSwapped(ScenarioParameters sp) {
    return twoGapsEndsSwapped(sp.integers.x, sp.integers.y != 0u);
}

vec2 twoGapsScenarioPosition(uint beaconIndex, float worldRadius, ScenarioParameters sp) {
    const bool swapped = twoGapsScenarioSwapped(sp);
    return beaconIndex == 0u ? twoGapsResourcePosition(worldRadius, swapped)
                             : twoGapsHomePosition(worldRadius, swapped);
}

vec3 twoGapsScenarioColor(uint beaconIndex) {
    return beaconIndex == 0u ? vec3(1.00, 0.82, 0.20) : vec3(0.20, 0.55, 1.00);
}

vec3 twoGapsScenarioBodyTint(Agent agent, vec3 bodyColor) {
    if (agent.internal.y < 0.5) {
        return bodyColor;
    }
    return mix(bodyColor, vec3(1.00, 0.82, 0.20), 0.35 + 0.45 * clamp(agent.internal.x, 0.0, 1.0));
}
