// Mirrors afterStep in src/worlds/scenarios/TwoGapsScenario.cpp. Measured from
// the target that has just become current, so setting off toward a gap is
// rewarded as progress once the agent is past it.
void twoGapsScenarioAfterStep(inout Agent agent, float distance) {
    scenarioDeliveryCycleAfterStep(agent, distance);
}
