// Mirrors rewardPuckProximity in include/vkexp/worlds/ScenarioMath.hpp. Scoring
// itself needs no hook -- the level is latched by puck_step.comp and mirrored
// onto every agent -- but the shaping does: an untrained population hardly
// reaches the puck, so being near it has to be worth something before pushing it
// can be.
void puckPushScenarioAfterStep(inout Agent agent) {
    const float distance = length(agent.penalties.yz - agent.pose.xy);
    const float closeness = clamp(1.0 - distance / params.lightSensorRange, 0.0, 1.0);
    agent.metrics.w += closeness * closeness * params.deltaTime * params.fitness.trackingReward;
}
