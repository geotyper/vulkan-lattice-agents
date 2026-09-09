// Mirrors rewardPuckProximity and rewardPuckWork in
// include/vkexp/worlds/ScenarioMath.hpp. Scoring itself needs no hook -- the
// level is latched by puck_step.comp and mirrored onto every agent -- but the
// shaping does, in two parts.
//
// The first pays for being near the puck, because an untrained population hardly
// reaches it and being near it has to be worth something before pushing it can
// be. The second pays for the pushing, and is the only term in this world that
// tells one agent in a world apart from another: everything else is read off a
// puck all twelve of them share.
void puckPushScenarioAfterStep(inout Agent agent) {
    const float radius = puckRadius(params.worldRadius, params.scenario.floats0.y);
    const float distance = length(agent.penalties.yz - agent.pose.xy);
    const float reach = PuckApproachReach * radius;
    const float closeness = clamp(1.0 - distance / max(reach, 1.0e-4), 0.0, 1.0);
    agent.metrics.w += closeness * closeness * PuckProximityShare * params.deltaTime *
                       params.fitness.trackingReward;
    const float contribution = puckPushContribution(agent.pose.xy, agent.motion.xy, agent.pose.w,
                                                    agent.penalties.yz, agent.target.xy, radius);
    const float moving = puckWorkMovingFraction(length(agent.target.xy), params.maximumSpeed);
    agent.metrics.w += contribution * moving * params.deltaTime * PuckWorkReward;
}
