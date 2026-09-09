// Mirrors GatePlateScenario.cpp. Two hooks, and they do different jobs.
//
// The before-step hook is the only place in this file that reads anything but
// its own agent. "Is the gate running" depends on where every agent in the world
// is standing, which is a reduction, and the spatial grid is already exactly
// that: built per world, once per step, before any agent moves. So there is no
// second buffer, no second pass and no shared record to keep in step -- every
// agent in a world scans the same cells over the same grid, reaches the same
// answer, and carries its own copy of the countdown. They cannot disagree
// because they are not communicating, they are recomputing.
bool gatePlateScenarioPressed(uint world) {
    const vec2 plate = gatePlatePosition(params.worldRadius);
    const float radius = gatePlateRadius(params.worldRadius);
    const ivec2 centreCell = gridCoordinate(plate);
    // One cell of slack, because an agent whose centre sits in a neighbouring
    // cell can still be inside the plate.
    const int reach = int(ceil(radius / params.gridCellSize)) + 1;
    for (int offsetY = -reach; offsetY <= reach; ++offsetY) {
        for (int offsetX = -reach; offsetX <= reach; ++offsetX) {
            const ivec2 cell = centreCell + ivec2(offsetX, offsetY);
            if (any(lessThan(cell, ivec2(0))) ||
                any(greaterThanEqual(cell, ivec2(int(params.gridWidth))))) {
                continue;
            }
            const uint cellIndex = world * params.gridCellsPerWorld +
                                   uint(cell.y) * params.gridWidth + uint(cell.x);
            int otherId = gridHeads[cellIndex];
            uint guard = 0;
            while (otherId >= 0 && guard++ < params.agentCount) {
                if (gateOnPlate(inputAgents[otherId].pose.xy, params.worldRadius)) {
                    return true;
                }
                otherId = gridNext[otherId];
            }
        }
    }
    return false;
}

void gatePlateScenarioBeforeStep(inout Agent agent) {
    const uint world = uint(max(agent.penalties.w, 0.0) + 0.5);
    agent.target.x = gateRemaining(agent.target.x, gatePlateScenarioPressed(world),
                                   params.scenario.floats0.x, params.deltaTime);
}

// And the after-step hook, which is where the two legs are. The leg is a fact
// about the room rather than about the agent: while the gate is shut the thing
// to do is press the plate, and while it runs the thing to do is go through --
// including for an agent whose neighbour opened it.
void gatePlateScenarioAfterStep(inout Agent agent, float distance) {
    const bool open = gatePlateScenarioOpen(agent);
    const bool wasOpen = agent.internal.y < 0.5;
    if (open != wasOpen) {
        // Bank what the finished leg earned and rebank against the new target,
        // exactly as the delivery cycle does at a pickup. Without it a gate
        // opening would read as a metre and a half of free progress.
        agent.metrics.w += max(agent.metrics.x - agent.metrics.y, 0.0) *
                           (wasOpen ? 1.0 : GatePlateProgressReward);
        agent.internal.y = open ? 0.0 : 1.0;
        const float rebanked = nearestBeaconDistance(agent, agent.pose.xy);
        agent.metrics.x = rebanked;
        agent.metrics.y = rebanked;
        return;
    }
    if (!open || distance >= params.arrivalRadius) {
        return;
    }
    agent.target.w = 1.0;
}
