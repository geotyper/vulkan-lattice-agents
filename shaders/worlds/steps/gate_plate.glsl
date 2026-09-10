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

// And the after-step hook, which is where the legs are. Two things decide which
// one the agent is on, and they are different kinds of thing: whether it is
// carrying, which is private to it, and whether the gate is running, which is a
// fact about the room. Both reasons to head for the plate -- "I have to open it"
// and "I am coming home" -- point at the same place.
void gatePlateScenarioAfterStep(inout Agent agent, float distance) {
    const bool open = gatePlateScenarioOpen(agent);
    const bool wasSeekingPlate = agent.internal.y >= 0.5;
    const bool wasCarrying = agent.internal.x >= 0.5;

    // Measured against whatever the agent was actually heading for. Reaching the
    // plate while pressing it is not an arrival: pressing is positional, so only
    // a carrying agent closes a trip.
    if (distance < params.arrivalRadius) {
        if (!wasSeekingPlate && !wasCarrying) {
            agent.internal.x = 1.0;
        } else if (wasSeekingPlate && wasCarrying) {
            agent.internal.x = 0.0;
            agent.target.w = floor(max(agent.target.w, 0.0) + 0.5) + 1.0;
        }
    }

    const bool seekPlate = agent.internal.x >= 0.5 || !open;
    if (seekPlate == wasSeekingPlate) {
        return;
    }
    // Bank what the finished leg earned and rebank against the new target,
    // exactly as the delivery cycle does at a pickup. Without it a gate opening
    // reads as a metre and a half of free progress. The leg that ended is
    // weighted by what it was: walking to the plate to press it is the leg with
    // no other signal, because the resource is invisible behind a shut gate.
    const float weight = (wasCarrying || !wasSeekingPlate) ? 1.0 : GatePlateProgressReward;
    agent.metrics.w += max(agent.metrics.x - agent.metrics.y, 0.0) * weight;
    agent.internal.y = seekPlate ? 1.0 : 0.0;
    const float rebanked = nearestBeaconDistance(agent, agent.pose.xy);
    agent.metrics.x = rebanked;
    agent.metrics.y = rebanked;
}
