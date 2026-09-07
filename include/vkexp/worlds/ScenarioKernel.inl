// Scenario math shared verbatim by the CPU reference and the shaders.
//
// This file is compiled twice: once as C++ through ScenarioKernel.hpp, which
// supplies a small vec2/uint shim, and once as GLSL through
// shaders/worlds/scenario_kernel.glsl, where the same names are built in. It is
// therefore restricted to the common subset of both languages:
//
//   * every function is introduced with VKEXP_KERNEL_FN;
//   * every float literal carries the `f` suffix, so C++ does not silently
//     promote the arithmetic to double and drift away from the GPU;
//   * only vec2, float, uint and bool cross function boundaries;
//   * no swizzles, matrices, references or standard library calls.
//
// Anything that needs SimulationStep, AgentState or ScenarioParameters stays in
// the per-language scenario files and calls into here.

const float ScenarioTau = 6.28318530718f;
const float ForageHomeRelocationSeconds = 8.0f;
const float ForageHomeMinimumRadiusRatio = 0.38f;
const float ForageHomeRadiusRange = 0.32f;
const float RandomMotionSegmentSeconds = 3.0f;
const float AlternatingDiagonalRatio = 0.62f;
const float BeaconVisualRadius = 0.060f;

// Body radius in metres, the scale every other length is chosen against. Here as
// well as in AgentTypes.hpp because geometry that should be sized by the agent
// has to be expressible in the shared kernel, and a `const float` -- which GLSL
// needs -- is not a constant expression to C++, so neither file can reference
// the other. testTwoDoorsGeometry asserts the two stay equal.
const float ScenarioAgentBodyRadius = 0.022f;

// --- shuttle ---------------------------------------------------------------
//
// Two beacons facing each other with a short wall between them, and a trial long
// enough to run the round trip more than once. The wall does not divide the
// arena -- it is shorter than the gap between the beacons is wide -- so there is
// no door to find and nothing to remember about which way is open. What it takes
// away is the straight line: the shortest path is around one end, and with light
// occluded the far beacon disappears behind it on the way.
//
//        resource
//   +-------------------+
//   |                   |
//   |     #########     |   the wall, at y = 0
//   |                   |
//   |       home        |
//   +-------------------+

const uint ShuttleBoxCount = 1u;
// Chosen so a round trip round the end of the wall fits the default 15 s trial
// at least twice at the speed limit: shuttling until the time runs out is the
// task, and a geometry with room for one trip and a wait is a different one.
// testShuttleGeometry asserts that, so the two numbers cannot drift apart from
// the trial length unnoticed.
const float ShuttleBeaconY = 0.38f;      // fraction of the world radius
const float ShuttleWallHalfLength = 0.26f;

VKEXP_KERNEL_FN vec2 shuttleResourcePosition(float worldRadius) {
    return vec2(0.0f, ShuttleBeaconY * worldRadius);
}

VKEXP_KERNEL_FN vec2 shuttleHomePosition(float worldRadius) {
    return vec2(0.0f, -ShuttleBeaconY * worldRadius);
}

VKEXP_KERNEL_FN vec2 shuttleBoxCentre() { return vec2(0.0f, 0.0f); }

// Sized by the agent across, like the two-door walls, and by the arena along:
// how far the detour is should scale with the room, how solid the wall is
// should not.
VKEXP_KERNEL_FN vec2 shuttleBoxHalfExtent(float worldRadius) {
    return vec2(ShuttleWallHalfLength * worldRadius, ScenarioAgentBodyRadius);
}

// --- two doors -------------------------------------------------------------
//
// A wall across the arena with two gaps. One leads through to the resource; the
// other opens into a closed pocket. From the home side the two are identical, so
// the only way to know is to have gone -- which is what makes the world worth
// building: a memory has something to hold, and a mark has something to say.
//
// The geometry is derived from the arena radius rather than stored, so it costs
// no parameter slot and the CPU and the shader cannot disagree about where a
// wall is. Every box is axis-aligned, which keeps the contact test to a clamp
// and a subtraction in both languages.
//
// Layout, in fractions of the world radius:
//
//        resource at (0, +0.72)
//     +-------------------------------+
//     |          #####                |   <- pocket cap over the blocked door
//     |          #   #                |   <- pocket sides
//     |######  ##########  ###########|   <- the wall, at y = 0
//     |      A            B           |
//     |          home at (0, -0.72)   |
//     +-------------------------------+

const uint TwoDoorsBoxCount = 6u;
// Thickness is in metres and set by the agent, not by the arena: a wall is a
// wall whatever room it stands in, and scaling it with the world radius made a
// 17 cm slab across a 3.7 m arena -- nearly four body diameters of masonry, which
// reads as architecture rather than as a divider. One body diameter is enough to
// be seen and far more than enough to be felt.
const float TwoDoorsWallHalfThickness = ScenarioAgentBodyRadius;
// Measured, not guessed, and re-measured after Two gaps was solved and this
// world was not. The failure is not that the task is hard: fitness shapes on the
// best straight-line approach, so the spot pressed against the middle of the
// wall scores best and sees nothing, and getting to a door pays nothing until
// the agent is well past it. Escaping that plateau needs the target to be
// visible from somewhere, and with the old numbers it was visible from 5.7% of
// the far side -- against 19% for Two gaps, which learns.
//
// The strongest lever turned out to be the opening, not the distance: widening
// the door from 0.07 to 0.11 and bringing the pair in from 0.40 to 0.30 takes
// visibility to 15.6% and cuts the plateau from 0.147 m to 0.097 m. Moving the
// beacons closer, which was the first guess, moves visibility by a point and a
// half on its own and makes the plateau worse.
//
// The door is now 9 body diameters wide and the divider between the two doors is
// wider than either of them, so the question this world asks is still which
// opening leads through, and not whether an agent can thread a slot.
const float TwoDoorsDoorOffset = 0.30f;
const float TwoDoorsDoorHalfWidth = 0.11f;
const float TwoDoorsPocketDepth = 0.30f;
const float TwoDoorsArenaReach = 1.05f; // past the arena edge, so no gap at the rim
// 0.84x the light range apart, so an agent standing on one end can see the other
// when nothing is in the way. At the old 0.72 the separation was 1.10x the range
// -- and that ratio is the same in every world size, since the range is a
// fraction of the arena radius -- so what hid the target was partly the distance,
// which an agent can do nothing about, rather than only the wall, which it can.
const float TwoDoorsHomeY = -0.55f;
const float TwoDoorsResourceY = 0.55f;

// Which gap is a dead end this trial. Alternating by trial means a genome is
// scored on both, so it cannot win by always turning the same way -- and the
// trial index is not among the network's inputs, so it cannot be read off.
// Which gap is a dead end, and on what clock. The two settings are a real
// trade, not a preference:
//
// By trial (the default), one genome meets both layouts inside a generation and
// is scored on the average of them. It cannot win by always turning the same
// way -- that caps at half the trials -- so selection asks for a policy that
// handles both from the start. The cost is that early on, when nothing works,
// a genome that happens to suit one layout is averaged back down by the other.
//
// By generation, the whole population meets one layout and its successors meet
// the other. Selection inside a generation is then clean and the gradient is
// undiluted, which should be faster. The risk it takes on is oscillation:
// generation N can select for "go right" and generation N+1 punish exactly that,
// and a population can thrash between the two without ever building the memory
// that would settle it. Which effect wins is measured, not argued.
VKEXP_KERNEL_FN uint twoDoorsBlockedDoor(uint trial, uint generation, bool perGeneration) {
    return (perGeneration ? generation : trial) & 1u;
}

// Signed x of a door centre: door 0 left, door 1 right.
VKEXP_KERNEL_FN float twoDoorsDoorCentre(uint door, float worldRadius) {
    const float side = door == 0u ? -TwoDoorsDoorOffset : TwoDoorsDoorOffset;
    return side * worldRadius;
}

VKEXP_KERNEL_FN vec2 twoDoorsHomePosition(float worldRadius) {
    return vec2(0.0f, TwoDoorsHomeY * worldRadius);
}

VKEXP_KERNEL_FN vec2 twoDoorsResourcePosition(float worldRadius) {
    return vec2(0.0f, TwoDoorsResourceY * worldRadius);
}

// Box `index` as a centre; `twoDoorsBoxHalfExtent` gives the matching extent.
// Two calls rather than one because only vec2, float, uint and bool may cross a
// shared-kernel boundary, and a box is four numbers.
VKEXP_KERNEL_FN vec2 twoDoorsBoxCentre(uint index, float worldRadius, uint blockedDoor) {
    const float leftDoor = twoDoorsDoorCentre(0u, worldRadius);
    const float rightDoor = twoDoorsDoorCentre(1u, worldRadius);
    const float doorHalf = TwoDoorsDoorHalfWidth * worldRadius;
    const float reach = TwoDoorsArenaReach * worldRadius;
    const float side = TwoDoorsWallHalfThickness;
    const float blockedX = twoDoorsDoorCentre(blockedDoor, worldRadius);
    if (index == 0u) { // wall, outer left
        return vec2((-reach + (leftDoor - doorHalf)) * 0.5f, 0.0f);
    }
    if (index == 1u) { // wall, between the doors
        return vec2(0.0f, 0.0f);
    }
    if (index == 2u) { // wall, outer right
        return vec2(((rightDoor + doorHalf) + reach) * 0.5f, 0.0f);
    }
    if (index == 3u) { // pocket cap
        return vec2(blockedX, TwoDoorsPocketDepth * worldRadius);
    }
    if (index == 4u) { // pocket side, left of the blocked door
        return vec2(blockedX - doorHalf - side, TwoDoorsPocketDepth * worldRadius * 0.5f);
    }
    return vec2(blockedX + doorHalf + side, TwoDoorsPocketDepth * worldRadius * 0.5f);
}

// No blockedDoor here: which gap is the dead end moves the pocket, never its
// size, and an unused parameter would only invite one to be passed wrongly.
VKEXP_KERNEL_FN vec2 twoDoorsBoxHalfExtent(uint index, float worldRadius) {
    const float leftDoor = twoDoorsDoorCentre(0u, worldRadius);
    const float rightDoor = twoDoorsDoorCentre(1u, worldRadius);
    const float doorHalf = TwoDoorsDoorHalfWidth * worldRadius;
    const float reach = TwoDoorsArenaReach * worldRadius;
    const float thickness = TwoDoorsWallHalfThickness;
    if (index == 0u) {
        return vec2(((leftDoor - doorHalf) + reach) * 0.5f, thickness);
    }
    if (index == 1u) {
        // Centred on the origin by construction, so its half width is just the
        // inner edge of the right-hand door.
        return vec2(rightDoor - doorHalf, thickness);
    }
    if (index == 2u) {
        return vec2((reach - (rightDoor + doorHalf)) * 0.5f, thickness);
    }
    if (index == 3u) {
        return vec2(doorHalf + 2.0f * thickness, thickness);
    }
    // The sides run from the wall up to the cap, so the pocket is closed on
    // three sides and the only way out is back through the door.
    return vec2(thickness, TwoDoorsPocketDepth * worldRadius * 0.5f);
}

// Colour with the cue taken out of it: both ends show the average of the two, so
// the information a colour carries is gone while the light itself is very nearly
// unchanged. Replacing one colour with the other would remove the information
// too, but would also move how much light each end emits, and a control that
// changes two things answers neither.
//
// Per channel because only vec2, float, uint and bool may cross the language
// boundary; the callers apply it three times.
VKEXP_KERNEL_FN float scenarioColorCueRemoved(float channelA, float channelB) {
    return 0.5f * (channelA + channelB);
}

// --- two gaps --------------------------------------------------------------
//
// A wall right across the arena with two ways through, neither of them a dead
// end, and the two ends of the delivery cycle optionally trading places from
// one generation to the next.
//
//        resource (or home)
//   +-------------------------+
//   |####       ####      ####|   wall, y = 0, two gaps
//   |    home (or resource)   |
//   +-------------------------+
//
// The measurement that set these numbers is worth writing down, because the
// obvious geometry does not work. Fitness shapes on the *best straight-line
// approach* to the target, so a wall between the two ends creates a place --
// pressed against the middle segment, on the line between them -- that is
// simultaneously the best score on offer and the one spot with no line of sight
// to the target at all. Getting to a gap costs distance and therefore earns
// nothing until the agent is well past it. Two doors has that plateau 0.17 m
// deep with the target visible from 5.7% of the far side, and does not solve it.
// Shuttle has it 0.12 m deep with 36% visible, and solves it quickly. These
// constants put this world at 0.11 m and 19%: the same plateau as the world that
// learns, and three times the sight line of the world that does not.
//
// Which is also why the ends sit at 0.50 rather than further apart. At 0.72, as
// in Two doors, the separation is 1.10x the light range at *every* world size --
// the ratio is scale-free, since the range is a fraction of the arena radius --
// so an agent standing on one end sees nothing whatever of the other, and the
// plateau has no perceptual gradient to climb out on. At 0.50 the separation is
// 0.77x the range, so what hides the target is the wall, which the agent can do
// something about, and not the range, which it cannot.
const uint TwoGapsBoxCount = 3u;
const float TwoGapsWallHalfThickness = ScenarioAgentBodyRadius;
const float TwoGapsGapOffset = 0.30f;    // fraction of the world radius
const float TwoGapsGapHalfWidth = 0.08f;
const float TwoGapsArenaReach = 1.05f;   // past the arena edge, so no gap at the rim
const float TwoGapsBeaconY = 0.50f;

// Whether the two ends have traded places this generation. Off, the resource is
// always north and home always south, and a genome can bake the direction in
// without ever reading the light. On, the direction is worth nothing and the
// colour is the only thing that says which end is which -- so the swap is what
// makes this world a test of perception rather than of memorisation.
//
// The generation number is what varies, because a trial index would let one
// genome be scored on both layouts within a generation and average them; the
// point is that a whole population meets one layout, and the next one meets the
// other. Both sides resolve it through this function rather than packing an
// already-resolved flag, so the rule itself is the shared thing.
VKEXP_KERNEL_FN bool twoGapsEndsSwapped(uint generation, bool swapEnabled) {
    return swapEnabled && (generation & 1u) == 1u;
}

VKEXP_KERNEL_FN vec2 twoGapsResourcePosition(float worldRadius, bool swapped) {
    const float side = swapped ? -TwoGapsBeaconY : TwoGapsBeaconY;
    return vec2(0.0f, side * worldRadius);
}

VKEXP_KERNEL_FN vec2 twoGapsHomePosition(float worldRadius, bool swapped) {
    const float side = swapped ? TwoGapsBeaconY : -TwoGapsBeaconY;
    return vec2(0.0f, side * worldRadius);
}

// Box `index` as a centre; `twoGapsBoxHalfExtent` gives the matching extent.
// Neither depends on the swap: the wall is the same wall whichever end is which.
VKEXP_KERNEL_FN vec2 twoGapsBoxCentre(uint index, float worldRadius) {
    const float gap = TwoGapsGapOffset * worldRadius;
    const float gapHalf = TwoGapsGapHalfWidth * worldRadius;
    const float reach = TwoGapsArenaReach * worldRadius;
    if (index == 0u) { // outer left, from the rim to the left gap
        return vec2((-reach + (-gap - gapHalf)) * 0.5f, 0.0f);
    }
    if (index == 1u) { // between the two gaps
        return vec2(0.0f, 0.0f);
    }
    return vec2((reach + (gap + gapHalf)) * 0.5f, 0.0f); // outer right
}

// The two outer segments are mirror images, so they share an extent and differ
// only in the centre above.
VKEXP_KERNEL_FN vec2 twoGapsBoxHalfExtent(uint index, float worldRadius) {
    const float gap = TwoGapsGapOffset * worldRadius;
    const float gapHalf = TwoGapsGapHalfWidth * worldRadius;
    const float reach = TwoGapsArenaReach * worldRadius;
    if (index == 1u) {
        return vec2(gap - gapHalf, TwoGapsWallHalfThickness);
    }
    return vec2((reach - (gap + gapHalf)) * 0.5f, TwoGapsWallHalfThickness);
}

// Does the segment from `start` to `finish` cross the box? The slab test, which
// is the whole of light occlusion: a wall that stops a body but not its light is
// a wall an agent can see through, and the light gradient then pulls it straight
// into the one place it cannot go.
//
// A lightmap would answer the same question by sampling, and would be the right
// tool for hundreds of sources over complex geometry. Here there are at most two
// beacons and six boxes, and the receptors are directional -- a map gives the
// light at a point and loses the direction the sharp receptor tuning needs, so
// it would have to be marched along each ray anyway. Twelve slab tests per agent
// per step sit beside a brain that already does more than a thousand multiplies.
VKEXP_KERNEL_FN bool segmentHitsBox(vec2 start, vec2 finish, vec2 centre, vec2 halfExtent) {
    float enter = 0.0f;
    float leave = 1.0f;

    const float spanX = finish.x - start.x;
    const float lowX = centre.x - halfExtent.x;
    const float highX = centre.x + halfExtent.x;
    if (spanX > -1.0e-8f && spanX < 1.0e-8f) {
        // Parallel to the slab: either the whole segment is inside it or the box
        // cannot be crossed at all.
        if (start.x < lowX || start.x > highX) {
            return false;
        }
    } else {
        float nearX = (lowX - start.x) / spanX;
        float farX = (highX - start.x) / spanX;
        if (nearX > farX) {
            const float held = nearX;
            nearX = farX;
            farX = held;
        }
        enter = max(enter, nearX);
        leave = leave < farX ? leave : farX;
    }

    const float spanY = finish.y - start.y;
    const float lowY = centre.y - halfExtent.y;
    const float highY = centre.y + halfExtent.y;
    if (spanY > -1.0e-8f && spanY < 1.0e-8f) {
        if (start.y < lowY || start.y > highY) {
            return false;
        }
    } else {
        float nearY = (lowY - start.y) / spanY;
        float farY = (highY - start.y) / spanY;
        if (nearY > farY) {
            const float held = nearY;
            nearY = farY;
            farY = held;
        }
        enter = max(enter, nearY);
        leave = leave < farY ? leave : farY;
    }

    return enter <= leave;
}

// Push-out for a circle against an axis-aligned box: zero when clear, otherwise
// the shortest vector that separates them. Written as the shallowest overlap
// axis rather than as a nearest-point normal so an agent that has sunk into a
// wall leaves the way it came instead of being ejected through it.
VKEXP_KERNEL_FN vec2 boxPushOut(vec2 position, float radius, vec2 centre, vec2 halfExtent) {
    const float dx = position.x - centre.x;
    const float dy = position.y - centre.y;
    const float overlapX = halfExtent.x + radius - (dx < 0.0f ? -dx : dx);
    const float overlapY = halfExtent.y + radius - (dy < 0.0f ? -dy : dy);
    if (overlapX <= 0.0f || overlapY <= 0.0f) {
        return vec2(0.0f, 0.0f);
    }
    if (overlapX < overlapY) {
        return vec2(dx < 0.0f ? -overlapX : overlapX, 0.0f);
    }
    return vec2(0.0f, dy < 0.0f ? -overlapY : overlapY);
}

VKEXP_KERNEL_FN uint scenarioHash(uint value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

VKEXP_KERNEL_FN float scenarioRandom01(uint value) {
    return float(scenarioHash(value) & 0x00ffffffu) / 16777215.0f;
}

// --- alternating diagonals -------------------------------------------------

// Unit-square offset of one diagonal beacon; callers scale by the world radius.
VKEXP_KERNEL_FN vec2 alternatingDiagonalOffset(uint beaconIndex, uint phase) {
    const float x = beaconIndex == 0u ? -AlternatingDiagonalRatio : AlternatingDiagonalRatio;
    const float first = phase == 0u ? -AlternatingDiagonalRatio : AlternatingDiagonalRatio;
    const float y = beaconIndex == 0u ? first : -first;
    return vec2(x, y);
}

// --- rotating orbit --------------------------------------------------------

// Rotates a unit direction by `angle`. Callers pass the normalised base heading
// and scale the result by the orbit radius.
VKEXP_KERNEL_FN vec2 rotatingOrbitOffset(vec2 baseDirection, float angle) {
    const float cosine = cos(angle);
    const float sine = sin(angle);
    return vec2(baseDirection.x * cosine - baseDirection.y * sine,
                baseDirection.x * sine + baseDirection.y * cosine);
}

// --- forage home -----------------------------------------------------------

VKEXP_KERNEL_FN uint forageHomeEpoch(float motionTime) {
    return uint(floor(max(motionTime, 0.0f) / ForageHomeRelocationSeconds));
}

VKEXP_KERNEL_FN bool forageHomeRelocated(float motionTime, float deltaTime) {
    const float currentTime = max(motionTime, 0.0f);
    const float previousTime = max(currentTime - deltaTime, 0.0f);
    const float epsilon = max(deltaTime * 0.01f, 0.000001f);
    return uint(floor((currentTime + epsilon) / ForageHomeRelocationSeconds)) !=
           uint(floor((previousTime + epsilon) / ForageHomeRelocationSeconds));
}

VKEXP_KERNEL_FN uint forageHomeKey(uint motionSeed, uint trial, uint epoch) {
    return motionSeed ^ (trial * 0x51ed270bu) ^ (epoch * 0x85ebca6bu) ^ 0xc2b2ae35u;
}

// Home position as a fraction of the world radius.
VKEXP_KERNEL_FN vec2 forageHomeOffset(uint key) {
    const float angle = scenarioRandom01(key) * ScenarioTau;
    const float radiusRatio =
        ForageHomeMinimumRadiusRatio + scenarioRandom01(key ^ 0x27d4eb2du) * ForageHomeRadiusRange;
    return vec2(cos(angle) * radiusRatio, sin(angle) * radiusRatio);
}

// --- random movement -------------------------------------------------------

VKEXP_KERNEL_FN uint randomMotionSegment(float motionTime) {
    return uint(floor(max(motionTime, 0.0f) / RandomMotionSegmentSeconds));
}

VKEXP_KERNEL_FN uint randomTeleportKey(uint motionSeed, uint trial, uint segment) {
    return motionSeed ^ (trial * 0x27d4eb2du) ^ (segment * 0x165667b1u) ^ 0xa511e9b3u;
}

VKEXP_KERNEL_FN bool randomTeleportSegment(uint motionSeed, uint trial, uint segment,
                                           float teleportProbability) {
    if (segment == 0u || teleportProbability <= 0.0f) {
        return false;
    }
    return scenarioRandom01(randomTeleportKey(motionSeed, trial, segment)) < teleportProbability;
}

VKEXP_KERNEL_FN uint randomWanderKey(uint motionSeed, uint trial, uint epoch) {
    return motionSeed ^ (trial * 0x9e3779b9u) ^ (epoch * 0x85ebca6bu);
}

// Wander position as a fraction of the roam radius.
VKEXP_KERNEL_FN vec2 randomWanderOffset(uint key, float scaledTime) {
    const float phase0 = scenarioRandom01(key) * ScenarioTau;
    const float phase1 = scenarioRandom01(key ^ 0x68bc21ebu) * ScenarioTau;
    const float phase2 = scenarioRandom01(key ^ 0x02e5be93u) * ScenarioTau;
    const float phase3 = scenarioRandom01(key ^ 0x967a889bu) * ScenarioTau;
    const float rawX = 0.62f * sin(scaledTime * 0.73f + phase0) +
                       0.28f * sin(scaledTime * 1.37f + phase1) +
                       0.18f * sin(scaledTime * 0.31f + phase2);
    const float rawY = 0.58f * sin(scaledTime * 0.83f + phase3) +
                       0.31f * sin(scaledTime * 1.19f + phase0) +
                       0.16f * sin(scaledTime * 0.27f + phase1);
    const vec2 raw = vec2(rawX, rawY);
    const float scale = 1.0f / (1.25f + length(raw));
    return vec2(raw.x * scale, raw.y * scale);
}
