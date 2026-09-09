// Shared by C++ and GLSL. See PuckKernel.hpp for the C++ shim and the rules
// this file has to stay inside: only vec2, float, uint and bool cross the
// boundary, literals carry the f suffix, and nothing here calls the standard
// library.
//
// The puck is the first piece of world state agents can *change*. Everything
// before it either stood still (walls, beacons) or was written and read back
// through a field nobody owned (the trail). A puck has one position, several
// agents in contact with it at once, and an outcome every one of them shares --
// which is exactly the setup group fitness sharing was built for and never
// measured against.
//
// How it is pushed, and why it is not the obvious thing. The obvious thing is
// to sum penetration depths: agent overlaps puck, puck moves out. That cannot
// work here, because the agent step resolves its own overlap first, so by the
// time the puck is integrated there is no penetration left to read. The push is
// therefore taken from the *approach velocity* along the contact normal --
// which is what a push is anyway -- and measured relative to the puck, so an
// agent cannot push a puck that is already outrunning it.

// How big the puck is, as a fraction of the arena radius, so it scales with the
// room the way every other length here does. Several bodies across on purpose:
// one agent should be able to move it and a group should be able to move it
// faster, which needs a contact arc wide enough for several agents to share.
//
// This is only the value a puck is created with -- each puck carries its own
// radius in its record, and the contact tests read that -- so the slider changes
// the world without anything else having to be told.
const float PuckRadiusRatio = 0.060f; // ~11 cm in the small arena, 5 body diameters

VKEXP_PUCK_FN float puckRadius(float worldRadius, float radiusRatio) {
    return worldRadius * radiusRatio;
}

// A skin on the contact test. The agent step pushes an agent clear of the puck
// in the same step, so a test at exactly touching distance would find nobody in
// contact and the puck would never move at all.
const float PuckContactSkin = 0.012f; // m

// How far from the puck the approach reward still pays anything, in puck radii.
// Tied to the puck rather than to the light range on purpose: at light range the
// reward is a broad haze over most of the arena, and loitering in the general
// area collects most of what pushing the puck to the middle would pay. Tied to
// the puck it pays for being *at* it, which is where pushing starts.
const float PuckApproachReach = 6.0f;

// And how much of the tracking weight that approach reward is allowed to be.
// The weight means "per second for being near the thing you are meant to track",
// which is the right rate in a world where being near the beacon *is* the task.
// Here it is not: being near the puck is how pushing starts, and at the full
// rate a trial spent leaning on the puck out-earns a trial spent delivering it.
// So the search reward keeps a quarter and the work below keeps the rest. The
// slider still scales it, and setting the slider to zero still turns the search
// reward off without touching what pushing pays.
const float PuckProximityShare = 0.25f;

// Per second, per agent in contact, applied to the approach speed. Chosen so
// one agent at the speed limit settles the puck at roughly a third of its own
// speed and a group moves it faster: cooperation has to pay, or the world is
// not asking the question it exists to ask.
const float PuckPushRate = 2.0f;  // 1/s
const float PuckDrag = 3.0f;      // 1/s, applied as exp(-drag * dt)

// What one agent's share of the pushing is, right now, in metres per second of
// useful approach. This is the answer to the credit-assignment problem the world
// created: the puck is one object, the outcome is joint, and every score derived
// from the puck's position is therefore identical for every agent in the world --
// the one that shoved it home and the one that stood in the way are scored the
// same. Selection cannot separate behaviours it cannot see apart, so what it
// saw was that standing near the puck pays and pushing costs motor effort, and
// it duly evolved agents that lean on the nearest face of the puck and block it.
//
// The fix is not to say which side to push from. That would be handing over the
// answer, and the world exists to ask the question. It is to pay each agent for
// the work it actually did, which is a physical quantity and not an opinion: the
// same approach speed the puck integrates, projected onto the direction the puck
// has to travel. An agent wedged between the puck and the middle projects
// negative and earns nothing -- but nothing told it that side was wrong, only
// that its pushing does not move the puck where the puck has to go.
//
// Zero and not a penalty, deliberately. Blocking should stop being paid for; it
// should not become a thing to actively avoid, or an agent learns to keep clear
// of the puck rather than to get behind it.
VKEXP_PUCK_FN float puckPushContribution(vec2 agentPosition, vec2 agentVelocity, float agentRadius,
                                         vec2 puckPosition, vec2 puckVelocity, float radius) {
    const float offsetX = puckPosition.x - agentPosition.x;
    const float offsetY = puckPosition.y - agentPosition.y;
    const float distance = length(vec2(offsetX, offsetY));
    // The same contact test the puck pass uses, skin included, so an agent is
    // credited exactly when it is one of the agents actually moving the puck.
    if (distance <= 1.0e-6f || distance > radius + PuckContactSkin + agentRadius) {
        return 0.0f;
    }
    const float normalX = offsetX / distance;
    const float normalY = offsetY / distance;
    const float approach = (agentVelocity.x - puckVelocity.x) * normalX +
                           (agentVelocity.y - puckVelocity.y) * normalY;
    if (approach <= 0.0f) {
        return 0.0f;
    }
    // Where the puck still has to go. Taken from the puck's own position rather
    // than passed in, because the destination is the middle in this world and a
    // puck already there has nowhere left to be pushed.
    const float remaining = length(puckPosition);
    if (remaining <= 1.0e-6f) {
        return 0.0f;
    }
    const float useful = -(puckPosition.x * normalX + puckPosition.y * normalY) / remaining;
    return approach * max(useful, 0.0f);
}

// Fitness per metre of useful approach. Sized against the delivery it is meant
// to lead to rather than picked: a lone agent at the speed limit holds the puck
// at roughly 0.24 m/s with 0.36 m/s of approach behind it, so the 1.1 m journey
// takes about 4.6 s and banks about 1.5 -- which at this weight is six, the same
// order as the whole-journey progress term the world already pays. That is the
// balance being aimed at. The joint part of the score says the puck arrived; this
// part says who moved it, and neither should drown the other out.
const float PuckWorkReward = 4.0f;

// Where the puck starts: on the world's vertical axis, this far up or down.
const float PuckStartOffset = 0.60f; // fraction of the world radius

// Which side of the centre line the puck starts on. By trial, so one genome
// meets both and pushing always the same way cannot stand in for perceiving
// where the puck actually is.
VKEXP_PUCK_FN float puckStartSide(uint trial) { return (trial & 1u) == 0u ? 1.0f : -1.0f; }

VKEXP_PUCK_FN vec2 puckStartPosition(float worldRadius, uint trial) {
    return vec2(0.0f, puckStartSide(trial) * PuckStartOffset * worldRadius);
}

// How far the puck got, in rungs, and why it is rungs on one journey rather than
// the two named goals it started as.
//
// The world was specified with a minimum -- push the puck past the arena's
// middle line -- and a maximum: push it into a disc around the centre. The
// geometry does not allow that order. The disc straddles the line and the puck
// arrives from outside, so it enters the disc *before* it reaches the line: with
// the default sliders, after 0.64 m of a 1.10 m journey. The minimum was the
// harder of the two, and never fired first, so the ladder had one rung where it
// looked like two: a world scored nothing at all until the puck was in, and then
// scored full marks. A puck brought fifty-seven per cent of the way counted the
// same as a puck nobody had touched, and the reported curve could only move in
// whole worlds.
//
// So the journey is the thing being measured, and the disc is where it ends. The
// rungs are equal fractions of the distance from where the puck was placed to
// the disc's edge, which makes the top rung mean exactly what the maximum always
// meant, makes every rung strictly harder than the one below it by construction,
// and follows the target-radius slider without anything having to be retuned.
//
// Four of them: enough that a population moving the puck at all shows up as a
// rising curve, few enough that each rung is a real step and not noise.
const uint PuckLevelNone = 0u;
const uint PuckLevelCount = 4u;

VKEXP_PUCK_FN float puckTargetRadius(float worldRadius, float targetRadiusRatio) {
    return worldRadius * targetRadiusRatio;
}

VKEXP_PUCK_FN bool puckInsideTarget(vec2 position, float targetRadius) {
    return length(position) <= targetRadius;
}

// What fraction of its journey the puck has covered: 0 where it was placed, 1 at
// the disc's edge, and clamped at both ends so a puck shoved backwards reports
// nothing rather than a negative.
VKEXP_PUCK_FN float puckJourneyFraction(float distance, float startDistance, float targetRadius) {
    const float journey = max(startDistance - targetRadius, 1.0e-4f);
    const float remaining = max(distance - targetRadius, 0.0f);
    const float covered = 1.0f - remaining / journey;
    return covered < 0.0f ? 0.0f : (covered > 1.0f ? 1.0f : covered);
}

// And which rung that is. The top rung is reached only at a fraction of exactly
// one, which is the disc, so "level == PuckLevelCount" and "inside the target"
// are the same statement -- the epsilon is there to make the boundary land on
// the rung rather than a float ulp below it, and the clamp keeps it from landing
// one above.
VKEXP_PUCK_FN uint puckLevelForJourney(float covered) {
    const float scaled = covered * float(PuckLevelCount) + 1.0e-4f;
    if (scaled <= 0.0f) {
        return PuckLevelNone;
    }
    const uint level = uint(scaled);
    return level > PuckLevelCount ? PuckLevelCount : level;
}
