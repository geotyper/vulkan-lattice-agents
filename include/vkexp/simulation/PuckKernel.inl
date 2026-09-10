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
// How it is pushed, and why it is not either of the two obvious things.
//
// The first obvious model sums penetration depths: agent overlaps puck, puck
// moves out. That cannot work here, because the agent step resolves its own
// overlap first, so by the time the puck is integrated there is no penetration
// left to read.
//
// The second was approach velocity along the contact normal, and it was what
// this world ran on first. It is wrong in a way that only shows up once the
// world asks for a group. Velocity is what an *impact* carries, so the model
// rewarded a run-up: one agent charging in cleanly moved the puck further than
// several leaning on it, because agents crowding a puck collide with each other
// and lose most of their inward speed. The world was asking to be solved by a
// battering ram at exactly the moment it was meant to start asking for tugboats.
//
// So the push is a *force*: how hard each agent in contact is driving into the
// puck, which is its motor command projected on the contact normal, and nothing
// to do with how fast it happens to be going. An agent wedged motionless in a
// crowd still pushes with everything it has, forces from several agents add, and
// a charge is worth no more than steady pressure. The puck's speed then follows
// from force against its own drag, the way a towed thing does.

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

// What one agent pressing at full drive does to the puck, and what resists it.
// Chosen against the equilibrium they settle at rather than picked: force over
// drag, so one agent holds the puck at 0.18 m/s, two at 0.37 and three at 0.55,
// which is the agents' own speed limit. Cooperation therefore pays linearly, and
// the puck cannot be made to outrun the things pushing it.
const float PuckPushAcceleration = 0.55f; // m/s^2 per agent pressing head-on
const float PuckDrag = 3.0f;              // 1/s, applied as exp(-drag * dt)

// How hard one agent presses, in agents: 1 is an agent driving at full throttle
// straight at the puck's centre. Off-axis it presses by the cosine, and an agent
// facing away presses nothing however close it is standing.
//
// `drive` is the agent's own forward command, already normalised, which is the
// whole point: it does not fall when the agent is blocked, so leaning works.
VKEXP_PUCK_FN float puckPressure(float drive, float alignment) {
    if (drive <= 0.0f || alignment <= 0.0f) {
        return 0.0f;
    }
    return drive * alignment;
}

// How hard the whole world has to push before the puck moves at all, counted in
// agents: 1.0 is one agent driving at full throttle straight at it.
//
// This is the knob that decides whether the world asks for cooperation or only
// permits it. Without it one agent moves the puck on its own, so a group is a
// convenience and never a requirement, and the interesting question -- can
// selection produce agents that push together -- is one the world never puts.
// Above one, a single agent cannot start it however hard it tries, and two have
// to be in contact at the same time and pushing the same way.
//
// Not a mass, deliberately. Mass makes one agent slower, not powerless: the puck
// still creeps, the score still rises, and the population still learns to solve
// it alone. A friction floor is a threshold, which is what "two or more" means.
//
// Subtracted from the push rather than switching it on and off, so a pair that
// barely clears the floor moves the puck slowly instead of the world flipping
// between nothing and everything. The same reason the journey is a fraction and
// not a completion: selection needs an increment, not a cliff.
const float PuckBreakawayPushes = 1.60f; // agents at full speed

// In agents, and the pressure it is compared against is in agents too, so the
// slider means exactly what it says. It did not, while the push was a velocity:
// the threshold then had to be scaled by the speed limit and "two agents" meant
// two agents *travelling at the limit*, which is not what standing on a puck
// looks like.
VKEXP_PUCK_FN float puckBreakawayPush(float breakawayPushes) {
    return max(breakawayPushes, 0.0f);
}

// What fraction of the world's push survives the friction floor. Zero below it,
// rising from zero above, so the transition has a slope.
VKEXP_PUCK_FN float puckFrictionFraction(float pushMagnitude, float breakaway) {
    if (pushMagnitude <= breakaway) {
        return 0.0f;
    }
    return (pushMagnitude - breakaway) / pushMagnitude;
}

// And how much of the work reward an agent collects, which has to follow the
// puck rather than the pushing. With a friction floor a lone agent can lean on a
// puck at full speed for a whole trial and move nothing, and paying for that
// would teach exactly the futile behaviour the floor exists to rule out. The
// reward therefore rides on the puck actually moving: nothing while it is stuck,
// full once it is under way.
//
// The ramp is short -- a quarter of an agent's speed limit -- because this is
// meant to separate stuck from moving, not to rank speeds. Speed is already paid
// for by the journey.
const float PuckWorkMovingSpeed = 0.25f; // fraction of the agent speed limit

VKEXP_PUCK_FN float puckWorkMovingFraction(float puckSpeed, float maximumSpeed) {
    const float full = max(maximumSpeed * PuckWorkMovingSpeed, 1.0e-4f);
    const float fraction = puckSpeed / full;
    return fraction < 0.0f ? 0.0f : (fraction > 1.0f ? 1.0f : fraction);
}

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
VKEXP_PUCK_FN float puckPushContribution(vec2 agentPosition, vec2 agentHeading, float drive,
                                         float agentRadius, vec2 puckPosition, float radius) {
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
    // The same pressure the pass integrates, so the reward and the physics
    // cannot answer differently about who is pushing.
    const float approach =
        puckPressure(drive, agentHeading.x * normalX + agentHeading.y * normalY);
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

// The other placement: anywhere in the arena rather than in front of the agents
// who have to move it. On the axis the puck is found by walking forward, and a
// population can learn the world without ever learning to look for it; scattered,
// finding it is part of the task and the journey is a different length every
// generation, so nothing about one layout can be memorised.
//
// The placement has to be the same number on both sides of the language boundary
// and the same number every time a generation is replayed, so it comes from
// scenarioRandom01 -- the hash the scenario kernel already uses to place a
// relocating home -- rather than from a second one written here. Seeded by the
// world and the generation together, so worlds differ from each other within a
// generation and every world differs from itself between them.
//
// Kept off both ends of the arena: too near the rim and there is no room to get
// behind it, too near the middle and it starts most of the way home.
const float PuckScatterInner = 0.35f; // fraction of the arena radius
const float PuckScatterOuter = 0.78f;

VKEXP_PUCK_FN vec2 puckScatteredPosition(float worldRadius, uint world, uint seed) {
    const uint key = world * 0x9e3779b9u + seed;
    const float angle = scenarioRandom01(key) * ScenarioTau;
    // Square-rooted so the placement is uniform over the ring's area rather than
    // over its radius, which would crowd the pucks toward the inner edge.
    const float unit = scenarioRandom01(key ^ 0x68bc21ebu);
    const float inner = PuckScatterInner * PuckScatterInner;
    const float span = PuckScatterOuter * PuckScatterOuter - inner;
    const float radius = worldRadius * sqrt(inner + unit * span);
    return vec2(cos(angle) * radius, sin(angle) * radius);
}

// The one entry point both the driver and the scenario's spawn go through, so
// the puck and the distance the shaping banks against cannot be placed from two
// different answers.
VKEXP_PUCK_FN vec2 puckStartPositionFor(float worldRadius, uint trial, uint world, uint seed,
                                        bool scattered) {
    if (scattered) {
        return puckScatteredPosition(worldRadius, world, seed);
    }
    return puckStartPosition(worldRadius, trial);
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
