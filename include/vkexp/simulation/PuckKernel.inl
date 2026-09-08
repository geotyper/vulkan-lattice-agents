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

// Radii. The puck is deliberately several bodies across: one agent should be
// able to move it and a group should be able to move it faster, which needs a
// contact arc wide enough for several agents to share.
const float PuckRadius = 0.075f; // m, about 3.4 body diameters

// A skin on the contact test. The agent step pushes an agent clear of the puck
// in the same step, so a test at exactly touching distance would find nobody in
// contact and the puck would never move at all.
const float PuckContactSkin = 0.012f; // m

// Per second, per agent in contact, applied to the approach speed. Chosen so
// one agent at the speed limit settles the puck at roughly a third of its own
// speed and a group moves it faster: cooperation has to pay, or the world is
// not asking the question it exists to ask.
const float PuckPushRate = 2.0f;  // 1/s
const float PuckDrag = 3.0f;      // 1/s, applied as exp(-drag * dt)

// Where the puck starts: on the world's vertical axis, this far up or down.
const float PuckStartOffset = 0.60f; // fraction of the world radius

// Which side of the centre line the puck starts on. By trial, so one genome
// meets both and pushing always the same way cannot stand in for perceiving
// where the puck actually is.
VKEXP_PUCK_FN float puckStartSide(uint trial) { return (trial & 1u) == 0u ? 1.0f : -1.0f; }

VKEXP_PUCK_FN vec2 puckStartPosition(float worldRadius, uint trial) {
    return vec2(0.0f, puckStartSide(trial) * PuckStartOffset * worldRadius);
}

// The two thresholds, in the order they are worth. Level 1 is the halfway line:
// the puck reached the far side of the arena's middle, wherever along it. Level
// 2 is the middle itself, a disc whose radius is a slider.
//
// Reported as a maximum and not a sum, so the number is monotone: a puck that
// is inside the disc counts 2 whether or not it happened to cross the line on
// the way, and a curve that rises always means a puck that got further.
const uint PuckLevelNone = 0u;
const uint PuckLevelCrossedLine = 1u;
const uint PuckLevelInsideTarget = 2u;
const uint PuckLevelCount = 2u;

VKEXP_PUCK_FN bool puckCrossedLine(float startSide, float positionY) {
    return startSide * positionY <= 0.0f;
}

VKEXP_PUCK_FN bool puckInsideTarget(vec2 position, float targetRadius) {
    return length(position) <= targetRadius;
}

VKEXP_PUCK_FN float puckTargetRadius(float worldRadius, float targetRadiusRatio) {
    return worldRadius * targetRadiusRatio;
}
