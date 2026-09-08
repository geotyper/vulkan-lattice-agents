// A shared puck and a lit disc in the middle to push it into. The puck itself
// is not a beacon: it is state in its own buffer, integrated by puck_step.comp,
// and what this file describes is only the marker standing on the target.
//
// floats0 = {target radius ratio, unused, unused, unused}.
// Packed by gpuParameters in src/worlds/scenarios/PuckPushScenario.cpp.

vec2 puckPushScenarioPosition() { return vec2(0.0); }

vec3 puckPushScenarioColor() { return vec3(0.35, 1.00, 0.55); }
