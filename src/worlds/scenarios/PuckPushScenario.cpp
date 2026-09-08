#include "vkexp/worlds/scenarios/PuckPushScenario.hpp"

#include "vkexp/simulation/PuckKernel.hpp"
#include "vkexp/worlds/ScenarioMath.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp::worlds::puck_push {
namespace {

namespace kernel = worlds::kernel;
namespace puck = ::vkexp::puck::kernel;

// The middle is lit, so where the puck has to go is perceivable rather than
// something an agent has to infer. This world is about moving a thing together;
// making the destination invisible as well would be two questions at once, and
// the scent relay already asks the other one.
constexpr Float4 targetColor{0.35F, 1.00F, 0.55F, 0.0F};

// Two levels, and the completion ratio is read against both: half means the
// pucks reached the halfway line, full means they are sitting in the middle.
constexpr std::uint32_t levelsPerWorld = puck::PuckLevelCount;

float fitness(const AgentState& agent, const FitnessWeights& weights) {
    return objectiveFitness(agent, static_cast<std::uint32_t>(std::max(agent.target.w, 0.0F)),
                            weights);
}

// Every agent in a world is scored on the same puck, which is the point: the
// outcome is joint, so an individual's contribution is worth something only if
// what the world achieves is worth something. That is the condition group
// fitness sharing was built for and has never been measured against.
std::uint32_t achievedObjectives(const AgentState& agent) {
    return std::min(static_cast<std::uint32_t>(std::max(agent.target.w, 0.0F)), levelsPerWorld);
}

// Being near the puck pays a little, per second. Without it an untrained
// population scores identically -- it barely reaches the puck at all -- and
// selection has nothing to work with. Squared, so the reward concentrates near
// the puck rather than spreading a shallow gradient over the whole arena.
void afterStep(AgentState& agent, const SimulationStep& settings, float) {
    rewardPuckProximity(agent, settings);
}

// One beacon, standing on the target disc. It is a marker and not a goal to
// arrive at: nothing here scores an agent for touching it.
ActiveBeacons activeBeacons(const SimulationStep&) {
    return {{{Beacon{{0.0F, 0.0F, 0.0F, 0.0F}, targetColor}}}, 1};
}

// Agents start spread over the side the puck starts on, so the first thing they
// have to do is find it rather than already be behind it.
void spawn(AgentState& agent, const SimulationStep& settings) {
    const auto trial = static_cast<kernel::uint>(std::max(agent.target.z, 0.0F));
    const float side = puck::puckStartSide(trial);
    agent.pose.x *= 0.55F;
    agent.pose.y = agent.pose.y * 0.28F + side * settings.worldRadius * 0.82F;
    // Seed the mirror with where the puck actually starts. Without this the
    // shaping opens the trial believing the puck is already at the origin, so
    // the distance it banks progress against is zero and no progress toward the
    // middle can ever be positive -- a world that silently scores nothing.
    const puck::vec2 start = puck::puckStartPosition(settings.worldRadius, trial);
    agent.penalties.y = start.x;
    agent.penalties.z = start.y;
}

// floats0 = {target radius ratio, unused, unused, unused}. The puck's own
// physics is in the shared kernel and needs nothing packed; what the scenario
// has to send is the one number that is a slider.
ScenarioParameterBlock gpuParameters(const SimulationStep& settings) {
    return {{settings.puckTargetRadiusRatio, 0.0F, 0.0F, 0.0F}, {}, {}};
}

constexpr neuro::BrainShape brain = neuro::maximumBrainShape;
static_assert(brain.fitsCapacity());

} // namespace

const ScenarioDefinition& definition() {
    static constexpr ScenarioDefinition value{
        .name = "Puck push",
        .key = "puck",
        .id = BeaconScenario::PuckPush,
        .brain = brain,
        .tunables = {.beaconRadiusRatio = false,
                     .beaconAngularSpeed = false,
                     .beaconRandomMotion = false,
                     .forageCargoDecay = false},
        .objectiveLabel = "Puck delivered",
        .radiusLabel = "Orbit radius",
        .description = "Push a shared puck to the middle; the outcome belongs to the whole world",
        .beacons = beacons,
        .beaconCount = 1,
        .targetDistance = targetDistance,
        .phaseForStep = nullptr,
        .fitness = fitness,
        .achievedObjectives = achievedObjectives,
        .objectivesPerAgent = levelsPerWorld,
        .beforeStep = nullptr,
        .afterStep = afterStep,
        .obstacleCount = 0,
        .puck = true,
        .obstacle = nullptr,
        .spawn = spawn,
        .gpuParameters = gpuParameters,
    };
    return value;
}

ActiveBeacons beacons(const AgentState&, const SimulationStep& settings) {
    return activeBeacons(settings);
}

// What the shaping measures. Not the agent's distance to the middle -- an agent
// that parks in the middle without the puck has done nothing -- but the puck's,
// which every agent in the world shares and which is the quantity the task is
// actually about. The puck rides on the agent record in penalties.yz for this,
// so the existing progress machinery, which knows how to bank one distance per
// leg, shapes on it with nothing added beside it.
float targetDistance(const AgentState& agent, const SimulationStep&) {
    return std::hypot(agent.penalties.y, agent.penalties.z);
}

} // namespace vkexp::worlds::puck_push
