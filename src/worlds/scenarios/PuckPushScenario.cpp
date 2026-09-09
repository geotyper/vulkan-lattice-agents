#include "vkexp/worlds/scenarios/PuckPushScenario.hpp"

#include "vkexp/simulation/PuckKernel.hpp"
#include "vkexp/worlds/ScenarioMath.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp::worlds::puck_push {
namespace {

namespace kernel = worlds::kernel;
namespace puck = ::vkexp::puck::kernel;

// Both the destination and the puck are lit, and the puck's light is the whole
// reason this world can be learned at all.
//
// It was not, at first. The photoreceptors see beacons and other agents' signals
// and nothing else, so a puck that was neither was invisible: an agent could
// only discover it by walking into it. The fitness paid for being near it and
// for moving it, but a population cannot climb a gradient it has no sense of --
// the reward existed and the handle on it did not. So the puck emits, its
// position taken from the mirror every agent already carries, and finding it
// becomes something an agent can steer at rather than something it stumbles on.
constexpr Float4 targetColor{0.35F, 1.00F, 0.55F, 0.0F};
constexpr Float4 puckColor{1.00F, 0.92F, 0.70F, 0.0F};

// What delivering is worth against what loitering is worth, which is the other
// half of why nothing was learned. Being near the puck used to pay trackingReward
// per second -- 0.25, so 3.75 over a fifteen-second trial for an agent that
// simply parks on it. Pushing the puck all the way in paid about the same: the
// raw progress term is metres, and there are only 1.1 of them to the middle.
// Two behaviours worth the same is not a gradient, and the easier one wins.
//
// Progress is normalised to the fraction of the way the puck has come, so it
// does not depend on arena size, and weighted so that delivering is worth
// several times loitering: 12 for the journey plus two levels of objective
// bonus, against the 0.94 a parked agent now collects once the proximity reward
// is cut to its share.
constexpr float puckProgressReward = 12.0F;

// Rungs on one journey, and the completion ratio is read against all of them:
// full means the pucks are sitting in the disc, and everything below it says how
// far they got. See PuckLevelCount for why the two named goals became a ladder.
constexpr std::uint32_t levelsPerWorld = puck::PuckLevelCount;

// What a delivery is worth in objective bonuses, held at what it was worth when
// the ladder had two rungs. The rungs are a reporting change: they gave the
// curve a gradient it did not have, and paying four bonuses where two were paid
// before would have moved the score at the same time and made the run before the
// change incomparable with the run after it.
constexpr float deliveryBonusLevels = 2.0F;

// Every agent in a world is scored on the same puck, which is the point: the
// outcome is joint, so an individual's contribution is worth something only if
// what the world achieves is worth something. That is the condition group
// fitness sharing was built for and has never been measured against.
//
// The reported ratio is this over levelsPerWorld, so it reads as the average
// fraction of the journey a world's puck covered rather than as the share of
// worlds that finished. That is the number that moves generation to generation;
// the share that finished only moves in whole worlds, which is what made it look
// flat when it was not.
std::uint32_t achievedObjectives(const AgentState& agent) {
    return std::min(static_cast<std::uint32_t>(std::max(agent.target.w, 0.0F)), levelsPerWorld);
}

float fitness(const AgentState& agent, const FitnessWeights& weights) {
    // Not objectiveFitness: that adds the progress term in metres, and here it
    // has to be a fraction of the journey and weighted against the loitering
    // reward. Everything else is the same shape as every other world's score.
    const float start = std::max(agent.metrics.x, 1.0e-4F);
    const float progress = std::clamp((agent.metrics.x - agent.metrics.y) / start, 0.0F, 1.0F);
    const float level = static_cast<float>(achievedObjectives(agent)) /
                        static_cast<float>(levelsPerWorld) * deliveryBonusLevels;
    return agent.metrics.w + progress * puckProgressReward + level * weights.objectiveBonus -
           agent.metrics.z * weights.motorCostWeight - agent.penalties.x;
}

// Two shaping terms, and they answer different questions.
//
// Being near the puck pays a little, per second. Without it an untrained
// population scores identically -- it barely reaches the puck at all -- and
// selection has nothing to work with. Squared, so the reward concentrates near
// the puck rather than spreading a shallow gradient over the whole arena.
//
// Pushing the puck toward the middle pays properly, and pays the agent doing it.
// Every other term here is read off the shared puck and is therefore the same
// number for all twelve agents in a world, which is what let a population settle
// on leaning against the near face of the puck and blocking it: that behaviour
// collected the proximity reward and cost nothing, and no term in the score
// could tell it apart from pushing. See puckPushContribution for why the answer
// is to price the work rather than to name the correct side.
void afterStep(AgentState& agent, const SimulationStep& settings, float) {
    rewardPuckProximity(agent, settings);
    rewardPuckWork(agent, settings);
}

// Two: the lit disc in the middle, which is a marker and not a goal to arrive
// at, and the puck itself, whose position comes from the mirror every agent
// already carries. Nothing here scores an agent for touching either one.
ActiveBeacons activeBeacons(const AgentState& agent) {
    return {{{Beacon{{0.0F, 0.0F, 0.0F, 0.0F}, targetColor},
              Beacon{{agent.penalties.y, agent.penalties.z, 0.0F, 0.0F}, puckColor}}},
            2};
}

// Agents start spread over the side the puck starts on, so the first thing they
// have to do is find it rather than already be behind it.
void spawn(AgentState& agent, const SimulationStep& settings) {
    const auto trial = static_cast<kernel::uint>(std::max(agent.target.z, 0.0F));
    const auto world = static_cast<kernel::uint>(std::max(agent.penalties.w, 0.0F));
    if (settings.puckRandomStart) {
        // Scattered: the agents stay where the driver's spiral put them and the
        // puck is somewhere in the arena, so the first thing they have to do is
        // look for it rather than walk forward.
        agent.pose.x *= 0.85F;
        agent.pose.y *= 0.85F;
    } else {
        const float side = puck::puckStartSide(trial);
        agent.pose.x *= 0.55F;
        agent.pose.y = agent.pose.y * 0.28F + side * settings.worldRadius * 0.82F;
    }
    // Seed the mirror with where the puck actually starts. Without this the
    // shaping opens the trial believing the puck is already at the origin, so
    // the distance it banks progress against is zero and no progress toward the
    // middle can ever be positive -- a world that silently scores nothing.
    const puck::vec2 start = puck::puckStartPositionFor(settings.worldRadius, trial, world,
                                                       settings.beaconMotionSeed,
                                                       settings.puckRandomStart);
    agent.penalties.y = start.x;
    agent.penalties.z = start.y;
    // And with the puck at rest, which is where the mirrored velocity the work
    // reward reads against starts. Without it the slot still holds the base
    // beacon this world does not use, and the first steps of a trial measure an
    // approach against a puck the agent believes is already moving.
    agent.target.x = 0.0F;
    agent.target.y = 0.0F;
}

// floats0 = {target radius ratio, puck radius ratio, breakaway pushes, unused}.
// The puck's physics is in the shared kernel and needs nothing packed; what has
// to be sent is the three numbers that are sliders.
ScenarioParameterBlock gpuParameters(const SimulationStep& settings) {
    return {{settings.puckTargetRadiusRatio, settings.puckRadiusRatio,
             settings.puckBreakawayPushes, 0.0F},
            {},
            {}};
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
        .objectiveLabel = "Puck journey",
        .radiusLabel = "Orbit radius",
        .description = "Push a shared puck to the middle; the outcome belongs to the whole world",
        .beacons = beacons,
        .beaconCount = 2,
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

ActiveBeacons beacons(const AgentState& agent, const SimulationStep&) {
    return activeBeacons(agent);
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
