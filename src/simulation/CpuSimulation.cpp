#include "vkexp/simulation/CpuSimulation.hpp"

#include "vkexp/simulation/Sensors.hpp"
#include "vkexp/worlds/WorldScenario.hpp"

#include <algorithm>
#include <cmath>

namespace vkexp {
namespace {

void setComponent(Float4& value, const std::size_t index, const float contact) {
    float* components[] = {&value.x, &value.y, &value.z, &value.w};
    *components[index] = std::max(*components[index], contact);
}

void recordWallContact(AgentState& agent, const float directionX, const float directionY,
                       const float strength) {
    constexpr float tau = 6.28318530718F;
    constexpr float halfSector = tau / 16.0F;
    float relative = std::fmod(std::atan2(directionY, directionX) - agent.pose.z + halfSector, tau);
    if (relative < 0.0F) {
        relative += tau;
    }
    const std::size_t sector = static_cast<std::size_t>(relative / tau * 8.0F) % 8;
    Float4& contacts = sector < 4 ? agent.wallTouch0 : agent.wallTouch1;
    setComponent(contacts, sector % 4, std::clamp(strength, 0.0F, 1.0F));
}

} // namespace

void stepAgentCpu(AgentState& agent,
                  const std::span<const float> weights,
                  const SimulationStep& settings) {
    const ScenarioDefinition& scenario = scenarioDefinition(settings.beaconScenario);
    if (scenario.beforeStep != nullptr) {
        scenario.beforeStep(agent, settings);
    }
    const neuro::BrainShape brain = resolvedBrain(scenario, settings);
    // The neuron state lives on the agent, so the CPU path carries it the same
    // way the shader does -- read it out, integrate, put it back -- rather than
    // keeping a parallel store that could drift out of step with the GPU's.
    // Every layer's neurons, not just the first layer's: the states of all
    // layers live end to end in the one block, so carrying `hiddenCount` of them
    // silently drops every neuron past the first layer -- which is a deep brain
    // that forgets everything behind its front layer on every step.
    neuro::HiddenState hidden{};
    const std::size_t neuronCount = brain.hiddenTotal();
    for (std::size_t neuron = 0; neuron < neuronCount; ++neuron) {
        hidden[neuron] = agentHiddenState(agent, neuron);
    }
    const neuro::Outputs output =
        neuro::evaluate(weights, sampleAgentInputs(agent, settings), hidden, settings.deltaTime,
                        static_cast<neuro::kernel::uint>(settings.neuronModel), brain);
    for (std::size_t neuron = 0; neuron < neuronCount; ++neuron) {
        setAgentHiddenState(agent, neuron, hidden[neuron]);
    }
    const float left = output[0];
    const float right = output[1];
    const float forwardX = std::cos(agent.pose.z);
    const float forwardY = std::sin(agent.pose.z);
    const float drive = 0.5F * (left + right) * settings.thrust;

    agent.motion.x += forwardX * drive * settings.deltaTime;
    agent.motion.y += forwardY * drive * settings.deltaTime;
    const float linearDamping = std::exp(-settings.linearDrag * settings.deltaTime);
    agent.motion.x *= linearDamping;
    agent.motion.y *= linearDamping;
    agent.motion.z += (right - left) * settings.turnAcceleration * settings.deltaTime;
    agent.motion.z *= std::exp(-settings.angularDrag * settings.deltaTime);
    const float speed =
        std::sqrt(agent.motion.x * agent.motion.x + agent.motion.y * agent.motion.y);
    if (speed > settings.maximumSpeed) {
        const float scale = settings.maximumSpeed / speed;
        agent.motion.x *= scale;
        agent.motion.y *= scale;
    }
    agent.motion.z =
        std::clamp(agent.motion.z, -settings.maximumAngularSpeed, settings.maximumAngularSpeed);
    agent.pose.x += agent.motion.x * settings.deltaTime;
    agent.pose.y += agent.motion.y * settings.deltaTime;
    agent.pose.z += agent.motion.z * settings.deltaTime;
    agent.wallTouch0 = {};
    agent.wallTouch1 = {};
    agent.agentTouch0 = {};
    agent.agentTouch1 = {};

    // Static geometry, before the arena boundary so the arena has the last word
    // on containment. An obstacle reports through the same tactile channel a
    // wall does: to the network a barrier is a barrier, and nothing here adds an
    // input the brain would have to be re-shaped for.
    for (std::uint32_t index = 0; index < scenario.obstacleCount; ++index) {
        const ObstacleBox box = scenario.obstacle(index, agent, settings);
        const worlds::kernel::vec2 push = worlds::kernel::boxPushOut(
            {agent.pose.x, agent.pose.y}, agent.pose.w, {box.centre.x, box.centre.y},
            {box.halfExtent.x, box.halfExtent.y});
        const float pushLength = std::sqrt(push.x * push.x + push.y * push.y);
        if (pushLength <= 0.0F) {
            continue;
        }
        agent.pose.x += push.x;
        agent.pose.y += push.y;
        const float normalX = push.x / pushLength;
        const float normalY = push.y / pushLength;
        const float into = -(agent.motion.x * normalX + agent.motion.y * normalY);
        if (into > 0.0F) {
            const float impulse = into * (1.0F + settings.collisionRestitution);
            agent.motion.x += normalX * impulse;
            agent.motion.y += normalY * impulse;
        }
        recordWallContact(agent, normalX, normalY,
                          0.25F + std::max(into, 0.0F) / settings.maximumSpeed);
    }

    const float maximumCenterDistance = std::max(settings.worldRadius - agent.pose.w, 0.0F);
    if (settings.worldShape == WorldShape::Circle) {
        const float centerDistance =
            std::sqrt(agent.pose.x * agent.pose.x + agent.pose.y * agent.pose.y);
        if (centerDistance > maximumCenterDistance) {
            const float normalX = agent.pose.x / centerDistance;
            const float normalY = agent.pose.y / centerDistance;
            agent.pose.x = normalX * maximumCenterDistance;
            agent.pose.y = normalY * maximumCenterDistance;
            const float outward = agent.motion.x * normalX + agent.motion.y * normalY;
            if (outward > 0.0F) {
                agent.motion.x -= normalX * outward * 1.5F;
                agent.motion.y -= normalY * outward * 1.5F;
            }
            recordWallContact(agent, normalX, normalY,
                              0.25F + std::max(outward, 0.0F) / settings.maximumSpeed);
        }
    } else {
        if (agent.pose.x > maximumCenterDistance) {
            agent.pose.x = maximumCenterDistance;
            recordWallContact(agent, 1.0F, 0.0F,
                              0.25F + std::max(agent.motion.x, 0.0F) / settings.maximumSpeed);
            agent.motion.x = std::min(agent.motion.x, 0.0F) - std::max(agent.motion.x, 0.0F) * 0.5F;
        } else if (agent.pose.x < -maximumCenterDistance) {
            agent.pose.x = -maximumCenterDistance;
            recordWallContact(agent, -1.0F, 0.0F,
                              0.25F - std::min(agent.motion.x, 0.0F) / settings.maximumSpeed);
            agent.motion.x = std::max(agent.motion.x, 0.0F) - std::min(agent.motion.x, 0.0F) * 0.5F;
        }
        if (agent.pose.y > maximumCenterDistance) {
            agent.pose.y = maximumCenterDistance;
            recordWallContact(agent, 0.0F, 1.0F,
                              0.25F + std::max(agent.motion.y, 0.0F) / settings.maximumSpeed);
            agent.motion.y = std::min(agent.motion.y, 0.0F) - std::max(agent.motion.y, 0.0F) * 0.5F;
        } else if (agent.pose.y < -maximumCenterDistance) {
            agent.pose.y = -maximumCenterDistance;
            recordWallContact(agent, 0.0F, -1.0F,
                              0.25F - std::min(agent.motion.y, 0.0F) / settings.maximumSpeed);
            agent.motion.y = std::max(agent.motion.y, 0.0F) - std::min(agent.motion.y, 0.0F) * 0.5F;
        }
    }
    const float wallContactAmount = agent.wallTouch0.x + agent.wallTouch0.y + agent.wallTouch0.z +
                                    agent.wallTouch0.w + agent.wallTouch1.x + agent.wallTouch1.y +
                                    agent.wallTouch1.z + agent.wallTouch1.w;
    agent.penalties.x += wallContactAmount * settings.wallCollisionPenalty * settings.deltaTime;

    const float motorCost = (std::abs(left) + std::abs(right)) * settings.deltaTime;
    const float signalIntensity = std::max(output[5], 0.0F);
    const float signalCost =
        signalIntensity * settings.deltaTime * settings.fitness.signalCostFactor;
    agent.motion.w =
        std::max(0.0F, agent.motion.w - (motorCost + signalCost) * settings.fitness.energyDrain);
    agent.signal = {output[2] * 0.5F + 0.5F, output[3] * 0.5F + 0.5F, output[4] * 0.5F + 0.5F,
                    signalIntensity};
    if (brain.outputCount >=
        neuro::Topology::actuatorOutputCount + neuro::Topology::recurrentMemoryCount) {
        agent.internal.z = output[6];
        agent.internal.w = output[7];
    } else {
        agent.internal.z = 0.0F;
        agent.internal.w = 0.0F;
    }

    const float distance = nearestBeaconDistance(agent, settings);
    agent.metrics.y = std::min(agent.metrics.y, distance);
    agent.metrics.z += motorCost + signalCost;
    scenario.afterStep(agent, settings, distance);
}

float agentFitness(const AgentState& agent, const BeaconScenario scenario,
                   const FitnessWeights& weights) {
    return scenarioDefinition(scenario).fitness(agent, weights);
}

} // namespace vkexp
