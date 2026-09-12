#pragma once

#include <cstdint>

// Space is discrete now: a position is a cell, a move is one cell, and there is
// no length to declare. What survives the move off the metric arena is the
// step's two time bases, which still have to meet somewhere, and this is where.
//
// Steps and seconds both stay, with different jobs:
//   * a step is the unit of reproducibility. A run is replayed by step count,
//     genome archives record steps, run snapshots record the step they stopped
//     on, and the CPU/GPU parity tests compare step for step. Nothing about a
//     replay depends on wall-clock time.
//   * a second is the unit of anything that decays or accumulates -- which,
//     since the lattice itself is discrete, now means the neurons: a time
//     constant is a duration, so how long a neuron remembers is a fact about
//     the brain rather than about the step rate it happened to run at.
//
// The rule that keeps the two consistent: a quantity accumulated by the step is
// multiplied by `deltaTime`, and a fraction removed per step is written as
// `1 - exp(-rate * deltaTime)`. `testNeuronTimeConstants` checks it where it
// still applies -- one time constant of stepping closes the same fraction of
// the gap at 30, 60 and 240 Hz.
//
// Costs charged per event are the other half of the same rule and do not take
// `deltaTime` at all: a move costs what a move costs, and a refusal likewise,
// because both are counted rather than integrated.
namespace vkexp::units {

// Integration rate. Sensing, the brain and the move all advance together at
// this rate; there is no substepping.
inline constexpr float simulationRateHz = 60.0F;
inline constexpr float fixedTimeStep = 1.0F / simulationRateHz;

[[nodiscard]] constexpr float secondsForSteps(const std::uint32_t steps, const float deltaTime) {
    return static_cast<float>(steps) * deltaTime;
}

} // namespace vkexp::units
