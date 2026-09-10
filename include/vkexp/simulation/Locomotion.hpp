#pragma once

#include "vkexp/simulation/AgentTypes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

// How a body answers its motors. The agent step integrates the same four
// numbers whatever is chosen here -- thrust, turn acceleration and the two
// drags -- so a locomotion style is a named point in that space and not a
// second physics model. Nothing new reaches the GPU.
//
// The axis is inertia, and only inertia. A preset never touches the two speed
// caps, so every style has the same top speed and the same top turn rate; what
// changes is how long it takes to get there and how far the body carries when
// the motors stop. That is what makes two runs at different styles comparable:
// the agent can ultimately do the same things, and the question is whether
// selection can control a body that answers slowly.
//
// Drag is the whole story, because the step applies it as exp(-drag * dt): the
// response time constant is 1/drag, the coast after the motors cut is a decay
// with that same constant, and thrust only sets how much headroom there is over
// the speed the drag will hold. Reading the table as time constants:
//
//   Robot   0.06 s   Rover  0.15 s   Table robot  0.59 s
//   Glider  1.2 s    Fish   2.0 s
//
// Velocity is a free vector and thrust is applied along the heading, so a low
// drag also means sideslip: a turning body keeps going the way it was going.
// That is the part that reads as "fish" rather than merely "slow".
namespace vkexp {

enum class LocomotionStyle : std::uint32_t {
    Robot = 0,
    Rover = 1,
    TableRobot = 2,
    Glider = 3,
    Fish = 4,
};

inline constexpr std::size_t locomotionStyleCount = 5;

struct LocomotionPreset {
    LocomotionStyle style{};
    // Shown in the window.
    const char* name{};
    // Stable identifier for command lines; unlike `name` it must not change
    // once runs have been recorded against it.
    const char* key{};
    float thrust{};           // m/s^2 at full forward drive
    float turnAcceleration{}; // rad/s^2 at full differential drive
    float linearDrag{};       // 1/s
    float angularDrag{};      // 1/s
    const char* description{};
};

// In LocomotionStyle order, from the body that answers instantly to the one
// that mostly carries. The middle row is the simulation's own defaults, value
// for value, so selecting it is a return to where every measurement so far was
// taken rather than an approximation of it.
inline constexpr std::array<LocomotionPreset, locomotionStyleCount> locomotionPresets{{
    {LocomotionStyle::Robot, "Robot", "robot", 13.4F, 66.8F, 16.7F, 16.7F,
     "Answers within a few steps and stops in half a body. Speed and heading are "
     "effectively commanded rather than integrated, so control is a lookup and "
     "not a skill."},
    {LocomotionStyle::Rover, "Rover", "rover", 6.0F, 33.0F, 6.67F, 8.33F,
     "A wheeled machine with real but short spin-up: it coasts about two bodies "
     "and has to begin its turns slightly early."},
    {LocomotionStyle::TableRobot, "Table robot", "default", 1.9F, 5.0F, 1.7F, 2.4F,
     "The simulation's own default, and the body every scenario was tuned "
     "against. A third of a second to full speed and seven bodies of coast."},
    {LocomotionStyle::Glider, "Glider", "glider", 0.75F, 4.5F, 0.833F, 1.25F,
     "Momentum is now the problem: a second to accelerate, fifteen bodies of "
     "coast, and enough sideslip that a turn does not immediately change where "
     "the body is going."},
    {LocomotionStyle::Fish, "Fish", "fish", 0.42F, 3.5F, 0.5F, 0.833F,
     "Nearly all glide. Two seconds to full speed, a coast of half the small "
     "arena, and a turn that keeps turning after the motors stop -- arriving "
     "anywhere means deciding well before being there."},
}};

[[nodiscard]] inline const LocomotionPreset& locomotionPreset(const LocomotionStyle style) {
    const auto index = static_cast<std::size_t>(style);
    return locomotionPresets[std::min(index, locomotionStyleCount - 1)];
}

inline void applyLocomotionPreset(SimulationStep& settings, const LocomotionStyle style) {
    const LocomotionPreset& preset = locomotionPreset(style);
    settings.thrust = preset.thrust;
    settings.turnAcceleration = preset.turnAcceleration;
    settings.linearDrag = preset.linearDrag;
    settings.angularDrag = preset.angularDrag;
}

// Which preset a set of sliders is currently sitting on, if any. The window
// needs this because the four values are also editable one at a time: a combo
// that could only ever be set and never read would say "Fish" over a body that
// had since been dragged somewhere else.
[[nodiscard]] inline bool matchesLocomotionPreset(const SimulationStep& settings,
                                                  const LocomotionStyle style) {
    const LocomotionPreset& preset = locomotionPreset(style);
    // A slider step at the coarse end of the widened ranges is worth more than
    // a fixed epsilon, so the comparison is relative.
    const auto near = [](const float value, const float reference) {
        return std::abs(value - reference) <= 0.005F * std::max(std::abs(reference), 1.0F);
    };
    return near(settings.thrust, preset.thrust) &&
           near(settings.turnAcceleration, preset.turnAcceleration) &&
           near(settings.linearDrag, preset.linearDrag) &&
           near(settings.angularDrag, preset.angularDrag);
}

[[nodiscard]] inline const LocomotionPreset* currentLocomotionPreset(const SimulationStep& settings) {
    for (const LocomotionPreset& preset : locomotionPresets) {
        if (matchesLocomotionPreset(settings, preset.style)) {
            return &preset;
        }
    }
    return nullptr;
}

[[nodiscard]] inline const LocomotionPreset* locomotionPresetForKey(const std::string_view key) {
    for (const LocomotionPreset& preset : locomotionPresets) {
        if (key == preset.key) {
            return &preset;
        }
    }
    return nullptr;
}

// What the four numbers actually mean for a body, in units that can be judged
// by eye. Derived rather than tabulated, so a hand-dragged slider is described
// as accurately as a preset -- and so a preset whose numbers are edited cannot
// keep advertising the behaviour it used to have.
struct LocomotionResponse {
    float topSpeed{};        // m/s actually attainable, cap or drag whichever binds first
    float timeToTopSpeed{};  // s to 98% of it from rest at full drive
    float coastDistance{};   // m travelled from top speed once the motors cut
    float topTurnRate{};     // rad/s actually attainable
    float timeToTopTurnRate{}; // s
    float spinCoast{};       // rad turned through after the motors cut
    // Whether the speed cap is what limits the body, or the drag gets there
    // first and the slider is inert. At the default settings this is false for
    // the turn: 5.0 rad/s^2 against 2.4/s holds 2.08 rad/s, so the maximum turn
    // speed slider has nothing to do.
    bool speedCapBinds{};
    bool turnCapBinds{};
};

[[nodiscard]] inline LocomotionResponse locomotionResponse(const SimulationStep& settings) {
    // Time from rest to `fraction` of the attainable value under a first-order
    // approach to `terminal`: tau * ln(terminal / (terminal - target)).
    const auto timeToReach = [](const float terminal, const float target, const float drag) {
        if (drag <= 0.0F || terminal <= target) {
            return 0.0F;
        }
        return std::log(terminal / (terminal - target)) / drag;
    };
    LocomotionResponse response{};
    const float linearTerminal =
        settings.linearDrag > 0.0F ? settings.thrust / settings.linearDrag : settings.maximumSpeed;
    response.speedCapBinds = linearTerminal > settings.maximumSpeed;
    response.topSpeed = std::min(linearTerminal, settings.maximumSpeed);
    response.timeToTopSpeed = timeToReach(linearTerminal, 0.98F * response.topSpeed,
                                          settings.linearDrag);
    response.coastDistance =
        settings.linearDrag > 0.0F ? response.topSpeed / settings.linearDrag : 0.0F;

    const float angularTerminal = settings.angularDrag > 0.0F
                                      ? settings.turnAcceleration / settings.angularDrag
                                      : settings.maximumAngularSpeed;
    response.turnCapBinds = angularTerminal > settings.maximumAngularSpeed;
    response.topTurnRate = std::min(angularTerminal, settings.maximumAngularSpeed);
    response.timeToTopTurnRate =
        timeToReach(angularTerminal, 0.98F * response.topTurnRate, settings.angularDrag);
    response.spinCoast =
        settings.angularDrag > 0.0F ? response.topTurnRate / settings.angularDrag : 0.0F;
    return response;
}

} // namespace vkexp
