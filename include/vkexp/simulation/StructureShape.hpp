#pragma once

#include "vkexp/simulation/LatticeTypes.hpp"

#include <cstdint>
#include <span>

namespace vkexp {

// What a finished block field looks like, said in numbers.
//
// None of this is fitness and none of it is fed to a brain. Fitness in the
// construction world is mass weighted by height, and mass weighted by height
// has exactly one maximum: fill the box. Whatever variety appears along the way
// is therefore invisible to the run -- two generations with the same score can
// have built a slab and a spire, and nothing in the log would say so. These are
// the numbers that would say so, measured first so that a later decision to
// select for shape is made against evidence instead of taste.
//
// The descriptors are deliberately cheap and scale-free where they can be, so
// that the same vector can later address a behaviour archive without being
// rewritten: a niche is a shape, not a size.
struct StructureShape {
    // Blocks placed, and how many floor columns carry at least one of them.
    std::uint32_t blocks{};
    std::uint32_t footprint{};
    // Height in cells of the tallest column, and the mean and population
    // standard deviation over the columns that carry anything. A slab and a
    // spire of equal mass differ here before they differ anywhere else.
    std::uint32_t peak{};
    float meanHeight{};
    float heightSpread{};
    // Occupied columns as a fraction of their own bounding rectangle: 1 for a
    // solid plan, lower for a ring, a cross or scattered piers. Zero when
    // nothing has been built.
    float compactness{};
    // Blocks standing on empty space. Only reachable when side support is
    // enabled, and the plainest evidence that a group built something other
    // than stacks.
    std::uint32_t overhangs{};
    // Empty cells with a block somewhere above them in the same column: the
    // roofed volume. A tower of solid courses has none.
    std::uint32_t enclosed{};
};

// Measures one world's slice of the structure field. The span must be exactly
// latticeCellsPerWorld(settings) long; anything else is a layout mistake rather
// than a smaller world, so it returns an empty measurement rather than guessing.
[[nodiscard]] StructureShape measureStructureShape(std::span<const std::int32_t> field,
                                                   const SimulationStep& settings);

} // namespace vkexp
