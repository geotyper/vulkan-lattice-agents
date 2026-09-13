#include "vkexp/simulation/StructureShape.hpp"

#include "vkexp/lattice/LatticeKernel.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace vkexp {
namespace {

namespace kern = vkexp::lattice::kernel;

} // namespace

StructureShape measureStructureShape(const std::span<const std::int32_t> field,
                                     const SimulationStep& settings) {
    StructureShape shape{};
    const std::uint32_t width = std::max(settings.latticeWidth, 1U);
    const std::uint32_t height = std::max(settings.latticeHeight, 1U);
    const std::uint32_t depth = std::max(settings.latticeDepth, 1U);
    if (field.size() != static_cast<std::size_t>(width) * height * depth) {
        return shape;
    }

    // One pass per column rather than per cell: every descriptor below is a
    // question about a column, and the column is short.
    std::vector<std::uint32_t> columnHeights;
    columnHeights.reserve(static_cast<std::size_t>(width) * depth);
    std::uint32_t minimumX = width;
    std::uint32_t maximumX = 0;
    std::uint32_t minimumZ = depth;
    std::uint32_t maximumZ = 0;
    for (std::uint32_t z = 0; z < depth; ++z) {
        for (std::uint32_t x = 0; x < width; ++x) {
            std::uint32_t top = 0;
            std::uint32_t filled = 0;
            for (std::uint32_t y = 0; y < height; ++y) {
                const std::size_t index = kern::latticeCellIndex(static_cast<int>(x),
                                                                 static_cast<int>(y),
                                                                 static_cast<int>(z), width, height);
                if (field[index] == kern::LatticeNoStructure) {
                    continue;
                }
                ++filled;
                top = y + 1U;
                if (y > 0) {
                    const std::size_t below =
                        kern::latticeCellIndex(static_cast<int>(x), static_cast<int>(y) - 1,
                                               static_cast<int>(z), width, height);
                    shape.overhangs += field[below] == kern::LatticeNoStructure ? 1U : 0U;
                }
            }
            if (top == 0) {
                continue;
            }
            // Everything under the top of the column that is not a block is
            // covered by one, whatever put it there.
            shape.enclosed += top - filled;
            shape.blocks += filled;
            ++shape.footprint;
            columnHeights.push_back(top);
            shape.peak = std::max(shape.peak, top);
            minimumX = std::min(minimumX, x);
            maximumX = std::max(maximumX, x);
            minimumZ = std::min(minimumZ, z);
            maximumZ = std::max(maximumZ, z);
        }
    }

    if (columnHeights.empty()) {
        return shape;
    }

    double total = 0.0;
    for (const std::uint32_t column : columnHeights) {
        total += static_cast<double>(column);
    }
    const auto count = static_cast<double>(columnHeights.size());
    const double mean = total / count;
    double variance = 0.0;
    for (const std::uint32_t column : columnHeights) {
        const double difference = static_cast<double>(column) - mean;
        variance += difference * difference;
    }
    shape.meanHeight = static_cast<float>(mean);
    shape.heightSpread = static_cast<float>(std::sqrt(variance / count));

    const auto bounding = static_cast<double>(maximumX - minimumX + 1U) *
                          static_cast<double>(maximumZ - minimumZ + 1U);
    shape.compactness = static_cast<float>(count / bounding);
    return shape;
}

} // namespace vkexp
