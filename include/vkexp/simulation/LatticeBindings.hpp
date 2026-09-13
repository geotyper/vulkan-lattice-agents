#pragma once

#include <cstdint>

namespace vkexp {

// How many storage buffers each compute pass declares, in its one descriptor
// set.
//
// Written here because three places have to agree and only one of them is the
// shader: the driver builds a layout, the parity harness builds the same layout
// again, and the shader declares what it actually reads. When the construction
// world added two buffers to the step pass, the driver was updated and the
// harness was not -- so the harness ran the step shader with two descriptors
// unbound, which is undefined behaviour rather than an unused slot, and the
// parity case it was supposed to protect failed for a reason nobody believed.
//
// testShaderBindingContract reads the compiled SPIR-V and checks these numbers
// against what the shaders declare, so adding a buffer to a pass fails as a
// wrong number here rather than as a descriptor nobody wrote.
inline constexpr std::uint32_t latticeStepBindings = 8;
inline constexpr std::uint32_t latticeResolveBindings = 6;
inline constexpr std::uint32_t latticeClearBindings = 2;
inline constexpr std::uint32_t latticeTrailCaptureBindings = 2;
// Not a pass of the step: the layout echo test's own shader, counted here so
// that it is covered by the same check as everything else.
inline constexpr std::uint32_t latticeLayoutEchoBindings = 3;

} // namespace vkexp
