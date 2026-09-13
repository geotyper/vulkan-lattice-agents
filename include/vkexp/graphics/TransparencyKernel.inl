// How much a see-through fragment counts, compiled by C++ and GLSL alike.
//
// Weighted blended order-independent transparency resolves a pixel as a
// weighted mean of the fragments that reached it, so the weight is the whole
// method: it is what makes a voxel at the front of the box read as being in
// front of the ones behind it. Get it wrong and the pass still runs, still
// blends and still produces a picture -- an evenly averaged one, in which
// moving any one fragment's opacity barely moves the result. That is a failure
// with no error in it, which is why the function lives here, alone, with a test
// on its shape rather than on its values.
//
// It is written about the box rather than about the eye. Both distances come in
// as multiples of the lattice's half-diagonal, and what the weight actually
// reads is where the fragment sits between the near side of the box and the far
// side -- so the spread from front to back is the same whether the camera is
// framing the whole lattice or pressed against one corner of it. Falling off
// with absolute distance instead would compress to nothing as the camera pulled
// back, which is precisely when seeing into a crowd matters most.
//
// **Not window depth.** The obvious form of this function, and the one the
// paper's supplement gives, is built on `gl_FragCoord.z`. With a near plane a
// hundredth of the box and a far plane six boxes out, window depth across the
// whole lattice runs from about 0.967 to 0.991 -- and the published constants
// turn that into 2.2e5 against 1.3e5, both of which clamp to the same ceiling.
// The front of the box and the back of it then weigh exactly the same, which is
// an unordered average wearing transparency's clothes.

// How steeply the weight falls from the near side of the box to the far side.
// Exponential, so the spread is e to this power -- about fifty to one -- at
// every camera distance, and the result needs no clamping: it is bounded by the
// opacity above and by e to the negative of this below. That bound is what
// keeps a crowded pixel inside a half-precision accumulation buffer, which the
// published constants, topping out in the thousands, did not.
const float LatticeWeightFalloff = 4.0f;

VKEXP_TRANSPARENCY_FN float latticeTransparencyWeight(float alpha, float fragmentDistance,
                                                      float eyeDistance) {
    // Where the fragment sits through the box's own depth: 0 at the near side
    // of the sphere around it, 1 at the far side. The box is one half-diagonal
    // in every direction from what the camera looks at, so the near side is one
    // closer than the eye distance and the far side one further.
    float depth = (fragmentDistance - (eyeDistance - 1.0f)) * 0.5f;
    if (depth < 0.0f) {
        depth = 0.0f;
    }
    if (depth > 1.0f) {
        depth = 1.0f;
    }

    // Opacity multiplies rather than saturating. The published form runs the
    // alpha through min(1, alpha * 10), which is flat for everything above 0.1
    // -- so the one control a viewer has stops reaching the weighted mean
    // exactly where it starts being useful.
    float opacity = alpha;
    if (opacity < 0.0f) {
        opacity = 0.0f;
    }
    if (opacity > 1.0f) {
        opacity = 1.0f;
    }
    return opacity * latticeWeightExp(-LatticeWeightFalloff * depth);
}
