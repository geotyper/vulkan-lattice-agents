#ifndef VKEXP_LATTICE_VIEW_GLSL
#define VKEXP_LATTICE_VIEW_GLSL

// What every drawing stage of the lattice view shares: the push constant the
// C++ side packs, where a cell sits in world space, and the cube itself.
//
// The cube is generated from gl_VertexIndex rather than uploaded. That is the
// one habit the deleted 2D renderer is worth keeping: there is no vertex
// buffer, no upload and no staging path in this program, and a unit cube is
// twelve triangles of arithmetic.

// Mirrors vkexp::LatticeViewParameters, asserted there field for field.
layout(push_constant) uniform ViewParameters {
    mat4 viewProjection;
    vec4 camera;  // eye xyz, voxel scale in w
    ivec4 lattice; // extents xyz, mode | (slice axis << 8) | (agent stride << 16)
    ivec4 beacon;  // beacon cell xyz, the visible world's first agent in w
    vec4 tint;     // opacity, background brightness, slice low, slice high
} view;

const uint LatticeViewModeAgents = 0u;
const uint LatticeViewModeBeacon = 1u;
const uint LatticeViewModeBounds = 2u;
const uint LatticeViewModeTrail = 3u;
const uint LatticeViewModeStructure = 4u;
const uint LatticeViewModeGoal = 5u;
const uint LatticeViewModeTerrain = 6u;

uint latticeViewMode() { return uint(view.lattice.w) & 0xffu; }
uint latticeViewSliceAxis() { return (uint(view.lattice.w) >> 8u) & 0xffu; }
// Consecutive genomes of one world are one trial count apart in the agent
// array, so the visible world is a strided run rather than a contiguous one.
uint latticeViewAgentStride() { return (uint(view.lattice.w) >> 16u) & 0xffffu; }

vec3 latticeExtent() { return vec3(view.lattice.xyz); }

// A cell's centre, with the box centred on the origin so the camera orbits the
// lattice rather than its corner. One cell is one unit: there is no scale to
// choose because the world has no length.
vec3 latticeCellCentre(ivec3 cell) {
    return vec3(cell) + vec3(0.5) - latticeExtent() * 0.5;
}

// Whether a cell is inside the visible slab. The slab is how the inside of a
// box gets seen without making everything see-through, so it applies whatever
// the voxel style is.
bool latticeCellVisible(ivec3 cell) {
    const uint axis = latticeViewSliceAxis();
    const float position = axis == 0u ? float(cell.x) : (axis == 1u ? float(cell.y) : float(cell.z));
    return position >= view.tint.z && position <= view.tint.w;
}

// Twelve triangles, six faces, no winding to keep straight: the pipelines cull
// nothing, because a transparent voxel needs both of its faces and an opaque
// one is closed anyway. The normal is the face's own and is exact.
vec3 latticeCubeVertex(uint vertex, out vec3 normal) {
    const uint face = vertex / 6u;
    const uint corner = vertex % 6u;
    const uint axis = face >> 1u;
    const float side = (face & 1u) == 0u ? 1.0 : -1.0;
    // Two triangles of the quad, in the face's own 2D frame.
    vec2 uv = vec2(-1.0, -1.0);
    if (corner == 1u) {
        uv = vec2(1.0, -1.0);
    } else if (corner == 2u || corner == 4u) {
        uv = vec2(1.0, 1.0);
    } else if (corner == 5u) {
        uv = vec2(-1.0, 1.0);
    }
    normal = vec3(0.0);
    vec3 position;
    if (axis == 0u) {
        position = vec3(side, uv.x, uv.y);
        normal.x = side;
    } else if (axis == 1u) {
        position = vec3(uv.x, side, uv.y);
        normal.y = side;
    } else {
        position = vec3(uv.x, uv.y, side);
        normal.z = side;
    }
    return position * 0.5;
}

// The twelve edges of the box, as a line list. Corner bits are x, y, z, so the
// table is the two faces and the four verticals between them.
vec3 latticeBoxEdgeVertex(uint vertex) {
    const uint table[24] = uint[24](0u, 1u, 1u, 3u, 3u, 2u, 2u, 0u,
                                    4u, 5u, 5u, 7u, 7u, 6u, 6u, 4u,
                                    0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u);
    const uint corner = table[vertex];
    const vec3 unitCorner =
        vec3(float(corner & 1u), float((corner >> 1u) & 1u), float((corner >> 2u) & 1u));
    return (unitCorner - vec3(0.5)) * latticeExtent();
}

// Where the objective is, drawn as a place rather than as a floating cube. Six
// vertices of a line list: a plumb line from the objective down to the floor
// plane, and a cross on that plane under it.
//
// One cube alone has no depth cue. In a box thirty-two cells deep, a cube at
// (4, 7, 28) and a cube at (4, 7, 4) project to nearly the same pixels, so the
// eye cannot say which column the group has to reach -- and in the chasm world
// that column is the whole question. The plumb line answers it, and the cross
// says whether there is floor under the answer or a hole.
const uint LatticeGoalGuideVertexCount = 6u;

vec3 latticeGoalGuideVertex(uint vertex, vec3 centre) {
    const float floorY = -latticeExtent().y * 0.5;
    const float arm = 2.5;
    if (vertex < 2u) {
        return vec3(centre.x, vertex == 0u ? floorY : centre.y, centre.z);
    }
    if (vertex < 4u) {
        return vec3(centre.x + (vertex == 2u ? -arm : arm), floorY, centre.z);
    }
    return vec3(centre.x, floorY, centre.z + (vertex == 4u ? -arm : arm));
}

// Far from the beacon to on top of it. Blue reads as cold at a glance and the
// warm end is the only bright thing in the box, so a group that has arrived is
// visible without reading a number.
vec3 latticeNearnessColour(float nearness) {
    const vec3 cold = vec3(0.10, 0.20, 0.62);
    const vec3 middle = vec3(0.13, 0.66, 0.60);
    const vec3 warm = vec3(0.98, 0.79, 0.30);
    const float t = clamp(nearness, 0.0, 1.0);
    return t < 0.5 ? mix(cold, middle, t * 2.0) : mix(middle, warm, (t - 0.5) * 2.0);
}

// A stable colour per genome. Trials of the same genome keep the same hue, so
// changing the visible trial does not silently change who a path belongs to.
vec3 latticeTrailColour(uint genome) {
    const float hue = fract(float(genome) * 0.61803398875);
    return 0.58 + 0.42 * cos(6.283185307 * (hue + vec3(0.0, 0.667, 0.333)));
}

// One key light and a floor bounce, with the normal turned toward the eye so an
// inside face of a see-through voxel is lit rather than black. Nothing here is
// trying to be a renderer: the job is to tell one face of a cube from the next.
vec3 latticeShade(vec3 colour, vec3 normal, vec3 worldPosition) {
    vec3 facing = normalize(normal);
    const vec3 toEye = view.camera.xyz - worldPosition;
    if (dot(facing, toEye) < 0.0) {
        facing = -facing;
    }
    const vec3 key = normalize(vec3(0.42, 0.86, 0.30));
    const float direct = clamp(dot(facing, key), 0.0, 1.0);
    const float bounce = 0.5 + 0.5 * facing.y;
    return colour * (0.30 + 0.55 * direct + 0.15 * bounce);
}

#endif
