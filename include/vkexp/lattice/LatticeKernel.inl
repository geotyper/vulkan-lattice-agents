// The lattice: how a cell is addressed, what counts as a neighbour, how a brain
// output becomes a move, and who wins when two agents want the same cell.
//
// Compiled twice -- as C++ through LatticeKernel.hpp and as GLSL through
// shaders/lattice/lattice_kernel.glsl -- so both sides move an agent by the same
// arithmetic. This is the file the CPU/GPU parity test exists to protect: there
// is no second copy of the movement rule to drift away from.
//
// Same common-subset rules as BrainKernel.inl: VKEXP_LATTICE_FN in front of
// every integer function, VKEXP_LATTICE_MATH_FN in front of every one that
// touches floats, only int/uint/float/bool across boundaries, no standard
// library, no structs and no arrays of parameters. GLSL reserves more words than
// C++ does -- `input`, `output`, `layout`, `filter`, `active`, `sample` and
// friends cannot be identifiers here.

// --- what a lattice is -------------------------------------------------------
//
// A dense box of cells, W x H x D, indexed x fastest and z slowest. The index is
// the whole spatial structure: there is no hash, no bucket list and no cell
// size, because a cell *is* a cell. Neighbours are a constant offset away, which
// is why the 2D build's atomicExchange spatial hash has no successor here.

// The Moore neighbourhood: every cell sharing a face, an edge or a corner.
const uint LatticeNeighborCount = 26u;
// The von Neumann subset of it: the six that share a face.
const uint LatticeFaceNeighborCount = 6u;

// Which of those an agent may actually step into. The sensor reads all 26 under
// both settings -- "how many directions can I see" and "how many can I walk"
// are separate claims, and only the second changes what the brain has to solve.
// Keeping the input vector one width under both is what lets a population be
// carried across the setting rather than retrained for it.
const uint LatticeNeighborhoodFaces = 0u;
const uint LatticeNeighborhoodMoore = 1u;
const uint LatticeNeighborhoodCount = 2u;

// Two tasks share the lattice machinery. The navigation baseline follows a
// beacon; construction replaces it with a persistent field of supported blocks
// and constrains agents to surfaces and climbable faces.
const uint LatticeWorldBeacon = 0u;
const uint LatticeWorldConstruction = 1u;
const uint LatticeWorldCount = 2u;

const int LatticeNoStructure = 0;

// Why a build attempt did or did not become a block, counted per world over a
// generation. Exactly one of these is recorded per agent per step in the
// construction world, so the nine of them sum to agents times steps and the
// shape of that sum says what is actually stopping the builders -- which the
// block count alone cannot, since it only ever says "few".
//
// The order is the order the decide pass tests them in, and it matters: an
// attempt that fails two tests is filed under the first. Reading the list top
// to bottom is reading the sequence an agent has to get through.
const uint LatticeBuildCooling = 0u;    // still inside the build interval
const uint LatticeBuildUnwilling = 1u;  // the build output was under threshold
const uint LatticeBuildNoFacing = 2u;   // no cardinal heading to build against
const uint LatticeBuildOffLattice = 3u; // the face points out of the world
const uint LatticeBuildBlocked = 4u;    // a block already stands there
const uint LatticeBuildUnsupported = 5u; // nothing under it, and no side support
const uint LatticeBuildAboveFrontier = 6u; // too far above the local foundation
const uint LatticeBuildInTheWay = 7u;   // an agent is standing in the cell
const uint LatticeBuildClaimed = 8u;    // bid placed; contention may still lose it
const uint LatticeBuildOutcomeCount = 9u;

// An empty cell, and a cell nobody has bid for. Two sentinels and not one: the
// occupancy grid stores agent indices and -1 for empty, while the bid grid is
// resolved by a minimum, so its empty value has to be larger than every agent
// index rather than smaller.
const int LatticeNoOccupant = -1;
const int LatticeNoClaim = 0x7fffffff;

// --- cell addressing ---------------------------------------------------------

VKEXP_LATTICE_FN uint latticeCellCount(uint width, uint height, uint depth) {
    return width * height * depth;
}

VKEXP_LATTICE_FN uint latticeCellIndex(int x, int y, int z, uint width, uint height) {
    return (uint(z) * height + uint(y)) * width + uint(x);
}

VKEXP_LATTICE_FN bool latticeInBounds(int x, int y, int z, uint width, uint height, uint depth) {
    return x >= 0 && y >= 0 && z >= 0 && x < int(width) && y < int(height) && z < int(depth);
}

// --- neighbour numbering -----------------------------------------------------
//
// Neighbour n is the n-th cell of the 3x3x3 block around the centre with the
// centre itself removed, walked x fastest. So 0 is (-1,-1,-1) and 25 is
// (+1,+1,+1), and the numbering is the same on both sides by construction
// rather than by two matching tables -- a table written twice is a table that
// can be edited once.

VKEXP_LATTICE_FN uint latticeNeighborSlot(uint neighbor) {
    return neighbor < 13u ? neighbor : neighbor + 1u;
}

VKEXP_LATTICE_FN int latticeNeighborX(uint neighbor) {
    return int(latticeNeighborSlot(neighbor) % 3u) - 1;
}

VKEXP_LATTICE_FN int latticeNeighborY(uint neighbor) {
    return int((latticeNeighborSlot(neighbor) / 3u) % 3u) - 1;
}

VKEXP_LATTICE_FN int latticeNeighborZ(uint neighbor) {
    return int(latticeNeighborSlot(neighbor) / 9u) - 1;
}

// The inverse, for a step that is known not to be (0,0,0). Undefined for the
// centre on purpose: "stay put" is not a neighbour, and giving it an index
// would put a 27th slot in every loop bound in the file.
VKEXP_LATTICE_FN uint latticeNeighborIndex(int stepX, int stepY, int stepZ) {
    const uint slot = uint((stepZ + 1) * 9 + (stepY + 1) * 3 + (stepX + 1));
    return slot < 13u ? slot : slot - 1u;
}

VKEXP_LATTICE_FN int latticeAbs(int value) { return value < 0 ? -value : value; }

VKEXP_LATTICE_FN bool latticeIsFaceNeighbor(uint neighbor) {
    return latticeAbs(latticeNeighborX(neighbor)) + latticeAbs(latticeNeighborY(neighbor)) +
               latticeAbs(latticeNeighborZ(neighbor)) ==
           1;
}

// Whether a move in this direction is legal under the selected neighbourhood.
// Sensing does not ask: see the note on LatticeNeighborhoodFaces.
VKEXP_LATTICE_FN bool latticeNeighborWalkable(uint neighborhood, uint neighbor) {
    return neighborhood == LatticeNeighborhoodMoore || latticeIsFaceNeighbor(neighbor);
}

// --- distance ----------------------------------------------------------------
//
// Distance is counted in moves, so it follows the neighbourhood rather than
// being a Euclidean length that happens to be measured on a grid. Under Moore
// the three axes advance together, which is a Chebyshev distance; under faces
// they advance one at a time, which is a Manhattan one. Getting this wrong
// makes the beacon shaping reward the wrong thing by up to a factor of three.

VKEXP_LATTICE_FN int latticeMaxOfThree(int first, int second, int third) {
    const int larger = first > second ? first : second;
    return larger > third ? larger : third;
}

VKEXP_LATTICE_FN uint latticeStepDistance(uint neighborhood, int deltaX, int deltaY, int deltaZ) {
    const int absoluteX = latticeAbs(deltaX);
    const int absoluteY = latticeAbs(deltaY);
    const int absoluteZ = latticeAbs(deltaZ);
    if (neighborhood == LatticeNeighborhoodMoore) {
        return uint(latticeMaxOfThree(absoluteX, absoluteY, absoluteZ));
    }
    return uint(absoluteX + absoluteY + absoluteZ);
}

// The longest journey the box admits: what the beacon shaping normalises
// against, so the reward for closing half the distance means the same thing in
// a 32-wide lattice and a 100-wide one.
VKEXP_LATTICE_FN uint latticeMaximumDistance(uint neighborhood, uint width, uint height,
                                             uint depth) {
    const int spanX = int(width) - 1;
    const int spanY = int(height) - 1;
    const int spanZ = int(depth) - 1;
    return latticeStepDistance(neighborhood, spanX, spanY, spanZ);
}

// --- turning three drives into one move --------------------------------------
//
// The brain produces one output per axis rather than one per direction. Twenty
// seven directions would need twenty seven outputs and an argmax over them; three
// signed drives with a dead zone span the same set of moves, cost three output
// slots, and leave "stay put" reachable by simply not committing -- which a
// softmax over directions can only approximate.
//
// The dead zone is the whole of the decision to stand still, so it is a
// parameter and not a constant: at zero an agent moves on every step whatever it
// thinks, and near one it has to be sure before it does.
const float LatticeMoveThresholdDefault = 0.25f;

VKEXP_LATTICE_MATH_FN int latticeAxisStep(float drive, float threshold) {
    if (drive > threshold) {
        return 1;
    }
    if (drive < -threshold) {
        return -1;
    }
    return 0;
}

// Which axis a face-only mover commits to when more than one drive clears the
// dead zone: the loudest, ties broken x then y then z. A fixed order and not a
// random one, because an arbitrary tie-break is a source of divergence between
// the two implementations that no amount of numeric tolerance would forgive.
VKEXP_LATTICE_MATH_FN uint latticeDominantAxis(float driveX, float driveY, float driveZ) {
    const float magnitudeX = driveX < 0.0f ? -driveX : driveX;
    const float magnitudeY = driveY < 0.0f ? -driveY : driveY;
    const float magnitudeZ = driveZ < 0.0f ? -driveZ : driveZ;
    if (magnitudeX >= magnitudeY && magnitudeX >= magnitudeZ) {
        return 0u;
    }
    return magnitudeY >= magnitudeZ ? 1u : 2u;
}

// The move one axis contributes, given all three drives and the neighbourhood.
// Written as one function of an axis index rather than three near-copies so the
// face-only reduction cannot be applied to two axes and forgotten on the third.
VKEXP_LATTICE_MATH_FN int latticeMoveComponent(uint neighborhood, uint axis, float driveX,
                                               float driveY, float driveZ, float threshold) {
    if (neighborhood == LatticeNeighborhoodFaces &&
        axis != latticeDominantAxis(driveX, driveY, driveZ)) {
        return 0;
    }
    if (axis == 0u) {
        return latticeAxisStep(driveX, threshold);
    }
    if (axis == 1u) {
        return latticeAxisStep(driveY, threshold);
    }
    return latticeAxisStep(driveZ, threshold);
}

// --- who gets the cell -------------------------------------------------------
//
// Two agents wanting one cell is the collision model: there is no impulse and no
// overlap resolution, only a cell that one of them ends up in. Resolving it with
// atomicCompSwap would hand it to whichever invocation arrived first, which is
// exactly the thing a GPU does not promise and a CPU reference cannot reproduce.
// So the bid is a minimum over agent indices -- associative, commutative, and
// therefore the same answer whatever order the atomics land in.
//
// The cost is a fixed bias: a low-numbered agent wins every contest it enters.
// That bias is spread by the population layout rather than by the rule, since
// the agents sharing a world are a contiguous block whose order within the world
// is as arbitrary as any other.
VKEXP_LATTICE_FN int latticeBetterClaim(int standing, int bid) {
    return bid < standing ? bid : standing;
}

// Building and walking can target the same cell in one decision pass. Build
// claims are negative, so a block wins over a body independent of invocation
// order; lower-numbered builders still win ties, as movers do.
VKEXP_LATTICE_FN int latticeBuildClaim(uint agent, uint agentCount) {
    return int(agent) - int(agentCount) - 1;
}

// A cell may be entered only if it was empty when the step began. So a queue of
// agents cannot shuffle forward together: the one behind still sees the cell in
// front as taken, even though its occupant is leaving. That is a real
// restriction on what the lattice can express in one step, and it is the price
// of an outcome that does not depend on which invocation ran first -- the
// alternative is a dependency chain whose length is the queue's.
VKEXP_LATTICE_FN bool latticeCellEnterable(int occupant) { return occupant == LatticeNoOccupant; }

// --- what the brain is told about a cell -------------------------------------

// Three channels per neighbour: something is standing there, the lattice ends
// there, and how loudly its occupant is signalling. The third is the whole of
// agent-to-agent perception -- an agent reads its neighbour's broadcast, not its
// neighbour's state -- which keeps what one agent can learn about another a
// property of the world rather than of the record layout.
const uint LatticeNeighborOccupied = 0u;
const uint LatticeNeighborBlocked = 1u;
const uint LatticeNeighborSignal = 2u;

VKEXP_LATTICE_MATH_FN float latticeClamp01(float value) { return clamp(value, 0.0f, 1.0f); }

// How near the beacon is, as a number the network can use directly: 1 standing
// on it, 0 at the far corner. Normalised by the longest journey the box admits,
// so a wider lattice is a longer task rather than a differently scaled input.
VKEXP_LATTICE_MATH_FN float latticeNearness(uint distance, uint maximumDistance) {
    if (maximumDistance == 0u) {
        return 1.0f;
    }
    return 1.0f - latticeClamp01(float(distance) / float(maximumDistance));
}

// The direction to a cell, as a unit vector. Euclidean and not lattice-metric on
// purpose: this is the one input that answers "which way", and a Chebyshev
// direction would report a diagonal and an axis move as the same heading.
VKEXP_LATTICE_MATH_FN float latticeVectorLength(int deltaX, int deltaY, int deltaZ) {
    return sqrt(float(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ));
}

VKEXP_LATTICE_MATH_FN float latticeDirectionComponent(int delta, float length) {
    return length > 0.0f ? float(delta) / length : 0.0f;
}

// --- scoring -----------------------------------------------------------------
//
// The per-step half of fitness, shared so the accumulation the shader does and
// the accumulation the reference does cannot differ. What the numbers are then
// worth is a weight, and lives with the other weights.

// Standing near enough to the beacon to count as having reached it. A radius and
// not an equality, because one cell can hold one agent: at radius 0 a world of
// twelve has eleven losers however well they all steered, which measures the
// arbitration rule rather than the policy.
VKEXP_LATTICE_FN bool latticeBeaconReached(uint distance, uint contactRadius) {
    return distance <= contactRadius;
}

// Fitness from a finished trial. Kept here beside the accumulation so the two
// halves of the score are read in one place: what the step counts, and what the
// counts are worth.
//
//   bestNearness   how close it ever got, 0..1
//   contacts       steps spent within the contact radius
//   moves          steps on which it actually changed cell
//   refusals       steps on which it tried to and could not
//
// Refusals are charged and moves are charged less, which is the whole of the
// pressure toward not crowding: a policy that walks into its neighbours pays for
// every attempt, and the cheapest way to stop paying is to go around.
VKEXP_LATTICE_MATH_FN float latticeTrialFitness(float bestNearness, float contacts, float moves,
                                                float refusals, float trackingReward,
                                                float objectiveBonus, float motorCost,
                                                float refusalPenalty) {
    return trackingReward * latticeClamp01(bestNearness) + objectiveBonus * contacts -
           motorCost * moves - refusalPenalty * refusals;
}
