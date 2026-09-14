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

// Four tasks share the lattice machinery. The navigation baseline follows a
// beacon; construction replaces it with a persistent field of supported blocks
// and constrains agents to surfaces and climbable faces; harvest keeps every one
// of construction's rules and moves the reward off the building and onto
// something only a building can reach; the chasm takes away half the ground and
// hangs the resource over the missing half, so the only route to it is one the
// group builds out of nothing.
const uint LatticeWorldBeacon = 0u;
const uint LatticeWorldConstruction = 1u;
const uint LatticeWorldHarvest = 2u;
const uint LatticeWorldChasm = 3u;
const uint LatticeWorldCount = 4u;

// Whether a world has a block field and the movement rules that go with it.
// Written once because it is asked in seven places across two languages, and a
// world mode that means "you may build" must not be a list somebody extends in
// six of them.
VKEXP_LATTICE_FN bool latticeWorldBuilds(uint worldMode) {
    return worldMode == LatticeWorldConstruction || worldMode == LatticeWorldHarvest ||
           worldMode == LatticeWorldChasm;
}

// Whether the reward is a load fetched from a resource and carried back down.
// The chasm differs from harvest in where the ground is and where the resource
// hangs, and in nothing else, so everything about picking up and delivering is
// asked through this rather than duplicated.
// Whether a world asks for a foundation under what is built high. Construction
// and harvest do; the chasm emphatically does not, and this is a rule rather
// than a slider because the two are incompatible by construction: a cantilever
// has nothing at all beneath it, so a foundation test refuses every block of a
// bridge. Leaving it switchable would let the chasm be turned into a world with
// no solution without saying so.
VKEXP_LATTICE_FN bool latticeWorldFrontier(uint worldMode) {
    return worldMode == LatticeWorldConstruction || worldMode == LatticeWorldHarvest;
}

VKEXP_LATTICE_FN bool latticeWorldHarvests(uint worldMode) {
    return worldMode == LatticeWorldHarvest || worldMode == LatticeWorldChasm;
}

// --- the body frame ----------------------------------------------------------
//
// An agent faces one of the four horizontal cardinals, and that facing is the
// frame every sense and every action is expressed in. Up and down stay absolute,
// because gravity is.
//
// Facing used to be a consequence rather than a choice: it was whichever way the
// agent last actually moved, written by the resolve pass. That made it a memory
// of locomotion -- a refused move left it pointing where the agent was stuck,
// and an agent that had never moved had no facing at all. Now it is state the
// agent owns and a turn is an action like any other.
const uint LatticeFacingCount = 4u;

// Rotate a body-frame horizontal offset into world axes. Facing zero is +x, and
// each step of the facing turns the frame from +x towards +z.
VKEXP_LATTICE_FN int latticeRotatedX(uint facing, int bodyX, int bodyZ) {
    if (facing == 0u) {
        return bodyX;
    }
    if (facing == 1u) {
        return -bodyZ;
    }
    if (facing == 2u) {
        return -bodyX;
    }
    return bodyZ;
}

VKEXP_LATTICE_FN int latticeRotatedZ(uint facing, int bodyX, int bodyZ) {
    if (facing == 0u) {
        return bodyZ;
    }
    if (facing == 1u) {
        return bodyX;
    }
    if (facing == 2u) {
        return -bodyZ;
    }
    return -bodyX;
}

// Straight ahead, as a world-axis step.
VKEXP_LATTICE_FN int latticeFacingX(uint facing) { return latticeRotatedX(facing, 1, 0); }
VKEXP_LATTICE_FN int latticeFacingZ(uint facing) { return latticeRotatedZ(facing, 1, 0); }

// One quarter turn. A positive drive turns towards +z, a negative one away.
VKEXP_LATTICE_FN uint latticeTurn(uint facing, bool positive) {
    return (facing + (positive ? 1u : LatticeFacingCount - 1u)) % LatticeFacingCount;
}

const int LatticeNoStructure = 0;

// The ground, stored as blocks rather than assumed. It used to be assumed: both
// support rules answered "yes" for anything at height zero, so the floor was
// solid everywhere and a hole in it was inexpressible. Worse, it would have been
// invisible -- an empty cell at height zero and a cell over a drop read
// identically to a sensor, which is the same mistake as a wall and a block
// sharing a channel.
//
// So the floor is a course of bedrock in the block field, and a chasm is where
// that course is missing. No new rule, no new buffer, no new sensor: the drop is
// visible because bedrock is visible, and support means what it says.
//
// Negative because a placed block stores its builder's index plus one, so
// everything positive is already spoken for and fitness, the shape descriptors
// and the renderer can all tell terrain from work by its sign.
const int LatticeBedrock = -1;

VKEXP_LATTICE_FN bool latticeIsBedrock(int cell) { return cell < LatticeNoStructure; }

// Where the ground stops. Solid for z below this and open beyond it, which makes
// the near half a place to stand and the far half a place that has to be built
// over. A depth equal to the lattice's is a world with no chasm at all, which is
// how the other building worlds ask for their floor.
VKEXP_LATTICE_FN bool latticeGroundColumn(int x, uint groundWidth) {
    return x >= 0 && uint(x) < groundWidth;
}

// --- placement ---------------------------------------------------------------

// The integer hash that places everything a world needs placed. Shared rather
// than host-only because the harvest world's resource has to be found by the
// shader as well: mirroring it onto the agent record would cost the lane that
// carries the per-step build intent, and a resource is a property of the world,
// not of the agent looking at it.
VKEXP_LATTICE_FN uint latticeMix(uint value) {
    value ^= value >> 16u;
    value *= 0x7FEB352Du;
    value ^= value >> 15u;
    value *= 0x846CA68Bu;
    value ^= value >> 16u;
    return value;
}

VKEXP_LATTICE_FN uint latticeMix(uint first, uint second) {
    return latticeMix(first ^ (latticeMix(second) + 0x9E3779B9u + (first << 6u) + (first >> 2u)));
}

// Where the resource stands in one world, as a pure function of the world index
// and the seed -- like a beacon, and for the same two reasons: a genome is
// scored on several placements rather than on one it could memorise, and world
// 91 can be placed without having placed world 90.
//
// Hashed against its own constant so that a harvest world and a beacon world
// built from the same seed do not put their objectives in the same cell.
VKEXP_LATTICE_FN uint latticeResourceHash(uint world, uint seed) {
    return latticeMix(seed ^ 0x8E5017u, world);
}

// Over the open half when there is one, and anywhere on the floor plan when
// there is not. A resource standing over ground the group can walk to is a
// resource it can reach by climbing; the point of the chasm is that it cannot.
//
// The split runs along x, so the open half is a range of columns rather than of
// rows. Which axis is arbitrary to the simulation and not to the eye: the box
// is widest on x and the camera starts side-on to it, so a chasm cut this way
// is the one you are already looking across.
VKEXP_LATTICE_FN int latticeResourceX(uint hash, uint width, uint groundWidth) {
    const uint open = groundWidth < width ? width - groundWidth : width;
    const uint first = groundWidth < width ? groundWidth : 0u;
    return int(first + hash % open);
}

VKEXP_LATTICE_FN int latticeResourceZ(uint hash, uint width, uint depth) {
    return int((hash / width) % depth);
}

// Somewhere in a band rather than at one height, hashed like the column. A fixed
// height is a number a genome can learn to count to: eight blocks and turn. A
// band makes the sensed direction the only thing that says when to stop, which
// is the difference between a policy and a memorised program.
//
// Clamped to the box rather than wrapped: a band asked for above the ceiling is
// a setting to correct, and wrapping it near the floor would hide that by making
// the world quietly easy.
VKEXP_LATTICE_FN int latticeResourceY(uint hash, uint lowest, uint highest, uint height) {
    const uint ceiling = height > 1u ? height - 1u : 1u;
    const uint low = lowest < ceiling ? lowest : ceiling;
    const uint high = highest < ceiling ? highest : ceiling;
    const uint span = high > low ? high - low + 1u : 1u;
    // A third mixing constant: the column already used the hash directly, and a
    // height drawn from the same bits would march with it across the worlds.
    return int(low + latticeMix(hash ^ 0x4E1A07u) % span);
}

// Why a build attempt did or did not become a block, counted per world over a
// generation. Exactly one of these is recorded per agent per step in the
// construction world, so the nine of them sum to agents times steps and the
// shape of that sum says what is actually stopping the builders -- which the
// block count alone cannot, since it only ever says "few".
//
// The order is the order the decide pass tests them in, and it matters: an
// attempt that fails two tests is filed under the first. Reading the list top
// to bottom is reading the sequence an agent has to get through.
// What one agent's tick was spent on, counted per world over a generation.
// Exactly one of these is recorded per agent per step, so they sum to agents
// times steps and the shape of that sum says what the group is actually doing.
//
// It covers walking as well as building now, because a tick is one action: a
// turn, a step, or a placement. That closes a hole the old funnel had -- it
// explained refused builds in detail and said nothing at all about an agent
// that simply stood there, which turned out to be most of what was happening.
//
// Two of the old entries are gone rather than renamed. "No facing" cannot occur
// when facing is state the agent owns, and "unwilling" cannot occur when not
// wanting to build means walking instead of standing still. Both were failure
// modes the body frame removes rather than fixes.
const uint LatticeActionTurning = 0u;   // the tick was spent turning in place
const uint LatticeActionWalking = 1u;   // a step forward, up a wall, or down
const uint LatticeActionWalled = 2u;    // the step could not be taken at all
const uint LatticeBuildCooling = 3u;    // wanted to build, still cooling, walked
const uint LatticeBuildOffLattice = 4u; // the cell in front is outside the world
const uint LatticeBuildBlocked = 5u;    // a block already stands there
const uint LatticeBuildUnsupported = 6u;   // nothing under it, and no side support
const uint LatticeBuildAboveFrontier = 7u; // too far above the local foundation
const uint LatticeBuildInTheWay = 8u;      // an agent is standing in the cell
// The last two are recorded by the resolve pass rather than the decide pass,
// because whether a bid won is not known until every bid is in.
const uint LatticeBuildPlaced = 9u;    // the bid won and a block stands there
const uint LatticeBuildContested = 10u; // the bid was placed and lost
const uint LatticeBuildOutcomeCount = 11u;

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

// --- reading a signed output as a decision -----------------------------------
//
// Every command the agent has is one signed output read through a dead zone: a
// turn is left, nothing or right, and the action is walk or build. Two outputs
// where the world-axis scheme needed five, because in a body frame the axes are
// not a choice the network has to make -- forward is wherever it is looking.
//
// The dead zone is the whole of the decision not to commit, so it is a parameter
// and not a constant: at zero an agent turns on every step whatever it thinks,
// and near one it has to be sure before it does.
//
// Wide by default, which the world-axis scheme did not need. A turn costs the
// whole tick, and an output that saturates clears a narrow dead zone on almost
// any input -- at 0.25 a fresh population spends 93% of every generation
// pivoting on the spot and never reaches a wall to climb. It cannot simply be
// pushed to one either: turning is also how an agent gets out of a corner, and
// the wider the zone the longer it is stuck against one. 0.7 is where the two
// costs met in a four-generation sweep of the construction world.
const float LatticeTurnThresholdDefault = 0.70f;

VKEXP_LATTICE_MATH_FN int latticeAxisStep(float drive, float threshold) {
    if (drive > threshold) {
        return 1;
    }
    if (drive < -threshold) {
        return -1;
    }
    return 0;
}

// How long an agent has to stand still before the stillness input saturates.
//
// A ramp and not a flag, which is the whole point of it. A deterministic policy
// in an unchanging neighbourhood produces the same output forever -- that is why
// an agent that parks itself stays parked for the rest of the generation -- and
// a flag that reads 1 for the entire stall is just as unchanging as the
// neighbourhood is. It would move the fixed point, not remove it. A count that
// rises every tick is an input that is never twice the same, so the output is
// never twice the same either, and a policy that has learned any threshold at
// all eventually crosses it and does something else.
//
// Distinct from the refusal flag beside it: refusal means a move was asked for
// and denied, and says nothing about an agent that never asked.
const float LatticeStillnessSpan = 48.0f;

VKEXP_LATTICE_MATH_FN float latticeStillness(float stillTicks) {
    return clamp(stillTicks / LatticeStillnessSpan, 0.0f, 1.0f);
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

// Four channels per neighbour: an agent is standing there, the lattice ends
// there, a block stands there, and how loudly the occupant is signalling. The
// last is the whole of agent-to-agent perception -- an agent reads its
// neighbour's broadcast, not its neighbour's state -- which keeps what one
// agent can learn about another a property of the world rather than of the
// record layout.
//
// The edge and the block are separate channels because they are opposite
// situations that used to read identically: one can never be built on and the
// other already has been. See BrainKernel.inl for what conflating them cost.
const uint LatticeNeighborOccupied = 0u;
const uint LatticeNeighborEdge = 1u;
const uint LatticeNeighborStructure = 2u;
const uint LatticeNeighborSignal = 3u;

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
