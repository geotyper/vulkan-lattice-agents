# Lattice Lab Plan

## Design rules

1. Domain algorithms remain usable without a window or Vulkan device.
2. GPU layouts have explicit size/offset assertions and a CPU reference.
3. Modules exchange published views and state, not ownership of each other's
   implementation details.
3a. Simulation logic lives in drivers, not in modules: anything needed for an
   experiment must be reachable without a window.
3b. A rule of the world is one function in a shared kernel, not a set of
   switches: how a cell is addressed, who may enter it and what a trial is worth
   are written once in `LatticeKernel.inl` and compiled by both languages.
3c. Anything the CPU and the GPU must agree on is either shared source or a
   runtime parameter -- never a constant written twice.
3d. The CPU path exists to inspect a few agents and to build the network from
   the same declaration, not to mirror the GPU. Parallel behaviour is verified
   by tests that need many agents, not by a second implementation.
3e. Steps are the unit of reproducibility; seconds are the unit of anything that
   decays. Space is discrete and has no unit at all: a move is one cell. What
   the step accumulates over time is scaled by `deltaTime`, what it counts per
   event is not, and any fraction removed per step is written as
   `1 - exp(-rate * dt)`.
3f. A fixed allocation with the configuration giving way, never a resize under
   live descriptors. The lattice is clamped to fit its buffers and the clamped
   value is written back, so the UI shows what is running.
3g. A struct written in both languages is checked by a shader that reads it
   back, not by a static assertion on one side. C++ and std430 align a struct by
   different rules, so `sizeof` can agree with itself and disagree with the
   shader; what a device actually read is the only evidence. The same goes for
   how many buffers a pass binds: the count is named once, and the compiled
   SPIR-V is what the test asks.
4. Add one evolutionary pressure at a time and keep deterministic replay tests.
5. Prefer measurable behavioral milestones over adding simulation features in
   parallel.
6. Every new parameter is a slider as well as a flag. A setting only reachable
   from the command line is a setting nobody tries.

## Runtime flow

```text
ImGui controls
     |
SimulationState
     |
CPU generation boundary: fitness -> GA -> genomes/initial agents/occupancy
     |
GPU per step: lattice_clear -> lattice_step (sense, brain, bid)
                            -> lattice_resolve (move, occupancy, metrics)
     |
storage barrier
     |
3D view of the selected lattice -> off-screen image -> ImGui
```

Generation transitions are intentionally synchronous. The GPU performs the
expensive per-agent/per-step work; the CPU reads results and evolves 512 small
genomes only once per generation. This is inspectable and easy to validate
before asynchronous readback or GPU-side selection is added.

## Milestones

The metric-arena milestones that got the project here -- phototaxis, locomotion,
the trail field, thirteen scenarios -- are recorded in `PROGRESS.md` and are no
longer in the tree. What follows is the lattice.

### L1. The lattice itself — complete

- [x] shared `LatticeKernel.inl` compiled as C++ and as GLSL: cell addressing,
      6- and 26-neighbourhoods, the move rule, the claim rule, the distance
      metric and the trial fitness;
- [x] order-independent arbitration of a contested cell by `atomicMin` over
      agent indices, with the "only a cell empty at the top of the step may be
      entered" rule that makes it reproducible;
- [x] three-dispatch step: clear, decide, resolve;
- [x] 224-byte `AgentState` with the hidden block last;
- [x] neighbourhood sensing, 26 cells x 3 channels, with the lattice edge read
      as blocked;
- [x] deterministic beacon and spawn placement as pure functions of the seed;
- [x] fixed grid allocations with the extents clamped to fit;
- [x] lockstep CPU/GPU parity with an accumulated signed-drift budget, across
      both neighbourhoods and all four neuron models;
- [x] contention, full-lattice and genome-addressing probes on the device;
- [x] every lattice parameter as a slider and as a flag.

Exit criterion met: a 40-generation run at 24x24x12 with 256 genomes takes
arrival from 0 to 0.95 and median fitness from -1.6 to +3.5, so the task is
learnable and the shaping is readable.

### L2. Seeing it

- [x] a 3D view of the selected lattice: instanced cubes for agents, the beacon
      marked, the block field, and a movement trail per agent;
- [x] a camera that turns, zooms, slides and can be switched between perspective
      and orthographic, with a slab control for seeing inside a full box;
- [x] a per-world picker that reaches trials as well as groups;
- [x] see-through voxels that answer the opacity control. The transparent
      style resolves without sorting -- weighted blended, for the same reason
      the lattice settles a contested cell with a minimum -- and the weight is
      now a shared kernel with a test on its shape, because a weight that has
      gone flat produces a picture rather than an error;
- [ ] inspect one agent: its neighbourhood vector, activations and drives;
- [ ] generation timing.

### L3. What lives in a cell

- [x] a second thing a cell can hold, sensed through the existing `blocked`
      channel and standable, climbable and buildable-against: the construction
      world's block field, persistent across a trial and stored in snapshots
      because a tower is history rather than a position;
- [ ] the same field placed by the world rather than by agents, so a run can be
      given obstacles it did not build;
- [ ] a per-cell deposit that decays, replacing the arena's trail field with no
      diffusion constant to tune. The drawn trail is not this: it is a
      breadcrumb history no agent reads;
- [ ] a second agent kind, so a neighbourhood channel distinguishes kin;
- [ ] procedural obstacle layouts with train/evaluation seed separation.

### L4. Tasks worth the lattice

- [x] structures: a scored arrangement of occupied cells, which is what a
      discrete world can state and a metric one cannot. Height weighted by
      course, scored per world rather than per agent, with a frontier that only
      rises once a course is broad enough -- so a single thin column cannot
      finance a taller one;
- [x] a target the group must reach together rather than individually: every
      genome in a construction world carries the same score, so a useful
      foundation is worth as much to its builder as the block on top;
- [ ] chains and formations, measured by neighbour occupancy rather than by
      distance;
- [ ] a task whose solution needs the broadcast channel, so signal-off is a
      falsifiable ablation rather than a free drift.

### L5. Selection and scale

- [ ] colony fitness and related genome batches;
- [ ] an evolution-strategy update as an alternative to tournament selection --
      natural once one genome drives many bodies;
- [ ] asynchronous double-buffered generation readback;
- [ ] optional GPU selection/mutation backend;
- [ ] pluggable multi-layer/recurrent evaluator implementations;
- [ ] shader hot reload and capture/replay tooling.

Carried over and still true of the brain:

- [x] one network preset driving the CPU evaluator, the shader and the tests;
- [x] continuous-time hidden neurons with evolved time constants;
- [x] gated neurons whose time constant is recomputed from the inputs;
- [x] leaky integrate-and-fire neurons on the same integrator;
- [x] a brain plan of up to three hidden layers, chosen at runtime;
- [x] the network's structure written down as a document and checked by
      derivation;
- [ ] an ablation mode comparing reactive and recurrent brains on one command.

## Runners

`vulkan_lattice_agents` drives `SimulationDriver` from the frame loop.
`vklat_headless` drives the same driver from an `ImmediateContext`, so batch
sweeps and ablations produce results comparable with what the window shows:

```sh
vklat_headless --generations 200 --csv runs/beacon.csv \
               --save-champion runs/beacon-champion.vkng
vklat_headless --lattice 48x48x24 --neighbourhood faces --generations 200 --quiet

# Fitness shaping is a parameter, so a sweep needs no rebuild.
for penalty in 0.0 0.01 0.05; do
  vklat_headless --generations 50 --seed 5 \
                 --refusal-penalty "$penalty" --csv "runs/refusal-$penalty.csv"
done
```

## The world, and where its rules live

`include/vkexp/lattice/LatticeKernel.inl` is the world. Cell indexing, both
neighbourhoods, the movement threshold, which axes a neighbourhood allows, what
makes a cell enterable, which of two bids wins, the distance metric and the
trial fitness are all functions in it, and both languages compile them. A rule
written anywhere else is a rule the CPU reference and the shader can disagree
about, which is rule 3b.

`LatticeWorld` places beacons and spawns as pure functions of the seed and the
world index -- not as a generator, so world 91 can be placed without having
placed world 90, which is what lets a snapshot resume and a test build one world
in isolation.

The occupancy grid is one `int32` per cell, sliced per logical world. The slice
bound is the isolation between worlds: there is no check to forget, because an
agent cannot address a cell outside its own slice.

## Changing the sensor suite or the brain

`include/vkexp/neuro/BrainKernel.inl` is the network preset. It declares how
many neighbours are sensed and in how many channels, how many beacon, self and
recurrent inputs there are, the hidden capacity, and what the outputs mean;
every offset, the input capacity, the genome size and the packed GPU layout are
derived from those numbers.

Both languages compile it, so raising `BrainSelfInputCount` from 4 to 5 moves
the CPU evaluator, the sensor sampler, the compute shader and the tests
together, and nothing else needs editing. The block offsets are asserted to tile
the input vector without gaps, so a sensor block that no longer fits fails
loudly, and `describeBrain` names the new block without being told about it.

The neighbour count is asserted equal on both sides: `BrainNeighborCount` and
`LatticeNeighborCount` are the same twenty-six, and a static assertion says so
rather than a comment.

## Neuron model

Hidden neurons are continuous-time: each holds a state and integrates toward its
activation, `y += (dt/tau) * (-y + y_in)`. Where `tau` comes from is a runtime
setting -- pinned to the step, one gene per neuron, or recomputed each step from
the inputs -- and a fourth model, leaky integrate-and-fire, keeps the same leak
and changes only what leaves the neuron. The integrator and the mapping into the
tau range live in `BrainKernel.inl` beside the rest of the preset, so the CPU
evaluator and the shader cannot integrate a neuron differently, and `dt` is
explicit so a memory is a number of seconds rather than a number of steps.

Two identities keep the models comparable instead of merely adjacent, and both
are asserted rather than inferred. `tau = dt` reduces the update to
`y = activation`, which is the memoryless network. And a gate fed a constant
reproduces the fixed-time-constant neuron exactly, because the gate and the gene
enter the same mapping.

The genome carries every model's genes at once, so switching is a parameter
change and not a reinterpretation of the population.

The state lives on the agent record, next to everything else a step carries, so
the CPU path and the GPU path store it the same way and multi-step parity covers
it without a separate harness. It is zero at the start of a generation, which is
the whole of the reset semantics.

## Structs written twice

Three structures exist in two languages at once: the agent record, the per-step
parameter block, and the fitness weights inside it. `static_assert` on `sizeof`
and `offsetof` guards the C++ side, and for a long time that was taken for the
whole guard. It is not, and the gap cost more than any other mistake in this
project so far.

C++ and std430 align a struct by different rules. `alignas(16)` on a block of
eight floats aligns it to sixteen; std430 gives a struct the alignment of its
widest member, which for eight floats is four. Put twenty-three scalars in
front of such a block and one language starts it at 96 while the other starts
it at 92 -- two strides, 128 against 124, and a C++ assertion that passes
because it is only ever comparing C++ to itself.

What makes that specific mistake so expensive is where it shows up. The step
parameters live in a buffer indexed by the step, so slot zero is very nearly
right and every slot after it is a struct read off the end of itself. One step
per submission never leaves slot zero. So the bug hid behind a batch size,
looked like a race, survived a full barrier between steps, and made every
batched measurement -- which is every measurement the window has ever shown --
a run against garbage extents.

Two rules follow, and both are cheap:

- a block shared with a shader is declared out of vectors, so both languages
  align it the same way by construction rather than by counting the scalars in
  front of it;
- the check is a shader that reads the struct back. `layout_echo.comp` hands
  every field of both records to the host as raw bits, at an index that is
  deliberately not zero and with a decoy in slot zero, and the host compares by
  name. It is the cheapest case in the suite.

The same reasoning covers descriptor layouts. How many buffers a pass binds is
named once, the driver and the parity harness both build from that name, and
`testShaderBindingContract` reads the compiled SPIR-V and checks it. A pass
that grows a buffer while a layout does not is undefined behaviour, not a
reported error -- which is how the parity harness came to run the step shader
with two descriptors unbound.

## Parity on a discrete world

The move rule is a threshold on a float, so a one-ulp difference in a drive that
happens to sit on the dead zone flips a discrete move and the two trajectories
part company. That is a property of the world and not an error, which is why
trajectory parity is run in lockstep: the CPU state is fed to the GPU each step
and one step of each is compared. Lockstep measures agreement; a free-running
comparison would measure chaos.

Because lockstep cannot see a bias smaller than the per-step tolerance, the
signed differences are summed across the run and held under a budget an order of
magnitude above the rounding noise.

Integers are compared exactly. There is no such thing as a cell that is nearly
right, and a tolerance on one would hide the class of bug the test exists for.

## Units

`include/vkexp/simulation/Units.hpp` reconciles the step's two time bases. A run
is replayed by step count, archives record steps, and parity compares step for
step. A second is the unit of anything that decays, which now means the neurons.
A cost charged per event -- a move, a refusal -- takes no `deltaTime` at all,
because it is counted rather than integrated. That simplification is what the
discrete world bought.

## Selection

Fitness sharing is an option and not a default, because the question it answers
is empirical. Individual selection makes a signal that only helps a neighbour a
net cost to its sender, so no network architecture can produce a code under it;
sharing removes that obstacle but also removes the pressure that distinguishes
genomes inside a world. Which trade wins is measured, not asserted, so the knob
spans both ends continuously and the reported numbers stay individual at every
setting so runs remain comparable.

A sweep is how the measurement is taken without leaving the window: stages that
differ in one value and in nothing else, each starting from the seeded initial
population rather than from the previous stage's result. It lives in the driver
and not in the UI module, because the comparison is an experiment and rule 3a
puts experiments where a window is not required to reach them.

## Replay

Watching is a mode of the same driver, not a second one: a replayed generation
is simulated, scored and reported through the ordinary path, and only selection
is skipped. That keeps a replay honest -- the fitness under the champion is
computed the way training computed it -- and it keeps the run repeatable,
because the generation counter that seeds beacon placement does not advance. The
smoke test asserts the repetition byte for byte, since a leak of evolution into
replay would show up as a different ending rather than as a worse number.

## Persistence

Two formats, deliberately separate. A genome archive (`.vkng`) carries weights
between runs and is the thing to keep. A run snapshot (`.vklr`) carries a whole
experiment between sessions -- population, which cell every agent stands in,
generation, step and every setting -- and is the thing to reopen. The snapshot
is written field by field through one visitor list walked in both directions, so
save and load order cannot diverge, and a size assertion on `SimulationStep`
makes a newly added tunable a build failure rather than a silently dropped
field.

Both refuse a file written for a different brain. Archives are version 4 and
version 4 is also the oldest accepted: an archive from the metric arena holds
weights addressed to photoreceptors that no longer exist, so loading one would
be silently wrong rather than usefully old. Snapshots use the new lattice magic;
version 5 adds optional cardinal side support after version 2 introduced the
built field, version 3 its scoring settings and version 4 the construction
frontier.

The occupancy grid is not stored. It is derived from where the agents stand, so
rebuilding it on load costs less than writing it and cannot disagree with the
agents it was built from. Construction is stored: a tower is history and cannot
be inferred from where its builders happen to stand.

## A third world: building towards something

The construction world scores blocks weighted by height, which makes the
structure the goal. That is the thing wrong with it. A goal that *is* the
building admits exactly one answer -- more building -- and every question worth
asking about architecture is a question about what a building is *for*.

So the third world puts the reward somewhere a building is the only way to
reach, and stops scoring the building at all.

**The shape of it.** A resource sits at a cell above the floor. Nothing else
changes about movement: in a construction world an agent may only climb a face
of an existing structure, so with nothing built nobody leaves height zero, and
any resource above the floor is unreachable by definition. To reach it, someone
has to build something climbable underneath it. The structure is the path, and
its shape is whatever gets an agent up there -- which is what makes it
architecture rather than mass.

**Fetch, not touch.** An agent that reaches the resource picks up a load and
scores when it carries the load back down to the floor. Touching would be
cheaper to implement and much weaker: a one-shot scramble scores as well as a
staircase, so nothing selects for a path anyone can use twice. A load that has
to come back down makes the route pay off every time it is used, which is what
turns a lucky pile into infrastructure -- and, with a group in one world sharing
a score, what makes one agent's ladder worth building for the others.

**Where the resource is.** A pure function of the world index and the seed, the
way beacons already are, so a genome is scored on several placements rather than
one it can memorise, and so world 91 can be placed without placing world 90. It
lives in `LatticeKernel.inl` and both languages compute it: nothing is mirrored
onto the agent record, which leaves `beacon.xyz` free to keep meaning the
per-step build intent, as it does in the construction world.

**What fitness is.** Deliveries, plus the height shaping the construction world
already accumulates. Deliveries alone is a needle in a haystack -- no early
population reaches height four by accident -- and the best height an agent
touched is a gradient pointing the right way that costs nothing new to compute.
Blocks score nothing. A block is time spent, and spending it well is the whole
problem.

**What to watch.** Not mean fitness. The shape descriptors: whether the thing
under the resource is a column, a ramp, or a staircase, and whether the same
world builds one path or several. A run where deliveries rise while block count
falls is the result worth having, because it means the group found a cheaper
route rather than a bigger pile.

**The construction frontier is off here, and that is a rule rather than a
default.** Whether a block may stand this far above the filled part of the world
is a fair question in a world built on ground and a nonsensical one over a hole:
a cantilever has nothing at all beneath it, so the test refuses every block of a
bridge. Construction and harvest ask it; the chasm does not, and the panel says
so instead of offering a slider that would quietly make this world unsolvable.

## A fourth world: the chasm

The harvest world hangs its resource above the floor, and the way up is a
structure. The chasm takes the floor away.

**The ground is blocks now.** Both support rules used to answer "yes" for
anything at height zero, so the floor was solid everywhere and a hole in it was
inexpressible -- and, worse, would have been invisible: an empty cell at height
zero and a cell over a drop read identically to a sensor, which is the same
mistake a wall and a block were making until recently. So height zero is a
course of bedrock in the block field, support means "something below me" with no
special case, and a chasm is where that course is missing. No new rule, no new
buffer, no new sensor: the drop is visible because bedrock is visible. Bedrock
is negative, so fitness, the shape descriptors and the renderer all tell terrain
from work by its sign.

**Half the floor, and the resource over the other half.** The group starts on
the ground; the resource hangs in a band of heights over the open half, hashed
per world like everything else that is placed. Walking there is impossible, and
climbing there is impossible, because there is nothing under it to climb. The
only route is one the group builds out from the edge.

The split runs along x, and the world defaults to a 32x32x32 cube. Which axis is
arbitrary to the simulation and not to the eye: the camera starts side-on to the
widest horizontal edge, so a chasm cut along x is the one you are already
looking across. The cube follows from that -- the span to cross is half the
width, so width is the number that sets the difficulty, and a shallow box would
make the far side read as a wall rather than as a far side.

**The objective is drawn as a place, not as a cube.** One cube hanging in air
has no depth cue: at thirty-two cells deep, a resource at the near edge and one
at the far edge project to nearly the same pixels, and which column the group
has to reach is the whole question. So the objective also gets a plumb line down
to the floor plane and a cross on it -- which doubles as the answer to whether
there is floor under it or a hole.

**The sequence that solves it already exists.** An agent standing on bedrock
builds beside itself, supported from below. It walks to the edge and aims into
the air: no support below, and the placement falls back a level onto the side
face of the block it just made. It steps onto that block, which puts it one
level up, and aims forward again -- unsupported at its own height, so the
fallback drops the target to the level of the cantilever, where a side face
holds it. That fallback was written for exactly this and the comment beside it
says so. The world is not new behaviour; it is a task for behaviour that had
nowhere to be useful.

**Side support is forced on and the build interval defaults to three.** A world
whose objective needs a cantilever must not be startable without one. And a
crossing is a long run of placements that pay nothing until the last one lands,
so twelve ticks of cooldown per block would price it out of reach; three does
not. Both stay sliders.

**What it will produce is a bridge, not an arch.** There is no load model, so a
cantilever may run forever and the cheapest crossing is a plank one block thick.
That is already a functional structure and a long way from a dense pile, and the
descriptors will show it -- overhangs up, compactness down. An arch needs a
reason to curve, and the cheap one is a limit on how far a side-supported block
may sit from a column-supported one. That limit interacts directly with how wide
the chasm is, so it is a knob for later, not a rule for now.

**What is deliberately not in the first version.** A depleting resource,
several resource nodes, a resource that has to be carried to a particular
deposit rather than to the floor, and any cost per block. Each is a separate
pressure and rule 4 says one at a time.

## Immediate next step

**Measure again from scratch.** Every run recorded before the step parameter
block was fixed batched more than one step per submission, and every step after
the first in each batch read its settings from four bytes off the end of the
previous block. Extents, cells per world and the world base were garbage, so
those runs were not measuring the settings they reported. Nothing in
`PROGRESS.md` from the lattice era survives as evidence, and the fitness curves
in it should be read as "a run happened", not as "this configuration behaves
like this". That is a cheap thing to redo and an expensive thing to forget.

Then the question construction was built for: whether sparse collective height
is sufficient. Every world now has a persistent, sensed block field and every
genome in it carries the same score. The first evidence to read is not mean
fitness but the shape of replayed structures -- isolated columns mean height
alone is enough, terraces or cooperating climbs mean the movement constraints
are selecting coordination. Only after that comparison should compactness,
symmetry or material cost enter fitness, since each defines a different
aesthetic rather than displaying this one.

One thing to watch while reading a crowded world from outside: the trails and
the agents resolve through the same transparency pass, so a long trail history
puts a thousand fragments in front of twelve. The depth weighting now gives the
nearest surface about fifty times the say of the far side, which is what makes
an agent read as an agent, but a box full of breadcrumbs is still a box full of
breadcrumbs. Turning the trails off, or shortening them, is what makes the
opacity control legible.
