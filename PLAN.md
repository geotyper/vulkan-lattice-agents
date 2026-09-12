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
(3D view of the selected lattice -- not written yet)
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

- [ ] a 3D view of the selected lattice: instanced cubes for agents, the beacon
      marked, an orbit camera and a slice control;
- [ ] a per-world picker that reaches trials as well as groups;
- [ ] inspect one agent: its neighbourhood vector, activations and drives;
- [ ] generation timing.

### L3. What lives in a cell

- [ ] static obstacles as a second occupancy value, sensed through the existing
      `blocked` channel;
- [ ] a per-cell deposit an agent can leave and read, replacing the arena's
      trail field with no diffusion constant to tune;
- [ ] a second agent kind, so a neighbourhood channel distinguishes kin;
- [ ] procedural obstacle layouts with train/evaluation seed separation.

### L4. Tasks worth the lattice

- [ ] a target the group must reach together rather than individually;
- [ ] structures: a scored arrangement of occupied cells, which is what a
      discrete world can state and a metric one cannot;
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
be silently wrong rather than usefully old. Snapshots restart at version 1 under
a new magic, for the same reason.

The occupancy grid is not stored. It is derived from where the agents stand, so
rebuilding it on load costs less than writing it and cannot disagree with the
agents it was built from.

## Immediate next step

The renderer. It is the one piece of the conversion deliberately left undone,
and it is now the thing blocking the questions worth asking: whether a champion
walks a straight line to the beacon or feels its way along neighbours, whether
groups clump or spread, and whether a refusal is a mistake or a queue. None of
that is visible in a fitness curve, and all of it is visible in a picture of
sixteen thousand cells.

Instanced cubes over the agent buffer, the beacon marked, an orbit camera and a
slice so the inside of a 3D box can be seen at all. The agent buffer is already
published and already read-only for a consumer, which is what the 2D renderer
established and what survives it.

After that, measure before extending. The four neuron models and both
neighbourhoods are one flag apart from each other, and none of the comparisons
has been run on the lattice: arrival is the number to read, not best fitness,
because a policy that never arrives and one that arrives and leaves both look
like adequate fitness.

Held deliberately: obstacles, deposits and a second agent kind are all the same
change -- another value a cell can hold -- and writing one of them well is worth
more than writing all three. The lattice makes them cheap enough that the reason
to wait is the renderer, not the difficulty.
