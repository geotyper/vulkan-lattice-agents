# Vulkan Lattice Agents

An experimental C++20/Vulkan playground for evolving thousands of small agent
brains on the GPU. The world is a discrete 3D lattice: an agent occupies one
cell, sees the twenty-six cells around it, and either steps into one of them or
does not. A genetic algorithm evolves the weights of a fixed dense network, and
the same network is evaluated identically on the CPU and on the GPU.

The lattice replaced a metric 2D arena -- bodies, velocities, drag, collision
impulses, a decaying trail field and thirteen scenarios. That history is still
reachable through `git log` and `git show`; it is not described here, because
none of it runs any more. What the move bought is that space is now countable:
a cell is occupied or it is not, a move happens or it is refused, and every
disagreement between the CPU and the GPU is an integer that differs rather than
a float that drifted.

Simulation, evolution, UI and Vulkan infrastructure are separate targets rather
than one application-specific module.

## Current experiment

There are two selectable tasks. **Construction** starts agents on the floor and
gives them a shared persistent block field; the interactive build opens in this
mode. **Beacon** remains the navigation baseline, with one hashed target cell per
logical world.

```
Lattice:    32x32x16 = 16384 cells per world, Moore (26)
Brain:      88 -> 20 -> 7
Trial:      900 steps = 15.0 s at 60.0 Hz
Population: 512 genomes x 4 trials = 2048 agents in 172 lattices
Movement:   threshold 0.25, beacon reached within 1 cell(s)
```

- 512 genomes, each evaluated in four trials (2048 GPU agents);
- the population partitioned into configurable logical groups from one agent to
  the whole population (12 agents per lattice by default), so with the default
  group size 512 genomes occupy 43 groups and 172 independent lattices;
- 26 neighbouring cells read as three channels each -- occupied, blocked,
  and what the occupant is broadcasting -- 78 inputs;
- four task inputs: beacon direction/nearness, or height, build readiness,
  previous build success and physical support;
- the agent's own heading as the unit step it last took, plus a flag saying its
  last move was refused, 4 inputs;
- two recurrent memory cells fed back, 2 inputs;
- `88 inputs -> 20 tanh neurons -> 7 outputs` by default, with the hidden layers
  configurable from the Brain window: up to three of them, 32 neurons in total;
- every hidden neuron holds its own state and a time constant that is evolved,
  recomputed from the inputs each step, or pinned to the step, so a memory is
  measured in seconds and can be held until something says to let go;
- outputs are three movement drives, one broadcast intensity, one build impulse,
  and the two recurrent cells;
- a move is one cell per step, along the faces or along any of the 26 diagonals,
  chosen by a setting;
- contested cells resolved by a rule that does not depend on the order agents
  are stepped in, so the CPU reference and the GPU agree exactly;
- elitism, tournament selection, uniform crossover, Gaussian mutation;
- adjustable lattice extents, generation length, agents per world, neighbourhood,
  move threshold, contact radius and every fitness coefficient, all as sliders
  and all as command-line flags;
- separate Simulation, Genetic Algorithm, Brain, view-settings, clean Lattice view and profiler
  windows;
- a live 3D view of one selected world, with instanced agent voxels, a marked
  beacon, perspective/orthographic orbit and zoom controls, an axis-aligned
  slice and optional see-through voxels and coloured breadcrumb trails;
- fitness/arrival history graphs for completed generations.

The four trials are not four independently trained populations. Every genome
controls four agents with the same weights but different spawn cells and a
different beacon, and their scores are averaged before selection, which
discourages a solution that only works from one corner of one lattice.

### Construction world

A construction block normally appears only on the floor or directly above
another block in the same `(x,z)` column. The optional `Side-supported bridges`
rule also accepts contact through one cardinal x/z face; diagonal edge and
corner contact never provide support. With the option enabled, an unsupported
placement in front of an elevated agent falls back one level, in front of its
feet. An agent standing on a column can therefore start a bridge, step onto it
and extend it. The build output has a configurable threshold and a successful
placement starts a configurable cooldown (12 ticks by default). An occupied
target is rejected both when actions are proposed and again when the winning
action is committed, so an existing block can never be built a second time.

Construction also has a shared moving frontier. A course advances the
foundation height only after at least 25% of its x/z area is occupied, and
courses are counted consecutively upward from the floor. New blocks may be
placed at most five levels above that foundation. Sparse high blocks therefore
cannot lift the frontier: a group that wants more height has to broaden every
lower course first. Both the fill threshold and the five-level headroom are
runtime settings. This encourages terraces, buttressed towers and clusters
without prescribing symmetry or a target silhouette.

Agents normally walk on the floor or on top of blocks. Moving into a one-block
ledge steps onto it; an upward drive while facing a column climbs the face while
preserving that facing. Moving away from every supporting face causes an
immediate fall to the nearest floor or block below. A construction block appears
in the blocked sensor channel like a wall, but contact with it is not counted as
a refused move and carries no wall penalty.

Fitness is deliberately collective: at the generation boundary every block
scores its one-based height (`y + 1`), and the values are summed for the whole
logical world. The sum is not divided by the number of blocks: doing so would
make a useful new foundation lower the average and favour one thin column again.
Every agent-tick spent on the horizontal perimeter is then charged the
configurable `boundaryPenalty` (0.002 by default); neither the construction
frontier nor the height ceiling is a boundary for this purpose. Every genome
sharing a world receives the same net total, averaged over its four trial worlds. The renderer gives blocks a
clay-to-sun gradient by height, a small deterministic maker variation and narrow
seams, so a growing structure reads as masonry rather than a single flat prism.

**The viewport reads the simulation; it does not participate in it.** Solid
voxels write depth and make a crowd's surface readable. See-through voxels use
weighted blended order-independent transparency, so looking inside a world does
not require reading the GPU agent buffer back and sorting it on the CPU. The
camera starts side-on and centred on the lattice, looking along the shorter
horizontal axis so the world's widest edge fills the view. It can switch between
perspective and orthographic projection. A slice
along x, y or z is the exact alternative when adjacency matters more than the
whole population. Drag the picture to orbit and use the wheel to zoom. In
construction mode the view reads the built-block field directly, without a CPU
copy or compaction pass.

`Trails` records the last 256 resolved cells of every agent in a fixed GPU ring
and draws a configurable newest span as smaller translucent voxels. Colour is
stable per genome, making paths separable when agents cross. This history is a
picture only: it is not an input to the brain, does not affect fitness, and is
cleared at a generation boundary or snapshot restore.

## The lattice

A cell is addressed as `(z * height + y) * width + x`, and a world's occupancy
is one `int32` per cell holding the index of the agent standing there or `-1`.
Every world gets its own slice of one grid, so an agent physically cannot read
or write a cell belonging to another world: the slice bound is the isolation.

Extents run from 4 to 128 per axis. The default 32x32x16 is 16384 cells, which
is 64 KiB of occupancy per world and the same again for the claim grid the move
resolution uses. That is deliberately small: **many worlds, each cheap**. The
alternative -- one large lattice with the whole population in it -- would have
meant giving up the group structure the genetic algorithm is built on, and with
it the four trials, the sweeps and every statistic that compares worlds.

**The allocation is fixed and the lattice gives way.** Occupancy, claims and the
construction field are allocated
once at a budget (256 MiB, or a sixteenth of device memory, whichever is
smaller) and never resized. A configuration that would not fit is shrunk to one
that does -- the longest extent halved until it fits -- and the clamped extents
are written back into the settings, so the sliders show the box that is running
rather than the one that was asked for. This is the lesson of the 2D trail
field, where every resize was a buffer-lifetime bug under live descriptors:
`vklat_reconfiguration_smoke` asserts that the step resources are built exactly
once across every reconfiguration the UI can produce.

**Two neighbourhoods, one input width.** `faces` allows the 6 axis-aligned
steps; `moore` allows all 26. The *sensed* neighbourhood is 26 cells either
way, so a population evolved under one movement rule loads into the other and
the comparison is an ablation rather than a different network. The setting
also chooses the distance the fitness is measured in -- Chebyshev under Moore,
Manhattan under faces -- because in each case that is the number of steps the
move rule would actually need.

## Moving, and who gets the cell

Three outputs are movement drives, one per axis. Each is turned into `-1`, `0`
or `+1` by a threshold: a drive has to be *sure* to become a step, and a drive
inside the dead zone is a decision to stand still. Under `faces` only the
dominant axis may move, which is where the two neighbourhoods differ and the
only place they differ.

A step is refused if the target is outside the lattice, or if somebody was
already standing in it when the step began. Both refusals are recorded on the
agent and read back as an input on the next step, so a policy can notice it is
stuck without having to infer it from the neighbourhood.

**Who wins a contested cell.** The obvious implementation -- `atomicCompSwap`,
first writer takes the cell -- would make the outcome depend on the order the
GPU happened to schedule its invocations in. No GPU promises that order and no
CPU reference can reproduce it, so the parity test would be measuring the
scheduler. Instead every bidder writes `atomicMin` of its own index into the
cell's claim slot, and the lowest-numbered bidder wins. A minimum is
associative and commutative, so the answer does not depend on the order, and
the CPU reference reaches it by plain iteration.

The price is that a chain cannot shuffle in one step: a cell may only be entered
if it was empty at the *top* of the step, so an agent stepping into the cell
somebody else is leaving is refused this step and succeeds the next. That is a
rule of the world rather than an artefact -- it is written down in
`LatticeKernel.inl` and asserted by `testLatticeContention` and by
`runContentionProbe` on the device.

The step is three dispatches, and each one exists because of the rule above:

| Pass | What it does |
| --- | --- |
| `lattice_clear` | empties the claim grid |
| `lattice_step` | senses, runs the brain, writes the intent and bids |
| `lattice_resolve` | moves the winners, updates occupancy, charges the metrics |

Nothing moves in the middle pass. An agent that moved there would change what
the agents after it see, and the answer would depend on the numbering again.

## What an agent senses

Eighty-eight numbers, and the preset that declares them is compiled by both
languages:

```
inputs 88                              hidden 20          outputs 6
  26 neighbours x 3                = 78   each with         3 move drives
    (occupied, blocked, broadcast)       its own state      1 broadcast level
  beacon direction + nearness      =  4   and its own       2 recurrent cells
  heading + refused flag           =  4   time constant       (fed back as inputs)
  2 recurrent cells fed back       =  2
```

The edge of the lattice reads as `blocked` rather than as an empty cell. There
is no boundary geometry and no push-out pass: a lattice simply ends, and the two
places that has to be said are the sensor and the move rule.

**Broadcasts are read as they were at the top of the step**, never as the
neighbour has just rewritten them. On the device that falls out of the
ping-pong buffers; on the CPU it takes an explicit snapshot, and
`sampleAgentInputs` therefore takes the broadcasts as a separate span instead of
reading them off the agent records. It is a contract, not an optimisation: a
reference that read the live record would drift from the shader by an amount
that depends on the agent numbering.

## The beacon, and what a trial is worth

```
fitness = trackingReward * bestNearness      how near it ever got, 0..1
        + objectiveBonus * contacts          steps spent within the contact radius
        - motorCost     * effort             moves made, plus broadcast held
        - refusalPenalty * refusals          steps walked into a wall or a neighbour
```

`bestNearness` is a maximum rather than a final value, so a trial is scored on
its best moment and an agent that arrives and then wanders is not scored as if
it never arrived. Contacts accumulate, so staying is worth more than touching.
Effort counts a move as one and a broadcast as `signalCost` of one, which is
what makes silence the default and a signal something that has to earn itself
back.

The beacon is a pure function of the seed and the world index -- `beaconCell`
is not a generator, so world 91's beacon can be computed without having computed
world 90's. The seed advances with the generation, so a population cannot learn
one fixed set of 172 positions. Spawn cells are hashed the same way and probed
forward from there, never onto the beacon: an agent that starts on the objective
has solved the world before the first step, which would make the shaping
unreadable for the whole group it is scored beside.

`arrivalRatio`, plotted next to the fitness curves, is the fraction of agents
that spent at least one step within the contact radius. It is the one number
that no fitness coefficient moves arithmetically, which makes it the honest
comparison between settings.

## Group fitness sharing

Selection is individual by default: a genome is scored on what it did, so a
broadcast that only helps a neighbour is pure cost to its sender. That is a
fitness property, not a network one, and no architecture fixes it -- which is
why `--fitness-sharing` (and the matching slider) exists as a comparable option
rather than a new default.

The setting blends each genome's score toward the mean of the genomes sharing
its logical world:

```
f' = (1 - s) * f + s * mean(f over the world)
```

At `s = 0` nothing changes; the input is returned unchanged, so the option off
is the old behaviour rather than a reimplementation of it. At `s = 1` a whole
world is scored together, selection acts on groups, and helping a neighbour pays
its sender back. Sharing only redistributes fitness inside a world and never
changes a world's total, so selection pressure between worlds survives at every
setting.

Only selection sees the shared numbers. Reported and plotted fitness stays
individual, because a shared run and an unshared one could not otherwise be
compared on their headline figures: sharing compresses spread by construction.
Arrival is untouched either way and is the cleanest comparison.

### Sweeping it from the window

Whether sharing helps is an empirical question, and one run cannot answer it.
The **Sharing sweep** panel runs the same experiment once per value -- up to six
stages -- and restarts evolution between them, so a stage measures its own
setting rather than that setting applied to whatever the previous one had
already evolved. Every stage keeps its own curves, and they are plotted on one
shared axis, because separately autoscaled plots would make a flat run and a
climbing one look alike.

Arrival is plotted first and on a fixed 0..1 axis. The last stage is not
restarted when it ends -- the run simply carries on at that setting with every
stage kept for reading.

The same comparison from the batch runner, which writes CSV instead of curves:

```sh
for share in 0.0 0.5 1.0; do
  vklat_headless --generations 60 --seed 5 \
                 --fitness-sharing "$share" --csv "runs/share-$share.csv"
done
```

Worth running with a non-zero `--signal-cost`. While broadcasting is free, the
channel stays a free drift even under shared fitness: what pays back has to cost
something first.

## Neuron time constants

Every hidden neuron carries its own state and integrates toward its activation
at its own evolved time constant:

```
y += (dt / tau) * (-y + activation)
h  = tanh(y)
```

`tau` is a gene, so how long a neuron remembers is selected for rather than
designed, and different time scales become a trait evolution can separate: a
fast neuron is a reflex that tracks its input within a step, a slow one holds a
fact across seconds. `dt` enters explicitly, so a memory is measured in seconds
and not in steps. A time constant of half a second closes `1 - 1/e` of the gap
to its input in half a second at 30 Hz, at 60 Hz and at 240 Hz, and the unit
test asserts exactly that.

The gene enters a bounded logarithmic range, from one step (16.7 ms) to four
seconds. Logarithmic because what matters about a memory is its order of
magnitude: a linear map would spend most of the gene range between two and four
seconds. A gene of zero lands on the geometric middle, about 260 ms.

This is where memory belongs. The two recurrent cells put it in the *output*
layer, which squeezes everything a brain might remember through a two-number
bottleneck; time constants give all twenty neurons a state and take no output
slot at all. The recurrent cells stay -- they are an explicit, inspectable
channel, and keeping them is what makes the recurrent-versus-time-constant
ablation possible -- but they are no longer the only thing holding the past.

### Four models, one integrator

Where the time constant comes from is a setting -- **Neuron model** in the
window, `--neuron-model` on the command line -- and it is nearly the only thing
that changes between the four. The integrator, the genome and the state are the
same in every case, which is what makes switching an ablation rather than a swap
between networks, and what lets a population keep its meaning across a switch
mid-experiment.

| Model | Time constant | What it is |
| --- | --- | --- |
| `reactive` | pinned to `dt` | The update collapses to `y = activation`: no state at all, the network from before time constants existed, reached by the same arithmetic. |
| `time` (default) | one gene per neuron | Fixed for the neuron's life. It forgets at one rate whatever is happening to it. |
| `gated` | recomputed each step from the inputs | The neuron can hold a value and then let go of it when something tells it to. |
| `spiking` | one gene per neuron | Leaky integrate-and-fire: the same leak, but the neuron emits 1 on crossing threshold and resets to zero instead of passing its state through `tanh`. Its output is a pulse train rather than a level. |

Gated is a strict generalisation: feed the gate a constant and it *is* the
fixed-time-constant neuron, which the unit test asserts directly for three
different constants. It costs one weight row and one bias per hidden neuron and
no extra state, because the state it needs is the one the neuron already carries.

Spiking is the one model that changes what leaves the neuron rather than only
how fast its state moves. It reads the same time-constant gene as `time`, so the
switch between the two is exactly the question of whether a level or a pulse
carries more here.

**Watch the sign.** The gate asks for a time constant, not for an update
fraction, so driving it up makes the neuron hold and leaving it low makes it
follow. That is the opposite of a GRU update gate. It is this way round because
the gate and the gene go through the same mapping, which is what makes the two
models comparable at all.

```sh
for model in reactive time gated spiking; do
  vklat_headless --generations 200 --seed 5 \
                 --neuron-model "$model" --csv "runs/model-$model.csv"
done
```

### The structure, and where each model lives in it

One preset, `include/vkexp/neuro/BrainKernel.inl`, declares the whole network,
and both languages compile it. Every offset, the genome size and the packed GPU
layout are derived from the counts in it, so raising one number moves the CPU
evaluator, the sensor sampler, the compute shader and the tests together.

The genome is one flat vector, as long as the plan needs. Under the default plan
-- one hidden layer of twenty -- that is 3727 floats in seven blocks:

| Block | Size | Read by |
| --- | --- | --- |
| inputs -> hidden 0 | 88 x 20 = 1760 | every model |
| hidden 0 bias | 20 | every model |
| hidden 0 -> output | 20 x 7 = 140 | every model |
| output bias | 7 | every model |
| time constants | 20 | `time`, `spiking` |
| gate 0 weights | 20 x 88 = 1760 | `gated` |
| gate 0 biases | 20 | `gated` |

A deeper plan has one weights-and-bias pair per layer, and one gate pair to
mirror it; the output layer always reads the last hidden layer. `12,8,8` comes
to 2579 weights in fifteen blocks -- *fewer* than the flat default, because the
first matrix is what dominates.

Every model carries every block, whichever one is selected. That is deliberate:
it makes switching a parameter change rather than a reinterpretation of the
population, so a saved run stays meaningful across a switch, and it is why the
four are comparable at all. The packing limit (`BrainStrideMask`, 4095 per
stride) is still not near.

State is one float per hidden neuron, on the 224-byte agent record beside
everything else a step carries, so the CPU path and the GPU path store it the
same way and multi-step parity covers it without a separate harness. It is zero
at the start of a generation, which is the whole of the reset semantics. The
gate needs no state of its own: what it needs is the state the neuron already
has.

### Choosing the structure

The hidden layers are a plan, not a constant. The **Brain** window sets how many
there are and how wide, `--hidden 12,8,8` says the same from a command line, and
both the CPU evaluator and the compute shader walk whatever is chosen:

| | |
| --- | --- |
| Layers | up to 3, dense from the front |
| Neurons | 32 in total, spent however the plan likes |
| Default | one layer of 20 |

**Only the hidden layers, and that is the design rather than a limitation.** How
many cells an agent can see and how many drives a move needs are statements
about the lattice, so the two ends belong to the world. How much brain to spend
on the world is the question worth asking, and it is the only one the window
asks.

**The capacity is compiled in; the plan is not.** GLSL sizes its arrays with
compile-time constants, so how many neurons there may be at most, and how many
layers, live in `BrainKernel.inl`. Everything inside that -- how many layers this
run uses, how wide each one is, where every weight of every layer lives -- is
computed at runtime by shared kernel functions that walk the plan, so the two
languages cannot walk it differently. `compute_smoke` runs a parity case at
`12,8,8` precisely because every other case in the file runs the single layer
the network usually has: a shader that read the plan even slightly differently
would drift there and nowhere else.

**The genome is exactly as long as its plan.** There is no fixed stride and no
tail: the flat default is 3727 weights, `12,8,8` is 2579, and a single 32-wide
layer is 5959. Interchangeability comes from the file saying which network it
holds, not from every run sharing one length -- an archive records the plan in
its header and the structure block beside it, and refuses to load into a build
that lays that network out differently, naming the block that moved.

A deeper plan is usually *cheaper* than a flat one, which is worth knowing before
reaching for it: the first matrix dominates, so a narrow first layer shrinks the
whole network even as it makes it deeper. That matters more here than it did in
the arena, because 88 inputs is a wider front than 61 was.

**The default width and the capacity are separate numbers**, and a test says so.
Sharing one constant would mean that raising how many neurons there *may* be
widens the brain behind its back -- which is exactly what happened once while
this was being built.

**Each layer holds its own state.** The time constants are per neuron, numbered
across all layers end to end, so a deep plan is not just a longer path but a path
with different memories along it -- a fast layer in front of a slow one is
something a run can be.

**A plan takes effect on a reset**, because it is a different layout of the same
genome: the population evolving under the old one does not carry over
meaningfully. The window says "not applied yet" rather than pretending
otherwise, and offers the default plan back in one button.

### The same structure, written down

The table above is hand-written, and every offset in it is really a function
call: the input vector is addressed by `brainNeighborChannelIndex` and its
siblings, the genome by `brainHiddenWeightIndex` and its siblings, and both
languages compile those from the one preset. That makes the layout impossible to
get *wrong* -- and impossible to *state*. Nothing could hand a file, or a
reader, the sentence "slots 78 to 82 are the task state".

`describeBrain` produces exactly that sentence, as a structure of named blocks,
and it produces it by asking the same index functions where each block begins.
It is derived, never restated, which is rule 3c applied to the layout itself.

```sh
vklat_headless --neuron-model gated --describe-brain brain.json
```

```json
{
  "inputs_count": 88, "hidden_count": 20, "outputs_count": 7,
  "weight_count": 3727, "neuron_model": "gated",
  "inputs": [
    { "name": "neighbourhood", "offset": 0, "count": 78, "rows": 26, "columns": 3 },
    { "name": "task", "offset": 78, "count": 4 },
    { "name": "self", "offset": 82, "count": 4 },
    { "name": "memory_in", "offset": 86, "count": 2 }
  ],
  "weights": [
    { "name": "hidden0_weights", "offset": 0, "count": 1760,
      "from": "inputs", "to": "hidden0", "rows": 20, "columns": 88 },
    ...
  ]
}
```

The window writes the same document with `Save structure`, next to the archive.

**The test is what makes it worth having.** `testBrainDescription` asserts that
the blocks tile the input vector, the output vector and the genome exactly --
no gap, which would be a slot nothing names, and no overlap, which would be two
names for one number -- and that both corners of every weight block are where
the kernel's own index function puts them. A description that merely looked
right would be worse than none, because the loader below acts on it.

**Both file formats notice a brain that changed shape**, and one of them can say
how. A genome archive carries this document plus the layer plan in its header,
so a file states which network it holds; it refuses to load into a build that
lays that network out differently, naming the block: *"input block 'beacon' is
missing"* rather than *"3727 weights, expected 2539"*. Archives are version 4,
and version 4 is also the oldest accepted: an archive from the metric arena
holds weights addressed to photoreceptors and tactile sectors that no longer
exist, so loading one would be silently wrong rather than usefully old. Run
snapshots start again at version 1 for the same reason, under a new magic and a
new extension.

**What this is not, yet.** The layers are chosen; the *connections* are not. A
plan says how many layers and how wide, and every layer is still fully connected
to the one before it. A genome with its own topology -- connections as data,
added and removed by mutation -- is a different change: it takes the arithmetic
away from the shader, and needs structural mutation and speciation to go with
it. Capacity also stays compiled in, because GLSL sizes its arrays with
compile-time constants, so a file cannot invent a sensor or a thirty-third
neuron.

## Replay

Watching trained weights is a different job from training them, and the
difference is one flag. **Replay only (no evolution)** scores and reports every
generation exactly as a training run does -- that is how loaded weights get
judged -- and then selects and mutates nothing. The population is left alone, so
the same genomes respawn; the beacon seed is derived from the generation number,
which does not advance either, so the next generation is the same run again
rather than a similar one. That repeatability is asserted by
`vklat_replay_smoke`, which compares two replayed generations byte for byte.

**Load genomes** takes a `.vkng` archive, which carries weights and nothing
else -- exactly what replaying a champion needs, since the lattice is whatever is
set up in the window. An archive normally holds one champion or a handful of
elites while a run has a population size fixed when its buffers were made, so the
archive is repeated across the population and every agent runs the loaded brain.
The status line says the repetition happened rather than leaving it to be
inferred.

To watch a headless champion:

```sh
vklat_headless --generations 200 --save-champion runs/beacon.vkng
# then in the window: Load genomes -> runs/beacon.vkng, tick Replay only
```

A run snapshot (`.vklr`) is the other way in, and carries the lattice and every
setting with it; a genome archive keeps the window's current lattice and changes
only the brain.

## Architecture

```text
vulkan_lattice_agents (windowed composition root)
vklat_headless        (batch composition root)
  |
  +-- vklat_domain            no Vulkan dependency
  |     neuro/
  |       BrainKernel.inl     network preset compiled by C++ and GLSL alike
  |       NeuralNetwork       C++ view of the preset + single-network evaluator
  |       BrainDescription    the layout, derived and written down
  |     lattice/
  |       LatticeKernel.inl   cells, neighbourhoods, the move rule, the fitness
  |       LatticeWorld        beacon and spawn placement, occupancy building
  |     LatticeSensors        CPU reference perception
  |     CpuLattice            CPU reference step and scoring
  |     GeneticAlgorithm      selection/crossover/mutation
  |     GenomeArchive         .vkng
  |     RunSnapshot           .vklr
  |     ExperimentSweep       staged runs of one varying setting
  |
  +-- vklat_simulation
  |     SimulationDriver      population, GA, and per-step dispatch recording
  |     SimulationModule      frame-loop adapter over the driver
  |     lattice/*.glsl        the shared kernel, compiled as GLSL
  |     neuro/*.glsl          the shared preset and the forward pass
  |     lattice_clear.comp    empties the claim grid
  |     lattice_step.comp     senses, runs the brain, bids for a cell
  |     lattice_resolve.comp  moves the winners and charges the metrics
  |     trail_capture.comp    records display-only per-agent breadcrumbs
  |
  +-- vklat_visualization
  |     LatticeRenderer       off-screen 3D view, camera, depth and OIT resolve
  |     view_*.vert|frag      procedural cubes and the transparency resolve
  |
  +-- vklat_ui
  |     SimulationUiModule    controls, statistics, brain, sweeps
  |
  +-- reusable infrastructure
        vkexp_core, vkexp_compute, vkexp_profiling, vkexp_imgui
```

The shared contracts are small:

- `SimulationState` carries controls, statistics, the published agent-buffer
  and occupancy views, the breadcrumb-history view, and the renderer's
  published viewport image;
- `AgentState` is an explicitly checked 224-byte std430-compatible structure,
  with the variable-length hidden block last so every earlier offset is fixed;
- `LatticeKernel.inl` is the world: cell indexing, both neighbourhoods, the move
  rule, the claim rule, the distance metric and the trial fitness. Both
  languages compile it, so the CPU reference and the shader cannot disagree
  about what the world is;
- `BrainKernel.inl` is the network preset: sensor block sizes, hidden capacity
  and output meanings, from which the input capacity, every block offset, the
  genome size and the packed GPU layout are derived. `Topology` is the C++ view
  of it and restates nothing;
- module order in `main.cpp` is the composition graph;
- `GpuStepParameters` travels in a storage buffer indexed by step rather than in
  push constants, so widening it is not bounded by the 128 bytes Vulkan
  guarantees;
- `FitnessWeights` carries the shaping coefficients to both the CPU reference
  and the shader, so a fitness experiment is a slider or a CLI flag and never a
  rebuild.

`SimulationDriver` holds the experiment; `SimulationModule` only maps frame
callbacks onto it. The batch runner drives the same driver from an
`ImmediateContext`, so a sweep and the window run identical code.

### Steps and seconds

`include/vkexp/simulation/Units.hpp` is where the step's two time bases are
reconciled. A step is the unit of reproducibility: replays, archives and parity
tests are all indexed by step count, and none of them depends on wall-clock
time. A second is the unit of anything that decays -- which, now that space is
discrete, means the neurons: a time constant is a duration, so how long a neuron
remembers is a fact about the brain rather than about the rate it happened to
run at.

Costs charged per event are the other half of the same rule and take no
`deltaTime` at all. A move costs what a move costs and a refusal likewise,
because both are counted rather than integrated. That is what the discrete world
bought: the quantity that used to need `1 - exp(-rate * dt)` to stay honest is
now an integer.

### Where the CPU path fits

The CPU code is not a mirror of the shader. It exists to build the network from
the shared preset, to score a finished generation, and to step and inspect a
handful of agents -- which the GPU cannot do usefully for 2048 of them at once.
What genuinely differs between the two, the parallel substrate, is verified by
tests that need many agents:

- `runGenomeAddressingProbe` gives six genomes distinctive move biases, one
  plane of the lattice each, and checks every agent follows its own; a wrong
  genome stride or base offset is invisible to a single-agent parity test.
- `runContentionProbe` puts two agents either side of one free cell so both bid
  for it, which is the one outcome a single-agent test structurally cannot see.
- `runFullLatticeProbe` fills a small lattice until refusals are the common case
  rather than the exception.

## CPU/GPU correctness

`vkexp_compute_smoke` builds the same agents, genomes and occupancy grid on the
CPU and on the GPU and advances both through the same three dispatches. On top
of the probes above, every combination of the two neighbourhoods and the four
neuron models runs a 120-step trajectory regression:

- **lockstep parity.** Each step feeds the CPU reference state to the GPU and
  compares one step of both. This is not a convenience: the move rule is a
  threshold on a float, so a one-ulp difference in a drive sitting on the dead
  zone flips a discrete move, and a free-running pair of trajectories diverges
  by construction rather than by error. Lockstep measures agreement; a
  free-running comparison would measure chaos.
- **accumulated drift budget.** Lockstep cannot see a systematic bias smaller
  than the per-step tolerance, because resetting to the CPU state each step
  stops it accumulating. Summing the *signed* per-step differences restores
  that: rounding noise cancels to about 4e-4 over the run, and the budget sits
  above that at 1e-3.
- **integers compare exactly.** Cells, headings, intents and the beacon are
  compared with no tolerance at all. There is no such thing as a cell that is
  nearly right, and a tolerance on one would hide exactly the bug this test
  exists for.
- **coverage assertions.** A run in which nobody moved, or in which nobody was
  ever refused a cell, fails rather than passing vacuously.
- **deep plan.** A second parity case runs `12,8,8` over a taller lattice, so a
  shader that walked the layer plan differently drifts where nothing else would
  catch it.
- **reconfiguration.** `vklat_reconfiguration_smoke` walks the lattice extents
  from 128^3 down to 1^3, the group sizes in both directions, four brain plans
  and the full cross product of boxes, group sizes and neighbourhoods, running a
  generation after each change. It asserts that the step resources were built
  once, that no buffer handle moved, that the chosen lattice fits the fixed
  allocation, and that every agent is still inside its own box. A stale
  descriptor here faults the device rather than returning a wrong number, so the
  test carries a timeout as part of its assertion.
- **replay determinism.** `vklat_replay_smoke` runs two replayed generations and
  compares them byte for byte.

Pure CPU tests cover cell addressing and its inverse, both neighbourhoods and
the walkability rule, the move rule including the threshold and the dominant
axis, spawn placement (in bounds, never doubled, never on the beacon), the
sensor vector block by block, contention, the trial fitness, logical-world
partition mapping, weight layout, neural evaluation, the four neuron models,
elite preservation, fitness sharing, step parameter packing, resolved step
settings, run-snapshot and genome-archive round trips including corruption and
truncation rejection, and reusable compute validation. Several guard the
contracts the architecture rests on: the shared lattice kernel is pinned on the
C++ side so a change to `LatticeKernel.inl` cannot slip through on a machine
without a GPU, and the brain preset is checked by derivation rather than by
snapshot -- the sensor blocks must tile the input vector without gaps or
overlaps, so adding a channel stays a one-line edit instead of a test rewrite.

## Build and run

Requirements: CMake 3.24+, Ninja, a C++20 compiler, Vulkan 1.3 development
files and driver, GLFW 3.3+, `glslangValidator`, and X11/Wayland for the GUI.
Dear ImGui v1.91.8 is fetched by CMake.

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
./build/debug/vulkan_lattice_agents
```

Disable validation if the validation layer is unavailable:

```bash
./build/debug/vulkan_lattice_agents --no-validation
```

### Batch runs

`vklat_headless` evolves without a window, which is what makes overnight runs,
parameter sweeps and ablation comparisons possible:

```bash
./build/release/vklat_headless --generations 200 \
    --csv runs/beacon.csv --save-champion runs/beacon-champion.vkng

# A different box, and the movement ablation against the same seed.
./build/release/vklat_headless --lattice 48x48x24 --neighbourhood faces \
    --generations 200 --seed 7 --csv runs/faces.csv

# Crowded: a small box with a large group is where refusals dominate.
./build/release/vklat_headless --lattice 8x8x8 --agents-per-world 24 \
    --generations 50 --csv runs/crowded.csv

# Resume a saved population.
./build/release/vklat_headless --generations 50 \
    --load-population runs/beacon-population.vkng

# Save and resume a whole experiment, not just its weights. The snapshot carries
# the lattice and every setting, so the resume restates none of them.
./build/release/vklat_headless --generations 40 --save-run runs/beacon.vklr
./build/release/vklat_headless --generations 40 --load-run runs/beacon.vklr
```

Fitness shaping coefficients are flags too (`--tracking-reward`,
`--objective-bonus`, `--motor-cost`, `--refusal-penalty`, `--signal-cost`,
`--fitness-sharing`), so sweeping them needs no rebuild:

```bash
for penalty in 0.0 0.01 0.05; do
    ./build/release/vklat_headless --generations 50 --seed 5 \
        --refusal-penalty "$penalty" --csv "runs/refusal-$penalty.csv"
done
```

`--help` lists every option. The banner prints the lattice that is *running*
rather than the one that was requested, so a box the budget clamped says so on
the first line.

Genome archives are versioned little-endian files that record the generation,
the beacon seed, the GA seed, the fitness and the brain shape, and refuse to
load into a build that lays that brain out differently.

A run snapshot (`.vklr`) is the heavier sibling: the population, which cell
every agent stands in, the generation and step it was on, and every setting,
which is what lets a resume start mid-generation with no flags. It is equally
strict, rejecting a file written for a different brain topology, agent layout or
settings list, and a population or trial count the running process cannot hold,
since both are buffer dimensions fixed at startup. The occupancy grid is
deliberately excluded: it is derived from where the agents stand, so rebuilding
it on load costs less than storing it and cannot disagree with the agents. The
interactive build has the same thing under **Snapshot** in the control panel.

The debug suite contains pure unit tests, CLI smoke tests, four short real
headless evolution runs covering both neighbourhoods, a crowded lattice and the
archive and snapshot round trips, and the Vulkan parity, replay and
reconfiguration tests. The Vulkan-dependent ones return CTest's skip code when
no compute device exists.

## What is next

The view makes the next step measurement rather than more rendering: compare
the four neuron models and both neighbourhoods from the same seeds, reading
arrival rather than best fitness and using replay plus the slice to distinguish
a direct route from a queue or a clump.

After that, the lattice makes a set of extensions cheap that the arena made
expensive: static obstacles are a second occupancy value, a second agent kind is
a third, and a cell that remembers what was broadcast into it is a field with no
diffusion constant to tune.

## Extension points

The next feature should enter through a focused contract:

- a new cell state is a value in the occupancy grid and a channel in the network
  preset; offsets, genome size and both implementations follow;
- a new sensor channel is a line in the preset, and `describeBrain` names it
  without being told;
- a new cost or reward is a field in `FitnessWeights`, a slider and a flag, and
  it reaches the CPU reference and the shader through the same struct;
- a rule about movement belongs in `LatticeKernel.inl`, where both languages
  compile it, and nowhere else;
- the neuron model is one shared integrator, so another unit replaces that
  function rather than the loop around it;
- colony scoring replaces fitness aggregation without touching the step;
- a different topology can become another evaluator/shader pair;
- GPU-side evolution can later replace the synchronous generation boundary.

## Project documentation

- [PROGRESS.md](PROGRESS.md) records what was implemented at each completed
  stage, the decisions behind it, and the verification status.
- [PLAN.md](PLAN.md) is the forward-looking roadmap and list of unfinished
  milestones.
