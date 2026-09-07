# Neuroevolution Lab Plan

## Design rules

1. Domain algorithms remain usable without a window or Vulkan device.
2. GPU layouts have explicit size/offset assertions and a CPU reference.
3. Modules exchange published views and state, not ownership of each other's
   implementation details.
3a. Simulation logic lives in drivers, not in modules: anything needed for an
   experiment must be reachable without a window.
3b. A scenario is one contract, not a set of switches: adding one means adding a
   source file, a shader pair and a registry line.
3c. Anything the CPU and the GPU must agree on is either shared source or a
   runtime parameter -- never a constant written twice.
3d. The CPU path exists to inspect one agent and to build the network from the
   same declaration, not to mirror the GPU. Parallel behaviour is verified by
   tests that need many agents, not by a second implementation.
3e. Steps are the unit of reproducibility; seconds and metres are the units of
   physics. Anything the step accumulates over time is scaled by `deltaTime`,
   and any fraction removed per step is written as `1 - exp(-rate * dt)`.
4. Add one evolutionary pressure at a time and keep deterministic replay tests.
5. Prefer measurable behavioral milestones over adding simulation features in
   parallel.

## Runtime flow

```text
ImGui controls
     |
SimulationState
     |
CPU generation boundary: fitness -> GA -> genomes/initial agents
     |
GPU per-step: spatial grid -> RGB/touch receptors -> brain -> collisions/physics
     |
storage barrier
     |
AgentRenderer -> off-screen image -> ImGui
```

Generation transitions are intentionally synchronous in the first version.
The GPU performs the expensive per-agent/per-step work; the CPU reads results
and evolves 512 small genomes only once per generation. This is inspectable and
easy to validate before asynchronous readback or GPU-side selection is added.

## Milestones

### 1. Phototaxis foundation — complete

- [x] fixed flattened neural topology and CPU evaluator;
- [x] std430 agent/parameter contracts;
- [x] CPU reference receptors, physics, and fitness;
- [x] GPU sensor/network/physics compute shader;
- [x] population with multiple trials per genome;
- [x] elitism, tournament selection, crossover, and mutation;
- [x] independent SSBO visualization and ImGui controls;
- [x] headless CPU/GPU one-step parity test;
- [x] unit tests and validation-layer launch smoke test.

Exit criterion: best and median fitness can be observed across generations,
and CPU/GPU parity fails loudly after a contract-breaking shader change.

### 2. Diagnostics and replay

- [x] champion-only replay with fixed seeds;
- [x] best/median/mean fitness and arrival-history plots;
- [ ] generation timing;
- [ ] inspect one agent's receptor values, activations, and motor outputs;
- [ ] render photoreceptor rays;
- [x] save/load versioned genome files;
- [x] deterministic multi-step CPU/GPU regression cases;
- [x] accumulated CPU/GPU drift budget that catches sub-tolerance bias.

### 3. Geometry and navigation

- [ ] world interface for circles, segments, and material properties;
- [ ] extend the spatial grid with wall segments and ray queries;
- [ ] wall distance/type receptor channels;
- [x] per-trial GPU spatial grid for agent queries;
- [x] circle-circle collision and tactile interaction tests;
- [x] configurable fitness penalty for world-boundary contacts;
- [x] static axis-aligned obstacles reported through the tactile channel;
- [x] a divided arena whose two ends optionally trade places by generation, so a
      fixed heading cannot stand in for reading the target;
- [x] the dead end keyed to the trial or to the generation, as a runtime option,
      so undiluted selection and forced generalisation can be compared;
- [x] scenario-owned spawn placement;
- [x] light occlusion by static obstacles, covered by parity;
- [ ] wall-ray and occlusion parity tests for a general segment world;
- [ ] procedural maze trials with train/evaluation seed separation.

### 4. Memory and task switching

- [x] stationary and mid-generation alternating beacon scenarios;
- [x] rotating beacon scenario with adjustable angular speed;
- [x] deterministic random beacon paths with configurable teleport chance;
- [x] discrete small, medium, and large training arenas;
- [x] recurrent/internal state with explicit reset semantics;
- [x] energy pickup and nest delivery;
- [x] repeated delivery cycles scored by count within one trial;
- [x] outbound/return behavior fitness;
- [x] runtime ablation switches reachable from the batch runner;
- [ ] ablation mode comparing reactive and recurrent brains.

### 5. Emergent communication

- [x] spatially accelerated additive RGB perception of nearby agents;
- [x] signal energy cost;
- [x] decaying ground trail field with antenna sensing (stigmergy);
- [x] wall occlusion;
- [ ] family/colony fitness and related genome batches;
- [x] hue ablation: both ends emit the average colour, so a solution that reads
      the colour and one that only alternates can be told apart;
- [ ] signal-off ablation to prove communication affects fitness;
- [ ] multiple colonies and optional interception of foreign signals.

### 6. Scale and extensibility

- [ ] asynchronous double-buffered generation readback;
- [ ] optional GPU selection/mutation backend;
- [x] scenario-owned active dense topology descriptors and recurrent outputs;
- [x] one network preset driving the CPU evaluator, the shader and the tests;
- [x] continuous-time hidden neurons with evolved time constants;
- [x] gated neurons whose time constant is recomputed from the inputs, selectable
      at runtime against the other two models;
- [ ] pluggable multi-layer/recurrent evaluator implementations;
- [x] scenario registry and data-driven experiment configuration;
- [x] batch/headless evolution executable;
- [ ] shader hot reload and capture/replay tooling.

## Runners

`vulkan_neuroevolution_agents` drives `SimulationDriver` from the frame loop.
`vkneuro_headless` drives the same driver from an `ImmediateContext`, so batch
sweeps and ablations produce results comparable with what the window shows:

```sh
vkneuro_headless --scenario rotating --generations 200 --csv runs/rotating.csv \
                 --save-champion runs/rotating-champion.vkng
vkneuro_headless --scenario forage --generations 200 --no-agent-light --quiet

# Fitness shaping is a parameter, so a sweep needs no rebuild.
for reward in 0.0 0.25 0.75; do
  vkneuro_headless --scenario rotating --generations 50 --seed 5 \
                   --tracking-reward "$reward" --csv "runs/tracking-$reward.csv"
done
```

## Changing the sensor suite or the brain

`include/vkexp/neuro/BrainKernel.inl` is the network preset. It declares how many
receptors, tactile sectors, self, task and recurrent inputs there are, how wide
the hidden layer is, and what the outputs mean; every offset, the input capacity,
the genome size and the packed GPU layout are derived from those numbers.

Both languages compile it, so raising `BrainSelfInputCount` from 4 to 5 moves the
CPU evaluator, the sensor sampler, the compute shader and the tests together, and
nothing else needs editing. The block offsets are asserted to tile the input
vector without gaps, so a sensor block that no longer fits fails loudly.

## Neuron model

Hidden neurons are continuous-time: each holds a state and integrates toward its
activation, `y += (dt/tau) * (-y + y_in)`. Where `tau` comes from is a runtime
setting with three values -- pinned to the step, one gene per neuron, or
recomputed each step from the inputs -- and that is the *only* difference
between them. The integrator and the mapping into the tau range live in
`BrainKernel.inl` beside the rest of the preset, so the CPU evaluator and the
shader cannot integrate a neuron differently, and `dt` is explicit so a memory is
a number of seconds rather than a number of steps.

Two identities keep the models comparable instead of merely adjacent, and both
are asserted rather than inferred. `tau = dt` reduces the update to
`y = activation`, which is the memoryless network. And a gate fed a constant
reproduces the fixed-time-constant neuron exactly, because the gate and the gene
enter the same mapping.

The genome carries every model's genes at once, so switching is a parameter
change and not a reinterpretation of the population. Adding a fourth model means
adding a value and a branch in one place on each side; per-block fixed time
constants -- fast for tactile, slow for task, no evolution -- is the obvious next
one, and is a control rather than an extension.

The state lives on the agent record, next to everything else a step carries, so
the CPU path and the GPU path store it the same way and multi-step parity covers
it without a separate harness. It is zero at the start of a generation, which is
the whole of the reset semantics.

## Trail field

A decaying RGB deposit on the ground, one field per logical world. Agents mark
where they walk and beacons mark where they pass; three antennae read the cell
under each tip. It is a storage buffer, not an image: deposits are integer
`atomicAdd`, so the order agents are scheduled in cannot change the field.

Constants and addressing live in `include/vkexp/simulation/TrailKernel.inl`,
compiled by both languages. Decay is a half-life in seconds, applied as
`exp(-rate * dt)`, so the field's lifetime does not depend on the step rate.

## Units

`include/vkexp/simulation/Units.hpp` declares the scale: one world unit is one
metre, the fixed rate is 60 Hz. The default arena is 3.68 m across, the body is
4.4 cm, top speed is 0.55 m/s and a 900-step trial is 15 seconds -- roughly an
e-puck on a large table.

Steps and seconds do different jobs and both stay. A run is replayed by step
count, archives record steps, and parity compares step for step; every physical
quantity is expressed per second, so `deltaTime` can change without changing
what a trial means. `testFixedStepIndependence` holds that invariant by driving
an agent into a wall for two simulated seconds at rates from 30 to 480 Hz.

## Adding a scenario

1. `src/worlds/scenarios/<Name>Scenario.cpp` fills in a `ScenarioDefinition`:
   brain shape, beacons, fitness, objective count, the optional before/after step
   hooks, the tunables the UI should offer, and the GPU parameter packer.
2. `shaders/worlds/<name>.glsl` unpacks the same parameter block for geometry,
   and `shaders/worlds/steps/<name>.glsl` mirrors the step hooks.
3. Add the enum value and one line to the registry in `src/worlds/WorldScenario.cpp`.

Shared math belongs in `include/vkexp/worlds/ScenarioKernel.inl`, which both
languages compile. Nothing else needs editing: the simulation, the scoring, the
batch runner's `--scenario` list and the UI controls all read the definition.

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
puts experiments where a window is not required to reach them; the stage
arithmetic is plain data and free functions, so the boundary is pinned by a unit
test rather than by watching curves.

## Replay

Watching is a mode of the same driver, not a second one: a replayed generation
is simulated, scored and reported through the ordinary path, and only selection
is skipped. That keeps a replay honest -- the fitness under the champion is
computed the way training computed it -- and it keeps the run repeatable,
because the generation counter that seeds beacon motion does not advance. The
smoke test asserts the repetition byte for byte, since a leak of evolution into
replay would show up as a different ending rather than as a worse number.

## Persistence

Two formats, deliberately separate. A genome archive (`.vkng`) carries weights
between runs and is the thing to keep. A world snapshot (`.vknw`) carries a whole
experiment between sessions -- population, agent positions, generation, step and
every physics setting -- and is the thing to reopen. The snapshot is written
field by field through one visitor list walked in both directions, so save and
load order cannot diverge, and a size assertion on `SimulationStep` makes a newly
added tunable a build failure rather than a silently dropped field.

## Immediate next step

Measure. The worlds and all three neuron models exist and are selected by one
parameter, so the questions are now experiments rather than opinions: run
`--scenario doors` and `--scenario shuttle` across `--neuron-model reactive`,
`time` and `gated` from the same seed, and run the sharing sweep on the same
world. Objective completion is the number to read, not best fitness -- a gate
that never closes and a gene that never varies both look like adequate fitness
and differ in whether the task is finished.

Then receptor visualization -- seeing what a champion perceives is what makes a
later failure attributable to perception, control, fitness or evolution instead
of only showing that population fitness stopped improving. The two-door world
makes that worth building: there is now a specific thing to look for in a
champion, which there was not on an orbiting beacon.

Held deliberately, and now with a run behind it: fitness shapes on the best
straight-line approach, which in a world with a wall between the two ends makes
the blindest spot the highest-scoring one. Two gaps is learned and the original
Two doors was not, and the sweep that followed says the binding term is how much
of the far side one opening lights -- opening width moves it several times as
much as beacon distance does. Both worlds are now held above a measured
visibility floor by their unit tests rather than by their constants.

The general fix is still a shaping term that measures progress along a route
rather than in a straight line. It should be written when a world needs geometry
that cannot be arranged around the plateau; two worlds have now been arranged
around it instead, which is evidence that the cheaper move is not yet exhausted.

Held deliberately: the obstacle interface is axis-aligned boxes, not the general
segment-and-material world interface milestone 3 describes. Boxes are what a
wall with gaps needs, they reuse the existing contact response exactly, and the
general interface should be written when a scenario needs something boxes cannot
express rather than in advance of one.
