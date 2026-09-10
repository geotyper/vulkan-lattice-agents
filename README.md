# Vulkan Neuroevolution Agents

An experimental C++20/Vulkan playground for evolving thousands of small agent
brains on the GPU. The first scenario is deliberately simple: agents use seven
directional photoreceptors to find stationary or periodically changing beacon
positions, while a genetic algorithm evolves the weights of a fixed dense
neural network.

The repository is a working vertical slice and a base for later worlds with
walls, memory, inter-agent light perception, colonies, and richer fitness
functions. Simulation, evolution, visualization, UI, and Vulkan infrastructure
are separate targets rather than one application-specific module.

## Current experiment

The world is metric: one unit is one metre, the step is fixed at 60 Hz, and the
default arena is 3.68 m across with a 4.4 cm body moving at up to 0.55 m/s, so a
900-step trial is 15 seconds. Steps remain the unit of reproducibility -- runs
replay by step count -- while every physical quantity is expressed per second.

- 512 genomes, each evaluated in four trials (2048 GPU agents);
- population partitioned into configurable logical groups (12 agents per world
  by default, with an all-agents mode);
- 7 forward light receptors with RGB and luminance channels;
- 8 full-body tactile sectors distinguishing walls from agents;
- 3 ground antennae reading the RGB of a decaying trail field;
- `61 inputs -> 20 tanh neurons -> 8 outputs`;
- every hidden neuron holds its own state and a time constant that is either
  evolved or recomputed from the inputs each step, so a memory is measured in
  seconds and can be held until something says to let go;
- outputs control left/right motors, RGB emission, emission intensity, and two
  recurrent memory cells;
- inertial movement with linear/angular drag and hard linear/angular speed
  limits;
- selectable circular or square world in small (x1), medium (x1.5), and large
  (x3) sizes;
- stationary per-trial beacons, alternating diagonal pairs, orbiting beacons,
  deterministic random movement with occasional teleportation, or an
  orbiting-resource/static-home foraging cycle;
- circle-circle agent collisions with impulse response and tactile pressure;
- configurable fitness penalty per second of contact with the world boundary;
- additive, softly tone-mapped RGB perception of nearby agent signals;
- runtime ablation switches for agent collisions and agent-light perception;
- average fitness across trials rewards progress and completion in every beacon
  phase and penalizes motor and signal energy use;
- elitism, tournament selection, uniform crossover, Gaussian mutation;
- adjustable simulation speed, generation length, agents per world, physics,
  and sensor FOV;
- generation length selectable from 120 to 15000 simulation steps;
- separate simulation, genetic-algorithm, world, and profiler windows;
- fitness/arrival history graphs for completed generations;
- an off-screen Vulkan renderer with persistent circular bodies, heading
  markers, and independently rendered coloured halos.

The population is divided into configurable groups, and each group keeps four
independent trial worlds. With the default group size of 12, 512 genomes occupy
43 groups and 172 logical worlds. A per-world uniform grid limits collision and
light queries to nearby agents in the same world. The viewport renders only the
selected world and can switch between groups and trials. Selecting all 512
agents produces one group and restores the original interaction density while
still avoiding the visual overlap of its four trials.

Individual fitness does not yet reward helping unrelated genomes, so the
communication channel is functional but meaningful signalling is not expected
until colony fitness is introduced.

The four trials are not four independently trained populations. Every genome
controls four agents with the same weights but different initial conditions.
Their scores are averaged before selection, which discourages solutions that
only work from one spawn position or heading.

## World and beacon scenarios

| Control | Variants | Behaviour |
| --- | --- | --- |
| World shape | Circle, Square | Changes the arena boundary and wall-contact response. |
| World size | Small x1, Medium x1.5, Large x3 | Scales the arena, spawn distribution, and beacon placement. Agent size, speed, and sensor range remain physical constants, so larger worlds are harder. |
| Arrival radius multiplier | x0.1 to x5.0 | Scales the fixed 0.060 beacon radius used for pickup/completion. At x1 the agent centre must enter the visible beacon circle. |
| Beacon scenario | Stationary | One fixed coloured target per logical trial. |
| Beacon scenario | Alternating diagonals | Two beacons occupy one diagonal during the first half of a generation, then two beacons occupy the opposite diagonal. |
| Beacon scenario | Rotating | One beacon per trial continuously orbits the world centre; speed and direction are adjustable from -90 to +90 degrees per second. |
| Beacon scenario | Random movement | Each trial follows a smooth bounded wandering path with adjustable speed and a configurable teleport chance checked every three seconds. |
| Beacon scenario | Forage + home | Agents collect an orange orbiting resource, then carry its decaying value to a blue home that relocates every eight seconds before seeking the resource again. |
| Beacon scenario | Scent relay | The same collect-and-deliver cycle, but home emits no light and lays no trail: it can only be found by dead reckoning or by a path the agents themselves marked. |
| Beacon scenario | Two doors | The same cycle across a wall with two gaps, one of which is a dead end. Which one swaps every trial -- or every generation, as an option -- and from the home side they are identical. Retuned after it went unsolved for 450 generations; see Two gaps for the measurement. |
| Beacon scenario | Shuttle | Fetch and carry back, over and over until the trial ends, around a short wall that closes the straight line between the two beacons. |
| Beacon scenario | Puck push | A round puck shared by every agent in a logical world, starting on one side of the centre line. Push it toward the lit disc in the middle; the objective is a ladder of quarters along that journey. |
| Beacon scenario | Gate and plate | A wall with one opening, shut by a gate that runs only while somebody stands on the plate in front of it. Press, cross, bring it back -- the gate has to be open both ways. How long it keeps running after the plate is let go is a slider, and at zero it cannot be done alone. |
| Beacon scenario | Two gaps | The same repeated cycle across a wall with two ways through, neither a dead end, with an option to make the two ends trade places every other generation. |

Changing the world size, shape, or beacon scenario resets the evolution because
fitness values gathered in different environments are not directly comparable.
In the alternating scenario, reaching either active beacon completes the
current phase. Progress and completion are scored independently for both halves
of the generation. The moving scenarios additionally reward sustained visible
closeness, so following the target scores better than a chance encounter.
Rotating and random scenarios expose an orbit/roaming-radius control; the
default random teleport probability is 25%.

The scent relay is the foraging cycle with the return trip made invisible. The
resource orbits and is lit as before; home sits opposite it and is black, so it
is absent from every light sensor and lays nothing on the ground. What the
agents do leave behind is their own trail, coloured by their own signal output,
which the antennae can read. Nothing in the fitness function mentions colour, so
whether a shared meaning emerges is the experiment rather than the design.

## Shuttle

Two fixed beacons facing each other with a short wall between them, and a trial
long enough to run the round trip more than once. Reach the resource, carry it
home, go again, until the time runs out.

```
        resource
   +-------------------+
   |                   |
   |     #########     |   the wall, at y = 0
   |                   |
   |        home       |
   +-------------------+
```

The wall does not divide the arena -- it is shorter than the beacons are far
apart -- so there is no door to find and nothing to remember about which way is
open. What it takes away is the straight line: the shortest route is around one
end, and with light occluded the far beacon disappears behind the wall on the
way. That is the whole of the world, deliberately: it is the cycle scenario with
the temptation to go straight removed, and nothing else added.

Unlike the other cycle scenarios this counts **trips** rather than asking
whether there was one, because shuttling until the time runs out is the task and
a run that manages it twice has to be distinguishable from one that manages it
once. Completion is reported against two round trips, which is what the default
15 s trial has room for at the speed limit -- the geometry is sized for that and
the unit test asserts it, so the wall and the trial length cannot drift apart
unnoticed. Fitness is not capped, so a faster agent still scores for every extra
trip. Longer trials make more of them fit:

```sh
vkneuro_headless --scenario shuttle --generations 200 --seed 5 --csv runs/shuttle.csv
vkneuro_headless --scenario shuttle --generations 200 --steps 2700   # 45 s trials
```

## Two doors

A wall runs across the arena with two gaps in it. One leads through to the
resource; the other opens into a closed pocket. Which gap is the dead end swaps
every trial, and a genome is evaluated on all four trials, so it cannot win by
always turning the same way. The trial index is not among the network's inputs,
so it cannot be read off either -- the only way to know is to have gone.

```
                resource
   +-------------------------------+
   |            #####              |   pocket cap
   |            #   #              |   pocket sides
   |#########  ###########  #######|   the wall, at y = 0
   |         A            B        |
   |                               |
   |             home              |
   +-------------------------------+
```

This is the world the neuron time constants were built for. On an orbiting
beacon there is nothing a memory is *for*, so a fitness curve was the only
evidence and it said nothing about what had been learned. Here there is
something specific to hold across seconds -- which gap was a dead end this
trial -- and something specific to watch for: an agent that blunders into the
pocket once and then goes straight to the other gap looks different from one
that does not.

Both beacons are lit, unlike the scent relay. The question this world asks is
which gap leads through, and an invisible home would stack a second, already
answered question on top of it and make the answer to neither legible.

**Why colour has an incentive here.** An agent that has been into the pocket is
the only one that knows, and it leaves a coloured trail as it comes back out.
Marking the dead end pays its own next cycle back within the same trial, not
only its neighbours' -- so unlike the scent relay, the signalling channel has an
individual incentive that does not depend on turning group fitness sharing on.
Nothing in the fitness function mentions colour; whether a mark acquires a
meaning is still the experiment rather than the design.

**Light is occluded.** A wall that stops a body but not its light is a wall an
agent can see through, and the gradient then pulls it straight at the one place
it cannot go. Occlusion is a slab test on the segment from the agent to the
light, done per beacon and per neighbour rather than per receptor: a light is a
point, so either the line to it is clear or it is not there at all.

That is roughly twelve slab tests per agent per step, beside a brain that
already does more than a thousand multiplies. A lightmap rebuilt each tick
would answer the same question by sampling and would be the right tool for
hundreds of sources over complex geometry; with two beacons and six boxes it is
both slower and coarser, and the receptors are directional -- a map gives the
light at a point and loses the direction the sharp receptor tuning needs, so it
would have to be marched along each ray anyway.

Note what this leaves in place: perception loses the resource behind the wall,
but fitness does not. Progress is still banked against the geometric distance to
the target, so the task stays learnable by shaping while being unsolvable by
looking.

**Geometry.** Six axis-aligned boxes. Thickness is set by the agent and not by
the arena -- one body diameter, 4.4 cm -- because a wall is a wall whatever room
it stands in; scaling it with the world radius made a 17 cm slab across a 3.7 m
arena, nearly four body diameters of masonry that read as architecture rather
than as a divider. Everything else is derived from the arena radius by
`ScenarioKernel.inl` rather than stored, so they cost no parameter slot and the
CPU and the shader cannot disagree about where a wall is. Contact reports
through the same tactile channel the arena boundary uses: to the network a
barrier is a barrier, and no new input had to be found room for. The unit test
sweeps the wall line for a seam between segments, sweeps the pocket boundary for
a way out, and asserts that no barrier is thinner than the distance an agent
covers in one step -- a wall a fast agent steps over between two contact tests
is decoration -- stated against the top of the speed slider and against the box
as the contact test sees it, inflated by the body radius, since a wall that
holds only at default speed is a wall that fails when the experiment is turned
up. Occlusion is asserted as the world rather than as the slab test:
from home the resource is hidden, from the open doorway it is not, and inside
the dead end it is hidden again.

The scenario also places its own spawn: the driver's default spiral covers the
whole arena, which here would start half the population already past the wall
with nothing left to solve.

### Which clock the dead end runs on

Off by default, the dead end swaps **every trial**: with four trials per genome
each one meets both layouts twice, so a policy that always turns the same way
caps at half the trials and selection asks for something that handles both from
the start. The cost is dilution -- early on, a genome that suits one layout is
averaged back down by the other.

**Dead end changes by generation** (`--doors-by-generation`) keys it to the
generation instead: a whole population trains on one door and its successors on
the other. Selection inside a generation is then undiluted, which should be
faster. The risk it takes on is oscillation -- generation N selecting for "go
right" and N+1 punishing exactly that, with the population thrashing between the
two and never building the memory that would settle it. That shows up as a
one-generation sawtooth rather than as a lower average, so read neighbouring
generations, not the mean.

Both settings put the ceiling for a door-blind policy at 50%, and they get there
differently: per trial it is half the trials every generation, per generation it
is every trial in half the generations. Above 50% is where memory has to be
doing work, because from the home side the two openings are identical -- the
only way to know is to enter one, meet the pocket, and come back out to the
other.

```sh
vkneuro_headless --scenario doors --generations 400 --seed 5 --csv runs/doors.csv
# the same run with the layout keyed to the generation
vkneuro_headless --scenario doors --generations 400 --seed 5 --doors-by-generation \
                 --csv runs/doors-by-generation.csv
# and with a memoryless brain, to see what the time constants bought
vkneuro_headless --scenario doors --generations 400 --seed 5 --neuron-model reactive \
                 --csv runs/doors-reactive.csv
```

## Two gaps

A wall right across the arena with two ways through, neither of them a dead end,
and an option that makes the resource and home trade places from one generation
to the next.

```text
        resource (or home)
   +-------------------------+
   |####       ####      ####|   two gaps, no dead end
   |     home (or resource)  |
   +-------------------------+
```

**Why the ends swap.** With a fixed layout a genome can win without ever reading
the light: carry north, deliver south. Nothing in the fitness function
distinguishes that from having understood the task, and the difference only
shows up when the world changes. Swapping the ends makes a heading worth
nothing, and since the two beacons differ only in colour, the colour becomes the
only thing that says which end is which. It is off by default: it is the harder
task, and a run that has not solved the fixed layout first says nothing about
the swapped one.

The swap follows the generation number rather than the trial, deliberately. Per
trial, one genome would meet both layouts inside a generation and be scored on
the average, which rewards a compromise; per generation, a whole population
meets one layout and its successor meets the other, so what carries over is
whatever generalised.

**Why the geometry is what it is.** Fitness shapes on the *best straight-line
approach* to the current target, and that makes any wall between the two ends a
trap: the spot pressed against the middle of the wall is simultaneously the best
score on offer and the one place with no line of sight to the target at all.
Reaching a gap costs distance, so it earns nothing until the agent is well past
it. Every world here has that plateau; what decides whether it is escapable is
how much of the far side can see the target at all:

| | plateau | target visible from | outcome |
| --- | --- | --- | --- |
| Two doors, as first built | 0.15 m | 5.7% of the far side | not solved in 450 generations |
| Shuttle | 0.12 m | 36% | solved quickly |
| Two gaps | 0.11 m | 19% | solved |
| Two doors, retuned | 0.10 m | 16% | see below |

Two doors is not hard because the arena is big. Light range is a fraction of the
arena radius rather than a fixed number of metres, so a bigger world does not
change any of these ratios at all.

What the sweep found, once Two gaps was learned and Two doors still was not, is
that the strongest lever is **how much of the far side one opening lights**, and
that the width of the opening moves it far more than the distance does. Bringing
the ends from 0.72 to 0.55 of the arena radius -- which was the first guess, and
does put them inside each other's range -- moves visibility from 5.7% to 7.1% on
its own and makes the plateau *worse*. Widening the door from 0.07 to 0.11 and
bringing the pair in from 0.40 to 0.30 takes it to 16% and cuts the plateau to
0.10 m. Two doors now carries all three changes.

Both worlds assert the result rather than the constants that produced it:
`visibleFractionOfFarSide` sweeps the far side in the unit tests and holds each
world above a floor set between the world that was not learned and the one that
was. Reverting any one of the three constants fails it.

```sh
# Learn the fixed layout first, then the swapped one from the same seed.
vkneuro_headless --scenario gaps --generations 200 --seed 5 --csv runs/gaps.csv
vkneuro_headless --scenario gaps --generations 200 --seed 5 --swap-ends \
                 --csv runs/gaps-swapped.csv
```

### What the swap does not prove

With the swap on, 250 generations reach about 75% of the round trips the trial
has room for. That is a real result -- an absolute heading is worth nothing
under the swap -- but it is *not* evidence that the agents read the beacon
colour, and it is worth being precise about why.

The task is to alternate between two ends. A policy that never looks at colour
solves it: **head for whichever beacon is further away.** Standing at home, the
resource is the far one; standing at the resource, home is the far one. Distance
is available without hue, because the nearer beacon is simply brighter. Swapping
the ends does nothing to this policy, since it is stated in terms of *here* and
*the other one* rather than north and south.

So the swap closes the direction shortcut and leaves the alternation shortcut
open. The control that tells the two apart is to remove the colour instead:

```sh
# Same world, same seed, hue carrying no information.
vkneuro_headless --scenario gaps --generations 250 --seed 5 --swap-ends \
                 --uniform-beacon-color --csv runs/gaps-no-hue.csv
```

Both ends then emit the *average* of the two colours -- averaged rather than one
copied onto the other, so the amount of light each end emits is unchanged and
the run answers one question instead of two. If the score holds, the solution
was alternation and colour was never being read. If it collapses, colour was
carrying the task.

Forcing colour to matter is a further step and not yet taken: it needs the two
ends to stop being distinguishable by "the one I am not at" -- a third beacon,
or a home that appears in one of two places after each pickup.

## Puck push

The first world where agents change something rather than only move through it,
and the first whose outcome belongs to a group rather than to an individual.

```text
        the side the puck starts on, by trial
   +-----------------------------+
   |            ( o )            |   the puck, where it is placed
   |            ( * )            |   the lit disc, the objective
   |            (   )            |
   +-----------------------------+
```

One puck per logical world, integrated by its own compute pass. Agents push it
by touching it; the puck is a body they cannot walk through, reported through
the same tactile channel a wall is, so no new sensor had to be found room for.

**The objective is a ladder on one journey.** It began as the two goals the
world was specified with -- a minimum, push the puck past the arena's middle
line, and a maximum, push it into a disc around the centre -- and the geometry
will not put those in that order. The disc straddles the line and the puck
arrives from outside, so it enters the disc *before* it reaches the line: after
0.64 m of a 1.10 m journey at the default sliders. The minimum was the harder of
the two and never fired first, so the ladder had one rung where it looked like
two. A world scored nothing at all until its puck was in, and then scored full
marks; a puck brought fifty-seven per cent of the way counted the same as a puck
nobody had touched, and the reported curve could only move in whole worlds.

So the journey is what is measured and the disc is where it ends. The rungs are
equal quarters of the distance from where the puck was placed to the disc's
edge. The top rung and "inside the disc" are the same statement, so the maximum
the world was specified with is intact; every rung below is strictly harder than
the one under it by construction; and the ladder follows the target-radius
slider without anything having to be retuned. Rungs are latched and taken as a
maximum, so the curve stays monotone -- a puck nudged in and back out still got
there.

The reported ratio therefore reads as the average fraction of the journey a
world's puck covered, not as the share of worlds that finished. It is not
comparable with the number this world reported before the ladder: the old one
counted deliveries, and this one counts distance. What a delivery is worth in
*fitness* was deliberately held where it was, so a run before the change and a
run after it are still comparable on the thing being selected for.

**Why the push is a pressure and not an impact.** Two models were tried. The
obvious one sums penetration depths and pushes the puck out of them; that cannot
work here, because the agent step resolves its own overlap first, so by the time
the puck is integrated there is no penetration left to read.

The second took the push from the *approach speed* along the contact normal,
which is what an impact is. It worked, and it taught the wrong thing. With the
friction floor low a single agent could run at the puck and knock it along, so
the world was solved by charging it; with the floor raised the agents did gather
around the puck -- and then stopped, because an agent already in contact has no
approach speed left. Standing on the puck and leaning, which is exactly the
behaviour the floor was meant to select for, registered as zero push. The world
punished the thing it was asking for.

The push is the agent's own motor drive projected on the contact normal instead:
`drive * dot(heading, normal)`, clamped at zero. Drive is what the brain asked
the wheels for, so an agent that has run out of room to accelerate still presses
at full strength -- a tugboat against a hull, not a hammer. Contact is a
geometric overlap test with a small skin, so leaning counts and passing by does
not, and alignment makes pushing straight worth more than pushing at an angle.

Three consequences follow. Pressure is dimensionless and per agent, so the
friction floor below is literally a count of agents rather than a speed in metres
per second. The sum is an acceleration rather than a velocity, so nothing in the
formulation bounds the puck any more -- enough agents would keep feeding a puck
they can no longer keep up with, and the world would be solved by launching it
once, so the pass clamps the puck to the agents' own speed limit and the smoke
test asserts the clamp. And a drag term, not the model, is what brings a released
puck to rest.


**Why the puck emits light.** The first version of this world did not learn at
all, and the reason is worth keeping: the photoreceptors see beacons and other
agents' signals and nothing else, so a puck that was neither was *invisible*. An
agent could only discover it by walking into it. The fitness paid for
approaching the puck and for moving it, and both rewards were real -- but a
population cannot climb a gradient it has no sense of. The reward existed and
the handle on it did not. The puck is a beacon now, at the position every agent
already mirrors, so reaching it is phototaxis, which is the one thing these
agents reliably evolve.

**Why the journey outweighs loitering.** The second reason, and the arithmetic
matters because the obvious version of the claim is wrong. Being near the puck
pays `trackingReward` per second, 3.75 over a fifteen-second trial for an agent
that simply parks on it. Pushing the puck all the way in and completing both
levels paid 8.85 -- more, so the endpoint was never the problem.

What was missing was the increment. The whole journey to the middle is 1.1 m, so
moving the puck a hand's width was worth 0.10 against that 3.75: under three per
cent. Evolution improves by increments, and there was none to find -- only the
completion, which nothing was going to stumble into. Progress is now a fraction
of the journey rather than a number of metres, weighted so the same push is
worth 29 per cent instead. The unit test asserts the increment, not the
endpoint, because the endpoint was never what failed.

**And the approach reward is tied to the puck, not to the light range.** At
light range it is a broad haze over most of the arena and loitering in the
general area collects most of what pushing would pay. Six puck radii pays for
being *at* it, which is where pushing starts.

**Why pushing is priced per agent.** The third reason, and the one the world
itself created. With the two fixes above a population does improve, slowly, and
it improves into the wrong shape: agents lean against whichever face of the puck
they arrive at, several of them on the side facing the middle, and hold it still.

That is not evolution failing to find the answer. It is the score paying for it.
Every term derived from the puck -- progress, level, the approach reward -- is
read off one object twelve agents share, so it is the *same number* for all
twelve. The agent that shoved the puck home and the agent standing in its way
were scored identically, and selection cannot separate behaviours it cannot see
apart. What it could see was that being near the puck pays and that moving costs
motor effort, and it evolved accordingly.

The fix is deliberately not "reward the agents pushing from the correct side".
That hands over the answer, and this world exists to ask the question. It is to
pay each agent for the work it actually did, which is a physical quantity rather
than an opinion: the same pressure `puck_step.comp` integrates, projected onto
the direction the puck still has to travel. An agent wedged between the
puck and the middle projects negative and earns nothing -- but nothing told it
that side was wrong, only that its pushing does not move the puck where the puck
has to go. Pushing at an angle pays less than pushing straight, so getting
further round the puck is a gradient and not a switch.

Zero, and not a penalty. Blocking should stop being paid for; it should not
become a thing to actively avoid, or an agent learns to keep clear of the puck
rather than to get behind it.

The approach reward is cut to a quarter of `trackingReward` at the same time.
The weight means "per second for being near the thing you are meant to track",
which is the right rate in a world where being near the beacon *is* the task;
here it is only how pushing starts, and at the full rate a trial spent leaning
on the puck out-earned a trial spent delivering it. A parked agent now collects
0.94 against the 12-plus-bonuses a delivery pays. The slider still scales it,
and setting it to zero still turns the search reward off without touching what
pushing pays -- which is the experiment worth running once the world moves.

So the score now has two parts that answer different questions: the joint part
says the puck arrived, and the per-agent part says who moved it. That split is
also what makes the sharing sweep below meaningful rather than circular.

**The first knob to reach for** is the puck's size. Bigger is easier twice over:
a wider contact arc for several agents to push at once, and a larger thing to
find. Both sliders -- `Puck radius` and `Target radius` -- take effect on reset.

**Whether one agent is enough is a slider.** `--puck-breakaway` (and `Breakaway
push`) is a friction floor on the puck, counted in agents leaning on it head-on:
how hard the *whole world* has to press before it moves at all. One agent at full
throttle, square to the contact normal, is exactly 1.0. Below one, a single
agent solves the world alone, a group is only a convenience, and the question
this world exists to ask -- can selection produce agents that push together -- is
one it never puts. Above one, no single agent can start it however hard it tries.

Not a mass, deliberately. Mass makes one agent slower, not powerless: the puck
still creeps, the score still rises, and the population still learns to solve it
alone. A floor is a threshold, which is what "two or more" means. It is
subtracted from the push rather than switching it on and off, so a pair that
barely clears it moves the puck slowly instead of the world flipping between
nothing and everything -- selection needs an increment here for the same reason
the journey is a fraction rather than a completion.

Pushes are summed as vectors before the floor is measured, so two agents on
opposite faces cancel and move nothing however hard they try, and two pushing at
an angle add up to less than two. A threshold of 2.0 therefore asks for more than
exactly two bodies: it asks for two pushing the same way.

The work reward follows the puck rather than the pushing, because with a floor in
the world a lone agent can lean on a stuck puck at full drive for a whole trial.
Paying for that would teach exactly the futile pushing the floor exists to rule
out, so the reward is scaled by whether the puck is actually moving.

**Where the puck starts is an option.** By default it is on the arena's axis with
the agents spawned on its side, so the first thing they do is reach it -- and
every measurement so far was taken that way. `--puck-scatter` (and `Scatter the
puck`) instead places it anywhere in a ring, a different place for every world
and a different place each generation, so finding it is part of the task and no
one layout can be memorised. The placement comes from the same hash the
relocating home already uses, seeded by the world and the generation, so a
replayed generation is the same generation.

**This is the world the sharing option was built for.** `--fitness-sharing`
blends a genome's score with its world's average, which is meant to make helping
a neighbour pay -- and until now every world scored an individual's own
journey, so there was little to share. Here the outcome is joint by
construction: one puck, one result, twelve agents. Whether sharing helps is the
measurement this world exists to make.

```sh
vkneuro_headless --scenario puck --generations 300 --seed 5 --csv runs/puck.csv
for share in 0.0 0.5 1.0; do
  vkneuro_headless --scenario puck --generations 300 --seed 5 \
                   --fitness-sharing "$share" --csv "runs/puck-share-$share.csv"
done
```

The puck is saved in world snapshots, unlike the trail field: the trail is
derived and recovers in a few half-lives, the puck's position is the state of
the experiment, and a resume that put it back at the start would read as a run
that had lost ground it had not lost.

## Gate and plate

```
              resource
   +-----------------------------+
   |              .              |
   |#########  ###[]#############|   the wall, the gate in its opening
   |                             |
   |     (plate)                 |
   |   o     o        o     o    |   everyone starts on this side
   +-----------------------------+
```

Three legs in a fixed order across two places. Press the plate, cross to the
resource, bring it back to the plate. Standing on the plate scores nothing.
Nothing about "press, then go" can be read off the current sensor values, so a
network that maps light to motors cannot do it -- "I have already opened it" has
to be held. That is the same claim the two-door world makes, except that here it
is held for seconds rather than latched once, and here somebody else can hold it
for you.

**Why it is a round trip and not a crossing.** Getting through was the first
version, and it is half a task: an agent that is through is done, the plate
behind it stops mattering to it, and the door being held is worth something
exactly once. Coming back makes the gate a thing that has to be open *twice*, so
whoever is holding it is worth something for as long as anybody is still out.

The plate is also home, which is what closes the cycle without a fourth
landmark: pressing it on the way back is the same act as pressing it on the way
out, and re-opens the gate for the next trip. Both reasons to head for the plate
-- "I have to open it" and "I am coming home" -- point at the same place, so the
world needs only its two beacons.

**This world wants about 1800 steps per generation**, twice the default. A leg is
2.1 m and a round trip about 840 steps at the speed limit, so the nominal two
trips do not fit in 900 -- the reported ratio would flatten near half with
nothing looking wrong. The scenario declares that number rather than leaving it
in a comment: the window says so beside the trial-length slider and offers a
button, and `vkneuro_headless` uses it when `--steps` is not given. The unit test
asserts both directions of it, because the geometry is what would quietly break
it.

**The latch is the difficulty, and it is one number.** `--gate-latch` (and the
`Gate latch (s)` slider) says how long the gate keeps running after the plate is
released.

- **Above zero** one agent presses and runs. Nothing has to be shared and no
  cooperation is needed; this is the end to start at, and the end that says
  whether the two-leg structure is learnable at all.
- **At zero** the gate shuts the instant the plate is let go. Only the far side
  scores, so somebody has to stay behind for nothing. That is the condition
  group fitness sharing exists for, reached by moving a slider rather than by
  adding a scenario.

**Why several agents in one arena is the point rather than a problem.** With a
dozen agents wandering, somebody stands on the plate by accident about an eighth
of the time, and those accidents are the world's bootstrap: the first crossings
happen because somebody happened to be standing in the right place. What
selection does with that is the question. At a positive latch it can learn to
press deliberately and go; at zero it has to keep somebody there, and the agent
that stays cannot be paid for it out of its own score.

**The reported number is round trips against the two a trial has room for.**
Uncapped in the score and capped in the report, the way every repeating world
here does it, so a quicker agent still gains from the extra trips. Lingering on
either end counts once: pressing the plate is positional and happens by standing
there, so only a carrying agent closes a trip. At a latch of zero the agent
holding the door completes none of its own, and that missing share is the cost of
the door being held.

**Where it is unlike the puck.** The gate is not a body agents move; it is a
fact about the room, computed fresh every step from where everybody is standing.
There is no gate buffer, no gate pass and no shared record to keep in step: the
spatial grid is already a per-world index of every agent, so each agent scans
the cells over the plate and reaches the same answer as its neighbours. They
cannot disagree because they are not communicating, they are recomputing. What
does have to be carried is the latch countdown, and each agent carries its own
copy in the slot this world does not use for a base beacon.

**What is on screen.** The plate is drawn as a disc at the radius the press test
actually reads, and lights up while the gate is running. It is a beacon as well,
because agents have to be able to find it -- but a beacon is drawn at the one
fixed visual radius every beacon uses, six centimetres against the plate's
twenty, so left at that the picture showed a dot where the rule tests a disc and
standing beside the dot looked like standing on the plate.

**What the assertions cover.** That the plate is not in the doorway, so the two
legs are two places. That a shut gate leaves the resource invisible from the
side the agents start on, and an open one shows it from 19 per cent of that side
-- the same figure as the two-gap wall that was learned, measured by the same
sweep. That the latch reloads on a press, runs down on release, stops at zero,
and at a latch of zero is open exactly during the step the plate is held. A gate
leaf that never parks and a latch that never runs down each fail a different one
of them.

## Group fitness sharing

Selection is individual by default: a genome is scored on what it did, so a
signal that only helps a neighbour is pure cost to its sender. That is a fitness
property, not a network one, and no architecture fixes it -- which is why
`--fitness-sharing` (and the matching slider) exists as a comparable option
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
Objective completion is untouched either way and is the cleanest comparison.

### Sweeping it from the window

Whether sharing helps is an empirical question, and one run cannot answer it.
The **Sharing sweep** panel runs the same experiment once per value -- up to six
stages -- and restarts evolution between them, so a stage measures its own
setting rather than that setting applied to whatever the previous one had
already evolved. Every stage keeps its own curves, and they are plotted on one
shared axis, because separately autoscaled plots would make a flat run and a
climbing one look alike.

Objective completion is plotted first and on a fixed 0..1 axis: it is the one
number sharing does not move arithmetically, so it is the honest comparison
between settings. The last stage is not restarted when it ends -- the run simply
carries on at that setting with every stage kept for reading.

The same comparison from the batch runner, which writes CSV instead of curves:

```sh
for share in 0.0 0.5 1.0; do
  vkneuro_headless --scenario scent --generations 60 --seed 5 \
                   --fitness-sharing "$share" --csv "runs/scent-share-$share.csv"
done
```

Worth running with a non-zero `--signal-cost`. While emitting is free, colour
stays a free drift even under shared fitness: what pays back has to cost
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
and not in steps -- the same rule the rest of the physics follows. A time
constant of half a second closes `1 - 1/e` of the gap to its input in half a
second at 30 Hz, at 60 Hz and at 240 Hz, and the unit test asserts exactly that.

The gene enters a bounded logarithmic range, from one step (16.7 ms) to four
seconds. Logarithmic because what matters about a memory is its order of
magnitude: a linear map would spend most of the gene range between two and four
seconds. A gene of zero lands on the geometric middle, about 260 ms.

This is where memory belongs. The two recurrent cells put it in the *output*
layer, which cost two of the eight output slots and squeezed everything a brain
might remember through a two-number bottleneck; time constants give all twenty
neurons a state and take no output slot at all. The recurrent cells stay --
they are an explicit, inspectable channel -- but they are no longer the only
thing holding the past.

### Three models, one integrator

Where the time constant comes from is a setting -- **Neuron model** in the
window, `--neuron-model` on the command line -- and it is the only thing that
changes between the three. The integrator, the genome and the state are the
same in every case, which is what makes switching an ablation rather than a
swap between networks, and what lets a population keep its meaning across a
switch mid-experiment.

| Model | Time constant | What it is |
| --- | --- | --- |
| `reactive` | pinned to `dt` | The update collapses to `y = activation`: no state at all, the network from before time constants existed, reached by the same arithmetic. |
| `time` (default) | one gene per neuron | Fixed for the neuron's life. It forgets at one rate whatever is happening to it. |
| `gated` | recomputed each step from the inputs | The neuron can hold a value and then let go of it when something tells it to. |

Gated is a strict generalisation: feed the gate a constant and it *is* the
fixed-time-constant neuron, which the unit test asserts directly for three
different constants. It costs one weight row and one bias per hidden neuron and
no extra state, because the state it needs is the one the neuron already carries.

**Watch the sign.** The gate asks for a time constant, not for an update
fraction, so driving it up makes the neuron hold and leaving it low makes it
follow. That is the opposite of a GRU update gate. It is this way round because
the gate and the gene go through the same mapping, which is what makes the two
models comparable at all.

```sh
for model in reactive time gated; do
  vkneuro_headless --scenario shuttle --generations 200 --seed 5 \
                   --neuron-model "$model" --csv "runs/shuttle-$model.csv"
done
```

### The structure, and where each model lives in it

One preset, `include/vkexp/neuro/BrainKernel.inl`, declares the whole network,
and both languages compile it. Every offset, the genome size and the packed GPU
layout are derived from the counts below, so raising one number moves the CPU
evaluator, the sensor sampler, the compute shader and the tests together.

```
inputs 61                            hidden 20            outputs 8
  7 receptors x 4 (RGB + luminance) = 28    each with        2 motors
  8 tactile sectors x 2 (wall, agent) = 16   its own state    3 signal colour
  3 antennae x 3 (trail RGB)          =  9   and its own      1 signal intensity
  speed, turn rate, energy, own signal =  4  time constant    2 recurrent cells
  cargo level, seeking-home flag       =  2                     (fed back as inputs)
  2 recurrent cells fed back           =  2
```

The genome is one flat vector of 2668 floats in seven blocks:

| Block | Size | Read by |
| --- | --- | --- |
| input -> hidden | 61 x 20 = 1220 | every model |
| hidden bias | 20 | every model |
| hidden -> output | 20 x 8 = 160 | every model |
| output bias | 8 | every model |
| time constants | 20 | `time` |
| gate weights | 20 x 61 = 1220 | `gated` |
| gate biases | 20 | `gated` |

Every model carries every block, whichever one is selected. That is deliberate:
it makes switching a parameter change rather than a reinterpretation of the
population, so a saved run stays meaningful across a switch, and it is why the
three are comparable at all. It was 1408 weights before time constants existed
and 1428 with them; the gate block roughly doubles it, and the packing limit
(`BrainStrideMask`, 4095 per stride) is still not near.

State is one float per hidden neuron, on the agent record beside everything
else a step carries -- 176 to 256 bytes -- so the CPU path and the GPU path
store it the same way and multi-step parity covers it without a separate
harness. It is zero at the start of a generation, which is the whole of the
reset semantics. The gate needs no state of its own: what it needs is the state
the neuron already has.

Both file formats notice a brain that changed shape. A genome archive from an
older brain is rejected by the weight count it already records, with a message
naming both counts, which is more use than a version number would be; world
snapshots carry a version of their own, currently 7.

## Replay

Watching trained weights is a different job from training them, and the
difference is one flag. **Replay only (no evolution)** scores and reports every
generation exactly as a training run does -- that is how loaded weights get
judged -- and then selects and mutates nothing. The population is left alone, so
the same genomes respawn; the beacon motion seed is the generation number, which
does not advance either, so the next generation is the same run again rather
than a similar one. That repeatability is asserted by `vkneuro_replay_smoke`,
which compares two replayed generations byte for byte.

**Load genomes** takes a `.vkng` archive, which carries weights and nothing
else -- exactly what replaying a champion needs, since the world is whatever is
set up in the window. An archive normally holds one champion or a handful of
elites while a run has a population size fixed when its buffers were made, so
the archive is repeated across the population and every agent on screen runs the
loaded brain. The status line says the repetition happened rather than leaving
it to be inferred from the picture.

To watch a headless champion:

```sh
vkneuro_headless --scenario scent --generations 200 --save-champion runs/scent.vkng
# then in the window: Load genomes -> runs/scent.vkng, tick Replay only
```

A world snapshot (`.vknw`) is the other way in, and carries the arena and every
setting with it; a genome archive keeps the window's current world and changes
only the brain.

The foraging scenario does not grant passive tracking fitness. Reaching the
resource switches an explicit task input from `seek resource` to `seek home`
and fills a cargo-level input. Cargo decays while being carried, so prompt home
delivery is worth more; delivery completes a cycle and switches the task back.
The home teleports to a deterministic random position every eight simulation
seconds, independently for each trial and generation.
Fitness remains cumulative—the expiring cargo is the decreasing reward
potential—so long generations do not erase already completed work. Two separate
learned memory values are fed back as inputs on the next simulation step and
updated by the final two network outputs.

## Architecture

```text
vulkan_neuroevolution_agents (windowed composition root)
vkneuro_headless             (batch composition root)
  |
  +-- vkneuro_domain          no Vulkan dependency
  |     neuro/
  |       BrainKernel.inl     network preset compiled by C++ and GLSL alike
  |       NeuralNetwork       C++ view of the preset + single-network evaluator
  |     worlds/
  |       ScenarioKernel.inl  scenario math compiled by C++ and GLSL alike
  |       WorldScenario       scenario contract + validated registry
  |       scenarios/          one file per experiment, whole contract each
  |     Sensors               CPU reference perception
  |     CpuSimulation         CPU reference physics/fitness
  |     GeneticAlgorithm      selection/crossover/mutation
  |
  +-- vkneuro_simulation
  |     SimulationDriver      population, GA, and per-step dispatch recording
  |     SimulationModule      frame-loop adapter over the driver
  |     worlds/*.glsl         per-scenario geometry, shared with the vertex shader
  |     worlds/steps/*.glsl   per-scenario step hooks mirroring the C++ ones
  |     agent_grid_*.comp     per-logical-world spatial acceleration
  |     agent_step.comp       RGB sensors + brain + collisions + physics
  |
  +-- vkneuro_visualization
  |     AgentRenderer         read-only consumer of the agent SSBO
  |
  +-- vkneuro_ui
  |     SimulationUiModule    controls/statistics/viewport only
  |
  +-- reusable infrastructure
        vkexp_core, vkexp_compute, vkexp_profiling, vkexp_imgui
```

The shared contracts are small:

- `SimulationState` carries controls, statistics, the published agent-buffer
  view, and the published viewport image;
- `AgentState` is an explicitly checked 176-byte std430-compatible structure;
- `BrainKernel.inl` is the network preset: sensor block sizes, hidden width and
  output meanings, from which the input capacity, every block offset, the genome
  size and the packed GPU layout are derived. Both languages compile it, so the
  CPU evaluator, the sensor sampler and the shader build the same network from
  one declaration; `Topology` is the C++ view of it and restates nothing. Each
  `ScenarioDefinition` then selects its active input/hidden/output counts;
- module order in `main.cpp` is the composition graph: compute publishes the
  buffer, rendering reads it, and ImGui composites the viewport;
- `GpuStepParameters` carries only what every scenario needs, plus a 48-byte
  `ScenarioParameterBlock` each scenario packs and unpacks itself. It travels in
  a storage buffer indexed by step rather than in push constants, so adding a
  scenario neither widens a shared struct nor approaches the 128-byte push
  constant size Vulkan guarantees;
- `ScenarioDefinition` is the whole contract of an experiment: brain shape,
  beacons, fitness, objective counting, per-step hooks, GPU packing, and the
  tunables the UI should offer. A validated registry replaces what used to be
  scenario switches in the simulation, the scoring, the renderer, the batch
  runner and the UI;
- `FitnessWeights` carries the shaping coefficients to both the CPU reference
  and the shader, so a fitness experiment is a slider or a CLI flag.

This lets a new sensor model, brain evaluator, selection policy, renderer, or
UI replace its counterpart without changing the application lifecycle.

`SimulationDriver` holds the experiment; `SimulationModule` only maps frame
callbacks onto it. The batch runner drives the same driver from an
`ImmediateContext`, so a sweep and the window run identical code.

### Units and the two time bases

`include/vkexp/simulation/Units.hpp` is where the scale is declared and where
the step's two time bases are reconciled. A step is the unit of reproducibility:
replays, archives and parity tests are all indexed by step count, and none of
them depends on wall-clock time. A second is the unit the physics is written in
-- speeds in m/s, drags and decay rates in 1/s -- so that `deltaTime` is a free
parameter rather than a hidden part of the fitness function.

The rule that keeps them consistent: a quantity accumulated over the step is
multiplied by `deltaTime`, and a fraction removed per step is written as
`1 - exp(-rate * dt)`, the form the drags already used. The wall penalty and the
contact solver were the two that broke it, and both were charged per step, which
is why `deltaTime` was pinned at 1/60 and never exposed.

### Where the CPU path fits

The CPU code is not a mirror of the shader. It exists to build the network from
the shared preset, to score a finished generation, and to step and inspect a
single agent -- which the GPU cannot do usefully for 2048 of them at once. What
genuinely differs between the two, the parallel substrate, is verified by tests
that need many agents:

- `runGenomeAddressingProbe` gives six genomes distinctive motor biases and
  checks every agent follows its own; a wrong genome stride or base offset is
  invisible to a single-agent parity test.
- `runMultiAgentDeterminism` and `runAgentInteractionTest` cover the shared
  spatial grid, barriers and logical-world isolation.

### Adding a scenario

One source file fills in a `ScenarioDefinition`, one `worlds/<name>.glsl`
unpacks the same parameter block for geometry, one `worlds/steps/<name>.glsl`
mirrors the step hooks, and one line joins the registry. Shared formulas go in
`include/vkexp/worlds/ScenarioKernel.inl`, which is compiled twice -- once as
C++ through a small `vec2`/`uint` shim, once as GLSL where those names are built
in -- so a hash constant or a beacon formula exists exactly once. The remaining
scenario identity checks live in two GLSL dispatchers, which is as far as a
language without function pointers allows.

The four beacon-following scenarios currently use a reactive `57 -> 20 -> 6`
brain. `Forage + home` and `Scent relay` declare `61 -> 20 -> 8`, adding task
state and two recurrent memory cells. Scenario changes still reset evolution, while the GA
and fixed-capacity GPU genome buffers remain shared.

## CPU/GPU correctness

`vkexp_compute_smoke` creates the same agent and genome on CPU and GPU, advances
both by one complete sensor/network/physics step, reads the SSBO back, and
compares every float with a small tolerance for both world shapes. A two-agent
GPU test verifies physical separation, tactile contact, and reception of an
emitted red signal, then verifies that the same colocated agents cannot collide
or exchange light across a logical-world boundary. Another parity case covers
the exact step at which the active beacon diagonal changes.

On top of that, every scenario runs a 540-step trajectory regression:

- **lockstep parity.** Each step feeds the CPU reference state to the GPU and
  compares one step of both, so the shader is checked at hundreds of genuinely
  reachable states -- including the alternating phase flip and the forage home
  relocation epoch -- without the chaotic drift a free-running trajectory would
  accumulate through tanh feedback.
- **accumulated drift budget.** Lockstep cannot see a systematic bias smaller
  than the per-step tolerance, because resetting to the CPU state each step stops
  it accumulating. Summing the signed per-step differences restores that:
  rounding noise cancels to ~1e-4 over 540 steps, while a changed shader constant
  reaches ~3e-2. The budget sits an order of magnitude above the noise.
- **coverage assertions.** A run whose beacon stayed out of sensor range, or whose
  alternating phase never flipped, fails rather than passing vacuously.
- **determinism.** 192 agents sharing one spatial grid are stepped twice; the
  results must be bit-identical, which is where a grid race would surface.
- **genome addressing.** Six genomes with distinctive motor biases; each agent
  must follow its own. This is the class of bug a single-agent parity test
  structurally cannot see.
- **trail coupling.** A mark is written under one antenna tip and the agent has
  to turn the way that tip is wired. A shader feeding every tip the same cell
  passes every other test and fails this one.
- **reconfiguration.** The arena size, the group size and the trail resolution
  are walked through the real driver, with a generation run after each change.
  These resize GPU buffers under a running simulation, and a stale descriptor
  there faults the device rather than returning a wrong number, so the test
  carries a timeout as part of its assertion.
- **step-rate independence.** An agent is driven into a wall and held there for
  two simulated seconds at 30, 60, 120, 240 and 480 Hz. The accumulated penalty
  has to stay within 25% of the 60 Hz value while the step count changes 16x,
  and has to stay closer to it than the step-count ratio would put it -- the
  second half is what fails if an accumulator goes back to counting steps.

Pure CPU tests cover world scaling, beacon layouts, logical-world partition
mapping, channel mapping, weight layout, neural evaluation, elite preservation,
scenario parameter packing, resolved step settings, genome archive round-trips
including corruption and truncation rejection, and reusable compute validation.
Several guard the contracts this architecture rests on: every registered scenario
is checked for registry order, a CLI key, a brain that fits the genome, a declared
beacon count that matches what it reports, and a step that survives its own hooks;
the shared scenario kernel is pinned on the C++ side so a change to
`ScenarioKernel.inl` cannot slip through on a machine without a GPU; and the brain
preset is checked by derivation rather than by snapshot -- the sensor blocks must
tile the input vector without gaps or overlaps, so adding a sensor stays a
one-line edit instead of a test rewrite.

## Build and run

Requirements: CMake 3.24+, Ninja, a C++20 compiler, Vulkan 1.3 development
files and driver, GLFW 3.3+, `glslangValidator`, and X11/Wayland for the GUI.
Dear ImGui v1.91.8 is fetched by CMake.

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
./build/debug/vulkan_neuroevolution_agents
```

Disable validation if the validation layer is unavailable:

```bash
./build/debug/vulkan_neuroevolution_agents --no-validation
```

### Batch runs

`vkneuro_headless` evolves without a window, which is what makes overnight runs,
parameter sweeps and ablation comparisons possible:

```bash
./build/release/vkneuro_headless --scenario rotating --generations 200 \
    --csv runs/rotating.csv --save-champion runs/rotating-champion.vkng

# Signal-off ablation against the same seed.
./build/release/vkneuro_headless --scenario forage --generations 200 --seed 7 \
    --no-agent-light --csv runs/forage-nolight.csv

# Resume a saved population.
./build/release/vkneuro_headless --scenario forage --generations 50 \
    --load-population runs/forage-population.vkng

# Save and resume a whole experiment, not just its weights. The snapshot carries
# the scenario, arena and every physics setting, so the resume restates none of
# them.
./build/release/vkneuro_headless --scenario scent --generations 40 \
    --save-world runs/scent.vknw
./build/release/vkneuro_headless --generations 40 --load-world runs/scent.vknw
```

Fitness shaping coefficients are flags too (`--objective-bonus`, `--motor-cost`,
`--tracking-reward`, `--signal-cost`, `--energy-drain`, `--fitness-sharing`), so
sweeping them needs no rebuild:

```bash
for reward in 0.0 0.25 0.75; do
    ./build/release/vkneuro_headless --scenario rotating --generations 50 --seed 5 \
        --tracking-reward "$reward" --csv "runs/tracking-$reward.csv"
done
```

`--help` lists every option, and its scenario list comes from the registry.
Genome archives are versioned little-endian files
that record the generation, scenario, seed, fitness and brain shape, and refuse
to load into a build with a different weight count.

A world snapshot (`.vknw`) is the heavier sibling: the population, where every
agent stands, the generation and step it was on, and every physics setting,
which is what lets a resume start mid-generation with no flags. It is equally
strict, rejecting a file written for a different brain topology, agent layout or
settings list, and a population or trial count the running process cannot hold,
since both are buffer dimensions fixed at startup. The trail field is
deliberately excluded: it is device-local, up to 256 MiB, and derived -- a
couple of half-lives of stepping rebuilds it, which costs less than storing it.
The interactive build has the same thing under **Snapshot** in the control
panel.

The debug suite contains pure unit tests, CLI smoke tests, two short real
headless evolution runs covering the archive round trip, and a headless Vulkan
parity test. The Vulkan-dependent ones return CTest's skip code when no compute
device exists.

## Extension points

The next world feature should enter through a focused contract:

- a new scenario adds one `.cpp`, two `.glsl` files and a registry line; it does
  not touch the shared step parameters, the simulation, the scoring or the UI;
- a new sensor channel is a line in the network preset; offsets, genome size and
  both implementations follow;
- a new accumulated cost or reward is written per second, so it does not silently
  become a function of the step rate;
- walls and occlusion extend sensor/world queries;
- richer recurrent cells or gated memory can extend the two-value recurrent state;
- the neuron model is one shared integrator, so a gated unit would replace that
  function rather than the loop around it;
- internal walls and occlusion extend grid-backed world queries;
- colony scoring replaces fitness aggregation without changing physics;
- a different topology can become another evaluator/shader pair;
- GPU-side evolution can later replace the synchronous generation boundary.

## Project documentation

- [PROGRESS.md](PROGRESS.md) records what was implemented at each completed
  stage, the decisions behind it, and the verification status.
- [PLAN.md](PLAN.md) is the forward-looking roadmap and list of unfinished
  milestones.
