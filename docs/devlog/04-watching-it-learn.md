# 4. Watching it learn

*2026-08-26*

![the app at generation 45](img/app-oval-gen45.png)

Three panels, one window. The track with all 100 cars — alive bright, dead
dimmed and frozen where they died, the current leader ringed in gold with its
seven sensor rays drawn. The leader's network on the right, every one of its 99
weights a line: green positive, red negative, thickness by magnitude, neurons
brightening with activation. And the progress chart underneath.

## The simulation does not know the renderer exists

Each generation is run **headless to completion first**, recording every tick,
and then the recorded frames are played back at whatever speed you asked for.

That ordering is deliberate. It means the render path cannot influence the
outcome of a run — what you are watching is exactly what the trainer scored,
not a second simulation that might drift. There is a test for it
(`test_recording_does_not_change_the_outcome`), and another that asserts the
simulation module never imports pygame at all, checked in a fresh interpreter.

It also means `H` can blast through ten generations with no rendering
whatsoever and then drop you back into the visuals, which is essential, because
generations 0 through 15 are not worth watching in real time.

## The chart had to change halfway through

The first version plotted best and mean fitness. It became useless the instant
a car finished a lap.

Fitness before any lap is completed is in the low hundreds. The lap bonus is
10,000. So the moment one car gets round, "best fitness" leaps to 10,274 and
then creeps to 10,340 over the next 78 generations — a flat line pinned to the
top of the axis, with the entire pre-lap history squashed into the floor.

So the chart switches what it plots. Before any lap: best and mean fitness.
After the first lap: **best lap time** becomes the headline series, because
that is what is actually being optimised from then on, with mean fitness kept
alongside on its own scale as the population-health signal. You can see both in
the screenshot — lap time falling 12.58s → 6.30s in gold, mean fitness climbing
underneath in white.

Mean fitness is the more informative of the two, and it is the one I did not
expect to care about. Best fitness plateaus early because one lucky genome gets
round. Mean fitness keeps climbing for another hundred generations as the *rest
of the population* learns to drive — it is the difference between "someone can
do this" and "everyone can do this".

## The learning curve

The oval is easy — a lap by generation 1. Snake is where it is worth watching:

```
  gen       best       mean  alive  laps     lap
    0      159.8       16.4      0     0      --
    5      285.2       53.7      0     0      --
   10      309.0       68.7      1     0      --
   12          -          -      -     1       -     <- first completed lap
   20    10246.7     1723.6     16     1  15.33s
   40    10282.3     6739.1     65     1  11.77s
   80    10306.8     9024.0     86     1   9.32s
  120    10311.8    10210.9     98     1   8.82s
  199    10315.3    10015.1     97     1   8.47s
```

Twelve generations of nothing but crashing. Then one car gets round, and
because elitism copies it forward unmutated, that solution never gets lost —
within eight generations 16 cars are finishing, and by generation 120 it is 98
out of 100.

Lap time keeps falling long after everyone can finish: 15.33s → 8.47s. That
second half of the run is the part where it stops learning to *drive* and starts
learning to drive *fast*.

## The staged fitness

This only works because scoring is staged. Two rules that decide whether the
whole thing functions:

**Crashing must not be punished harder than doing nothing.** If it is,
generation 1 evolves into parked cars — sitting still scores zero, crashing
scores negative, and the population never explores the track at all. So a crash
just ends the run and the car keeps everything it earned on the way.

**Lap time cannot be the objective until laps are possible.** For the first
twelve generations, nothing on snake finishes, so a time-based score would be
uniformly "infinity" and carry no gradient. Fitness is progress-based until a
genome completes a lap, and only then does time take over.

There is also an anti-idle rule: three seconds without new forward progress and
the car is culled. Without it, spinning gently on the spot is a perfectly
survivable strategy that burns the whole 40-second episode.

Next: whether any of this actually generalises.
