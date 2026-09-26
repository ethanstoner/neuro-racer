# Prediction: is the one-way bias in the data or the network?

[Devlog 09](09-the-generated-test.md) found that the snake champion laps every
built-in track clockwise and none of them counter-clockwise. All of its training
was clockwise, so the obvious explanation is the data: it was never asked to
turn the other way. The alternative is that something about this setup (an
8-8-3 network, this fitness, 200 generations) makes a two-way driver hard to
evolve even when one is asked for.

The published champion is also one seed. Before blaming the data, the
experiment has to show the bias isn't seed 1's luck.

## Design

Snake, 200 generations, population 100, seeds 1 to 5, three conditions:

| Condition | Trained on |
| --- | --- |
| forward | snake as built (clockwise). Seed 1 rebuilds the published champion: training is deterministic, and the first 8 generations match its history exactly. |
| reverse | snake driven counter-clockwise |
| both | each car drives both; fitness is the mean of the two scores |

That's 15 runs. Each final champion is scored by `direction.py` on snake both
ways, on all seven built-ins both ways, and on devlog 09's 100 generated tracks
(50 clockwise, 50 counter-clockwise), from 72 starts per track.

A run **learned** if its champion laps its own training condition from more than
90% of starts. For both, that means both directions. Predictions 1 and 2 count
only runs that learned, because a run that learned nothing trivially fails the
other direction too.

## Predictions

1. **Forward is one-way by default.** At least 4 of the forward runs that
   learned lap reversed snake from under 10% of starts.
2. **Reverse is its mirror.** At least 4 of the reverse runs that learned lap
   forward snake from under 10%.
3. **Asking for both gets both.** At least 4 of 5 both-runs learned, lapping
   snake from more than 90% of starts in each direction.
4. **And it transfers.** At least 4 of 5 both-runs lap within 10 of each other
   on the 50 clockwise and 50 counter-clockwise generated tracks.
5. **Two-way costs little one-way.** The median both-run laps at most 5 fewer
   clockwise generated tracks than the median forward run.

## What each outcome would mean

- 1 and 2 hold, 3 and 4 hold: the bias is in the data. One direction of training
  teaches one direction of driving, and mixing directions fixes it.
- 1 fails: other seeds drive both ways from clockwise-only training, so seed 1
  was unlucky and devlog 09 describes a seed, not a method.
- 3 fails: the network or budget can't hold both. The bias is structural, and
  the fix is capacity or training time, not data.
- 3 holds but 4 fails: it learned snake both ways by memorising two tracks, not
  by learning a direction-free policy.
- 5 fails: generality has a measurable price in one-way skill.

Result: [11-both-ways-round.md](11-both-ways-round.md). This file was committed (8f2f845) before any of the 15 runs.
