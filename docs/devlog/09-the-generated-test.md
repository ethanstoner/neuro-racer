# 100 tracks nobody drew, and the direction nobody tested

The prediction is in [08](08-prediction-generated.md), committed before this
ran. 100 tracks from `src/procgen.py` (seed 7), 72 starts per champion per
track, 1m37s in total:

```
python -m experiments.generalise --n 100 --seed 7 --json docs/results/generalisation.json
```

| Champion | floor | own track | generated mean | tracks lapped | predicted | correct |
| --- | --- | --- | --- | --- | --- | --- |
| oval | none | 6% | 0.1% | 0 / 100 | 0 | 100% |
| chicane | 131px | 100% | 23.1% | 19 / 100 | 8 | 87% |
| snake | 51px | 100% | 50.1% | 48 / 100 | 85 | **53%** |

240 of 300 pairs are correct, which is exactly 80%: on the line I set, not
under it. But the third failure condition in 08 fired outright. **Snake's misses
all lean one way**: it failed 42 tracks *wider* than its floor and lapped only
5 tighter than it.

## What the 42 have in common

It isn't corners. The tracks snake lapped and the ones it failed have almost the
same tightest corner, 93.7px against 92.0px. Tight-corner count (1.5 against
2.0 corners under 100px), reverse curvature and lap length differ a little, but
none of them splits the two groups. This does, completely:

| | clockwise | counter-clockwise |
| --- | --- | --- |
| generated tracks at or above snake's floor | 43 | 42 |
| snake lapped | **43** | **0** |

The generator picks the driving direction with a coin flip. The seven built-in
tracks don't: every one is written as `x = cos t, y = sin t`, which runs
clockwise on screen. Training, the held-out four and the corner sweep all ran
one way round, so no earlier test could have seen this.

The direct check is to reverse the built-in tracks and nothing else:

| Champion | track | forward | reversed |
| --- | --- | --- | --- |
| snake | snake (its own) | 100% | **0%** |
| snake | oval | 100% | 0% |
| snake | chicane | 100% | 0% |
| snake | clover | 99% | 0% |
| snake | peanut | 100% | 0% |
| snake | ripple | 97% | 0% |
| snake | keyhole | 100% | 0% |
| chicane | chicane (its own) | 100% | 93% |
| chicane | oval | 100% | 100% |
| chicane | peanut | 100% | 94% |
| chicane | keyhole | 58% | 22% |
| oval | oval (its own) | 6% | 0% |

![snake champion on its own track, reversed](img/eval-snake199-on-snake-reversed.png)

The snake champion on its own training track, driven the other way. It gets
through the S-bend along the top at speed, reaches the tight left-hand end of
the track, and goes into the outside wall 3.9 seconds in.

So the champion this project held up as a real, generalising policy works one
way round only. The chicane champion, which the corner story ranked below it,
mostly holds up when reversed. Keyhole is the exception: 58% forward, 22%
reversed.

## Is the corner floor still worth anything?

Split by direction, yes:

| Champion | direction | tracks | lapped | floor rule correct | misses |
| --- | --- | --- | --- | --- | --- |
| snake | clockwise | 50 | 48 | 90% | 5 lapped tighter than the floor |
| snake | counter-clockwise | 50 | 0 | 16% | 42 failed wider than it |
| chicane | clockwise | 50 | 15 | 80% | 10 lapped tighter |
| chicane | counter-clockwise | 50 | 4 | 94% | 2 tighter, 1 wider |

In the direction a champion trained in, the corner floor is a conservative
bound. Almost every miss is the champion doing *better* than the floor
predicts. Why the chicane floor (131px) is too high here is not yet measured.
The obvious suspect is devlog 07's recovery-room caveat: the sweep rung that
set that floor has four tight corners a lap packed close together. The
generated tracks vary from none to eight corners under 100px, so this set could
answer the question, but I haven't run that split yet.

## Corrected claim

> In the direction it trained in, a champion that learned a policy laps corners
> down to roughly 0.75 of its hardest training corner, and the floor is
> conservative: on random tracks it laps some tighter than that. Driving
> direction is a separate axis that corner radius cannot see, and a champion can
> be completely competent one way round and completely unable the other.

## What changed in the tooling

- Tracks load from files (`--track any.track.json`) in the format the
  virtual-world editor exports. The editor validates live against the same
  three rules the loader enforces. A fixture exported from the editor's UI
  pins its measurements to numpy's to 0.01px.
- `src/procgen.py` generates held-out sets from a seed, and `generalise.py`
  reports train against held-out lap rates.
- `test_heldout.py` now pins the direction result, so it can't be quietly
  fixed away or forgotten.
- Found while building the file format: `min_centerline_radius` is noisy.
  Rounding a 250px circle's points to 0.01px moves its reading from 247.7px to
  240.1px, because it takes the maximum of second differences. Track files are
  now measured on exactly the rounded points they contain. Floors quoted to the
  pixel (51px, 131px) are good to a few percent, not to the pixel.

The obvious next experiment is to train on both directions of the same track
and see whether the snake result is a property of the training data or of the
network.
