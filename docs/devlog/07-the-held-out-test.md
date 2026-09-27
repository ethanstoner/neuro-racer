# The held-out test, and the prediction it broke

[The prediction](06-prediction.md), written before the run: a champion laps a
track if and only if that track's tightest corner is at or above the tightest
corner it saw in training.

Three committed champions, four tracks none of them had ever seen, 72 spawn
poses each.

```
champion   ceiling  track      corner  starts         lapped  predicted
oval          161p  clover        52p      72      0 (  0%)       fail  ok
oval          161p  peanut       108p      72      0 (  0%)       fail  ok
oval          161p  ripple        51p      72      0 (  0%)       fail  ok
oval          161p  keyhole      100p      72      0 (  0%)       fail  ok
chicane       172p  clover        52p      72      0 (  0%)       fail  ok
chicane       172p  peanut       108p      72     72 (100%)       fail  MISS
chicane       172p  ripple        51p      72      0 (  0%)       fail  ok
chicane       172p  keyhole      100p      72     42 ( 58%)       fail  MISS
snake          70p  clover        52p      72     71 ( 99%)       fail  MISS
snake          70p  peanut       108p      72     72 (100%)       pass  ok
snake          70p  ripple        51p      72     70 ( 97%)       fail  MISS
snake          70p  keyhole      100p      72     72 (100%)       pass  ok

prediction correct on 8/12 champion-track pairs
```

8 of 12. The prediction is wrong, and every one of the four misses is wrong in
the same direction: champions cleared corners **tighter** than anything they
trained on. Snake, trained on a 70px hairpin, lapped ripple's 51px corner from
97% of starts.

So the ceiling is not the training minimum. But it is not absent either — snake
cleared all four held-out tracks while chicane cleared two and oval cleared
none, which is exactly the ordering the training corners predict. Something is
setting a limit; the tidy version just names it wrongly.

## Finding the real floor

The four held-out tracks are too coarse a ruler. Their corners cluster at 51,
52, 100 and 108px, so all they establish is that snake's limit is below 51 and
chicane's is somewhere in a 48px-wide gap.

`tools/corner_sweep.py` builds a family of tracks that differ only in corner
tightness — same size, same lap length, same four direction changes — and walks
each champion down it until it fails.

| Champion | Trained on | Measured floor | floor / ceiling |
| --- | --- | --- | --- |
| oval | 161px | *never cleared even the 173px rung* | — |
| chicane | 172px | **131px** | 0.76 |
| snake | 70px | **51px** | 0.73 |

Two independent champions, trained on different tracks, land on the same ratio.
Each generalises to corners about **a quarter tighter** than the hardest corner
it was shown, and no further.

The failure is a cliff, not a slope. Chicane holds 48/48 starts at 131px and
drops to 2/48 at 106px. Snake holds 48/48 all the way down through eleven rungs
to 51px, then goes to 0/48 at 46px. There is no band of partial competence —
one rung either side of the floor is the difference between every start and
none of them.

And the oval champion fails the *widest* rung in the family, one with gentler
corners than the oval it trained on. That is the cleanest statement of the
memorisation result in the project: it is not that oval's policy is narrow, it
is that there is no policy. Put it on any shape other than the exact one it was
born on and it is finished.

## The same track, the two champions

Both of these are clover, a track neither champion has ever seen. Trajectory
coloured by speed — yellow fast, blue slow.

![snake champion on clover](img/eval-snake199-on-clover.png)

The snake champion laps it in 8.28s and is visibly *driving*: yellow down the
straights, braking to blue for the tight left-hand lobe and again into the
right-hand hairpin, on corners it was never shown.

![oval champion on clover](img/eval-oval79-on-clover.png)

The oval champion, same track. Flat out — the trace never leaves yellow, so it
never brakes once — into the first corner, and DID NOT FINISH. It is not
driving badly. It is not reading anything at all.

## What I'd correct in the earlier write-up

"The generalisation ceiling is set by the hardest corner in the training
distribution" was too strong. The measured version:

> A champion that learns a real policy generalises to corners roughly 25%
> tighter than the hardest one it trained on, then fails abruptly. A champion
> that memorises a trajectory has no floor at all, because it has no policy to
> extend.

## The caveat I can't remove

Corner radius is not a complete description of difficulty. The chicane champion
laps peanut's 108px corner from every start, but fails the sweep's 106px rung.
Nearly the same radius, opposite outcomes — because the sweep rung presents four
tight corners a lap with short straights between them, and peanut presents one
with a long fast approach. Recovery room matters and a single scalar does not
capture it.

So the 0.75 ratio is a property of these champions measured on this family, not
a constant of nature. What survives regardless of family is the ordering, the
cliff, and the oval champion's total absence of a floor.

Reproduce:

```bash
python -m experiments.heldout --json docs/results/heldout.json
python tools/corner_sweep.py --json docs/results/corner-sweep.json
```
