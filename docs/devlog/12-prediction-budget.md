# Prediction: direction, or just more training?

In [devlog 11](11-both-ways-round.md), all five both-ways champions lapped all
100 generated tracks, including 13 whose tightest corner is under the one-way
snake floor of 51px. But both-ways training ran every car on two tracks each
generation, twice the simulation of a one-way run. Two controls match that
budget without adding a second direction:

| Condition | Trained on | Budget |
| --- | --- | --- |
| forward-400 | snake, clockwise, 400 generations | 400 track-evaluations per car lineage, the same as both-ways |
| snake+chicane | snake and chicane, both clockwise, 200 generations, ranked by the mean | 2 tracks a generation, the same as both-ways |

Chicane's tightest corner is 172px, looser than snake's 70px. It adds variety,
but no tighter corners and no second direction. Seeds 1 to 5 each, scored by
`direction.py` exactly as in devlog 11.

Of the 13 generated tracks under 51px, 7 run clockwise. A one-way driver can
only be expected on those 7, so the corner question is asked of them alone.
Both-ways champions lapped all 7 on every seed.

## Predictions

1. **More training doesn't teach the other direction.** At most 2 of 5
   forward-400 runs lap 45 or more of the 50 counter-clockwise generated tracks.
   (Both-ways: 5 of 5, every one at 50.)
2. **Clockwise variety doesn't either.** At most 2 of 5 snake+chicane runs do.
3. **The corner gain isn't just budget.** Each control's median is 4 or fewer
   of the 7 tight clockwise tracks. (Both-ways: 7 on every seed.)

## What each outcome would mean

- 1 and 2 hold: direction data is what makes a two-way driver. More training or
  more tracks in one direction don't substitute for it.
- 3 holds: driving each corner both ways, which is turning left and right
  through it, also improved tight-corner handling, not only direction.
- 3 fails for forward-400: the corner gain in devlog 11 was the extra
  training, and the "0.75 floor" of devlog 07 was an under-trained floor.
- 3 fails for snake+chicane only: any second track helps corners, and
  direction was incidental to that part of the result.

Result: to follow in 13. This file is committed before any of the 10 runs.
