# Prediction, written before the generated-track run

The corner-floor result in [07](07-the-held-out-test.md) was measured on four
held-out tracks and one sweep family, all drawn by me. This time the held-out set
is 100 tracks from `src/procgen.py`, seed 7. Their shapes were fixed by the
generator's rules before any champion was run on them, and the only thing I
looked at in advance was their tightest corners:

| Tightest corner | Tracks |
| --- | --- |
| under 51px | 13 |
| 51 to 131px | 79 |
| 131px and over | 8 |

(40.6px to 178.9px. Every track also passed the editor's three rules and turns
both ways for at least 3% of the lap.)

## The prediction

The measured floors from the corner sweep: chicane 131px, snake 51px. The oval
champion has no floor.

A champion laps a generated track (more than half of its 72 starts finish a lap)
**if and only if** the track's tightest corner is at or above that champion's
floor.

| Champion | floor | predicted passes |
| --- | --- | --- |
| oval | none | 0 of 100 |
| chicane | 131px | the 8 tracks at 131px or over |
| snake | 51px | the 87 tracks at 51px or over |

## What would count as wrong

- **Accuracy under 80%** across the 300 champion-track pairs means one corner
  radius is not a usable difficulty measure on random tracks, and devlog 07's
  caveat (recovery room matters) is the main effect, not a footnote.
- **The oval champion lapping any track** means it has more of a policy than the
  memorisation story allows.
- **Misses that all lean one way** (passing tighter than the floor, or failing
  wider) mean the floor is biased on this family, not just noisy.

Result: to follow in 09-the-generated-test.md. This file is committed before that run.
