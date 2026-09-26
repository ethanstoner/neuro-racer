# Prediction: the real corner floor, and random start poses

Two follow-ups from [devlog 13](13-more-training.md), predicted together and
committed before any champion is run on the new sweep and before any
random-start run.

## A. Where is the floor?

Devlog 13's champions lapped 40.6px corners, the tightest the generated set
has, so their floor is somewhere below that. `floors.py` sweeps the corner-sweep
wave family on a finer ruler: 31 rungs from 173px down to 14px, spaced 2 to 5px
through the range that matters. 14px is where the family stops being a legal
track, and about the circle the car can hold at 40px/s (the physics measures
14.3px, or 22.7px at 60px/s).

Each champion is swept only in the direction it trained. A champion's **floor**
is the tightest rung it reaches from wide to tight without a failure (more than
half of 72 starts lapped). A both-ways champion's floor is the worse of its two
directions. Only runs that learned their own track (over 90% of starts) count.

1. **Physics bounds it.** No champion's floor is under 20px.
2. **The finer ruler agrees with the old one.** The published champion (forward
   seed 1) lands within one rung of its coarse-sweep floor of 51px: somewhere
   from 46.4px to 54.7px.
3. **Training lowers the floor.** The forward-400 median floor is at least two
   rungs tighter than the forward-200 median.
4. **Direction adds nothing on corners.** The both-ways median floor is within
   one rung of the forward-400 median.

## B. Random start poses

`train.py --random-starts`: every generation, all cars start from one freshly
drawn pose. It can be anywhere round the lap and up to 0.7 of the usable
half-width off the line, which is the distribution the 72-start assessment
covers. The budget is the same as ordinary forward training. Snake, clockwise,
seeds 1 to 5, 400 generations, each run read at generation 199 and 399.

1. **It still learns.** At least 4 of 5 lap their own track from more than 90%
   of starts at generation 199.
2. **No drift.** None of the 5 loses more than 5 points on its own track
   between generation 199 and 399. (Forward-400 seed 2 lost 17.)
3. **It doesn't teach direction.** At most 2 of 5 lap 45 or more of the 50
   counter-clockwise generated tracks at generation 199.
4. **It generalises better in its own direction.** At least 4 of 5 lap 45 or
   more of the 50 clockwise generated tracks at generation 199. Forward at 200
   generations managed 2 of 5.

## What would change my mind

- A1 fails: the line through a corner matters more than its centerline radius.
  A 90px-wide track leaves room to take a tight corner on a much wider arc.
- A2 fails: the old floors came from a ruler too coarse to trust, and devlog
  07's numbers need re-measuring, not just re-reading.
- B2 fails: the drift isn't about start poses, and the fix has to be somewhere
  else, such as keeping elites across generations or averaging fitness.
- B4 fails: start-pose variety isn't the kind of variety that transfers to new
  tracks.

## Amendment, before any champion was scored

The first attempt at the sweep crashed on its opening champion. Below a 26px
corner the full-size wave family runs off the 800px arena, and `floors.py` was
missing the edge check `tools/corner_sweep.py` has. So the ruler described
above ("31 rungs down to 14px") included illegal tracks. No rate was computed.

The fixed ruler is the same wave shape at 0.9 scale, which stays legal down to
an 11.6px corner: **32 rungs, 153px down to 12px**, every one checked for
overlap and arena bounds. The predictions stand as written, with one change.
A2's "within one rung of 51px" on the new rungs is **48.2px to 55.4px**, not
46.4 to 54.7. A2 also now compares across a slightly smaller track than the
original sweep, so a miss by one rung shouldn't be read as much.

Result: to follow in 15. The original predictions were committed in b6e2c51,
this amendment before any champion was scored.
