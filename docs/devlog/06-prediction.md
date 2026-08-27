# Prediction, written before the held-out run

The three-champion result says the generalisation ceiling is set by the hardest
corner in the training distribution. That was measured by testing each champion
on the *other training tracks* — tracks chosen, by me, partly because they made
the point. It is suggestive, not a test.

Four tracks were built as a held-out set and never trained on. Their corners
were measured before any champion was evaluated on them:

| Track | Tightest corner | vs. snake's 70px |
| --- | --- | --- |
| ripple | 51px | **tighter** |
| clover | 52px | **tighter** |
| keyhole | 100px | wider |
| peanut | 108px | wider |

Each champion's training ceiling is the tightest corner it ever saw:
oval 161px, chicane 172px, snake 70px.

## The prediction

A champion completes laps on a held-out track if and only if that track's
tightest corner is **at or above** its training ceiling.

| Champion | ceiling | ripple 51 | clover 52 | keyhole 100 | peanut 108 |
| --- | --- | --- | --- | --- | --- |
| oval | 161px (memorised) | fail | fail | fail | fail |
| chicane | 172px | fail | fail | fail | fail |
| snake | 70px | **fail** | **fail** | **pass** | **pass** |

The snake row is the one that can actually be wrong. Snake scored 100% on all
three training tracks, so the lazy expectation is "snake is just the good one"
— it should handle anything. The theory says something much narrower and much
easier to falsify: snake must **fail on ripple and clover**, because at 51 and
52px they are tighter than anything it has ever been shown, and it must pass
keyhole and peanut.

If snake clears all four, the ceiling claim is wrong and what I actually built
is a track-difficulty ranking. If snake clears none, the claim is untestable
because the held-out tracks are simply undriveable. Both are recorded here so
that neither can be quietly rewritten after the fact.

Result: [07-the-held-out-test.md](07-the-held-out-test.md)
