# 2. Is there actually anything to learn?

*2026-08-26*

The pitch for this project includes "go the correct speed to get the best time".
That is only a real thing to learn if the track contains corners the car
**cannot take at full throttle**. If flooring it works everywhere, then the
optimal policy is a constant, there is no speed to discover, and the most
interesting part of the project quietly does not exist.

It would be easy to assume this and never check. So I measured it.

## Steering that fades with speed

The car is arcade physics: acceleration along its heading, drag, and a lateral
grip term that damps sideways velocity. The one line that makes it a car rather
than a tank:

```python
rate = cfg.max_steer_rate / (1.0 + speed / cfg.steer_speed_falloff)
```

Steering authority falls off as you go faster. That single term is the entire
reason braking for corners is a thing the network has to discover instead of
something it gets for free.

## The numbers

`tools/premise_report.py` drives the car at full steering lock at each speed and
measures the radius of the circle it actually traces — measured from the
simulation, not derived from the formula, so it stays honest if the physics
changes.

```
turn radius achievable at full steering lock
     90 px/s  ->      37 px
    150 px/s  ->      71 px
    210 px/s  ->     112 px
    270 px/s  ->     161 px
    330 px/s  ->     217 px
    420 px/s  ->     314 px      <- top speed

tightest corner on each track
   oval         161 px
   chicane      172 px
   snake         70 px

max speed each track's worst corner can be taken at
   oval       270 px/s   (64% of top speed)
   chicane    280 px/s   (67% of top speed)
   snake      145 px/s   (35% of top speed)
```

So there is genuinely something to learn. On snake the car has to be down to
about a third of its top speed for the tightest corner, which means a policy
that only knows how to accelerate physically cannot get round.

Note also that this inverted my assumption. I had labelled `chicane` as the
hardest track and `snake` as the medium one. Measurement said the opposite —
snake's worst corner is less than half chicane's radius. I fixed the docs to
match the measurement rather than the other way round.

## Pinning it down

The risk with a number like this is that someone (me, in three weeks) widens a
track or raises the steering rate, the optimal policy silently collapses to
"floor it", and the project just becomes less interesting without anything
appearing to break.

So `tests/test_premise.py` asserts it:

- At top speed the car **must not** be able to hold the tightest corner —
  otherwise there is no correct speed to learn.
- At 90 px/s it **must** be able to — otherwise the track is undriveable and no
  amount of learning will ever complete a lap.
- The gap between "can corner" and "top speed" must be wide enough to matter.

Those are tests of the design premise rather than of any one function, and they
are the ones I would least like to delete.

## Driving it by hand

`drive.py` is arrow-key control on the same physics and the same collision
checks the AI gets. It stays in the repo permanently: it is the human baseline
the evolved champion gets measured against at the end, and it is the fastest
way to tell whether a tuning change made the car better or worse to actually
drive.

Next: what the car can see.
