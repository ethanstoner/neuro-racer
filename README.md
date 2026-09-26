# NeuroRacer

[![tests](https://github.com/ethanstoner/neuro-racer/actions/workflows/tests.yml/badge.svg)](https://github.com/ethanstoner/neuro-racer/actions/workflows/tests.yml)

Cars that teach themselves to race by neuroevolution, used as a small lab for
one question: when a trained policy looks like it generalises, does it? Every
claim here was tested with a prediction committed to git before the run, and
several of those predictions were wrong.

![80 generations of learning to drive](docs/devlog/img/learning.gif)

*80 generations on `snake`, unedited. Generation 0 puts 7 of 100 cars through a
wall in the first corner; by generation 79, 83 of them are lapping and the best
has gone from 23.70s to 9.37s. The panel on the right is the leading car's
actual network, redrawn every generation as it evolves.*

### Highlights

- **Found a hidden bias in the headline result.** The champion presented as the
  one that generalises laps **0%** of starts on all seven built-in tracks driven
  the other way, including its own. Every built-in track ran clockwise, so
  nothing had tested it. Training both directions made **5 of 5** seeds lap
  all 100 generated held-out tracks and all 7 built-ins both ways.
- **Separated direction from compute with budget-matched controls.** At equal
  simulation budget, one-way training produced a two-way driver in 1 of 5 seeds
  (400 generations) and 0 of 5 (two clockwise tracks), against 5 of 5 for
  both-ways.
- **Showed the "corner floor" was a training artefact.** Measured on a 32-rung
  sweep: 43px after 200 one-way generations, 27px after 400, and 12px (the end
  of the ruler) when trained both ways.
- **100 cars for 8.7× the cost of one.** The whole population steps as NumPy
  arrays, so a full 200-generation run takes about 5 minutes on one core.

**Python · NumPy · pygame**, with a TypeScript track editor
([virtual-world](https://github.com/ethanstoner/virtual-world)) that shares
the track format.

## What it found

Each step below came from testing the previous claim harder. The devlog has the
full write-ups, including the predictions that failed.

**1. One champion learned to drive; another memorised a track.** Three
champions, trained identically on different tracks, each dropped at 72 start
poses:

| Champion | oval | snake | chicane |
| --- | --- | --- | --- |
| trained on **oval** | 6% | 0% | 0% |
| trained on **chicane** | **100%** | 0% | **100%** |
| trained on **snake** | **100%** | **100%** | **100%** |

The oval champion had the fastest lap and couldn't drive. It had memorised one
open-loop trajectory and failed its own track from 30px off the line
([devlog 05](docs/devlog/05-it-memorised-the-track.md)).

**2. A held-out test broke the first explanation.** "A champion laps a track iff
its corners are no tighter than its training" scored 8 of 12 on four unseen
tracks, and every miss was a champion doing *better*
([06](docs/devlog/06-prediction.md), [07](docs/devlog/07-the-held-out-test.md)).
A corner sweep put the floor at about 0.75 of the hardest training corner.

**3. 100 generated tracks exposed direction.** A seeded generator
(`src/procgen.py`) made held-out tracks nobody picked. The snake champion lapped
**43 of 43** clockwise tracks above its floor and **0 of 42** counter-clockwise.
Reversed, it laps none of the seven built-ins, its own included
([08](docs/devlog/08-prediction-generated.md), [09](docs/devlog/09-the-generated-test.md)).

| snake champion, its own track reversed: crashes 3.9s in | both-ways champion, same track: 8.48s |
| --- | --- |
| ![](docs/devlog/img/eval-snake199-on-snake-reversed.png) | ![](docs/devlog/img/eval-snake-both199-on-snake-reversed.png) |

**4. Direction has to be trained.** Five seeds per condition, with controls
matched to the both-ways budget
([10](docs/devlog/10-prediction-direction.md) to [13](docs/devlog/13-more-training.md)):

| Training | drives both ways | generated tracks lapped (median of 5) |
| --- | --- | --- |
| forward, 200 generations | 2 of 4 that learned | 48 / 100 |
| forward, 400 generations (same budget) | 1 of 5 | 62 / 100 |
| snake + chicane, both clockwise (same budget) | 0 of 5 | 60 / 100 |
| **both directions** | **5 of 5** | **100 / 100** |

The published champion was seed 1 of the forward runs: the extreme case, not
the typical one.

**5. The corner floor depends on training, not the network.** A finer sweep, 32
rungs down to 12px ([14](docs/devlog/14-prediction-floors.md),
[15](docs/devlog/15-floors-and-random-starts.md)):

| Training | corner floor (median, runs that learned) |
| --- | --- |
| forward, 200 generations | 43px |
| forward, 400 generations | 27px |
| **both directions** | **12px**, the end of the ruler |

Below about 45px, centerline radius stops describing the corner the car drives.
Champions lap "12px corners" on a far wider line. Training from a random start
pose every generation fixed a robustness drift on the training track, but
didn't improve generalisation: it made robust specialists.

## Engineering highlights

- Designed the evaluation around **pre-registration**: five prediction documents
  committed before their runs (commits cited in each devlog). Six of the 16
  scored predictions in devlogs 10, 12 and 14 were wrong, and each is written up
  as a miss rather than re-explained.
- **Made training deterministic and proved it**: re-running seed 1 rebuilds the
  published champion bit for bit, so every result can be re-derived from a seed.
- **Vectorised the whole population**: a tick for 100 cars costs 0.69ms, 8.7× a
  single car, because per-call NumPy overhead is shared. The 30 training runs
  behind devlogs 11 to 15 ran in three parallel batches of 15 to 17 minutes
  each.
- **Represented a track as three rasterised arrays** (`drivable`, `progress`,
  `checkpoint`), so collision, lap progress and raycasting are array lookups
  with no geometry code in the hot loop.
- **Kept the measurements honest across languages**: the virtual-world editor
  ports the track metrics to TypeScript. Parity is tested both ways through
  fixture files, which also exposed that rounding coordinates to 0.01px moves
  the corner-radius reading by about 3%.

## Quick start

```bash
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

venv\Scripts\python.exe main.py --track snake      # watch it learn
venv\Scripts\python.exe drive.py snake             # drive it yourself
venv\Scripts\python.exe train.py --track snake --generations 200
venv\Scripts\python.exe train.py --track snake --direction both      # both ways round
venv\Scripts\python.exe train.py --track snake chicane               # several tracks
venv\Scripts\python.exe train.py --track snake --random-starts       # new start pose each generation
venv\Scripts\python.exe train.py --track my-track.track.json         # a track file
```

| Key | |
| --- | --- |
| `1` `2` `3` | playback speed 1× / 4× / 16× |
| `H` | headless burst: 10 generations with no rendering |
| `SPACE` | pause |
| `S` | screenshot |
| `ESC` | quit |

Track files come from the virtual-world editor, which checks the same rules the
loader enforces: no overlap, inside the arena, and a tightest centerline radius
of at least 40px.

### Reproducing the results

The champions are committed under `champions/`, so none of this has to be taken
on trust:

```bash
venv\Scripts\python.exe heldout.py                 # the four hand-made held-out tracks
venv\Scripts\python.exe generalise.py              # champions vs 100 generated tracks
venv\Scripts\python.exe direction.py               # every condition, both directions
venv\Scripts\python.exe floors.py                  # the 32-rung corner sweep
venv\Scripts\python.exe robustness.py --run champions/snake --all-tracks
```

`direction.py` and `floors.py` read training runs from `runs/`, which isn't
committed. The command for each run is at the top of its devlog.

## How it works

**A track is three arrays.** A centerline plus a width, rasterised once into
`drivable`, `progress` and `checkpoint` maps over the 1200×800 world. Raycasting
is 48 samples along each ray gathered from the mask.

**The whole population moves in lockstep.** Positions, velocities and genomes
are stacked arrays, and the population's forward pass is two `einsum` calls:

| Population | ms / tick | µs per car-tick | vs. 1 car |
| --- | --- | --- | --- |
| 1 | 0.079 | 79.4 | 1.0× |
| 10 | 0.120 | 12.0 | 1.5× |
| 100 | 0.692 | 6.9 | 8.7× |
| 500 | 2.940 | 5.9 | 37.0× |

**Sensors → network → controls.** Seven raycast distances spread ±90°, plus
speed, into 8 → 8 → 3 (throttle, brake, steer): 99 weights, small enough that
every connection is drawn in the visualiser.

**Fitness is staged.** Progress along the lap until someone finishes, then lap
time. A crash ends the run but keeps what the car earned; punishing crashes
harder than idling makes generation 1 evolve parked cars. Three seconds without
progress and the car is culled.

**Steering fades with speed**, which is the only reason braking is something to
learn. `tests/test_premise.py` measures this from the simulation and fails if
tuning ever makes flooring it optimal everywhere.

## Layout

Anything you run lives at the top level. `src/` is library code with no CLI,
`tools/` is instrumentation.

```
config.py            every tunable, one frozen dataclass

main.py              the live app
train.py             headless training: directions, several tracks, random starts
drive.py             arrow-key driving, human baseline
evaluate.py          replay a champion, draw its trajectory by speed
preview_track.py     rasterise a track to PNG
robustness.py        policy, or one memorised trajectory?
heldout.py           every champion vs. the four unseen tracks
generalise.py        every champion vs. a seeded set of generated tracks
direction.py         every training condition, scored in both directions
floors.py            the 32-rung corner-floor sweep

src/                 library -- imported, never executed
  track.py           centerline + width -> the three masks, corner geometry
  tracks.py          built-in tracks; load() also takes a file path or reverse=True
  track_io.py        track files: the editor's format and the rules they must pass
  procgen.py         seeded star-shaped tracks for held-out sets
  physics.py         arcade step over population arrays
  sensors.py         vectorised mask-sampling raycast
  net.py             batched 8-8-3 MLP, flat genome
  fitness.py         staged scoring, wrapped progress, idle culling
  evolve.py          elitism, tournament selection, annealed mutation
  robustness.py      spawn grid, random start poses, lap-rate assessment
  simulation.py      headless generation runner (never imports pygame)
  artifacts.py       run recorder -- history.csv and champions.json
  pump.py            background generation worker for the live app
  render/            track, network, chart and HUD panels

tools/               instrumentation
tracks/              track files used by the devlog images
champions/           oval, chicane, snake and snake-both, committed so the
                     published numbers can be re-measured
tests/               all headless, so they run in CI
docs/                devlog, and the JSON behind every results table
```

## Tests

```bash
venv\Scripts\python.exe -m pytest        # 213 tests
```

All headless, so they run in CI. The ones worth knowing about:

- `test_premise.py`: the tracks must contain corners the car can't take flat
  out, so a tuning change can't quietly make the problem trivial.
- `test_car_cannot_ride_the_wall`: collision uses the car's body. The first
  random population found the wall-hugging exploit immediately.
- `test_a_parked_car_scores_below_a_car_that_progressed_then_crashed`: the most
  important property in the fitness function.
- `test_heldout.py`: pins the published claims to the committed champions,
  including `test_the_snake_champion_only_drives_one_way_round` and
  `test_training_both_ways_round_fixes_it`.
- `test_editor_measurements_match_numpy`: the editor's TypeScript measurements
  against the numpy originals, on a file exported from the editor's UI.
- `test_spawn_poses_face_along_the_track`: a broken harness would read as
  "nothing generalises".

## What I learned

- **The test set decides the result.** Every tidy conclusion here came from
  tracks I'd drawn, and every one changed on a set nobody picked. The
  clockwise-only built-ins hid the biggest effect in the project for a month.
- **Writing the prediction down first is what makes a miss useful.** Four
  predictions failed in the last round alone. Because they were in git before
  the runs, each failure pointed at what to test next instead of getting
  quietly re-explained.
- **One seed is an anecdote.** The published champion was the most one-sided of
  five forward seeds. Five seeds per condition cost about 16 minutes in
  parallel.

## Build log

1. [A track is three arrays](docs/devlog/01-a-track-is-three-arrays.md)
2. [Is there actually anything to learn?](docs/devlog/02-is-there-anything-to-learn.md)
3. [Generation one cheats](docs/devlog/03-generation-one-cheats.md)
4. [Watching it learn](docs/devlog/04-watching-it-learn.md)
5. [One of them learned to drive. The other memorised a track.](docs/devlog/05-it-memorised-the-track.md)
6. [A prediction, written down first](docs/devlog/06-prediction.md)
7. [The held-out test, and the prediction it broke](docs/devlog/07-the-held-out-test.md)
8. [A prediction for 100 tracks nobody drew](docs/devlog/08-prediction-generated.md)
9. [100 tracks nobody drew, and the direction nobody tested](docs/devlog/09-the-generated-test.md)
10. [Prediction: is the one-way bias in the data or the network?](docs/devlog/10-prediction-direction.md)
11. [Both ways round](docs/devlog/11-both-ways-round.md)
12. [Prediction: direction, or just more training?](docs/devlog/12-prediction-budget.md)
13. [Direction, and more training](docs/devlog/13-more-training.md)
14. [Prediction: the real corner floor, and random start poses](docs/devlog/14-prediction-floors.md)
15. [The corner floor, and random start poses](docs/devlog/15-floors-and-random-starts.md)

## License

MIT, see [LICENSE](LICENSE).
