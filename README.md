# Snake / TabPFN

A local Snake experiment: TabPFN predicts the future reward of turning left,
going straight, or turning right. A replay table provides its experience.
The interesting variable is **what information we put in that table**.

## Run

```sh
uv sync
cp .env.example .env
# Add your TabPFN token to .env to enable fitting and TabPFN play.
uv run snake-pfn serve
```

Open <http://127.0.0.1:8000>. No key is needed for collection or CLI baseline evaluation.
Get a token from [Prior Labs](https://platform.priorlabs.ai/account/api-keys).
Restart the server after editing `.env`. It binds to localhost only.

The first visit opens on a welcome card over the dimmed app. A pulsing *Live* badge
identifies hosted predictions. **Show me** starts a guided run of about 20 seconds plus
training; it cannot be skipped.

1. **TabPFN never sees the board.** Earlier saved moves are archived (`/api/clear`) and the
   board is reset with the apple in A1 (`/api/reset` and `/api/random-steps` take an
   optional `food` square). Board and empty table light up together.
2. **Every move becomes a row.** Five random moves play at 0.7 s each (`/api/random-steps`
   with `delay: 0.7`), each appearing as a highlighted row. **Watch 1,000 moves** continues.
3. **Watching random moves.** Random moves play until the table holds 1,000
   (`Runner.practice_moves`, `delay: 0.003`), a few seconds with a live counter.
4. **The rules.** The board resets and training starts in the background (`/api/fit`,
   three fitted Q rounds, roughly 15–20 s on the hosted API) while the card lists the rules.
5. **Watch TabPFN play** starts the game. If training is still running, the card says
   *Training TabPFN…* and the game starts by itself when it is done.

The tour lives in the browser (`snake_pfn/web/static/tour.js`, remembered in
`localStorage`) and only calls existing endpoints. An empty table means TabPFN has never
seen Snake, so the intro starts over on a fresh page load. Every step starts when you press its button, never on its own. On later visits the
**Play** button starts a game directly when a model is fitted, or runs steps 3–5 first. Reloading the page during practice or training rejoins the
running step. Practice and the demo use the random policy on purpose: TabPFN builds its
skill from noise.

During a game the snake's head glows while TabPFN scores the three candidate rows
against its cached context. There is no player turn or apple control. The board shows
an arrow into each candidate square; the selected turn is bright. Wall collisions are
marked with a red bar and body collisions with a red cross. If TabPFN selects a crash,
the board flashes while the prediction is shown, then executes that choice. Nothing
overrides the highest predicted return.

After every fifth completed TabPFN game it refits for one round on everything saved so
far before the game ends, shown as a slim **Learning from that game…** strip over the board. The
context is *not* reloaded after each move: it is rebuilt only at the first TabPFN game,
after an input change, and at those refits.

The watch button stops at the end of one game, or after 100 moves per press. A run
that reaches the cap can be continued; a finished game offers **Play again**. The button becomes **Stop** while an operation is
running; stopping waits for an in-flight API request to finish. Every board change is
recorded as a frame (`/api/frames?after=<seq>`) of kind `reset`, `query`, `prediction`
(values known, snake not yet moved), or `move`; the browser fetches frames in batches and
draws them on each screen refresh, skipping ahead when hundreds are queued, so the board
follows the run even when the server runs faster than the page polls. TabPFN's moves are
always shown frame by frame.

**Clear table and play random games**, at the bottom of the table panel, archives the saved moves
to a timestamped file, then runs steps 3–5 on the empty table. The page uses the `board_outcomes` inputs; other presets,
individual column selection, and evaluation are available through `/api/features`, the JSON configs, and the CLI below.

Each TabPFN move needs one hosted prediction, which takes roughly 0.7–1.8 s on
`v3.5-fast`; a fit takes about 4–5 s. The server adds no waits: the next move starts as
soon as TabPFN answers, and the browser alone holds each prediction on screen (200 ms
for the query, 450 ms for the answer) while the next request is already in flight. Under
the board, the page shows how long learning took (the whole fit job, all rounds included)
and how long the latest answer took. `SNAKE_API_INTERVAL` can add a minimum gap between
calls (default 0). That latency, not the game, sets TabPFN's playing speed. All played moves are saved, including failures.
Individual column selection and evaluation are available through the JSON configs
and CLI below, keeping the demo to the board, inputs, and two buttons.

The board is 5 × 5 by default (`DEFAULT_SIZE` in `snake_pfn/game/engine.py`). Set
`SNAKE_BOARD_SIZE` (4–16) to change it; a log holds one board size, so on startup the server
archives saved moves from a different size. The snake starts four segments long
(`START_LENGTH`) in the middle row heading right; on 4 × 4 and 5 × 5 boards the tail bends up the
left edge.

The page pairs **TabPFN plays Snake**, with a live apple score and move count, with
**What TabPFN sees**: one table where every row is one saved move (the board in words,
the direction, and the observed reward), 20 per page. While TabPFN plays, its three
candidate moves (↰ left, ↑ straight, ↱ right) are appended under the latest saved move
with the reward shown as `?`. When TabPFN answers, the `?` becomes the predicted reward
and the chosen row is highlighted. Candidates show only on the last page and stay until
the next query or a fresh board; reloading restores the latest decision.

Rewards are signed and coloured blue (good) or orange (bad), and the chosen row is marked
by weight and a bar, so nothing depends on red versus green. On the board, crash marks are
pink with a dark outline. The last column is always the observed reward, never the fitted
targets. Hover a cell or header for its raw value or column name. While context is being
uploaded, the table dims and a light sweeps over it. Inspection is read-only and makes no
additional TabPFN calls. The heuristic **collect** seeding is still available through the
CLI and `/api/collect`.

The `snake_pfn` Python logger records everything the code does: jobs starting
and finishing, each move with its square, direction, Q values, and reward, fit rounds
with target statistics, TabPFN API calls with timing, input changes, and errors with
tracebacks. The page doesn't show it; read it from the in-memory buffer of the newest 1,000 entries
(`/api/log?after=<id>`). Nothing is written to disk.

## Snake rules

TabPFN controls the snake; the apple stays still until eaten. Eating earns `+1`, grows
the snake by one segment, and spawns a new apple in a free square. The same game and
score continue. Collision with a wall or the body ends the game; filling the board wins.
The original loop safeguard also ends a game after `size × size × 2` moves without food
(50 on the default 5 × 5 board). Eating resets that timer, and `hunger_fraction` uses the
same board-sized limit.

The browser sends `/api/play` with `stop_after_episode: true`. The final board, score,
and last decision stay visible. Press **Play again** to start a fresh game. API callers
can omit that flag to play across multiple episodes. Practice also runs across episodes.
Only terminal states count as completed games or trigger the every-fifth-game refit;
apples remain non-terminal transitions. There is no `/api/apple` control endpoint.

## Debug mode without TabPFN

```sh
uv run snake-pfn serve --stub
```

`--stub` (or `SNAKE_STUB_MODEL=1`) swaps the hosted model for a placeholder that sleeps
like the API (1.5 s per fit, 1 s per prediction; `SNAKE_STUB_FIT_SECONDS` and
`SNAKE_STUB_PREDICT_SECONDS`) and answers with random values, so TabPFN play becomes a
random policy that still goes through the load, predict, reveal, and move choreography.
No requests leave the machine and no quota is spent. The page shows a "debug · stub
model" badge. Use a separate `--data-dir` to keep stub moves out of your real log.

## Code layout

```text
snake_pfn/
  game/
    engine.py       # State, movement, food, collisions, rewards
  control/
    policies.py     # Baseline action selection
    features.py     # Configurable state → model input columns
    experience.py   # Transition log and seed collection
    learner.py      # TabPFN adapter, fitted Q iteration, evaluation
    runner.py       # Coordinates play, collection, and learning jobs
    log.py          # In-memory event log behind the debug panel
  web/
    api.py          # Thin HTTP adapter to the controller
    static/         # HTML, CSS, canvas rendering, and browser controls
  cli.py            # Local server and headless experiment commands
```

The game engine uses only the standard library and has no knowledge of TabPFN or
the UI. Controllers depend on the game, never the web layer. The browser renders
state and sends commands; action selection and learning stay in `control/`.

## Change the inputs without recollecting

Each JSONL row saves the full state before and after a move, action, observed
reward, terminal flag, episode ID, seed, and collection policy. No feature preset
affects logging. Features are derived from the pre-action state when fitting or
predicting; outcomes and metadata never become input columns.

| Preset (5 × 5 board) | Inputs, including candidate action |
| --- | --- |
| `board` | 30: 25 ordered cells, heading, length, hunger, action |
| `compact` | 16: context, food offsets/distances, danger, reachable space, action |
| `augmented` | 41: board, context, food, danger, space, action |
| `outcomes` | 5: crash, eats apple, closer to apple (±1), room left, action |
| `board_outcomes` (default) | 30: 25 ordered cells, the four outcomes, action |

Cell counts grow with the board: on 4 × 4, `board` and `board_outcomes` have 21 columns and `augmented` 32.

Board cells are logical state, rather than RGB pixels: empty `0`, food `-1`,
tail `1`, increasing to the head's length. This preserves the order in which the
body will vacate cells. The hunger timer is included because it affects termination.

Use a preset name or a JSON config in the CLI. For example:

```json
{
  "groups": ["context", "food", "danger", "space"],
  "exclude": ["will_eat", "next_food_distance"]
}
```

Save it as `configs/my-inputs.json`, then use `--features configs/my-inputs.json`.
The candidate `action` column is always included. All other columns can be
excluded. Group order is canonical, so train and predict use the same schema.

To add a new observation, extend `GROUPS` and `all_features()` in
[`snake_pfn/control/features.py`](snake_pfn/control/features.py). The feature API
exposes its columns automatically. Old logs can supply any new feature derivable from their
saved game states. Candidate features preview the move without sampling the next
food location. Reachable space treats the projected body as static; it is a
heuristic, not a guarantee of eventual safety.

Changing inputs resets the fitted model and its Q iterations, **not the log**.
In this MVP, UI input selection and fitted state live for the server session;
use JSON configs and CLI model saves to retain experiments across restarts.

## Learning loop

Each row is `(state features, candidate action) → estimated discounted return`.

```text
Full transition log → selected input columns → TabPFN Q regressor
       ↑                                         ↓
       └──────── actual reward ← Snake ← best of 3 action values
```

Fitted Q iteration uses:

```text
target = reward + 0.95 × (1 − terminal) × max Q_previous(next_state, action)
```

The first round uses observed rewards alone. Each subsequent round computes all
targets with the frozen previous model and fits a fresh estimator. Every saved move
goes into TabPFN's context; `--max-rows` in the CLI can cap it by random sample. Terminal
transitions never bootstrap. The reward is `+1` for food, `−1` for collision or
starvation, and `−0.01` for other moves. Starvation occurs after `size × size × 2`
moves without food; filling the board wins.

TabPFN provides the action-value function; the heuristic is used only for the
explicit baseline player and seed collection. Learned actions are not filtered
through a safety heuristic, and nothing overrides the largest Q value.
This is in-context regression: pretrained weights are never updated.

The hosted adapter explicitly selects `v3.5-fast` and uses `fit_with_cache`. Requests time
out after `SNAKE_API_TIMEOUT` seconds (default 20); the client retries once, then the job
stops with an error rather than leaving a game hanging. Its retry warnings appear in the Log
panel. After the last training round the learner makes one small warm-up prediction so the
first move of a game does not pay the fresh context's setup cost.
Set `SNAKE_MODEL_VERSION=v3.5` to use the larger variant. All three candidate actions
are predicted in one request per move. Raise `SNAKE_API_INTERVAL` if your account
hits rate limits. Fits and predictions consume hosted quota,
and request latency determines TabPFN's playing speed. Errors stop the job and
are displayed; there is no silent substitute model. The status line and the log
show which round or prediction is in progress.

This project is inspired by [ICR-RL](https://arxiv.org/abs/2509.11259), which studies
reinforcement learning through in-context regression. Learning to play Snake well
with these features and this budget remains an experiment, not a guaranteed result.
See the [official SDK](https://github.com/PriorLabs/tabpfn-client) for hosted model
and cache details.

## Headless experiments

```sh
# These two commands need no token.
uv run snake-pfn collect --episodes 30 --seed 0
uv run snake-pfn export --features compact --output data/compact.csv

# Fit, save a hosted model reference + its input schema, then evaluate it.
uv run snake-pfn train --features configs/compact.json --rounds 3
uv run snake-pfn evaluate --model data/model.json --episodes 10 --seed 10000

# Food-seeking baseline, without TabPFN.
uv run snake-pfn evaluate --episodes 10 --seed 10000

# Compare all input presets and the baseline.
uv run snake-pfn compare --episodes 10 --rounds 3 --seed 10000
```

`compare` uses the same frozen log, replay sample, number of fitted Q rounds, and
held-out game seeds for all presets. It rejects evaluation seeds present in the
training log. Evaluation has no exploration, updates, or data collection. It batches
live evaluation games in each prediction request. The main metric is mean apples
per game; results include per-seed scores and survival steps. By default results
go to `data/comparison.json`. Start with a few games to check API usage and latency.

`export` writes **only the selected input columns**, useful for inspecting exactly
what the regressor sees. Training targets depend on the previous fitted Q model
and are computed during each fit, so the CSV intentionally omits a target column.

Experience persists in `data/experience.jsonl`. The API's `/api/clear` action moves it to a
timestamped file and resets the session model. Saved hosted models may expire or
be deleted on the server; the experience log lets you fit again. Do not run two
writers against the same log simultaneously; use `--data` / `--data-dir` to isolate
CLI experiments and the browser session.

## Checks

```sh
uv run pytest -q
uv run ruff check snake_pfn tests
node --check snake_pfn/web/static/app.js snake_pfn/web/static/data-view.js
```

Tests cover game rules, feature schemas, log round trips, terminal masking,
frozen-target iteration, failed-fit recovery, random steps, continuous growth and game endings,
frame streaming, the table's follow-latest mode, and the local API flow/concurrency.
Tests use explicit regression doubles; they do not consume hosted API quota or
establish that TabPFN improves its score. Real model quality needs the evaluation
run above with a valid token.
