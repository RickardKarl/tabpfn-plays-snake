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

1. **Play 1000 random steps** plays random moves across as many games as needed, in
   about five seconds. Every move is appended to the table next to the board; the board
   animates the run, skipping frames when they arrive faster than the screen refreshes.
2. Choose **Board**, **Measurements**, or **Board + measurements** under model inputs.
   Changes apply immediately and reset the fitted model.
3. **Play game with TabPFN**. The first press trains TabPFN on the saved moves (three
   fitted Q rounds, roughly 15–20 s on the hosted API), then TabPFN chooses the moves.
   A status card under the board says what TabPFN is doing, with a timer: **Sleeping**
   when idle, **Loading table** while the sampled moves are uploaded as its server-side
   context, and **Predicting moves** while the three candidate rows are scored against
   that cached context (about 1 s per move, where the time goes). The context is *not*
   reloaded after each move: it is rebuilt only at the first TabPFN game, after an input
   change, and after every fifth TabPFN game. TabPFN's view of the next move is drawn on
   the board itself: the square each turn leads to shows a dashed "?" tile while the
   prediction is pending, then the predicted expected future reward, coloured green,
   amber, and rust from best to worst, with the winner outlined. After a 1.5 s pause the
   snake slides onto the winning tile. A turn into the wall is marked as a red band on the
   head square's edge, and a turn into the body as a red dashed outline on that segment,
   each with its value, so you can see when TabPFN rates a crash highly. TabPFN always takes the highest
   value; there is no exploration. Later presses reuse the model. Play with TabPFN
   unlocks after the first random-steps run of the session.
4. Try another input configuration using exactly the same collected experience;
   the next game trains again with the new columns.
5. **Clear table** archives the saved moves to a timestamped file and starts fresh.

Play is capped at 100 moves per click. The button becomes **Stop** while an
operation is running; stopping waits for an in-flight API request to finish.
TabPFN play is greedy: the highest predicted value always wins, with no exploration.
It refits after every fifth completed episode.
Every board change is recorded as a frame (`/api/frames?after=<seq>`) of kind `reset`,
`prediction` (values known, snake not yet moved), or `move`; the browser
fetches frames in batches and draws them on each screen refresh, skipping ahead when
hundreds are queued, so the board follows the run even when the server runs faster
than the page polls. TabPFN's one move per second is always shown frame by frame.

Each TabPFN move needs one hosted prediction, which takes roughly 0.7–1.8 s on
`v3.5-fast`; a fit takes about 4–5 s. `SNAKE_API_INTERVAL` adds a pause between calls
(default 0.2 s). That latency, not the game, sets TabPFN's playing speed. All played moves are saved, including failures.
Individual column selection and evaluation are available through the JSON configs
and CLI below, keeping the demo to the board, inputs, and two buttons.

The board is 5 × 5 by default. Set `SNAKE_BOARD_SIZE` (5–16) to change it; a log holds
one board size, so on startup the server archives saved moves from a different size.

The page is split in half: the light **What you see** panel holds the game and
controls; the dark **What TabPFN sees** panel is the table the model receives. The
decision happens in that table too: while TabPFN plays, the three candidate rows it is
asked about appear as ghost rows at the bottom with a blurred "?" target, fill with the
predicted values when the answer lands, and the winning row becomes the next saved row.
While the table is being uploaded as TabPFN's context, it dims and a light sweeps over it. Values are
shown in words rather than numbers: the move as up/right/down/left, board squares as
empty/apple/snake/head/tail (named like a chessboard, A1 top-left), and yes/no for
danger columns. Hover any cell or header for the raw value or column name. While a
game runs, the table follows the newest saved move and highlights it; its inputs
describe the board just before that move. Before training, the last column is the
observed reward. After training it shows the exact sampled inputs and fitted-Q targets
from the latest successful fit, 20 rows per page. The latest prediction's three candidate
rows remain available from `/api/table` as `prediction`. Inspection is read-only and
makes no additional TabPFN calls.
Fit/prediction tables live in memory for the server session and are cleared when the
input configuration changes. The 20-game **collect** seeding is still available through
the CLI and `/api/collect`.

A **Log** panel at the bottom of the page shows everything the code does: jobs starting
and finishing, each move with its square, direction, Q values, and reward, fit rounds
with target statistics, TabPFN API calls with timing, input changes, and errors with
tracebacks. It is fed by the `snake_pfn` Python logger through an in-memory buffer of
the newest 1,000 entries (`/api/log?after=<id>`), so nothing is written to disk.

## Play as the apple

While TabPFN plays, you are the apple, and the game is turn-based. After each snake move
there is a 1.5 s **Your move** window (`Runner.turn_window`): the arrow keys, or a click
on a neighbouring square, pick where the apple steps; pressing again changes the pick and
clicking the apple cancels it. When the window closes the step is applied and only then
is TabPFN asked, so the ghost query rows always show the board it actually saw. Moves
made while TabPFN is predicting count for the following turn. Squares under the snake are
not allowed. The header counts how many moves you have evaded; if the
snake goes 50 moves (2 × board cells) without eating, it starves and you win. Apple moves
are logged and saved transitions use the moved apple.

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
| `augmented` | 41: all available features |

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
starvation, and `−0.01` for other moves. Starvation occurs after 2 × board cells moves
without food (50 on the 5 × 5 board); filling the board wins.

TabPFN provides the action-value function; the heuristic is used only for the
explicit baseline player and seed collection. Learned actions are not filtered
through a safety heuristic, and nothing overrides the largest Q value.
This is in-context regression: pretrained weights are never updated.

The hosted adapter explicitly selects `v3.5-fast` and uses `fit_with_cache`.
Set `SNAKE_MODEL_VERSION=v3.5` to use the larger variant. All three candidate actions
are predicted in one request per move. API calls are paced; adjust
`SNAKE_API_INTERVAL` for your account. Fits and predictions consume hosted quota,
and request latency determines TabPFN's playing speed. Errors stop the job and
are displayed; there is no silent substitute model. The status line and the log panel
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

# Compare all three input presets and the baseline.
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
node --check snake_pfn/web/static/app.js snake_pfn/web/static/data-view.js snake_pfn/web/static/log-view.js
```

Tests cover game rules, feature schemas, log round trips, terminal masking,
frozen-target iteration, failed-fit recovery, random steps, frame streaming, the table's
follow-latest mode, and the local API flow/concurrency.
Tests use explicit regression doubles; they do not consume hosted API quota or
establish that TabPFN improves its score. Real model quality needs the evaluation
run above with a valid token.
