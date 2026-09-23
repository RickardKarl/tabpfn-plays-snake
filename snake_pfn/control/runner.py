"""One worker owns gameplay, logging, and fits. UI reads published snapshots."""

import logging
import os
import random
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..game.engine import ACTIONS, DEFAULT_SIZE, STARVATION_MOVES, Snake, candidate
from .experience import Store, Transition, collect
from .features import GROUPS, PRESETS, all_features
from .learner import Learner, stub_enabled
from .log import EventLog
from .policies import heuristic_action

log = logging.getLogger("snake_pfn.runner")
DIRECTION_NAMES = ("up", "right", "down", "left")


class Runner:
    def __init__(self, data_dir="data", learner_factory=Learner, size=None):
        self.data_dir = Path(data_dir)
        self.events = EventLog()
        self.size = int(size or os.getenv("SNAKE_BOARD_SIZE", str(DEFAULT_SIZE)))
        self.store = Store(self.data_dir / "experience.jsonl")
        log.info("Loaded %d saved moves from %s", len(self.store.rows), self.store.path)
        self.learner_factory = learner_factory
        self.learner = learner_factory()
        self.env = Snake(0, self.size)
        self.message = "Press Play."
        foreign = {row.state.size for row in self.store.rows} - {self.size}
        if foreign:
            # One log holds one board size; keep the old game instead of mixing sizes.
            archive = self.store.clear()
            old = ", ".join(f"{n}×{n}" for n in sorted(foreign))
            self.message = f"Archived saved moves from a {old} board to {archive}."
            log.warning("Board is %d×%d; %s", self.size, self.size, self.message)
        self.episode_id = uuid4().hex
        self.rng = random.Random(42)
        self.episodes = 0
        self.history = []
        self.q_values = None
        self.last_action = None
        self.last_state = None
        self.error = None
        self.job = None
        self.phase = "idle"  # idle · training · waiting (player's turn) · predicting · random
        self.play_reveal = 0.5  # Seconds the predicted values stay on screen before the move
        self.practice_moves = 1000  # Random moves TabPFN watches before its first game
        self.pending_query = None  # Rows TabPFN is scoring right now, for late-joining pages
        self.pending_food = None  # Where the player wants the apple next; applied between moves
        self.apple_moves = 0  # Player moves in the current game
        self.outcome = None  # Versus game result: eaten · starved · crashed
        self.turn_window = 1.5  # Seconds after each snake move for the player to (re)choose
        # Every board change becomes a frame so the browser can animate each step.
        self.frames = deque(maxlen=600)
        self.seq = 0
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.worker = None
        self._snapshot = {}
        log.info("Board %d×%d · inputs %s · TabPFN token %s%s", self.size, self.size,
                 self.learner.spec.to_dict(), "found" if os.getenv("TABPFN_TOKEN") else "missing",
                 " · STUB MODEL, no TabPFN calls" if stub_enabled() else "")
        self.publish()

    def publish(self):
        with self.lock:
            self._snapshot = {
                "state": self.env.state.to_dict(),
                "seq": self.seq,
                "episodes": self.episodes,
                "history": self.history[-30:],
                "rows": len(self.store.rows),
                "job": self.job,
                "phase": self.phase,
                "hunger_limit": STARVATION_MOVES,
                "turn_window": self.turn_window,
                "practice_moves": self.practice_moves,
                "query": self.pending_query,
                "pending_food": self.pending_food,
                "apple_moves": self.apple_moves,
                "outcome": self.outcome,
                "error": self.error,
                "message": self.message,
                "q_values": self.q_values,
                "last_action": self.last_action,
                "decision_state": self.last_state.to_dict() if self.last_state else None,
                "features": self.learner.spec.to_dict(),
                "columns": list(self.learner.spec.row(self.env.state, 1)),
                "rounds": self.learner.rounds,
                "fit_rows": self.learner.fit_rows,
                "fit_seconds": self.learner.last_fit_seconds,
                "fitted": self.learner.model is not None,
                "has_token": bool(os.getenv("TABPFN_TOKEN")) or stub_enabled(),
                "model_mode": "stub" if stub_enabled() else "hosted",
                "model_version": "stub" if stub_enabled() else os.getenv("SNAKE_MODEL_VERSION", "v3.5-fast"),
            }

    def snapshot(self):
        with self.lock:
            return dict(self._snapshot)

    def set_message(self, text):
        self.message = text
        self.publish()

    def set_phase(self, phase, message=None):
        """What TabPFN is doing right now, for the pipeline indicator in the UI."""
        self.phase = phase
        if message is not None:
            self.message = message
        self.publish()

    def frame(self, state, q_values=None, action=None, reward=None, kind="move", query=None):
        """kind: reset (new board) · query (rows sent to TabPFN, answer pending) ·
        prediction (values known, snake not moved yet) · move (snake moved)."""
        with self.lock:
            self.seq += 1
            self.frames.append(
                {
                    "seq": self.seq,
                    "kind": kind,
                    "state": state.to_dict(),
                    "q_values": q_values,
                    "action": action,
                    "reward": reward,
                    "query": query,
                }
            )

    def move_apple(self, cell):
        """Player move: the apple steps to a neighbouring free square before the snake's next move."""
        with self.lock:
            state = self.env.state
            if self.phase not in ("waiting", "predicting", "random") or state.done or state.food is None:
                raise ValueError("You can move the apple only while a game is running")
            x, y = cell
            fx, fy = state.food
            if (x, y) == (fx, fy):  # Changed your mind: stay put.
                self.pending_food = None
                self.publish()
                return
            neighbour = abs(x - fx) + abs(y - fy) == 1
            free = 0 <= x < state.size and 0 <= y < state.size and (x, y) not in state.snake
            if not (neighbour and free):
                return  # Impossible step: ignore it, the board already shows where the apple can go.
            self.pending_food = (x, y)
            self.publish()

    def apply_apple_move(self):
        with self.lock:
            target = self.pending_food
            self.pending_food = None
            state = self.env.state
            if target is None or state.done or target in state.snake:
                return state
            self.env.state = state = replace(state, food=target)
            self.apple_moves += 1
            self.frame(state, kind="apple")
            log.info("Apple moved to %s by the player", self.square(target))
            return state

    def query_rows(self, state):
        """The exact three input rows TabPFN is about to score, one per candidate turn."""
        spec = self.learner.spec
        return {
            "columns": list(spec.row(state, 1)),
            "rows": [list(spec.row(state, action).values()) for action in range(3)],
            "directions": [candidate(state, action)[0] for action in range(3)],
        }

    def frames_since(self, after=0, limit=600):
        with self.lock:
            frames = [f for f in self.frames if f["seq"] > after][:limit]
            return {"frames": frames, "latest": self.seq}

    def catalog(self):
        sample = all_features(Snake(0, self.size).state, 1)
        return {
            "groups": [
                {"id": key, "description": text, "columns": list(sample[key])}
                for key, text in GROUPS.items()
            ],
            "presets": PRESETS,
        }

    def table(self, offset=0, limit=20, source="auto", latest=False):
        """Read captured fit inputs without making any model/API calls."""
        with self.lock:
            learner = self.learner
            fitted = learner.fit_table
            prediction = learner.prediction_table
            if fitted is not None and source != "experience":
                if latest:
                    offset = max(0, ((len(fitted["rows"]) - 1) // limit) * limit)
                return {
                    "source": "fit",
                    "columns": fitted["columns"],
                    "rows": fitted["rows"][offset : offset + limit],
                    "targets": fitted["targets"][offset : offset + limit],
                    "rewards": fitted["rewards"][offset : offset + limit],
                    "directions": fitted["directions"][offset : offset + limit],
                    "total": len(fitted["rows"]),
                    "offset": offset,
                    "round": fitted["round"],
                    "prediction": prediction,
                    "latest_index": None,
                }
            # Before the first fit, show saved observations, explicitly as a preview.
            if latest:
                offset = max(0, ((len(self.store.rows) - 1) // limit) * limit)
            rows = list(self.store.rows[offset : offset + limit])
            spec = learner.spec
            columns = list(spec.row(self.env.state, 1))
            return {
                "source": "preview",
                "columns": columns,
                "rows": [list(spec.row(row.state, row.action).values()) for row in rows],
                "targets": [None] * len(rows),
                "rewards": [row.reward for row in rows],
                "directions": [candidate(row.state, row.action)[0] for row in rows],
                "total": len(self.store.rows),
                "offset": offset,
                "round": 0,
                "prediction": prediction,
                "latest_index": len(self.store.rows) - 1 if self.store.rows else None,
            }

    def launch(self, name, work):
        with self.lock:
            if self.job is not None:
                raise ValueError("Pause and wait for the current operation to finish first")
            self.job = name
            self.error = None
            self.stop.clear()
            self.publish()
            log.info("Job started: %s", name)

            def task():
                started = time.monotonic()
                try:
                    work()
                except Exception as exc:
                    self.error = str(exc)
                    # Never silently switch to a different player after an API error.
                    self.message = "Stopped. Resolve the error and retry."
                    log.error("Job %s failed: %s", name, exc, exc_info=True)
                finally:
                    with self.lock:
                        self.job = None
                        self.phase = "idle"
                        self.publish()
                    log.info("Job finished: %s in %.1f s", name, time.monotonic() - started)

            self.worker = threading.Thread(target=task, daemon=True)
            self.worker.start()

    def pause(self):
        if self.job:
            log.info("Stop requested during %s", self.job)
        self.stop.set()

    def configure(self, spec):
        with self.lock:
            if self.job:
                raise ValueError("Pause and wait before changing inputs")
            spec.row(self.env.state, 1)  # Validate before replacing any fitted state.
            self.learner = self.learner_factory(spec=spec)
            self.q_values = None
            self.last_action = None
            self.last_state = None
            self.message = "Inputs changed. Experience kept; fit a new model to use these columns."
            self.error = None
            log.info("Inputs changed to %s → %d columns; fitted model reset",
                     spec.to_dict(), len(spec.row(self.env.state, 1)))
            self.publish()

    def reset(self, food=None):
        with self.lock:
            if self.job:
                raise ValueError("Pause and wait before resetting")
            self.new_game(food=food)
            self.publish()

    def new_game(self, seed=None, food=None):
        """Fresh board. `food` places the first apple on a chosen free square (the intro
        uses A1) instead of the seeded random one."""
        self.env = Snake(self.env.seed + 1 if seed is None else seed, self.size)
        if food is not None:
            cell = tuple(food)
            state = self.env.state
            if 0 <= cell[0] < self.size and 0 <= cell[1] < self.size and cell not in state.snake:
                self.env.state = replace(state, food=cell)
        self.episode_id = uuid4().hex
        self.q_values = None
        self.last_action = None
        self.last_state = None
        self.pending_food = None
        self.apple_moves = 0
        self.outcome = None
        self.frame(self.env.state, kind="reset")

    def clear(self):
        with self.lock:
            if self.job:
                raise ValueError("Pause and wait before clearing memory")
            count = len(self.store.rows)
            archive = self.store.clear()
            log.info("Cleared %d saved moves; archived to %s; fitted model reset", count, archive)
            self.learner = self.learner_factory(spec=self.learner.spec)
            self.history = []
            self.q_values = None
            self.last_action = None
            self.last_state = None
            self.message = f"Memory archived to {archive}" if archive else "Memory is empty."
            self.error = None
            self.publish()

    def collect(self, episodes):
        # Never restart collection on the same seeds after a process restart.
        seed = max((r.seed for r in self.store.rows), default=99) + 1
        log.info("Collecting %d seed games from seed %d", episodes, seed)
        games = collect(self.store, episodes, seed, self.size, stop=self.stop.is_set)
        self.message = f"Collected {len(games)} episodes. Full states saved to disk."
        log.info("Collected %d games: %s", len(games), games)

    def fit(self, rounds, message="Training TabPFN on the saved moves…"):
        self.set_phase("training", message)
        log.info("Fitting %d round(s) from %d saved moves with inputs %s",
                 rounds, len(self.store.rows), self.learner.spec.to_dict())
        self.learner.fit(self.store.rows, rounds=rounds, stop=self.stop.is_set,
                         progress=self.set_message)
        self.message = f"Model ready: {self.learner.rounds} fitted Q rounds."
        log.info("Model ready: %d fitted Q round(s) on %d rows",
                 self.learner.rounds, self.learner.fit_rows)

    def random_steps(self, moves, delay=0.002, food=None):
        """Play many random moves across games quickly; the browser skips frames to keep up."""
        seed = max((row.seed for row in self.store.rows), default=self.env.seed) + 1
        self.new_game(seed=seed, food=food)
        log.info("Random steps from seed %d", seed)
        self.play(moves, "random", 1, False, delay=delay)

    def play(self, moves, policy, epsilon, auto_fit, stop_after_episode=False, delay=0.12,
             rounds=3, reveal=None, turn=None, versus=False):
        """versus: one game against the apple player; it ends when the snake eats, crashes,
        or starves, and the outcome is published for the board."""
        reveal = self.play_reveal if reveal is None else reveal
        turn = self.turn_window if turn is None else turn
        if versus and (self.env.state.steps or self.env.state.done or self.outcome):
            self.new_game()  # A versus game always starts on a fresh board.
        self.outcome = None
        if policy == "tabpfn" and self.learner.model is None:
            # First Play trains from the saved moves; later runs reuse the fitted model.
            self.fit(rounds)
            if self.stop.is_set():
                return
        if policy == "random":
            self.set_phase("random", f"Playing {moves} random steps…")
        else:
            self.set_message(f"Playing with {policy} · up to {moves} moves in this run.")
        log.info("Playing with %s policy · up to %d moves%s", policy, moves,
                 "" if policy == "tabpfn" else f" · epsilon {epsilon:.2f}")
        for index in range(moves):
            if self.stop.is_set():
                break
            if policy == "random":
                self.message = f"Playing random steps · {index + 1} of {moves}"
            if self.env.state.done:
                self.new_game()
                log.info("New game on seed %d", self.env.seed)
            if policy == "tabpfn" and turn:
                # The player's turn: pick, change, or cancel the apple's step until TabPFN is asked.
                self.set_phase("waiting", "Your move: arrow keys step the apple. "
                               "Change your mind until TabPFN starts predicting.")
                if self.stop.wait(turn):
                    break
            state = self.apply_apple_move()
            if policy == "tabpfn":
                # Greedy: TabPFN's highest value always wins. No exploration.
                self.set_phase("predicting", f"TabPFN is predicting the value of move {state.steps + 1}…")
                self.pending_query = self.query_rows(state)
                self.frame(state, kind="query", query=self.pending_query)
                self.publish()
                q_values = self.learner.values([state])[0].tolist()
                action = int(np.argmax(q_values))
                # Show the values on the still board, then move. The browser holds too.
                self.frame(state, q_values, action, kind="prediction")
                self.set_message(
                    f"Move {state.steps + 1}: TabPFN picks {ACTIONS[action]} · Q {self.format_q(q_values)}"
                )
                if self.stop.wait(reveal):
                    break
            else:
                q_values = None
                action = heuristic_action(state, self.rng, 1 if policy == "random" else epsilon)
            if self.stop.is_set():
                break  # A pending network request can finish after Pause.
            self.q_values = q_values
            next_state, reward = self.env.step(action)
            self.frame(next_state, q_values, action, reward=reward)
            self.pending_query = None
            if policy == "tabpfn":
                self.message = (
                    f"Move {next_state.steps}: {DIRECTION_NAMES[next_state.direction]}"
                    f" · Q {self.format_q(q_values)}"
                )
            log.info(
                "Move %d: head %s → %s (%s)%s · reward %+.2f%s",
                next_state.steps,
                self.square(state.snake[0]),
                DIRECTION_NAMES[next_state.direction],
                ACTIONS[action],
                f" · Q {self.format_q(q_values)}" if q_values else "",
                reward,
                f" · game over: {next_state.reason}" if next_state.done else "",
            )
            self.last_action, self.last_state = action, state
            self.store.append(
                Transition(
                    state, action, reward, next_state, self.episode_id, self.env.seed, policy
                )
            )
            eaten = reward == 1 and not next_state.done
            if versus and eaten:
                self.outcome = "eaten"
            elif versus and next_state.done:
                self.outcome = "starved" if next_state.reason == "starvation" else "crashed"
            ended = next_state.done or eaten  # Eating ends the game in every mode.
            if ended:
                self.episodes += 1
                self.history.append(
                    {
                        "episode": self.episodes,
                        "score": next_state.score,
                        "steps": next_state.steps,
                        "policy": policy,
                        "outcome": self.outcome,
                    }
                )
                log.info("Episode %d over: %s · %d apples · %d moves · %d saved moves total%s",
                         self.episodes, next_state.reason, next_state.score,
                         next_state.steps, len(self.store.rows),
                         f" · player moved the apple {self.apple_moves} times" if self.apple_moves else "")
            self.publish()
            if ended and auto_fit and policy == "tabpfn" and self.episodes % 5 == 0:
                log.info("Refitting after episode %d", self.episodes)
                self.fit(1, "Learning from that game…")
            if ended and (stop_after_episode or versus):
                break
            if eaten:
                self.new_game()
                log.info("The snake ate: new game on seed %d", self.env.seed)
            if self.stop.wait(delay):
                break
        if self.stop.is_set():
            self.message = "Paused."
        elif self.outcome:
            self.message = {
                "eaten": f"Game over: TabPFN ate the apple after {self.moves_text()}.",
                "starved": "You win: the snake starved.",
                "crashed": "You win: the snake crashed.",
            }[self.outcome]
        else:
            self.message = "Run finished. Experience saved."
        self.publish()

    def moves_text(self):
        steps = self.env.state.steps
        return f"{steps} move{'' if steps == 1 else 's'}"

    @staticmethod
    def square(cell):
        """Chessboard-style name used in the table and on the board, e.g. C3."""
        x, y = cell
        return f"{'ABCDEFGHIJKLMNOP'[x]}{y + 1}"

    @staticmethod
    def format_q(q_values):
        return " ".join(f"{name} {q:+.2f}" for name, q in zip(ACTIONS, q_values))
