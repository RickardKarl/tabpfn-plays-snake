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

from ..game.engine import ACTIONS, DEFAULT_SIZE, Snake, candidate, starvation_limit
from .experience import Store, Transition
from .learner import Learner, stub_enabled
from .policies import heuristic_action

log = logging.getLogger("snake_pfn.runner")
DIRECTION_NAMES = ("up", "right", "down", "left")
PICKS = ("turns left", "goes straight", "turns right")


class Runner:
    def __init__(self, data_dir="data", learner_factory=Learner, size=None, fresh=False):
        self.data_dir = Path(data_dir)
        self.size = int(size or os.getenv("SNAKE_BOARD_SIZE", str(DEFAULT_SIZE)))
        self.store = Store(self.data_dir / "experience.jsonl")
        if fresh and self.store.rows:
            # Every server start begins at the intro; earlier moves are archived, not deleted.
            count = len(self.store.rows)
            log.info("Archived %d saved moves to %s; starting fresh", count, self.store.clear())
        log.info("Loaded %d saved moves from %s", len(self.store.rows), self.store.path)
        self.learner_factory = learner_factory
        self.learner = learner_factory()
        self.env = Snake(0, self.size)
        self.message = "Press play."
        foreign = {row.state.size for row in self.store.rows} - {self.size}
        if foreign:
            # One log holds one board size; keep the old game instead of mixing sizes.
            archive = self.store.clear()
            old = ", ".join(f"{n}×{n}" for n in sorted(foreign))
            self.message = f"Archived saved moves from a {old} board to {archive}."
            log.warning("Board is %dx%d; %s", self.size, self.size, self.message)
        self.episode_id = uuid4().hex
        self.rng = random.Random(42)
        self.episodes = 0  # Finished games this session; every fifth triggers a refit
        self.q_values = None
        self.last_action = None
        self.last_state = None
        self.error = None
        self.job = None
        self.phase = "idle"  # idle · training · predicting · random
        self.practice_moves = 1000  # Random moves TabPFN watches before its first game
        self.practice_seed = 1000  # Seeds those moves and their games; the intro demo keeps 42
        self.pending_query = None  # Rows TabPFN is scoring right now, for late-joining pages
        self.learn_seconds = None  # Wall time of the last fit job, rounds and warm-up included
        self.learn_rows = 0
        # Every board change becomes a frame so the browser can animate each step.
        self.frames = deque(maxlen=600)
        self.seq = 0
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.worker = None
        self._snapshot = {}
        log.info("Board %dx%d, inputs %s, TabPFN token %s%s", self.size, self.size,
                 self.learner.spec.to_dict(), "found" if os.getenv("TABPFN_TOKEN") else "missing",
                 ", STUB MODEL, no TabPFN calls" if stub_enabled() else "")
        self.publish()

    def publish(self):
        with self.lock:
            self._snapshot = {
                "state": self.env.state.to_dict(),
                "seq": self.seq,
                "rows": len(self.store.rows),
                "job": self.job,
                "phase": self.phase,
                "hunger_limit": starvation_limit(self.size),
                "practice_moves": self.practice_moves,
                "practice_seed": self.practice_seed,
                "query": self.pending_query,
                "error": self.error,
                "message": self.message,
                "q_values": self.q_values,
                "last_action": self.last_action,
                "decision_state": self.last_state.to_dict() if self.last_state else None,
                "decision_query": self.query_rows(self.last_state) if self.last_state else None,
                "learn_seconds": self.learn_seconds,
                "learn_rows": self.learn_rows,
                "predict_seconds": self.learner.last_predict_seconds,
                "fitted": self.learner.model is not None,
                "has_token": bool(os.getenv("TABPFN_TOKEN")) or stub_enabled(),
                "model_mode": "stub" if stub_enabled() else "hosted",
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

    def frame(self, state, q_values=None, action=None, reward=None, kind="move", query=None,
              seconds=None):
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
                    "seconds": seconds,  # TabPFN's answer time, on prediction frames
                }
            )

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

    def table(self, offset=0, limit=20, latest=False):
        """One page of saved moves as model inputs and observed rewards. No API calls."""
        with self.lock:
            if latest:
                offset = max(0, ((len(self.store.rows) - 1) // limit) * limit)
            rows = list(self.store.rows[offset : offset + limit])
            spec = self.learner.spec
            return {
                "columns": list(spec.row(self.env.state, 1)),
                "rows": [list(spec.row(row.state, row.action).values()) for row in rows],
                "rewards": [row.reward for row in rows],
                "directions": [candidate(row.state, row.action)[0] for row in rows],
                "total": len(self.store.rows),
                "offset": offset,
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
                    self.message = "Something went wrong. See the error below."
                    log.error("Job %s failed: %s", name, exc, exc_info=True)
                finally:
                    with self.lock:
                        self.pending_query = None
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
        self.frame(self.env.state, kind="reset")

    def clear(self):
        with self.lock:
            if self.job:
                raise ValueError("Pause and wait before clearing memory")
            count = len(self.store.rows)
            archive = self.store.clear()
            log.info("Cleared %d saved moves; archived to %s; fitted model reset", count, archive)
            self.learner = self.learner_factory(spec=self.learner.spec)
            self.q_values = None
            self.last_action = None
            self.last_state = None
            self.message = "Table cleared." if archive else "Table is empty."
            self.error = None
            self.publish()

    def fit(self, rounds, message="Training TabPFN…"):
        self.set_phase("training", message)
        started = time.monotonic()
        log.info("Fitting %d round(s) from %d saved moves with inputs %s",
                 rounds, len(self.store.rows), self.learner.spec.to_dict())
        self.learner.fit(self.store.rows, rounds=rounds, stop=self.stop.is_set,
                         progress=self.set_message)
        if self.learner.model is not None and not self.stop.is_set():
            self.learn_seconds = round(time.monotonic() - started, 1)
            self.learn_rows = self.learner.fit_rows
        self.message = "TabPFN is ready."
        log.info("Model ready: %d fitted Q round(s) on %d rows",
                 self.learner.rounds, self.learner.fit_rows)

    def random_steps(self, moves, delay=0.002, food=None, seed=None):
        """Play many random moves across games quickly; the browser skips frames to keep up.
        `seed` restarts the move sequence and game seeds there, unless the table already
        holds games from that seed on (then it continues them)."""
        start = max((row.seed for row in self.store.rows), default=self.env.seed) + 1
        if seed is not None and start <= seed:
            self.rng = random.Random(seed)
            start = seed
        seed = start
        self.new_game(seed=seed, food=food)
        log.info("Random steps from seed %d", seed)
        self.play(moves, "random", 1, False, delay=delay)

    def play(self, moves, policy, epsilon, auto_fit, stop_after_episode=False, delay=0,
             rounds=3):
        """Play continuous Snake; only terminal states finish an episode. TabPFN moves as
        soon as it answers: the browser holds each prediction on screen by itself."""
        if policy == "tabpfn" and self.learner.model is None:
            # First Play trains from the saved moves; later runs reuse the fitted model.
            self.fit(rounds)
            if self.stop.is_set():
                return
        if policy == "random":
            self.set_phase("random", "Random moves…")
        else:
            self.set_message("TabPFN is playing.")
        log.info("Playing with %s policy, up to %d moves%s", policy, moves,
                 "" if policy == "tabpfn" else f", epsilon {epsilon:.2f}")
        for index in range(moves):
            if self.stop.is_set():
                break
            if policy == "random":
                self.message = f"Random move {index + 1} of {moves}"
            if self.env.state.done:
                self.new_game()
                log.info("New game on seed %d", self.env.seed)
            state = self.env.state
            if policy == "tabpfn":
                # Greedy: TabPFN's highest value always wins. No exploration.
                self.set_phase("predicting")
                self.q_values = self.last_action = self.last_state = None
                self.pending_query = self.query_rows(state)
                self.frame(state, kind="query", query=self.pending_query)
                self.publish()
                q_values = self.learner.values([state])[0].tolist()
                if self.stop.is_set():
                    break
                action = int(np.argmax(q_values))
                self.q_values, self.last_action, self.last_state = q_values, action, state
                self.frame(state, q_values, action, kind="prediction", query=self.pending_query,
                           seconds=self.learner.last_predict_seconds)
                self.set_message(f"TabPFN {PICKS[action]}.")
            else:
                q_values = None
                action = heuristic_action(state, self.rng, 1 if policy == "random" else epsilon)
            if self.stop.is_set():
                break  # A pending network request can finish after Pause.
            self.q_values = q_values
            next_state, reward = self.env.step(action)
            self.frame(next_state, q_values, action, reward=reward)
            self.pending_query = None
            log.debug(
                "Move %d: head %s -> %s (%s)%s, reward %+.2f%s",
                next_state.steps,
                self.square(state.snake[0]),
                DIRECTION_NAMES[next_state.direction],
                ACTIONS[action],
                f", Q {self.format_q(q_values)}" if q_values else "",
                reward,
                f", game over: {next_state.reason}" if next_state.done else "",
            )
            self.last_action, self.last_state = action, state
            self.store.append(
                Transition(
                    state, action, reward, next_state, self.episode_id, self.env.seed, policy
                )
            )
            ended = next_state.done
            if ended:
                self.episodes += 1
                log.info("Episode %d over: %s, %d apples, %d moves, %d saved moves total",
                         self.episodes, next_state.reason, next_state.score,
                         next_state.steps, len(self.store.rows))
            self.publish()
            if ended and auto_fit and policy == "tabpfn" and self.episodes % 5 == 0:
                log.info("Refitting after episode %d", self.episodes)
                self.fit(1, "Learning from that game…")
            if ended and stop_after_episode:
                break
            if delay and self.stop.wait(delay):
                break
        self.pending_query = None
        if self.stop.is_set():
            self.message = "Stopped."
        elif stop_after_episode and self.env.state.done:
            self.message = "Game over."
        else:
            self.message = "Done."
        self.publish()

    @staticmethod
    def square(cell):
        """Chessboard-style name used in the table and on the board, e.g. C3."""
        x, y = cell
        return f"{'ABCDEFGHIJKLMNOP'[x]}{y + 1}"

    @staticmethod
    def format_q(q_values):
        return " ".join(f"{name} {q:+.2f}" for name, q in zip(ACTIONS, q_values))
