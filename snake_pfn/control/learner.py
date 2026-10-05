"""Fitted Q iteration. Fit changes TabPFN's context, not its pretrained weights."""

import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np

from ..game.engine import Snake
from .features import FeatureSpec
from .policies import heuristic_action

log = logging.getLogger("snake_pfn.tabpfn")


class HostedRegressor:
    def __init__(self, version=None, interval=None):
        from tabpfn_client import TabPFNRegressor
        from tabpfn_client.estimator import ClientOptions

        if not os.getenv("TABPFN_TOKEN"):
            raise ValueError("Set TABPFN_TOKEN in .env and restart the server to use TabPFN.")
        self.version = version or os.getenv("SNAKE_MODEL_VERSION", "v3.5-fast")
        self.model = TabPFNRegressor.create_default_for_version(self.version)
        # One request may not stall a game: the client retries once, then the job errors out.
        timeout = float(os.getenv("SNAKE_API_TIMEOUT", "20"))
        self.model.set_params(fit_mode="fit_with_cache", client_options=ClientOptions(timeout=timeout))
        self.interval = (
            float(os.getenv("SNAKE_API_INTERVAL", "0")) if interval is None else interval
        )
        self.last_call = 0.0

    def _pace(self):
        wait = max(0, self.interval - (time.monotonic() - self.last_call))
        if wait > 0.05:
            log.debug("Waiting %.2f s before the next API call", wait)
        time.sleep(wait)
        self.last_call = time.monotonic()

    def fit(self, x, y):
        self._pace()
        started = time.monotonic()
        log.info("API fit (%s): %d rows x %d columns", self.version, len(x), x.shape[1])
        self.model.fit(x, y)
        log.info("API fit done in %.2f s", time.monotonic() - started)
        return self

    def predict(self, x):
        self._pace()
        started = time.monotonic()
        values = np.asarray(self.model.predict(x), dtype=float).reshape(-1)
        log.debug("API predict: %d rows in %.2f s", len(x), time.monotonic() - started)
        if len(values) != len(x) or not np.isfinite(values).all():
            raise ValueError("TabPFN returned invalid action values")
        return values


class StubRegressor:
    """Debug stand-in for the hosted API: sleeps like it, answers with random values.

    Enabled with `snake-pfn serve --stub` or SNAKE_STUB_MODEL=1. Nothing leaves the machine,
    so the UI choreography can be exercised without spending TabPFN quota.
    """

    def __init__(self):
        self.fit_seconds = float(os.getenv("SNAKE_STUB_FIT_SECONDS", "1.5"))
        self.predict_seconds = float(os.getenv("SNAKE_STUB_PREDICT_SECONDS", "1.0"))
        self.rng = np.random.default_rng()

    def fit(self, x, y):
        log.info("Stub fit: %d rows x %d columns, sleeping %.1f s", len(x), x.shape[1],
                 self.fit_seconds)
        time.sleep(self.fit_seconds)
        return self

    def predict(self, x):
        time.sleep(self.predict_seconds)
        log.debug("Stub predict: %d rows, random values", len(x))
        return self.rng.uniform(-0.6, 0.7, len(x))


def stub_enabled():
    return os.getenv("SNAKE_STUB_MODEL", "").lower() in ("1", "true", "yes")


def default_factory():
    return StubRegressor if stub_enabled() else HostedRegressor


def bellman_targets(rows, previous, spec, gamma=0.95):
    targets = np.array([r.reward for r in rows], dtype=float)
    live = [i for i, row in enumerate(rows) if not row.next_state.done]
    if previous is not None and live:
        log.info("Bootstrapping targets: predicting %d next-state candidates for %d live rows",
                 3 * len(live), len(live))
        frame = spec.frame((rows[i].next_state, a) for i in live for a in range(3))
        # Split large contexts to keep request sizes bounded, preserving row order.
        predictions = np.concatenate(
            [previous.predict(frame.iloc[i : i + 3000]) for i in range(0, len(frame), 3000)]
        )
        targets[live] += gamma * predictions.reshape(-1, 3).max(axis=1)
    return targets


class Learner:
    def __init__(self, spec=None, factory=None, max_rows=None, gamma=0.95):
        self.spec = spec or FeatureSpec()
        self.factory = factory or default_factory()
        self.max_rows = max_rows
        self.gamma = gamma
        self.model = None
        self.rounds = 0
        self.fit_rows = 0
        self.last_predict_seconds = None

    def fit(self, experience, rounds=1, seed=42, stop=lambda: False, progress=lambda text: None):
        if len(experience) < 10:
            raise ValueError("Collect at least 10 transitions first")
        if len({r.state.size for r in experience}) != 1:
            raise ValueError("A training log must contain one board size")
        experience = list(experience)
        rows = (
            random.Random(seed).sample(experience, min(len(experience), self.max_rows))
            if self.max_rows
            else experience  # Whole log: every saved move is in TabPFN's context.
        )
        x = self.spec.frame((r.state, r.action) for r in rows)
        log.info("Sampled %d of %d saved moves -> %d columns: %s",
                 len(rows), len(experience), x.shape[1], ", ".join(x.columns))
        total = self.rounds + rounds
        for _ in range(rounds):
            if stop():
                log.info("Fit stopped before round %d", self.rounds + 1)
                break
            started = time.monotonic()
            progress(f"Training · round {self.rounds + 1} of {total}…")
            y = bellman_targets(rows, self.model, self.spec, self.gamma)
            log.info("Round %d targets: mean %+.3f, min %+.3f, max %+.3f",
                     self.rounds + 1, y.mean(), y.min(), y.max())
            if stop():
                break
            # Always create a fresh estimator. The previous Q stays frozen for targets.
            candidate_model = self.factory()
            candidate_model.fit(x, y)
            self.model = candidate_model  # Publish only a successful complete fit.
            self.rounds += 1
            self.fit_rows = len(rows)
            log.info("Round %d fitted in %.2f s", self.rounds, time.monotonic() - started)
        if self.model is not None and not stop():
            # The first prediction against a fresh context pays its setup cost; pay it here,
            # while the player reads the rules, rather than on the first move of the game.
            progress("Training · warming up…")
            self.model.predict(x.iloc[:3])

    def values(self, states):
        if self.model is None:
            raise ValueError("Fit TabPFN before selecting the TabPFN player")
        states = list(states)
        frame = self.spec.frame((state, action) for state in states for action in range(3))
        started = time.monotonic()
        values = np.asarray(self.model.predict(frame)).reshape(-1, 3)
        self.last_predict_seconds = round(time.monotonic() - started, 2)
        return values

    def save(self, path):
        if not isinstance(self.model, HostedRegressor):
            raise ValueError("No fitted hosted model to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.model.save_model(str(path))
        path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "features": self.spec.to_dict(),
                    "rounds": self.rounds,
                    "fit_rows": self.fit_rows,
                    "gamma": self.gamma,
                    "max_rows": self.max_rows,
                    "version": self.model.version,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path):
        from tabpfn_client import TabPFNRegressor

        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        spec = FeatureSpec(tuple(meta["features"]["groups"]), tuple(meta["features"]["exclude"]))
        learner = cls(spec, max_rows=meta["max_rows"], gamma=meta["gamma"])
        learner.model = HostedRegressor(meta["version"])
        learner.model.model = TabPFNRegressor.load_model(str(path))
        learner.rounds = meta["rounds"]
        learner.fit_rows = meta["fit_rows"]
        return learner


def evaluate(learner=None, episodes=10, seed=10000, size=5):
    """Frozen policy, no exploration or logging; batch live environments per tick."""
    envs = [Snake(seed + i, size) for i in range(episodes)]
    rngs = [random.Random(seed + i) for i in range(episodes)]
    while live := [i for i, env in enumerate(envs) if not env.state.done]:
        if learner is not None:
            values = learner.values([envs[i].state for i in live])
            actions = [int(np.argmax(q)) for q in values]
        else:
            actions = [heuristic_action(envs[i].state, rngs[i]) for i in live]
        for i, action in zip(live, actions):
            envs[i].step(action)
    games = [
        {
            "seed": env.seed,
            "score": env.state.score,
            "steps": env.state.steps,
            "reason": env.state.reason,
        }
        for env in envs
    ]
    return {
        "policy": "tabpfn" if learner else "heuristic",
        "games": games,
        "mean_apples": float(np.mean([g["score"] for g in games])),
        "mean_steps": float(np.mean([g["steps"] for g in games])),
    }
