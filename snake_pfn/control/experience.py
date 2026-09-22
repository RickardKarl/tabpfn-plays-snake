import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..game.engine import Snake, State
from .policies import heuristic_action


@dataclass(frozen=True)
class Transition:
    state: State
    action: int
    reward: float
    next_state: State
    episode: str
    seed: int
    policy: str

    def to_dict(self):
        return {
            "schema": 1,
            "state": self.state.to_dict(),
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state.to_dict(),
            "done": self.next_state.done,
            "episode": self.episode,
            "seed": self.seed,
            "policy": self.policy,
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") != 1:
            raise ValueError("Unsupported experience schema")
        return cls(
            State.from_dict(data["state"]),
            data["action"],
            data["reward"],
            State.from_dict(data["next_state"]),
            data["episode"],
            data["seed"],
            data["policy"],
        )


class Store:
    def __init__(self, path="data/experience.jsonl"):
        self.path = Path(path)
        self.rows = []
        if self.path.exists():
            for line_number, line in enumerate(self.path.read_text().splitlines(), 1):
                if line.strip():
                    try:
                        self.rows.append(Transition.from_dict(json.loads(line)))
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ValueError(
                            f"Invalid experience at {self.path}:{line_number}"
                        ) from exc

    def append(self, row: Transition):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as file:
            file.write(json.dumps(row.to_dict()) + "\n")
        self.rows.append(row)

    def clear(self):
        """Archive instead of destroying the original experiment."""
        archive = None
        if self.path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            archive = self.path.with_name(f"{self.path.stem}-{stamp}-{uuid4().hex[:6]}.jsonl")
            self.path.rename(archive)
        self.rows.clear()
        return str(archive) if archive else None


def collect(store: Store, episodes=20, seed=0, size=5, stop=lambda: False):
    """Mix whole random episodes with exploratory food-seeking episodes."""
    results = []
    for i in range(episodes):
        env = Snake(seed + i, size)
        rng = random.Random(seed + i + 1_000_000)
        random_policy = i % 4 == 0
        policy = "random" if random_policy else "heuristic-explore"
        episode = uuid4().hex
        while not env.state.done:
            if stop():
                return results
            state = env.state
            action = heuristic_action(state, rng, 1.0 if random_policy else 0.15)
            next_state, reward = env.step(action)
            store.append(Transition(state, action, reward, next_state, episode, seed + i, policy))
        results.append({"seed": seed + i, "score": env.state.score, "steps": env.state.steps})
    return results
