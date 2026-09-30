"""Only this module decides what the model sees; logs always retain full states."""

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..game.engine import DIRECTIONS, State, candidate, starvation_limit

GROUPS = {
    "board": "Ordered board: empty 0, food -1, tail 1 … head N",
    "context": "Heading, snake length, and time since eating",
    "food": "Food position relative to the head and candidate move",
    "danger": "Immediate collision in each direction and for this move",
    "space": "Reachable space after the candidate move (static-body approximation)",
    "outcome": "What this move leads to: a crash, the apple, closer to it, and room left",
}
PRESETS = {
    "board": ("board", "context"),
    "compact": ("context", "food", "danger", "space"),
    "augmented": ("board", "context", "food", "danger", "space"),
    "outcomes": ("outcome",),
    "board_outcomes": ("board", "outcome"),
}


def reachable(size, head, body):
    blocked = set(body[1:])
    seen = {head}
    queue = deque([head])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            cell = (x + dx, y + dy)
            if (
                0 <= cell[0] < size
                and 0 <= cell[1] < size
                and cell not in blocked
                and cell not in seen
            ):
                seen.add(cell)
                queue.append(cell)
    return len(seen)


def delta_distance(food, head, next_head):
    """+1 if the move steps toward the apple, −1 if away, 0 when there is no apple."""
    if food is None:
        return 0
    before = abs(food[0] - head[0]) + abs(food[1] - head[1])
    after = abs(food[0] - next_head[0]) + abs(food[1] - next_head[1])
    return before - after


def all_features(state: State, action: int):
    """Derive ALL candidate features from pre-action information only."""
    size = state.size
    board = {f"cell_{y}_{x}": 0.0 for y in range(size) for x in range(size)}
    if state.food is not None:
        x, y = state.food
        board[f"cell_{y}_{x}"] = -1.0
    for rank, (x, y) in enumerate(reversed(state.snake), 1):
        board[f"cell_{y}_{x}"] = float(rank)
    x, y = state.snake[0]
    fx, fy = state.food or (x, y)
    forward = DIRECTIONS[state.direction]
    right = DIRECTIONS[(state.direction + 1) % 4]
    delta = (fx - x, fy - y)
    _, head, eating, collision, body = candidate(state, action)
    free = 0 if collision else reachable(size, head, body)
    return {
        "board": board,
        "context": {
            "heading_x": forward[0],
            "heading_y": forward[1],
            "length": len(state.snake),
            "hunger_fraction": state.hungry / starvation_limit(state.size),
        },
        "food": {
            "food_forward": sum(a * b for a, b in zip(delta, forward)),
            "food_right": sum(a * b for a, b in zip(delta, right)),
            "food_distance": abs(fx - x) + abs(fy - y),
            "next_food_distance": abs(fx - head[0]) + abs(fy - head[1]),
            "will_eat": int(eating),
        },
        "danger": {
            "danger_left": int(candidate(state, 0)[3]),
            "danger_straight": int(candidate(state, 1)[3]),
            "danger_right": int(candidate(state, 2)[3]),
            "will_collide": int(collision),
        },
        "space": {"reachable_cells": free, "space_per_segment": free / len(body)},
        # Four plain questions about the move; will_eat is shared with "food".
        "outcome": {
            "will_crash": int(collision),
            "will_eat": int(eating),
            "closer_to_apple": delta_distance(state.food, (x, y), head),
            "room_left": free,
        },
    }


@dataclass(frozen=True)
class FeatureSpec:
    groups: tuple[str, ...] = PRESETS["board_outcomes"]
    exclude: tuple[str, ...] = ()

    def __post_init__(self):
        unknown = set(self.groups) - GROUPS.keys()
        if unknown or not self.groups or len(set(self.groups)) != len(self.groups):
            raise ValueError(f"Select unique feature groups from {list(GROUPS)}; got {self.groups}")
        if "action" in self.exclude:
            raise ValueError("The candidate action is always required")

    @classmethod
    def read(cls, name_or_path: str):
        if name_or_path in PRESETS:
            return cls(PRESETS[name_or_path])
        data = json.loads(Path(name_or_path).read_text())
        return cls(tuple(data["groups"]), tuple(data.get("exclude", [])))

    def to_dict(self):
        return {"groups": list(self.groups), "exclude": list(self.exclude)}

    def row(self, state: State, action: int):
        values = all_features(state, action)
        known = {key for group in values.values() for key in group}
        if set(self.exclude) - known:
            raise ValueError(f"Unknown excluded columns: {sorted(set(self.exclude) - known)}")
        row = {"action": action}
        # Registry order is canonical, independent of checkbox order.
        for group in GROUPS:
            if group in self.groups:
                row.update({k: v for k, v in values[group].items() if k not in self.exclude})
        if len(row) == 1:
            raise ValueError("Select at least one state feature")
        return row

    def frame(self, pairs):
        rows = [self.row(s, a) for s, a in pairs]
        if not rows:
            raise ValueError("Cannot encode an empty dataset")
        frame = pd.DataFrame(rows, dtype=float)
        if frame.isna().any().any():
            raise ValueError("Mixed board sizes are not supported in one fit")
        return frame
