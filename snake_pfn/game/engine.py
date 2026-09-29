"""Small deterministic Snake environment. Coordinates are (x, y), head first."""

import random
from dataclasses import asdict, dataclass

ACTIONS = ("left", "straight", "right")
DEFAULT_SIZE = 5
MIN_SIZE, MAX_SIZE = 4, 16
START_LENGTH = 4
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
TURNS = (-1, 0, 1)


def starvation_limit(size: int) -> int:
    """Bound fruitless loops to two board traversals, as in the original experiment."""
    return size * size * 2


@dataclass(frozen=True)
class State:
    size: int
    snake: tuple[tuple[int, int], ...]
    direction: int
    food: tuple[int, int] | None
    hungry: int = 0
    steps: int = 0
    score: int = 0
    done: bool = False
    reason: str | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(
            **{
                **data,
                "snake": tuple(map(tuple, data["snake"])),
                "food": tuple(data["food"]) if data["food"] is not None else None,
            }
        )


def candidate(state: State, action: int):
    """Preview a move without spawning food or touching the environment RNG."""
    if action not in range(3):
        raise ValueError("Action must be 0 (left), 1 (straight), or 2 (right)")
    direction = (state.direction + TURNS[action]) % 4
    dx, dy = DIRECTIONS[direction]
    x, y = state.snake[0]
    head = (x + dx, y + dy)
    eating = head == state.food
    occupied = state.snake if eating else state.snake[:-1]
    collision = not (0 <= head[0] < state.size and 0 <= head[1] < state.size)
    collision = collision or head in occupied
    body = (head,) + (state.snake if eating else state.snake[:-1])
    return direction, head, eating, collision, body


class Snake:
    def __init__(self, seed: int = 0, size: int = DEFAULT_SIZE):
        if not MIN_SIZE <= size <= MAX_SIZE:
            raise ValueError(f"Board size must be between {MIN_SIZE} and {MAX_SIZE}")
        self.seed = seed
        self.rng = random.Random(seed)
        mid = size // 2
        # Head in the middle row heading right; the body runs left and, if the row is too
        # short, the tail bends upward along the left edge.
        body = tuple(
            (mid - i, mid) if mid - i >= 0 else (0, mid - (i - mid)) for i in range(START_LENGTH)
        )
        self.state = State(size, body, 1, self._food(size, body))

    def _food(self, size, body):
        free = [(x, y) for y in range(size) for x in range(size) if (x, y) not in body]
        return self.rng.choice(free) if free else None

    def step(self, action: int):
        s = self.state
        if s.done:
            raise ValueError("Episode has ended")
        direction, _, eating, collision, body = candidate(s, action)
        hungry = 0 if eating else s.hungry + 1
        reason = "collision" if collision else None
        food = s.food
        if collision:
            body = s.snake  # Keep the final board valid for rendering.
        elif eating:
            food = self._food(s.size, body)
            if food is None:
                reason = "board filled"
        if reason is None and hungry >= starvation_limit(s.size):
            reason = "starvation"
        reward = -1.0 if reason in ("collision", "starvation") else (1.0 if eating else -0.01)
        self.state = State(
            s.size,
            body,
            direction,
            food,
            hungry,
            s.steps + 1,
            s.score + int(eating and not collision),
            bool(reason),
            reason,
        )
        return self.state, reward
