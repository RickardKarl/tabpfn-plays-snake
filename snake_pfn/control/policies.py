"""Baseline policies for collection and comparison."""

import random

from ..game.engine import State, candidate


def heuristic_action(state: State, rng: random.Random, explore: float = 0.0):
    if rng.random() < explore:
        return rng.randrange(3)
    choices = []
    for action in range(3):
        _, head, _, collision, _ = candidate(state, action)
        distance = sum(abs(a - b) for a, b in zip(head, state.food or head))
        choices.append((-1000 if collision else 0) - distance)
    best = max(choices)
    return rng.choice([i for i, value in enumerate(choices) if value == best])
