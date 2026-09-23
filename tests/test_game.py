import random
from dataclasses import replace

import pytest

from snake_pfn.control.policies import heuristic_action
from snake_pfn.game.engine import DEFAULT_SIZE, STARVATION_MOVES, Snake, State, candidate


def test_seed_reproducibility_and_food_never_on_body():
    left, right = Snake(13), Snake(13)
    rng = random.Random(12)
    for _ in range(300):
        assert left.state == right.state
        assert left.state.food not in left.state.snake
        if left.state.done:
            break
        action = heuristic_action(left.state, rng)
        assert left.step(action) == right.step(action)


def test_default_board_is_small_and_size_is_configurable():
    assert Snake().state.size == DEFAULT_SIZE == 4
    assert Snake(size=4).state.snake == ((2, 2), (1, 2), (0, 2), (0, 1))  # Tail bends up.
    assert Snake(size=8).state.snake == ((4, 4), (3, 4), (2, 4), (1, 4))
    assert Snake(size=16).state.size == 16
    with pytest.raises(ValueError, match="between"):
        Snake(size=3)


def test_eating_and_collision_rewards():
    env = Snake()
    env.state = State(8, ((3, 3), (2, 3), (1, 3)), 1, (4, 3))
    after, reward = env.step(1)
    assert reward == 1 and after.score == 1 and len(after.snake) == 4
    assert after.food not in after.snake
    env.state = replace(after, snake=((7, 3), (6, 3), (5, 3)))
    terminal, reward = env.step(1)
    assert reward == -1 and terminal.done and terminal.reason == "collision"
    with pytest.raises(ValueError, match="ended"):
        env.step(1)


def test_vacating_tail_is_legal_but_body_is_not():
    state = State(8, ((2, 2), (2, 3), (1, 3), (1, 2)), 0, (5, 5))
    assert candidate(state, 0)[3] is False  # Turn left onto the vacating tail.
    env = Snake()
    env.state = state
    after, _ = env.step(0)
    assert not after.done and len(set(after.snake)) == 4
    env.state = State(8, ((2, 2), (2, 3), (1, 3), (1, 2), (1, 1)), 0, (5, 5))
    assert env.step(0)[0].reason == "collision"


def test_starvation_and_food_resets_timer():
    env = Snake()
    limit = STARVATION_MOVES
    env.state = replace(env.state, hungry=limit - 1, food=(0, 0))
    state, reward = env.step(1)
    assert state.reason == "starvation" and reward == -1
    env = Snake()
    ahead = (env.state.snake[0][0] + 1, env.state.snake[0][1])
    env.state = replace(env.state, hungry=limit - 1, food=ahead)
    state, reward = env.step(1)
    assert not state.done and state.hungry == 0 and reward == 1


def test_filling_board_wins():
    env = Snake()
    # A legal serpentine path with the one remaining cell above the head.
    body = tuple((x, y) for y in range(8) for x in (range(8) if y % 2 == 0 else range(7, -1, -1)))
    env.state = State(8, body[1:], 1, body[0])
    # Head (1,0), turn up from south to west via right.
    env.state = replace(env.state, direction=2)
    state, reward = env.step(2)
    assert state.done and state.reason == "board filled" and state.food is None
    assert len(state.snake) == 64 and reward == 1
