import json
from dataclasses import replace

import numpy as np
import pytest

from snake_pfn.control.experience import Store, Transition, collect
from snake_pfn.control.features import FeatureSpec, all_features
from snake_pfn.control.learner import HostedRegressor, Learner, bellman_targets, evaluate
from snake_pfn.game.engine import Snake


class ActionRegressor:
    """Test oracle only; never used as a product backend."""

    def fit(self, x, y):
        self.x, self.y = x.copy(), np.array(y)
        return self

    def predict(self, x):
        return x["action"].to_numpy() * 2.0


def test_feature_presets_and_individual_columns():
    state = Snake(size=5).state
    assert len(FeatureSpec.read("board").row(state, 1)) == 30  # 25 cells + 4 context + action
    assert len(FeatureSpec.read("compact").row(state, 1)) == 16
    assert len(FeatureSpec.read("augmented").row(state, 1)) == 41
    assert len(FeatureSpec.read("board").row(Snake(size=4).state, 1)) == 21  # 16 cells on 4 × 4
    assert list(FeatureSpec.read("outcomes").row(Snake(size=4).state, 1)) == [
        "action", "will_crash", "will_eat", "closer_to_apple", "room_left"
    ]
    assert len(FeatureSpec.read("board_outcomes").row(Snake(size=4).state, 1)) == 21
    spec = FeatureSpec(("food", "context"), ("will_eat",))
    row = spec.row(state, 1)
    assert "will_eat" not in row and "cell_0_0" not in row and row["action"] == 1
    assert list(row) == list(FeatureSpec(("context", "food"), ("will_eat",)).row(state, 0))
    with pytest.raises(ValueError, match="always required"):
        FeatureSpec(exclude=("action",))
    with pytest.raises(ValueError, match="Unknown excluded"):
        FeatureSpec(exclude=("typo",)).row(state, 0)


def test_outcome_features_describe_the_candidate_move():
    # 4 × 4 start: head (2, 2) heading right, body (1, 2), (0, 2), (0, 1).
    state = replace(Snake(size=4).state, food=(3, 2))
    straight = all_features(state, 1)["outcome"]
    assert straight == {"will_crash": 0, "will_eat": 1, "closer_to_apple": 1,
                        "room_left": straight["room_left"]}
    assert straight["room_left"] > 0
    left = all_features(state, 0)["outcome"]  # Up, away from the apple.
    assert left["closer_to_apple"] == -1 and left["will_eat"] == 0
    edge = replace(state, snake=((3, 2), (2, 2), (1, 2), (0, 2)), food=(0, 0))
    crash = all_features(edge, 1)["outcome"]
    assert crash["will_crash"] == 1 and crash["room_left"] == 0
    assert all_features(replace(state, food=None), 1)["outcome"]["closer_to_apple"] == 0


def test_features_use_only_current_state_and_do_not_touch_rng():
    env, control = Snake(30), Snake(30)
    for action in range(3):
        all_features(env.state, action)
    assert env.step(1) == control.step(1)
    # Stats outside the observation (score and total elapsed steps) do not leak in.
    state = env.state
    assert FeatureSpec().row(state, 0) == FeatureSpec().row(replace(state, score=99, steps=999), 0)


def test_board_order_and_candidate_danger():
    state = Snake().state
    board = all_features(state, 0)["board"]
    for i, (x, y) in enumerate(reversed(state.snake), 1):
        assert board[f"cell_{y}_{x}"] == i
    state = replace(state, snake=((7, 3), (6, 3), (5, 3)), direction=1)
    features = all_features(state, 1)
    assert features["danger"]["will_collide"] == 1
    assert features["space"]["reachable_cells"] == 0


def test_round_trip_and_reencoding_same_log(tmp_path):
    path = tmp_path / "experience.jsonl"
    store = Store(path)
    collect(store, episodes=5, seed=20)
    restored = Store(path)
    assert restored.rows == store.rows
    raw_before = path.read_bytes()
    for preset in ("board", "compact", "augmented"):
        frame = FeatureSpec.read(preset).frame((r.state, r.action) for r in restored.rows)
        assert len(frame) == len(restored.rows) and not frame.isna().any().any()
        assert not {"reward", "done", "next_state", "seed", "episode"} & set(frame.columns)
    assert path.read_bytes() == raw_before
    archive = restored.clear()
    assert not restored.rows and not path.exists()
    assert Store(archive).rows == store.rows


def test_bellman_terminal_mask_and_frozen_previous_model(tmp_path):
    state = Snake().state
    live = Transition(state, 1, 1.0, state, "ep", 0, "test")
    dead = replace(live, reward=-1, next_state=replace(state, done=True))
    spec = FeatureSpec.read("compact")
    assert np.allclose(bellman_targets([live, dead], None, spec), [1, -1])
    assert np.allclose(bellman_targets([live, dead], ActionRegressor(), spec, 0.5), [3, -1])
    created = []

    def factory():
        model = ActionRegressor()
        created.append(model)
        return model

    learner = Learner(spec, factory=factory, gamma=0.5)
    learner.fit([live, dead] * 5, rounds=2)
    assert len(created) == 2 and created[0] is not created[1]
    assert set(created[0].y) == {-1, 1}
    assert set(created[1].y) == {-1, 3}
    assert list(created[0].x) == list(created[1].x)
    assert learner.fit_table["columns"] == list(created[1].x)
    assert learner.fit_table["rows"] == created[1].x.to_numpy().tolist()
    assert learner.fit_table["targets"] == created[1].y.tolist()
    assert learner.fit_table["round"] == 2
    assert learner.values([state]).shape == (1, 3)
    assert (
        learner.prediction_table["rows"]
        == spec.frame((state, a) for a in range(3)).to_numpy().tolist()
    )
    assert learner.prediction_table["values"] == [0, 2, 4]


def test_failed_fit_keeps_previous_model():
    state = Snake().state
    row = Transition(state, 1, 1, state, "ep", 0, "test")
    learner = Learner(factory=ActionRegressor)
    learner.fit([row] * 10)
    previous = learner.model
    previous_table = learner.fit_table

    def fail():
        raise RuntimeError("API unavailable")

    learner.factory = fail
    with pytest.raises(RuntimeError, match="unavailable"):
        learner.fit([row] * 10)
    assert learner.model is previous and learner.rounds == 1
    assert learner.fit_table is previous_table


def test_hosted_adapter_requires_token_and_pins_version(monkeypatch):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    with pytest.raises(ValueError, match="TABPFN_TOKEN"):
        HostedRegressor()
    # Construct the real SDK object without contacting the hosted service.
    monkeypatch.setenv("TABPFN_TOKEN", "test-not-a-real-token")
    hosted = HostedRegressor(version="v3.5-fast")
    assert hosted.model.model_path == "v3.5-fast_default"
    assert hosted.model.fit_mode == "fit_with_cache"


def test_evaluation_is_deterministic_and_does_not_fit():
    first = evaluate(episodes=3, seed=10000)
    assert first == evaluate(episodes=3, seed=10000)
    learner = Learner(factory=ActionRegressor)
    learner.model = ActionRegressor()
    result = evaluate(learner, episodes=2)
    assert learner.rounds == 0 and len(result["games"]) == 2


def test_corrupt_log_has_actionable_error(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps({"schema": 99}) + "\n")
    with pytest.raises(ValueError, match="broken.jsonl:1"):
        Store(path)


def test_stub_model_is_selected_by_env_and_makes_no_hosted_calls(monkeypatch):
    from snake_pfn.control.learner import HostedRegressor, StubRegressor

    monkeypatch.delenv("SNAKE_STUB_MODEL", raising=False)
    assert Learner().factory is HostedRegressor
    monkeypatch.setenv("SNAKE_STUB_MODEL", "1")
    monkeypatch.setenv("SNAKE_STUB_FIT_SECONDS", "0")
    monkeypatch.setenv("SNAKE_STUB_PREDICT_SECONDS", "0")
    learner = Learner()
    assert learner.factory is StubRegressor
    env = Snake(3)
    rows = []
    for _ in range(12):
        state = env.state
        next_state, reward = env.step(1)
        rows.append(Transition(state, 1, reward, next_state, "e", 3, "random"))
        if next_state.done:
            env = Snake(4)
    learner.fit(rows, rounds=2)
    values = learner.values([Snake(5).state])
    assert values.shape == (1, 3) and (-0.6 <= values).all() and (values <= 0.7).all()
