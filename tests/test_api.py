import threading
from dataclasses import replace

import numpy as np
from fastapi.testclient import TestClient

from snake_pfn.control.experience import collect
from snake_pfn.control.runner import Runner
from snake_pfn.web.api import create_app


def finish(runner):
    runner.worker.join(timeout=10)
    assert not runner.worker.is_alive()


def seed(runner, episodes):
    """Fill the table with heuristic seed games without going through the API."""
    start = max((r.seed for r in runner.store.rows), default=99) + 1
    collect(runner.store, episodes, start, runner.size)
    runner.publish()


def test_full_local_flow(tmp_path):
    runner = Runner(tmp_path)
    with TestClient(create_app(runner=runner)) as client:
        page = client.get("/")
        assert page.status_code == 200 and page.headers["cache-control"] == "no-cache"
        assert "cache-control" not in client.get("/api/state").headers
        seed(runner, 2)
        rows = client.get("/api/state").json()["rows"]
        assert rows >= 10
        assert client.post("/api/play", json={"moves": 1}).status_code == 200
        finish(runner)
        assert client.get("/api/state").json()["rows"] == rows + 1
        assert client.post("/api/clear").status_code == 200
        assert client.get("/api/state").json()["rows"] == 0
        assert list(tmp_path.glob("experience-*.jsonl"))


def test_busy_runner_rejects_mutations_but_allows_reads_and_pause(tmp_path):
    runner = Runner(tmp_path)
    release = threading.Event()
    runner.launch("test", lambda: release.wait(5))
    try:
        with TestClient(create_app(runner=runner)) as client:
            assert client.get("/api/state").json()["job"] == "test"
            assert client.post("/api/reset").status_code == 409
            assert client.post("/api/clear").status_code == 409
            assert client.post("/api/play", json={}).status_code == 409
            assert client.post("/api/pause").status_code == 200
            assert runner.stop.is_set()
    finally:
        release.set()
        finish(runner)


def test_missing_model_error_and_validation(tmp_path):
    runner = Runner(tmp_path)
    with TestClient(create_app(runner=runner)) as client:
        assert client.post("/api/play", json={"policy": "tabpfn"}).status_code == 200
        finish(runner)
        data = client.get("/api/state").json()
        assert "at least 10" in data["error"] and data["rows"] == 0 and data["job"] is None
        assert client.post("/api/play", json={"moves": -2}).status_code == 422


def test_pause_during_prediction_does_not_execute_or_log_move(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class BlockingPredictor:
        def predict(self, frame):
            entered.set()
            assert release.wait(5)
            return np.zeros(len(frame))

    runner = Runner(tmp_path)
    runner.learner.model = BlockingPredictor()
    initial = runner.env.state
    runner.launch("playing", lambda: runner.play(10, "tabpfn", 0, False))
    try:
        assert entered.wait(5)
        runner.pause()
    finally:
        release.set()
        finish(runner)
    assert runner.env.state == initial and not runner.store.rows
    assert runner.q_values is None and runner.last_action is None


def test_table_shows_saved_moves_as_model_inputs_without_model_calls(tmp_path):
    class NoCalls:
        def predict(self, x):
            raise AssertionError("The table must not call the model")

    runner = Runner(tmp_path)
    runner.learner.model = NoCalls()
    with TestClient(create_app(runner=runner)) as client:
        assert client.get("/api/table").json()["total"] == 0
        seed(runner, 3)
        page = client.get("/api/table?offset=2&limit=3").json()
        spec = runner.learner.spec
        assert page["columns"] == list(spec.row(runner.env.state, 1))
        assert page["rows"] == [
            list(spec.row(r.state, r.action).values()) for r in runner.store.rows[2:5]
        ]
        assert page["rewards"] == [r.reward for r in runner.store.rows[2:5]]
        # While a game runs the UI follows the latest saved move.
        latest = client.get("/api/table?latest=true&limit=4").json()
        total = len(runner.store.rows)
        assert latest["total"] == total and latest["offset"] == ((total - 1) // 4) * 4
        assert latest["latest_index"] == total - 1
        assert latest["rewards"] == [r.reward for r in runner.store.rows[latest["offset"] :]]
        assert client.get("/api/table?offset=-1").status_code == 422
        assert client.get("/api/table?limit=1001").status_code == 422


def test_random_steps_play_across_games_and_log_every_move(tmp_path):
    runner = Runner(tmp_path)
    with TestClient(create_app(runner=runner)) as client:
        assert client.post("/api/random-steps", json={"moves": 60}).status_code == 200
        assert client.get("/api/state").json()["job"] == "random-steps"
        finish(runner)
        data = client.get("/api/state").json()
        assert data["job"] is None and data["error"] is None and data["rows"] == 60
        assert runner.episodes >= 1
        assert {row.policy for row in runner.store.rows} == {"random"}
        assert data["message"] == "Done."
        table = client.get("/api/table?source=experience&latest=true&limit=5").json()
        assert table["latest_index"] == 59
        assert all(direction in range(4) for direction in table["directions"])
        # More steps continue from a fresh seed, adding to the same table.
        client.post("/api/random-steps", json={"moves": 5})
        finish(runner)
        assert client.get("/api/state").json()["rows"] == 65
        assert len({row.seed for row in runner.store.rows}) >= 2
        assert client.post("/api/random-steps", json={"moves": 0}).status_code == 422


def test_play_trains_first_when_no_model_is_fitted(tmp_path):
    class CountingModel:
        fits = 0

        def fit(self, x, y):
            CountingModel.fits += 1

        def predict(self, x):
            return np.zeros(len(x))

    runner = Runner(tmp_path)
    runner.learner.factory = CountingModel
    seed(runner, 2)
    with TestClient(create_app(runner=runner)) as client:
        body = {"moves": 2, "policy": "tabpfn", "epsilon": 0, "auto_fit": False, "rounds": 2}
        assert client.post("/api/play", json=body).status_code == 200
        finish(runner)
        data = client.get("/api/state").json()
        assert data["error"] is None and data["fitted"] and runner.learner.rounds == 2
        assert data["phase"] == "idle" and data["message"] == "Done."
        assert CountingModel.fits == 2
        # The speed readout: the whole fit job, and each answer's time on its frame.
        assert data["learn_seconds"] >= 0 and data["learn_rows"] == len(runner.store.rows) - 2  # Before the 2 moves
        predictions = [f for f in runner.frames if f["kind"] == "prediction"]
        assert all(f["seconds"] >= 0 for f in predictions)
        assert data["predict_seconds"] == predictions[-1]["seconds"]
        # Every TabPFN move is announced by a prediction frame on the unmoved board.
        frames = client.get("/api/frames").json()["frames"]
        kinds = [f["kind"] for f in frames if f["kind"] != "reset"]
        assert kinds == ["query", "prediction", "move"] * 2
        query, prediction, move = frames[-3], frames[-2], frames[-1]
        assert prediction["q_values"] == move["q_values"] == [0.0, 0.0, 0.0]
        assert prediction["action"] == move["action"]
        assert prediction["state"] == query["state"]
        assert prediction["state"]["steps"] + 1 == move["state"]["steps"]
        # The query frame carries the exact rows TabPFN scores, in the table's columns.
        assert query["query"]["columns"] == list(runner.learner.spec.row(runner.env.state, 1))
        assert len(query["query"]["rows"]) == 3 and query["query"]["rows"][0][0] == 0
        assert query["query"]["rows"][2][0] == 2 and len(query["query"]["directions"]) == 3
        assert data["query"] is None  # Cleared once the move is made.
        client.post("/api/play", json=body)
        finish(runner)
        assert CountingModel.fits == 2  # The fitted model is reused.


def test_random_steps_accept_a_bounded_delay(tmp_path):
    runner = Runner(tmp_path)
    with TestClient(create_app(runner=runner)) as client:
        assert client.post("/api/random-steps", json={"moves": 2, "delay": 0.01}).status_code == 200
        finish(runner)
        assert client.get("/api/state").json()["rows"] == 2
        assert client.post("/api/random-steps", json={"moves": 2, "delay": 10}).status_code == 422


def test_saved_moves_from_another_board_size_are_archived_on_start(tmp_path):
    first = Runner(tmp_path, size=8)
    seed(first, 1)
    assert first.store.rows and first.store.rows[0].state.size == 8
    runner = Runner(tmp_path, size=5)
    assert runner.store.rows == [] and runner.env.state.size == 5
    assert "8×8" in runner.snapshot()["message"]
    assert list(tmp_path.glob("experience-*.jsonl"))


def test_server_start_archives_saved_moves_so_the_intro_replays(tmp_path):
    seed(Runner(tmp_path), 1)
    with TestClient(create_app(tmp_path)) as client:
        state = client.get("/api/state").json()
        assert state["rows"] == 0 and not state["fitted"]
    assert list(tmp_path.glob("experience-*.jsonl"))


def test_frames_capture_every_board_change_for_smooth_playback(tmp_path):
    runner = Runner(tmp_path)
    with TestClient(create_app(runner=runner)) as client:
        assert client.get("/api/frames").json() == {"frames": [], "latest": 0}
        client.post("/api/random-steps", json={"moves": 60})
        finish(runner)
        state = client.get("/api/state").json()
        data = client.get("/api/frames").json()
        seqs = [f["seq"] for f in data["frames"]]
        # One start frame, one per move, and one fresh board per finished game.
        games = runner.episodes
        assert seqs == list(range(1, 60 + 1 + games + 1 - int(state["state"]["done"])))
        assert data["latest"] == state["seq"] == seqs[-1]
        assert data["frames"][0]["state"]["steps"] == 0 and data["frames"][0]["reward"] is None
        assert data["frames"][0]["kind"] == "reset" and data["frames"][1]["kind"] == "move"
        assert {f["kind"] for f in data["frames"]} == {"reset", "move"}
        assert data["frames"][-1]["state"] == state["state"]
        # Only terminal moves finish a game; eating normally continues it.
        assert sum(f["kind"] == "move" and f["state"]["done"] for f in data["frames"]) == games
        assert all(f["q_values"] is None for f in data["frames"])
        assert client.get(f"/api/frames?after={data['latest']}").json()["frames"] == []
        assert client.get(f"/api/frames?after={seqs[-2]}").json()["frames"] == data["frames"][-1:]
        client.post("/api/reset")
        assert client.get("/api/frames?after=%d" % data["latest"]).json()["frames"][0]["state"]["steps"] == 0


def test_training_reports_progress_messages(tmp_path):
    class ZeroModel:
        def fit(self, x, y):
            pass

        def predict(self, x):
            return np.zeros(len(x))

    runner = Runner(tmp_path)
    runner.learner.factory = ZeroModel
    seed(runner, 2)
    seen = []
    phases = []
    runner.publish = lambda: (seen.append(runner.message), phases.append(runner.phase),
                              Runner.publish(runner))
    runner.fit(2)
    assert set(phases) == {"training"}
    assert "Training · round 1 of 2…" in seen and "Training · round 2 of 2…" in seen
    assert runner.message == "TabPFN is ready."


def test_eating_grows_snake_and_continues_the_same_episode(tmp_path):
    class StraightAhead:
        def predict(self, x):
            return np.tile([0.0, 1.0, 0.0], len(x) // 3)

    runner = Runner(tmp_path, size=4)  # Moves below assume the 4 × 4 layout.
    runner.learner.model = StraightAhead()
    head = runner.env.state.snake[0]
    runner.env.state = replace(runner.env.state, food=(head[0] + 1, head[1]))
    initial_length, episode = len(runner.env.state.snake), runner.episode_id
    runner.play(1, "tabpfn", 0, False, delay=0, stop_after_episode=True)
    data = runner.snapshot()
    assert data["state"]["score"] == 1 and data["state"]["steps"] == 1
    assert len(runner.env.state.snake) == initial_length + 1
    assert runner.env.state.food not in runner.env.state.snake
    assert runner.episode_id == episode and runner.episodes == 0
    assert runner.store.rows[0].reward == 1 and not runner.store.rows[0].next_state.done
    assert [f["kind"] for f in runner.frames] == ["query", "prediction", "move"]
    # The next prediction uses the grown snake, instead of a fresh board.
    runner.play(20, "tabpfn", 0, False, delay=0, stop_after_episode=True)
    data = runner.snapshot()
    assert data["state"]["done"] and data["state"]["reason"] == "collision"
    assert data["state"]["steps"] == 2 and data["state"]["score"] == 1
    assert runner.episodes == 1
    assert runner.store.rows[1].state == runner.store.rows[0].next_state
    assert runner.store.rows[1].episode == episode
    # Final decision survives the end of a game and can be restored on page reload.
    assert data["decision_query"]["rows"] == runner.frames[-2]["query"]["rows"]
    assert data["decision_state"]["steps"] == 1 and data["q_values"] == [0, 1, 0]
    runner.play(1, "tabpfn", 0, False, delay=0, stop_after_episode=True)
    assert runner.episode_id != episode and runner.env.state.steps == 1


def test_apple_cannot_be_moved_and_api_can_stop_at_game_over(tmp_path):
    class StraightAhead:
        def predict(self, x):
            return np.tile([0.0, 1.0, 0.0], len(x) // 3)

    runner = Runner(tmp_path, size=4)
    runner.learner.model = StraightAhead()
    runner.env.state = replace(runner.env.state, food=(0, 0))
    with TestClient(create_app(runner=runner)) as client:
        assert client.post("/api/apple", json={"x": 0, "y": 1}).status_code in (404, 405)
        assert client.post("/api/play", json={
            "policy": "tabpfn", "moves": 20, "auto_fit": False, "stop_after_episode": True,
        }).status_code == 200
        finish(runner)
        state = client.get("/api/state").json()
        assert state["state"]["done"] and state["state"]["steps"] == 2
        assert len(runner.store.rows) == 2
        assert all(row.state.food == (0, 0) for row in runner.store.rows)
        assert "Game over" in state["message"]


def test_only_finished_games_trigger_refitting(tmp_path):
    runner = Runner(tmp_path, size=4)
    runner.episodes = 4
    calls = []
    runner.fit = lambda *args: calls.append(args)

    class StraightAhead:
        def predict(self, x):
            return np.tile([0.0, 1.0, 0.0], len(x) // 3)

    runner.learner.model = StraightAhead()
    head = runner.env.state.snake[0]
    runner.env.state = replace(runner.env.state, food=(head[0] + 1, head[1]))
    runner.play(1, "tabpfn", 0, True, delay=0)
    assert not calls and runner.episodes == 4
    runner.play(1, "tabpfn", 0, True, delay=0)
    assert len(calls) == 1 and runner.episodes == 5


def test_first_apple_can_be_placed_for_the_intro(tmp_path):
    runner = Runner(tmp_path)
    with TestClient(create_app(runner=runner)) as client:
        assert client.post("/api/reset", json={"food": [0, 0]}).json()["state"]["food"] == [0, 0]
        assert client.post("/api/reset").status_code == 200  # Body stays optional.
        client.post("/api/random-steps", json={"moves": 1, "delay": 0, "food": [3, 3]})
        finish(runner)
        assert runner.store.rows[-1].state.food == (3, 3)
        # A square under the snake is ignored and the seeded apple is kept.
        head = runner.env.state.snake[0]
        state = client.post("/api/reset", json={"food": list(head)}).json()["state"]
        assert state["food"] not in state["snake"]


def test_practice_seed_changes_practice_moves_but_not_the_demo(tmp_path):
    def intro(path, practice_seed):
        runner = Runner(path)
        with TestClient(create_app(runner=runner)) as client:
            client.post("/api/random-steps", json={"moves": 5, "delay": 0, "food": [0, 0]})
            finish(runner)
            client.post("/api/random-steps", json={"moves": 40, "delay": 0, "seed": practice_seed})
            finish(runner)
            return [(r.seed, r.action) for r in runner.store.rows]

    first, same, other = intro(tmp_path / "a", 1000), intro(tmp_path / "b", 1000), intro(tmp_path / "c", 7)
    assert first == same
    assert first[:5] == other[:5] and first[5:] != other[5:]
    assert first[5][0] == 1000 and other[5][0] == 7


def test_practice_seed_continues_a_partly_filled_table(tmp_path):
    runner = Runner(tmp_path)
    with TestClient(create_app(runner=runner)) as client:
        client.post("/api/random-steps", json={"moves": 40, "delay": 0, "seed": 1000})
        finish(runner)
        last = runner.store.rows[-1].seed
        client.post("/api/random-steps", json={"moves": 10, "delay": 0, "seed": 1000})
        finish(runner)
        assert runner.store.rows[40].seed == last + 1
