import threading

import numpy as np
from fastapi.testclient import TestClient

from snake_pfn.control.runner import Runner
from snake_pfn.web.api import create_app


def finish(runner):
    runner.worker.join(timeout=10)
    assert not runner.worker.is_alive()


def test_full_local_flow_and_feature_invalidation(tmp_path):
    runner = Runner(tmp_path)
    with TestClient(create_app(runner=runner)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/features").json()["presets"]["board"] == ["board", "context"]
        assert client.post("/api/collect", json={"episodes": 2}).status_code == 200
        finish(runner)
        rows = client.get("/api/state").json()["rows"]
        assert rows >= 10
        runner.learner.model = object()  # Stand-in for previously fitted state.
        response = client.post("/api/features", json={"groups": ["food"], "exclude": ["will_eat"]})
        assert response.status_code == 200
        data = response.json()
        assert data["rows"] == rows and not data["fitted"] and len(data["columns"]) == 5
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
            assert client.post("/api/features", json={"groups": ["food"]}).status_code == 409
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
        assert "random steps first" in data["error"] and data["rows"] == 0
        runner.random_runs = 1
        client.post("/api/play", json={"policy": "tabpfn"})
        finish(runner)
        data = client.get("/api/state").json()
        assert "at least 10" in data["error"] and data["rows"] == 0 and data["job"] is None
        assert client.post("/api/play", json={"moves": -2}).status_code == 422
        assert client.post("/api/features", json={"groups": ["future_reward"]}).status_code == 409
        assert client.post("/api/features", json={"groups": []}).status_code == 409


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
    runner.random_runs = 1
    runner.launch("playing", lambda: runner.play(10, "tabpfn", 0, False, reveal=0, turn=0))
    try:
        assert entered.wait(5)
        runner.pause()
    finally:
        release.set()
        finish(runner)
    assert runner.env.state == initial and not runner.store.rows
    assert runner.q_values is None and runner.last_action is None


def test_inspection_uses_actual_fit_and_prediction_without_model_calls(tmp_path):
    class RecordingModel:
        def __init__(self):
            self.calls = 0

        def fit(self, x, y):
            self.x, self.y = x.copy(), y.copy()

        def predict(self, x):
            self.calls += 1
            self.query = x.copy()
            return x["action"].to_numpy() * 0.25

    runner = Runner(tmp_path)
    runner.learner.factory = RecordingModel
    with TestClient(create_app(runner=runner)) as client:
        assert client.get("/api/table").json()["total"] == 0
        runner.collect(3)
        preview = client.get("/api/table?offset=2&limit=3").json()
        assert preview["source"] == "preview" and preview["targets"] == [None] * 3
        expected = [
            list(runner.learner.spec.row(r.state, r.action).values())
            for r in runner.store.rows[2:5]
        ]
        assert preview["rows"] == expected
        runner.learner.fit(runner.store.rows, rounds=2)
        model = runner.learner.model
        runner.learner.values([runner.env.state])
        calls_before = model.calls
        fitted = client.get("/api/table?offset=2&limit=3").json()
        assert fitted["source"] == "fit" and fitted["round"] == 2
        assert fitted["columns"] == list(model.x)
        assert fitted["rows"] == model.x.iloc[2:5].to_numpy().tolist()
        assert fitted["targets"] == model.y[2:5].tolist()
        assert fitted["rewards"] == [r.reward for r in runner.store.rows[2:5]]  # whole log, in order
        # While a game runs the UI asks for the saved moves even though a model is fitted.
        saved = client.get("/api/table?source=experience&latest=true&limit=4").json()
        total = len(runner.store.rows)
        assert saved["source"] == "preview" and saved["total"] == total
        assert saved["offset"] == ((total - 1) // 4) * 4
        assert saved["latest_index"] == total - 1
        assert saved["rewards"] == [r.reward for r in runner.store.rows[saved["offset"] :]]
        assert saved["prediction"] is not None  # The last prediction stays visible.
        assert fitted["prediction"]["rows"] == model.query.to_numpy().tolist()
        assert fitted["prediction"]["values"] == [0, 0.25, 0.5]
        assert model.calls == calls_before
        # New experience does not change what the last successful fit received.
        runner.collect(1)
        assert client.get("/api/table?offset=2&limit=3").json() == fitted
        assert client.get("/api/table?offset=-1").status_code == 422
        assert client.get("/api/table?limit=1001").status_code == 422
        client.post("/api/features", json={"groups": ["food"]})
        changed = client.get("/api/table").json()
        assert changed["source"] == "preview" and changed["prediction"] is None
        assert changed["columns"] == [
            "action",
            "food_forward",
            "food_right",
            "food_distance",
            "next_food_distance",
            "will_eat",
        ]


def test_random_steps_play_across_games_and_log_every_move(tmp_path):
    runner = Runner(tmp_path)
    with TestClient(create_app(runner=runner)) as client:
        assert client.post("/api/random-steps", json={"moves": 60}).status_code == 200
        assert client.get("/api/state").json()["job"] == "random-steps"
        finish(runner)
        data = client.get("/api/state").json()
        assert data["job"] is None and data["error"] is None and data["rows"] == 60
        assert data["episodes"] >= 1  # Starvation after 50 moves guarantees a game over.
        assert {row.policy for row in runner.store.rows} == {"random"}
        assert data["message"] == "Run finished. Experience saved."
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
    runner.collect(2)
    runner.random_runs = 1
    with TestClient(create_app(runner=runner)) as client:
        body = {"moves": 2, "policy": "tabpfn", "epsilon": 0, "auto_fit": False, "rounds": 2}
        runner.play_reveal = runner.turn_window = 0
        assert client.post("/api/play", json=body).status_code == 200
        finish(runner)
        data = client.get("/api/state").json()
        assert data["error"] is None and data["fitted"] and data["rounds"] == 2
        assert data["phase"] == "idle" and data["message"].startswith("Run finished")
        assert CountingModel.fits == 2
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
        assert query["query"]["columns"] == runner.snapshot()["columns"]
        assert len(query["query"]["rows"]) == 3 and query["query"]["rows"][0][0] == 0
        assert query["query"]["rows"][2][0] == 2 and len(query["query"]["directions"]) == 3
        assert data["query"] is None  # Cleared once the move is made.
        client.post("/api/play", json=body)
        finish(runner)
        assert CountingModel.fits == 2  # The fitted model is reused.


def test_saved_moves_from_another_board_size_are_archived_on_start(tmp_path):
    first = Runner(tmp_path, size=8)
    first.collect(1)
    assert first.store.rows and first.store.rows[0].state.size == 8
    runner = Runner(tmp_path, size=5)
    assert runner.store.rows == [] and runner.env.state.size == 5
    assert "8×8" in runner.snapshot()["message"]
    assert list(tmp_path.glob("experience-*.jsonl"))


def test_event_log_records_jobs_and_moves_and_supports_incremental_polling(tmp_path):
    runner = Runner(tmp_path)
    with TestClient(create_app(runner=runner)) as client:
        first = client.get("/api/log").json()
        assert first["entries"] and first["entries"][0]["id"] == 1
        assert any("Loaded 0 saved moves" in e["message"] for e in first["entries"])
        client.post("/api/random-steps", json={"moves": 60})
        finish(runner)
        data = client.get(f"/api/log?after={first['latest']}").json()
        messages = [e["message"] for e in data["entries"]]
        assert all(e["id"] > first["latest"] for e in data["entries"])
        assert any(m.startswith("Job started: random-steps") for m in messages)
        assert any(m.startswith("Move 1: head C3") for m in messages)
        assert any(m.startswith("Episode 1 over") for m in messages)
        assert any(m.startswith("Job finished: random-steps") for m in messages)
        assert data["latest"] == data["entries"][-1]["id"]
        assert client.get(f"/api/log?after={data['latest']}").json()["entries"] == []
        assert client.get("/api/log?after=-1").status_code == 422
        # Failures are logged with their traceback.
        def broken_factory():
            raise RuntimeError("model unavailable in tests")

        runner.learner.factory = broken_factory
        client.post("/api/play", json={"policy": "tabpfn"})
        finish(runner)
        errors = [e for e in client.get("/api/log").json()["entries"] if e["level"] == "error"]
        assert errors and "Traceback" in errors[-1]["message"]


def test_event_log_survives_uvicorn_logging_configuration(tmp_path):
    import logging.config

    from uvicorn.config import LOGGING_CONFIG

    runner = Runner(tmp_path)
    before = runner.events.since()["latest"]
    logging.config.dictConfig(LOGGING_CONFIG)  # Closes all pre-existing handlers.
    runner.pause()
    runner.launch("probe", lambda: None)
    finish(runner)
    messages = [e["message"] for e in runner.events.since(before)["entries"]]
    assert any(m.startswith("Job started: probe") for m in messages)
    runner.events.detach()
    runner.launch("silent", lambda: None)
    finish(runner)
    assert runner.events.since()["entries"][-1]["message"].startswith("Job finished: probe")


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
        assert seqs == list(range(1, 60 + 1 + state["episodes"] + 1))
        assert data["latest"] == state["seq"] == seqs[-1]
        assert data["frames"][0]["state"]["steps"] == 0 and data["frames"][0]["reward"] is None
        assert data["frames"][0]["kind"] == "reset" and data["frames"][1]["kind"] == "move"
        assert {f["kind"] for f in data["frames"]} == {"reset", "move"}
        assert data["frames"][-1]["state"] == state["state"]
        assert sum(f["reward"] == -1.0 for f in data["frames"]) == state["episodes"]
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
    runner.collect(2)
    seen = []
    phases = []
    runner.publish = lambda: (seen.append(runner.message), phases.append(runner.phase),
                              Runner.publish(runner))
    runner.fit(2)
    assert set(phases) == {"training"}
    assert any("round 1 of 2 · using observed rewards" in m for m in seen)
    assert any("round 2 of 2 · predicting future values" in m for m in seen)
    assert any(m.startswith("Training TabPFN · round 2 of 2 · fitting") for m in seen)
    assert runner.message == "Model ready: 2 fitted Q rounds."


def test_player_moves_the_apple_one_step_between_snake_moves(tmp_path):
    class ZeroModel:
        def fit(self, x, y):
            pass

        def predict(self, x):
            return np.zeros(len(x))

    runner = Runner(tmp_path)
    runner.learner.factory = ZeroModel
    runner.collect(2)
    runner.random_runs = 1
    with TestClient(create_app(runner=runner)) as client:
        assert client.post("/api/apple", json={"x": 0, "y": 0}).status_code == 409  # not playing
        runner.job = "playing"  # Stand-in for a live TabPFN game.
        fx, fy = runner.env.state.food
        far = (fx + 2) % 5, fy
        assert client.post("/api/apple", json={"x": far[0], "y": far[1]}).status_code == 409
        step = next(
            (x, y) for x, y in ((fx + 1, fy), (fx - 1, fy), (fx, fy + 1), (fx, fy - 1))
            if 0 <= x < 5 and 0 <= y < 5 and (x, y) not in runner.env.state.snake
        )
        data = client.post("/api/apple", json={"x": step[0], "y": step[1]}).json()
        assert tuple(data["pending_food"]) == step and runner.env.state.food == (fx, fy)
        # Changing your mind: clicking the apple itself cancels the step; a new step replaces it.
        assert client.post("/api/apple", json={"x": fx, "y": fy}).json()["pending_food"] is None
        assert tuple(client.post("/api/apple", json={"x": step[0], "y": step[1]}).json()["pending_food"]) == step
        runner.job = None
        before = len(runner.store.rows)
        runner.play(1, "tabpfn", 0, False, reveal=0, turn=0)  # Applies the step before asking TabPFN.
        saved = runner.store.rows[before]
        assert saved.state.food == step and runner.snapshot()["apple_moves"] == 1
        assert runner.snapshot()["pending_food"] is None
        kinds = [f["kind"] for f in runner.frames_since()["frames"]][-4:]
        assert kinds == ["apple", "query", "prediction", "move"]
