from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..control.runner import Runner


class PlayRequest(BaseModel):
    moves: int = Field(default=100, ge=1, le=1000)
    policy: Literal["heuristic", "random", "tabpfn"] = "heuristic"
    epsilon: float = Field(default=0.1, ge=0, le=1)  # Baseline policies only; TabPFN is greedy.
    auto_fit: bool = True
    rounds: int = Field(default=3, ge=1, le=10)
    stop_after_episode: bool = False  # Stop on collision, starvation, or a full board


class FitRequest(BaseModel):
    rounds: int = Field(default=3, ge=1, le=10)


Cell = tuple[int, int]


class RandomStepsRequest(BaseModel):
    moves: int = Field(default=1000, ge=1, le=5000)
    delay: float = Field(default=0.002, ge=0, le=5)  # Seconds between moves; slow for demos
    food: Cell | None = None  # Where the first apple goes; random when omitted
    seed: int | None = Field(default=None, ge=0)  # Restart random moves and games from here


class ResetRequest(BaseModel):
    food: Cell | None = None


def create_app(data_dir="data", runner=None):
    load_dotenv(find_dotenv(usecwd=True))  # The .env next to where you run the server
    runner = runner or Runner(data_dir, fresh=True)  # A restarted server replays the intro

    @asynccontextmanager
    async def lifespan(app):
        yield
        runner.pause()

    app = FastAPI(title="Snake / TabPFN", lifespan=lifespan)

    @app.middleware("http")
    async def revalidate_static(request, call_next):
        # A stale cached app.js against fresh HTML leaves the board blank after an update.
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    def command(fn):
        try:
            fn()
            return runner.snapshot()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/state")
    def state():
        return runner.snapshot()

    @app.get("/api/frames")
    def frames(after: int = Query(0, ge=0), limit: int = Query(600, ge=1, le=600)):
        return runner.frames_since(after, limit)

    @app.get("/api/table")
    def table(offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100),
              latest: bool = False):
        return runner.table(offset, limit, latest)

    @app.post("/api/random-steps")
    def random_steps(body: RandomStepsRequest):
        return command(
            lambda: runner.launch("random-steps", lambda: runner.random_steps(**body.model_dump()))
        )

    @app.post("/api/play")
    def play(body: PlayRequest):
        return command(lambda: runner.launch("playing", lambda: runner.play(**body.model_dump())))

    @app.post("/api/pause")
    def pause():
        runner.pause()
        return {"message": "Stopping after the current request completes"}

    @app.post("/api/reset")
    def reset(body: ResetRequest | None = None):
        return command(lambda: runner.reset(body.food if body else None))

    @app.post("/api/fit")
    def fit(body: FitRequest):
        return command(lambda: runner.launch("fitting", lambda: runner.fit(body.rounds)))

    @app.post("/api/clear")
    def clear():
        return command(runner.clear)

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
    return app
