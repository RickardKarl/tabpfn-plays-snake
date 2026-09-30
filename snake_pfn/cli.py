import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .control.experience import Store, collect
from .control.features import PRESETS, FeatureSpec
from .control.learner import Learner, evaluate


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return n


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Snake × TabPFN experiment lab")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--data-dir", default="data")
    serve.add_argument("--stub", action="store_true",
                       help="Debug: placeholder model with API-like delays, no TabPFN calls")
    gather = sub.add_parser("collect")
    gather.add_argument("--episodes", type=positive, default=20)
    gather.add_argument("--seed", type=int, default=0)
    for name in ("export", "train", "compare"):
        cmd = sub.add_parser(name)
        if name != "compare":
            cmd.add_argument("--features", default="board_outcomes", help="Preset name or JSON config")
        if name in ("train", "compare"):
            cmd.add_argument("--rounds", type=positive, default=3)
            cmd.add_argument("--max-rows", type=positive, default=None,
                             help="Cap context rows by random sample; default uses all")
        cmd.add_argument(
            "--output",
            default={
                "export": "data/features.csv",
                "train": "data/model.json",
                "compare": "data/comparison.json",
            }[name],
        )
    ev = sub.add_parser("evaluate")
    ev.add_argument("--model", help="Saved model; omit for food-seeking baseline")
    ev.add_argument("--output", default="data/evaluation.json")
    for name in ("evaluate", "compare"):
        cmd = sub.choices[name]
        cmd.add_argument("--episodes", type=positive, default=10)
        cmd.add_argument("--seed", type=int, default=10000)
    for name in ("collect", "export", "train", "compare"):
        sub.choices[name].add_argument("--data", default="data/experience.jsonl")
    args = parser.parse_args()
    if args.command == "serve":
        import os

        import uvicorn

        from .web.api import create_app

        if args.stub:
            os.environ["SNAKE_STUB_MODEL"] = "1"

        uvicorn.run(create_app(args.data_dir), host="127.0.0.1", port=args.port)
        return
    if args.command == "evaluate":
        model = Learner.load(args.model) if args.model else None
        result = evaluate(model, args.episodes, args.seed)
    else:
        store = Store(args.data)
        if args.command == "collect":
            result = collect(store, args.episodes, args.seed)
            print(json.dumps({"episodes": len(result), "total_transitions": len(store.rows)}))
            return
        if not store.rows:
            parser.error("No experience yet. Run collect first.")
        if args.command == "export":
            frame = FeatureSpec.read(args.features).frame((r.state, r.action) for r in store.rows)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index=False)
            print(f"Exported {len(frame)} rows × {len(frame.columns)} inputs to {args.output}")
            return
        if args.command == "train":
            learner = Learner(FeatureSpec.read(args.features), max_rows=args.max_rows)
            learner.fit(store.rows, rounds=args.rounds)
            learner.save(args.output)
            print(f"Saved fitted model and feature metadata to {args.output}")
            return
        # Same frozen experience, replay sample, FQI budget and unseen evaluation seeds.
        train_seeds = {r.seed for r in store.rows}
        eval_seeds = set(range(args.seed, args.seed + args.episodes))
        if train_seeds & eval_seeds:
            parser.error("Evaluation seeds overlap with collection seeds; choose a new --seed")
        result = {
            "training_rows": len(store.rows),
            "rounds": args.rounds,
            "max_rows": args.max_rows,
            "results": {},
        }
        result["results"]["heuristic"] = evaluate(None, args.episodes, args.seed)
        for preset in PRESETS:
            print(f"Fitting and evaluating {preset}…", flush=True)
            learner = Learner(FeatureSpec.read(preset), max_rows=args.max_rows)
            learner.fit(store.rows, rounds=args.rounds)
            result["results"][preset] = evaluate(learner, args.episodes, args.seed)
            result["results"][preset]["features"] = learner.spec.to_dict()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
