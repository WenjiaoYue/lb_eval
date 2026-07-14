#!/usr/bin/env python3
"""record_recipe.py — Positive knowledge base of working quantization recipes.

Unlike lessons (which capture FAILURES), this records SUCCESSES: for a given
model, which scheme/method/params produced a working quantized model and what
accuracy it reached. This is the "quantization recipes" registry.

Store: recipes/recipes.jsonl (one JSON object per successful run).

Usage:
    # Record a recipe after a successful quantize+evaluate run
    python3 record_recipe.py record \
        --model Qwen/Qwen3-0.6B --scheme W4A16 --method RTN --iters 0 \
        --export-format auto_round --device cuda \
        --accuracy-json /path/to/run/accuracy.json \
        --source-task Qwen/Qwen3-0.6B_W4A16_RTN

    # Query the best known recipe(s) for a model
    python3 record_recipe.py query --model Qwen/Qwen3-0.6B
    python3 record_recipe.py query --model Qwen/Qwen3-0.6B --json

    # List all recipes
    python3 record_recipe.py list
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STORE = SCRIPT_DIR / "recipes.jsonl"


def _load(store: Path) -> list[dict]:
    if not store.exists():
        return []
    out = []
    for line in store.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _parse_accuracy(accuracy_json: str) -> tuple[dict, float | None]:
    """Return (per_task_scores, mean_score) from an accuracy.json, best-effort."""
    if not accuracy_json or not os.path.isfile(accuracy_json):
        return {}, None
    try:
        data = json.loads(Path(accuracy_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, None

    scores: dict[str, float] = {}

    def _walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    # Heuristic: accuracy-like metrics
                    if any(m in k.lower() for m in ("acc", "score", "exact_match", "f1")):
                        scores[f"{prefix}{k}"] = float(v)
                elif isinstance(v, dict):
                    _walk(v, prefix=f"{k}.")
    _walk(data)

    mean = round(sum(scores.values()) / len(scores), 6) if scores else None
    return scores, mean


def cmd_record(args) -> int:
    store = Path(args.store or DEFAULT_STORE)
    store.parent.mkdir(parents=True, exist_ok=True)

    scores, mean = _parse_accuracy(args.accuracy_json)

    def _maybe_json(s):
        if not s:
            return None
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return s

    recipe = {
        "id": f"recipe-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "scheme": args.scheme,
        "method": args.method,
        "iters": args.iters,
        "export_format": args.export_format,
        "device": args.device,
        "num_gpus": args.num_gpus,
        "layer_config": _maybe_json(args.layer_config),
        "ignore_layers": args.ignore_layers or "",
        "auto_round_ref": args.auto_round_ref or "",
        "transformers_ref": args.transformers_ref or "",
        "accuracy": scores,
        "accuracy_mean": mean,
        "status": args.status,
        "source_task": args.source_task or f"{args.model}_{args.scheme}_{args.method}",
    }

    with open(store, "a", encoding="utf-8") as f:
        f.write(json.dumps(recipe, ensure_ascii=False) + "\n")

    print(f"[recipe] Recorded: {args.model} | {args.scheme}/{args.method} "
          f"(iters={args.iters}, mean_acc={mean})")
    print(f"[recipe] Store: {store}")
    return 0


def _rank_key(r: dict):
    # Prefer verified/successful, then higher mean accuracy, then newer.
    status_rank = 1 if str(r.get("status", "")).lower() in ("finished", "success", "verified") else 0
    mean = r.get("accuracy_mean")
    return (status_rank, mean if isinstance(mean, (int, float)) else -1, r.get("timestamp", ""))


def cmd_query(args) -> int:
    store = Path(args.store or DEFAULT_STORE)
    recipes = _load(store)
    matches = [r for r in recipes if r.get("model") == args.model]
    if args.scheme:
        matches = [r for r in matches if r.get("scheme") == args.scheme]
    if not matches:
        print(f"[recipe] No recipe found for {args.model}"
              + (f" / {args.scheme}" if args.scheme else ""))
        return 1

    matches.sort(key=_rank_key, reverse=True)
    if args.json:
        print(json.dumps(matches if args.all else matches[0], ensure_ascii=False, indent=2))
        return 0

    shown = matches if args.all else matches[:1]
    for r in shown:
        print(f"● {r['model']} | {r['scheme']}/{r['method']} "
              f"iters={r.get('iters')} format={r.get('export_format')} "
              f"device={r.get('device')}")
        print(f"    mean_acc={r.get('accuracy_mean')} status={r.get('status')} "
              f"ts={r.get('timestamp')}")
        if r.get("ignore_layers"):
            print(f"    ignore_layers={r['ignore_layers']}")
        if r.get("layer_config"):
            print(f"    layer_config={json.dumps(r['layer_config'], ensure_ascii=False)}")
    return 0


def cmd_list(args) -> int:
    store = Path(args.store or DEFAULT_STORE)
    recipes = _load(store)
    if args.json:
        print(json.dumps(recipes, ensure_ascii=False, indent=2))
        return 0
    print(f"[recipe] {len(recipes)} recipe(s) in {store}")
    for r in sorted(recipes, key=_rank_key, reverse=True):
        print(f"  {r.get('model'):40s} {r.get('scheme')}/{r.get('method'):8s} "
              f"mean_acc={r.get('accuracy_mean')} status={r.get('status')}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Quantization recipe registry (positive KB).")
    p.add_argument("--store", help=f"Path to recipes.jsonl (default: {DEFAULT_STORE})")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record", help="Record a successful recipe")
    pr.add_argument("--model", required=True)
    pr.add_argument("--scheme", default="W4A16")
    pr.add_argument("--method", default="RTN")
    pr.add_argument("--iters", default="0")
    pr.add_argument("--export-format", dest="export_format", default="auto_round")
    pr.add_argument("--device", default="cuda")
    pr.add_argument("--num-gpus", dest="num_gpus", default="1")
    pr.add_argument("--layer-config", dest="layer_config", default="")
    pr.add_argument("--ignore-layers", dest="ignore_layers", default="")
    pr.add_argument("--auto-round-ref", dest="auto_round_ref", default="")
    pr.add_argument("--transformers-ref", dest="transformers_ref", default="")
    pr.add_argument("--accuracy-json", dest="accuracy_json", default="")
    pr.add_argument("--status", default="finished")
    pr.add_argument("--source-task", dest="source_task", default="")
    pr.set_defaults(func=cmd_record)

    pq = sub.add_parser("query", help="Query best recipe for a model")
    pq.add_argument("--model", required=True)
    pq.add_argument("--scheme", default="")
    pq.add_argument("--all", action="store_true", help="Show all matches, ranked")
    pq.add_argument("--json", action="store_true")
    pq.set_defaults(func=cmd_query)

    pl = sub.add_parser("list", help="List all recipes")
    pl.add_argument("--json", action="store_true")
    pl.set_defaults(func=cmd_list)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
