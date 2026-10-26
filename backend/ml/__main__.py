"""CLI entry: python -m ml <command>."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Medical QA ML CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("preprocess", help="Prepare HealthCareMagic splits")
    p_prep.add_argument("--source", type=str, default=None)
    p_prep.add_argument("--out-dir", type=str, default=None)
    p_prep.add_argument("--seed", type=int, default=42)

    p_train = sub.add_parser("train", help="Fine-tune with LoRA/QLoRA")
    p_train.add_argument("--config", type=str, default="configs/train.yaml")

    p_eval = sub.add_parser("evaluate", help="Evaluate on held-out test set")
    p_eval.add_argument("--max-samples", type=int, default=50)
    p_eval.add_argument("--compare-base", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "preprocess":
        from ml.preprocess import run_preprocess

        stats = run_preprocess(source=args.source, out_dir=args.out_dir, seed=args.seed)
        print(stats)
        return 0
    if args.command == "train":
        from ml.train import run_train

        result = run_train(config_path=args.config)
        print(result)
        return 0
    if args.command == "evaluate":
        from ml.evaluate import run_evaluate

        result = run_evaluate(max_samples=args.max_samples, compare_base=args.compare_base)
        print(result)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
