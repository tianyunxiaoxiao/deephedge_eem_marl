"""Command-line interface for the synthetic study."""

from __future__ import annotations

import argparse
import json

from deephedge.config import StudyConfig
from deephedge.experiments import run_full_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the DeepHedge EEM-MARL synthetic study.")
    parser.add_argument("--output", type=str, default=None, help="Optional directory for summary.json.")
    parser.add_argument("--paths", type=int, default=StudyConfig.n_paths)
    parser.add_argument("--steps", type=int, default=StudyConfig.n_steps)
    parser.add_argument("--seed", type=int, default=StudyConfig.seed)
    args = parser.parse_args(argv)

    cfg = StudyConfig(n_paths=args.paths, n_steps=args.steps, seed=args.seed)
    report = run_full_study(cfg, output=args.output)
    print(json.dumps(report["acceptance"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
